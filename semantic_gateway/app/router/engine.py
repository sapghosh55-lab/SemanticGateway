import time
import re
import os
import hashlib
from typing import List, Tuple, Optional, Any, Dict
from pathlib import Path
import numpy as np
import yaml
import httpx

try:
    from app.config import Settings
    from app.router.models import RouteDecision, RouteDefinition
except (ImportError, ModuleNotFoundError):
    from semantic_gateway.app.config import Settings
    from semantic_gateway.app.router.models import RouteDecision, RouteDefinition


class SemanticEngine:
    """
    Lightweight Serverless Semantic Router Engine (No PyTorch / SentenceTransformer dependencies):
    Layer 1: Fast Heuristic & Complexity Pre-Filter (<1ms execution before embedding).
    Layer 2: Dense 384-dimensional Vector Cosine Matching using Hugging Face Inference API & NumPy.
    """

    def __init__(self, settings: Settings, model: Optional[Any] = None):
        self.settings = settings
        self.routes: List[RouteDefinition] = []
        self.anchor_embeddings: Optional[np.ndarray] = None  # Shape (N, 384) float32 normalized
        self.anchor_metadata: List[Tuple[str, str, str]] = []  # (route_name, target_model, utterance)
        self.fast_indices: List[int] = []
        self.deep_indices: List[int] = []
        self._is_indexed = False
        self.hf_client = httpx.Client(timeout=httpx.Timeout(4.0, connect=2.0))

    def load_routes(self, routes_path: Optional[str] = None) -> None:
        path_str = routes_path or self.settings.ROUTES_FILE
        path = Path(path_str)

        if not path.exists():
            candidates = [
                Path(__file__).resolve().parent.parent.parent.parent / path_str,
                Path(__file__).resolve().parent.parent.parent / path_str,
                Path(__file__).resolve().parent.parent / path_str,
                Path.cwd() / path_str,
            ]
            for candidate in candidates:
                if candidate.exists():
                    path = candidate
                    break

        if path.exists():
            with open(path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
        else:
            data = {
                "routes": [
                    {
                        "name": "fast_lane",
                        "description": "Fast lane for simple/lightweight queries",
                        "target_model": self.settings.FAST_MODEL_NAME,
                        "utterances": ["hello", "hi", "what is 2+2", "convert units", "what is the capital of france"]
                    },
                    {
                        "name": "deep_lane",
                        "description": "Deep lane for complex queries",
                        "target_model": self.settings.FRONTIER_MODEL_NAME,
                        "utterances": ["explain quantum physics", "design a system", "audit smart contract", "implement consensus algorithm"]
                    }
                ]
            }

        raw_routes = data.get("routes", [])
        self.routes = []

        all_utterances: List[str] = []
        self.anchor_metadata = []

        for r in raw_routes:
            r_name = r.get("name", "deep_lane")
            target_model = r.get("target_model") or (
                self.settings.FAST_MODEL_NAME if r_name == "fast_lane" else self.settings.FRONTIER_MODEL_NAME
            )
            utterances = r.get("utterances", [])

            self.routes.append(
                RouteDefinition(
                    name=r_name,
                    description=r.get("description", ""),
                    target_tier="fast" if r_name == "fast_lane" else "frontier",
                    utterances=utterances
                )
            )

            for utt in utterances:
                all_utterances.append(utt)
                self.anchor_metadata.append((r_name, target_model, utt))

        if not all_utterances:
            raise ValueError("No anchor utterances defined in routes configuration.")

        # Pre-calculate route index groups
        self.fast_indices = [i for i, meta in enumerate(self.anchor_metadata) if meta[0] == "fast_lane"]
        self.deep_indices = [i for i, meta in enumerate(self.anchor_metadata) if meta[0] == "deep_lane"]

        # Load precomputed 384-dimensional normalized anchor embeddings
        npy_path = Path(__file__).resolve().parent / "anchor_embeddings.npy"
        if npy_path.exists():
            try:
                loaded = np.load(npy_path).astype(np.float32)
                if loaded.shape[0] == len(all_utterances):
                    self.anchor_embeddings = loaded
                    self._is_indexed = True
                    return
            except Exception:
                pass

        # Fallback: compute or generate normalized anchor vectors
        self.anchor_embeddings = self._generate_embeddings(all_utterances)
        self._is_indexed = True

    def _generate_embeddings(self, texts: List[str]) -> np.ndarray:
        """Batch encode texts into normalized 384-dimensional embeddings."""
        vecs = [self._encode_query(t) for t in texts]
        return np.vstack(vecs).astype(np.float32)

    def extract_prompt(self, messages: list) -> str:
        """Extract the last user message text from chat messages list."""
        if not messages:
            return ""
        for msg in reversed(messages):
            role = getattr(msg, "role", None) or (msg.get("role") if isinstance(msg, dict) else "")
            if role == "user":
                content = getattr(msg, "content", None) or (msg.get("content") if isinstance(msg, dict) else "")
                if isinstance(content, str):
                    return content
                elif isinstance(content, list):
                    texts = []
                    for part in content:
                        if isinstance(part, dict) and part.get("type") == "text":
                            texts.append(part.get("text", ""))
                    return " ".join(texts)
        last_msg = messages[-1]
        content = getattr(last_msg, "content", None) or (last_msg.get("content") if isinstance(last_msg, dict) else "")
        return str(content)

    def _normalize_query(self, prompt: str) -> str:
        """Normalize query casing and strip excess punctuation before computing embeddings."""
        cleaned = prompt.strip().lower()
        cleaned = re.sub(r'[^\w\s\+\-\*\/\%\.]', ' ', cleaned)
        cleaned = re.sub(r'\s+', ' ', cleaned).strip()
        return cleaned or prompt.strip().lower()

    def _heuristic_prefilter(self, prompt: str) -> Optional[Tuple[str, str, float]]:
        """
        Layer 1: Fast Heuristic & Complexity Pre-Filter (Executes in <1ms before embedding).
        Returns (target_route, matched_reason, similarity_score) if matched, otherwise None.
        """
        stripped = prompt.strip()
        if not stripped:
            return ("fast_lane", "heuristic:empty_input", 1.0)

        words = stripped.split()
        word_count = len(words)

        # 1. High Complexity Guard:
        # Token length > 120 words
        if word_count > 120:
            return ("deep_lane", "heuristic:token_length_exceeded", 1.0)

        # Explicit multi-step reasoning signals and code generation keywords
        complexity_pattern = r'\b(implement|architect|byzantine|concurrency|smart\s+contract|dockerfile|debug\s+this\s+stacktrace)\b'
        if re.search(complexity_pattern, stripped, re.IGNORECASE):
            return ("deep_lane", "heuristic:high_complexity_keyword", 1.0)

        # 2. Fast Trivial Filter:
        # If input text (stripped) is <= 25 characters and matches conversational intents
        if len(stripped) <= 25:
            conversational_pattern = r'\b(hi|hii|hello|hey|bye|byebye|thanks|thank\s+you|ok|okay)\b'
            if re.search(conversational_pattern, stripped, re.IGNORECASE):
                return ("fast_lane", "heuristic:conversational_intent", 1.0)

        # 3. Pure Math / Arithmetic Filter:
        # If the stripped input contains only numbers, math operators, whitespace, or percentages
        if re.match(r'^[\d\s\+\-\*\/\(\)\.\=\%]+$', stripped) and re.search(r'[\d\+\-\*\/\=\%]', stripped):
            return ("fast_lane", "heuristic:pure_arithmetic", 1.0)

        # Starts with simple math terms like "calculate", "solve", "what is [number]"
        simple_math_pattern = r'^(calculate|solve|what\s+is\s+\d+)\b'
        if re.match(simple_math_pattern, stripped, re.IGNORECASE):
            return ("fast_lane", "heuristic:simple_math_intent", 1.0)

        return None

    def _encode_query(self, text: str) -> Tuple[Optional[np.ndarray], bool]:
        """
        Generate 384-dimensional query vector using Hugging Face Inference API.
        Returns (embedding_vector, True) if HF Inference API succeeds,
        or (None, False) if HF Inference API is unavailable/unauthenticated.
        """
        hf_token = (
            getattr(self.settings, "HF_TOKEN", "")
            or os.getenv("HF_TOKEN", "")
            or getattr(self.settings, "HUGGINGFACE_API_KEY", "")
            or os.getenv("HUGGINGFACE_API_KEY", "")
        )
        model_id = self.settings.EMBEDDING_MODEL_NAME or "sentence-transformers/all-MiniLM-L6-v2"
        if not model_id.startswith("sentence-transformers/") and "/" not in model_id:
            model_id = f"sentence-transformers/{model_id}"

        headers = {"Content-Type": "application/json"}
        if hf_token:
            headers["Authorization"] = f"Bearer {hf_token}"

        api_urls = [
            f"https://router.huggingface.co/hf-inference/models/{model_id}",
            f"https://api-inference.huggingface.co/models/{model_id}"
        ]

        for url in api_urls:
            try:
                resp = self.hf_client.post(url, json={"inputs": [text]}, headers=headers)
                if resp.status_code == 200:
                    data = resp.json()
                    raw = data[0] if isinstance(data, list) and len(data) > 0 and isinstance(data[0], list) else data
                    vec = np.array(raw, dtype=np.float32)
                    if vec.shape == (384,):
                        norm = float(np.linalg.norm(vec))
                        if norm > 0:
                            return ((vec / norm).astype(np.float32), True)
            except Exception:
                pass

        return (None, False)

    def classify(self, prompt: str) -> RouteDecision:
        """
        Hybrid Intent & Complexity Routing:
        Layer 1: Heuristic & Complexity Pre-Filter (<1ms).
        Layer 2: Adaptive Semantic Cosine Matching using dense embeddings.
        """
        start_time = time.perf_counter()

        # ==========================================
        # LAYER 1: Fast Heuristic & Complexity Pre-Filter (<1ms)
        # ==========================================
        prefilter_result = self._heuristic_prefilter(prompt)
        if prefilter_result is not None:
            target_route, matched_reason, score = prefilter_result
            elapsed_ms = (time.perf_counter() - start_time) * 1000.0

            if target_route == "fast_lane":
                selected_model = self.settings.FAST_MODEL_NAME
                base_url = self.settings.FAST_MODEL_BASE_URL
                api_key = self.settings.FAST_MODEL_API_KEY
            else:
                selected_model = getattr(self.settings, "FRONTIER_MODEL_NAME", None) or getattr(self.settings, "DEEP_MODEL_NAME", "llama-3.3-70b-versatile")
                base_url = getattr(self.settings, "FRONTIER_MODEL_BASE_URL", getattr(self.settings, "DEEP_MODEL_BASE_URL", "https://api.groq.com/openai/v1"))
                api_key = getattr(self.settings, "FRONTIER_MODEL_API_KEY", getattr(self.settings, "DEEP_MODEL_API_KEY", ""))

            return RouteDecision(
                target_route=target_route,
                selected_model=selected_model,
                base_url=base_url,
                api_key=api_key,
                similarity_score=round(score, 4),
                classification_time_ms=round(elapsed_ms, 3),
                decision_source="HEURISTIC_PREFILTER",
                matched_utterance=matched_reason
            )

        # ==========================================
        # LAYER 2: Adaptive Semantic Cosine Matching
        # ==========================================
        if not self._is_indexed or self.anchor_embeddings is None:
            raise RuntimeError("SemanticEngine is not initialized with routes.")

        # Normalize query casing and strip excess punctuation
        normalized_prompt = self._normalize_query(prompt)

        # Compute normalized 384-dim embedding for the query prompt via Hugging Face Inference API
        query_vec, is_hf = self._encode_query(normalized_prompt)

        if is_hf and query_vec is not None:
            # Vectorized Cosine Similarity via dot product against 384-dim normalized matrix
            similarities = np.dot(self.anchor_embeddings, query_vec)
        else:
            # High-precision Lexical & Sub-Word Semantic Vector against Anchor Utterances
            similarities = np.zeros(len(self.anchor_metadata), dtype=np.float32)
            q_norm = normalized_prompt.lower()
            q_tokens = set(re.findall(r'\w+', q_norm))
            q_grams = set(q_norm[i:i+3] for i in range(len(q_norm)-2)) if len(q_norm) >= 3 else set([q_norm])

            for idx, (r_name, t_model, utt) in enumerate(self.anchor_metadata):
                u_norm = self._normalize_query(utt).lower()
                if q_norm == u_norm:
                    similarities[idx] = 1.0
                    continue
                u_tokens = set(re.findall(r'\w+', u_norm))
                if not u_tokens or not q_tokens:
                    continue
                intersection = q_tokens & u_tokens
                union = q_tokens | u_tokens
                jaccard = len(intersection) / len(union) if union else 0.0

                u_grams = set(u_norm[i:i+3] for i in range(len(u_norm)-2)) if len(u_norm) >= 3 else set([u_norm])
                gram_overlap = len(q_grams & u_grams) / max(len(q_grams | u_grams), 1)

                similarities[idx] = max(jaccard * 0.7 + gram_overlap * 0.3, jaccard)

        # Compute separate max similarity scores for fast_lane and deep_lane
        fast_sims = similarities[self.fast_indices] if self.fast_indices else np.array([])
        deep_sims = similarities[self.deep_indices] if self.deep_indices else np.array([])

        sim_fast = float(np.max(fast_sims)) if len(fast_sims) > 0 else -1.0
        fast_best_local_idx = int(np.argmax(fast_sims)) if len(fast_sims) > 0 else -1

        sim_deep = float(np.max(deep_sims)) if len(deep_sims) > 0 else -1.0
        deep_best_local_idx = int(np.argmax(deep_sims)) if len(deep_sims) > 0 else -1

        threshold = self.settings.SIMILARITY_THRESHOLD
        word_count = len(prompt.strip().split())

        # Adaptive routing decision logic
        if sim_deep > sim_fast and (sim_deep >= threshold or sim_deep > (sim_fast + 0.05)):
            target_route = "deep_lane"
            best_idx = self.deep_indices[deep_best_local_idx] if deep_best_local_idx != -1 else 0
            best_score = sim_deep
        elif word_count < 8 and sim_fast >= 0.35:
            # Short query adaptive sensitivity (< 8 words and sim_fast >= 0.35)
            target_route = "fast_lane"
            best_idx = self.fast_indices[fast_best_local_idx] if fast_best_local_idx != -1 else 0
            best_score = sim_fast
        elif sim_fast >= threshold and sim_fast >= sim_deep:
            target_route = "fast_lane"
            best_idx = self.fast_indices[fast_best_local_idx] if fast_best_local_idx != -1 else 0
            best_score = sim_fast
        else:
            # Preserve deep_lane as fallback for high-entropy and unmatched inputs
            target_route = "deep_lane"
            best_idx = self.deep_indices[deep_best_local_idx] if deep_best_local_idx != -1 else (self.fast_indices[fast_best_local_idx] if fast_best_local_idx != -1 else 0)
            best_score = sim_deep if sim_deep > -1.0 else 0.0

        matched_route, route_model, matched_utt = self.anchor_metadata[best_idx]

        if target_route == "fast_lane":
            selected_model = self.settings.FAST_MODEL_NAME or route_model
            base_url = self.settings.FAST_MODEL_BASE_URL
            api_key = self.settings.FAST_MODEL_API_KEY
        else:
            selected_model = getattr(self.settings, "FRONTIER_MODEL_NAME", None) or getattr(self.settings, "DEEP_MODEL_NAME", None) or route_model
            base_url = getattr(self.settings, "FRONTIER_MODEL_BASE_URL", getattr(self.settings, "DEEP_MODEL_BASE_URL", "https://api.groq.com/openai/v1"))
            api_key = getattr(self.settings, "FRONTIER_MODEL_API_KEY", getattr(self.settings, "DEEP_MODEL_API_KEY", ""))

        elapsed_ms = (time.perf_counter() - start_time) * 1000.0

        return RouteDecision(
            target_route=target_route,
            selected_model=selected_model,
            base_url=base_url,
            api_key=api_key,
            similarity_score=round(best_score, 4),
            classification_time_ms=round(elapsed_ms, 3),
            decision_source="SEMANTIC_EMBEDDING",
            matched_utterance=matched_utt
        )

    # Alias method for route calls
    route = classify


# Alias class for backward compatibility
SemanticRouterEngine = SemanticEngine
