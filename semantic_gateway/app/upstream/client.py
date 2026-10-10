import os
import json
from typing import AsyncGenerator, Dict, Any, Tuple, Optional
import httpx
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi import HTTPException

try:
    from app.router.models import ChatCompletionRequest, RouteDecision
    from app.telemetry.tracker import TelemetryTracker
    from app.config import Settings, get_settings
except (ImportError, ModuleNotFoundError):
    from semantic_gateway.app.router.models import ChatCompletionRequest, RouteDecision
    from semantic_gateway.app.telemetry.tracker import TelemetryTracker
    from semantic_gateway.app.config import Settings, get_settings


def is_valid_api_key(key: Optional[str]) -> bool:
    """Check if an API key is present and not a dummy placeholder string."""
    if not key:
        return False
    k = key.strip().lower()
    if k in ("", "none", "null", "ollama", "your_openai_api_key_here", "your_gemini_api_key_here", "your_groq_api_key_here", "your_groq_or_openai_api_key_here"):
        return False
    if k.startswith("your_"):
        return False
    return True


DECOMMISSIONED_MODELS = {
    "llama3-8b-8192",
    "llama3-70b-8192",
    "gemma2-9b-it",
    "gemma-7b-it",
    "llama2-70b-4096",
    "llama-3.1-70b-versatile",
    "mixtral-8x7b-32768",
    "mixtral-8x7b",
    "gemma2:2b",
}

VALID_GROQ_MODELS = [
    "llama-3.1-8b-instant",
    "llama-3.3-70b-versatile",
    "llama-3.2-3b-preview",
    "llama-3.2-1b-preview",
]


def build_groq_candidate_models(route_name: str, requested_model: str) -> list[str]:
    req = requested_model.strip() if requested_model else ""
    if not req or req in DECOMMISSIONED_MODELS:
        req = "llama-3.1-8b-instant" if route_name == "fast_lane" else "llama-3.3-70b-versatile"

    candidates = [req]
    for m in VALID_GROQ_MODELS:
        if m not in candidates and m not in DECOMMISSIONED_MODELS:
            candidates.append(m)
    return candidates


class UpstreamClient:
    """Async client for forwarding chat completion requests to upstream LLM providers."""

    def __init__(self, tracker: TelemetryTracker, settings: Optional[Settings] = None):
        self.tracker = tracker
        self.settings = settings or get_settings()
        self.client = httpx.AsyncClient(timeout=httpx.Timeout(30.0, connect=10.0))

    async def close(self) -> None:
        await self.client.aclose()

    def resolve_api_key(self, preferred_key: str) -> str:
        """Dynamically find a valid API key from preferred key, GROQ_API_KEY, or environment variables."""
        if is_valid_api_key(preferred_key):
            return preferred_key.strip()

        candidates = [
            getattr(self.settings, "GROQ_API_KEY", ""),
            os.getenv("GROQ_API_KEY", ""),
            getattr(self.settings, "FRONTIER_MODEL_API_KEY", ""),
            os.getenv("FRONTIER_MODEL_API_KEY", ""),
            getattr(self.settings, "DEEP_MODEL_API_KEY", ""),
            os.getenv("DEEP_MODEL_API_KEY", ""),
            getattr(self.settings, "FAST_MODEL_API_KEY", ""),
            os.getenv("FAST_MODEL_API_KEY", ""),
            getattr(self.settings, "GEMINI_API_KEY", ""),
            os.getenv("GEMINI_API_KEY", ""),
        ]
        for key in candidates:
            if is_valid_api_key(key):
                return key.strip()
        return ""

    def map_groq_model_name(self, model_name: str, route_name: str) -> str:
        """Map any local or legacy model name to Groq's active production flagship model."""
        m = model_name.strip() if model_name else ""
        if not m or m in DECOMMISSIONED_MODELS:
            return "llama-3.1-8b-instant" if route_name == "fast_lane" else "llama-3.3-70b-versatile"
        return m

    def resolve_target(self, decision: RouteDecision) -> Tuple[str, str, str, str, str]:
        """
        Resolves (base_url, selected_model, api_key, target_route, fallback_reason).
        """
        if decision.target_route == "fast_lane":
            base_url = self.settings.FAST_MODEL_BASE_URL.strip()
            raw_model = decision.selected_model.strip() if decision.selected_model else self.settings.FAST_MODEL_NAME.strip()
            pref_key = self.settings.FAST_MODEL_API_KEY.strip()
            api_key = self.resolve_api_key(pref_key)
            model_name = self.map_groq_model_name(raw_model, "fast_lane") if "groq.com" in base_url else raw_model
            return (base_url, model_name, api_key, "fast_lane", "")

        # DEEP_LANE configuration
        deep_base_url = getattr(self.settings, "DEEP_MODEL_BASE_URL", "https://api.groq.com/openai/v1").strip()
        raw_deep_model = decision.selected_model.strip() if decision.selected_model else getattr(self.settings, "DEEP_MODEL_NAME", "llama-3.3-70b-versatile").strip()
        pref_deep_key = getattr(self.settings, "DEEP_MODEL_API_KEY", "").strip()
        deep_api_key = self.resolve_api_key(pref_deep_key)
        deep_model_name = self.map_groq_model_name(raw_deep_model, "deep_lane") if "groq.com" in deep_base_url else raw_deep_model

        return (deep_base_url, deep_model_name, deep_api_key, "deep_lane", "")

    async def forward(
        self,
        request: ChatCompletionRequest,
        decision: RouteDecision
    ):
        base_url, model_name, api_key, route_name, fallback_reason = self.resolve_target(decision)

        target_url = f"{base_url.rstrip('/')}/chat/completions"
        target_model = model_name

        headers = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"

        print(f"[SemanticRouter] Routing upstream to {route_name.upper()} ({target_model}) at {target_url}")

        payload = request.model_dump(exclude_none=True)

        if not request.stream:
            return await self._forward_sync(
                target_url, headers, payload, decision, target_model, route_name, api_key
            )
        else:
            return await self._forward_stream(
                target_url, headers, payload, decision, target_model, route_name, api_key
            )

    async def _forward_sync(
        self,
        url: str,
        headers: Dict[str, str],
        payload: Dict[str, Any],
        decision: RouteDecision,
        model_name: str,
        route_name: str,
        api_key: str
    ) -> JSONResponse:
        resp_headers = {
            "x-semantic-route": route_name,
            "x-semantic-model": model_name,
            "x-semantic-classification-ms": str(decision.classification_time_ms),
            "x-semantic-decision-source": decision.decision_source,
            "x-estimated-cost-saved": "0.000000",
            "X-Semantic-Router-Tier": "fast" if route_name == "fast_lane" else "deep",
            "X-Semantic-Router-Model": model_name,
            "X-Semantic-Router-Score": str(decision.similarity_score),
            "X-Semantic-Router-Latency-MS": str(decision.classification_time_ms),
            "X-Semantic-Router-Decision-Source": decision.decision_source
        }

        # Check if API key is present before attempting cloud API call
        if "api.groq.com" in url and not is_valid_api_key(api_key):
            notice_msg = (
                "[Configuration Notice]: GROQ_API_KEY is not set in Render environment variables. "
                "Please add GROQ_API_KEY to your Render Dashboard -> Environment Settings to enable live LLM responses."
            )
            return JSONResponse(
                status_code=200,
                headers=resp_headers,
                content={
                    "id": "chatcmpl-config-notice",
                    "object": "chat.completion",
                    "created": 1700000000,
                    "model": model_name,
                    "choices": [{
                        "index": 0,
                        "message": {"role": "assistant", "content": notice_msg},
                        "finish_reason": "stop"
                    }],
                    "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
                }
            )

        clean_payload = {
            "model": model_name.strip(),
            "messages": payload.get("messages", []),
            "temperature": payload.get("temperature", 0.7)
        }
        if "max_tokens" in payload and payload["max_tokens"] is not None:
            clean_payload["max_tokens"] = payload["max_tokens"]

        candidate_models = build_groq_candidate_models(route_name, clean_payload.get("model", ""))

        last_err = ""
        for alt_model in candidate_models:
            fb_payload = clean_payload.copy()
            fb_payload["model"] = alt_model
            try:
                print(f"[DEBUG GROQ SENDER] URL: {url} | Model: {alt_model}")
                response = await self.client.post(url, json=fb_payload, headers=headers)
                if response.status_code == 200:
                    data = response.json()
                    usage = data.get("usage", {})
                    prompt_tokens = usage.get("prompt_tokens", 100)
                    completion_tokens = usage.get("completion_tokens", 50)
                    cost_saved = self.tracker.record_usage(route_name, prompt_tokens, completion_tokens)
                    resp_headers["x-estimated-cost-saved"] = f"{cost_saved:.6f}"
                    resp_headers["x-semantic-model"] = alt_model
                    resp_headers["X-Semantic-Router-Model"] = alt_model
                    return JSONResponse(content=data, status_code=200, headers=resp_headers)
                else:
                    last_err = f"[Upstream Error (HTTP {response.status_code})]: {response.text}"
                    print(f"[DEBUG GROQ ERROR]: {response.status_code} - {response.text}")
            except Exception as e:
                last_err = f"[Upstream Error ({type(e).__name__})]: {str(e)}"
                print(f"[DEBUG GROQ EXCEPTION]: {repr(e)}")

        return JSONResponse(
            status_code=200,
            headers=resp_headers,
            content={
                "id": "chatcmpl-upstream-error",
                "object": "chat.completion",
                "created": 1700000000,
                "model": model_name,
                "choices": [{
                    "index": 0,
                    "message": {"role": "assistant", "content": last_err or "Upstream LLM endpoint temporary failure."},
                    "finish_reason": "stop"
                }],
                "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
            }
        )

    async def _forward_stream(
        self,
        url: str,
        headers: Dict[str, str],
        payload: Dict[str, Any],
        decision: RouteDecision,
        model_name: str,
        route_name: str,
        api_key: str
    ) -> StreamingResponse:
        estimated_saved = self.tracker.calculate_cost_saved(route_name, prompt_tokens=100, completion_tokens=50)

        response_headers = {
            "x-semantic-route": route_name,
            "x-semantic-model": model_name,
            "x-semantic-classification-ms": str(decision.classification_time_ms),
            "x-semantic-decision-source": decision.decision_source,
            "x-estimated-cost-saved": f"{estimated_saved:.6f}",
            "X-Semantic-Router-Tier": "fast" if route_name == "fast_lane" else "deep",
            "X-Semantic-Router-Model": model_name,
            "X-Semantic-Router-Score": str(decision.similarity_score),
            "X-Semantic-Router-Latency-MS": str(decision.classification_time_ms),
            "X-Semantic-Router-Decision-Source": decision.decision_source
        }

        if "api.groq.com" in url and not is_valid_api_key(api_key):
            notice_msg = (
                "[Configuration Notice]: GROQ_API_KEY is not set in Render environment variables. "
                "Please add GROQ_API_KEY to your Render Dashboard -> Environment Settings to enable live LLM responses."
            )
            async def notice_generator() -> AsyncGenerator[bytes, None]:
                err_json = {"choices": [{"message": {"role": "assistant", "content": notice_msg}}]}
                yield f"data: {json.dumps(err_json)}\n\n".encode("utf-8")
            return StreamingResponse(notice_generator(), media_type="text/event-stream", headers=response_headers)

        clean_payload = {
            "model": model_name.strip(),
            "messages": payload.get("messages", []),
            "temperature": payload.get("temperature", 0.7),
            "stream": True
        }
        if "max_tokens" in payload and payload["max_tokens"] is not None:
            clean_payload["max_tokens"] = payload["max_tokens"]

        candidate_models = build_groq_candidate_models(route_name, clean_payload.get("model", ""))

        async def event_generator() -> AsyncGenerator[bytes, None]:
            last_err = ""
            for alt_model in candidate_models:
                fb_payload = clean_payload.copy()
                fb_payload["model"] = alt_model
                try:
                    print(f"[DEBUG GROQ STREAM] URL: {url} | Model: {alt_model}")
                    async with self.client.stream("POST", url, json=fb_payload, headers=headers) as response:
                        if response.status_code == 200:
                            async for chunk in response.aiter_bytes():
                                yield chunk
                            return
                        else:
                            err_bytes = await response.aread()
                            last_err = err_bytes.decode("utf-8", errors="replace")
                            print(f"[DEBUG GROQ STREAM ERROR]: {response.status_code} - {last_err}")
                except Exception as e:
                    last_err = str(e)
                    print(f"[DEBUG GROQ STREAM EXCEPTION]: {repr(e)}")

            err_json = {
                "choices": [{
                    "message": {
                        "role": "assistant",
                        "content": f"[Upstream Error]: {last_err or 'Stream connection failed'}"
                    }
                }]
            }
            yield f"data: {json.dumps(err_json)}\n\n".encode("utf-8")

        return StreamingResponse(event_generator(), media_type="text/event-stream", headers=response_headers)
