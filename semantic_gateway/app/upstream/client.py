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
        """Map local or legacy/decommissioned model names to guaranteed active Groq API model identifiers."""
        m = (model_name or "").strip().lower()
        if m in ("llama-3.3-70b-versatile", "llama3-70b-8192", "mixtral-8x7b-32768"):
            return m
        if route_name == "fast_lane":
            return "llama3-8b-8192"
        return "llama-3.3-70b-versatile"

    def resolve_target(self, decision: RouteDecision) -> Tuple[str, str, str, str, str]:
        """
        Resolves (base_url, selected_model, api_key, target_route, fallback_reason).
        """
        if decision.target_route == "fast_lane":
            base_url = self.settings.FAST_MODEL_BASE_URL.strip()
            raw_model = decision.selected_model.strip() if decision.selected_model else self.settings.FAST_MODEL_NAME.strip()
            pref_key = self.settings.FAST_MODEL_API_KEY.strip()
            api_key = self.resolve_api_key(pref_key)

            if "groq.com" in base_url:
                model_name = self.map_groq_model_name(raw_model, "fast_lane")
            else:
                model_name = raw_model

            return (base_url, model_name, api_key, "fast_lane", "")

        # DEEP_LANE configuration
        deep_base_url = getattr(self.settings, "DEEP_MODEL_BASE_URL", "https://api.groq.com/openai/v1").strip()
        raw_deep_model = decision.selected_model.strip() if decision.selected_model else getattr(self.settings, "DEEP_MODEL_NAME", "llama-3.3-70b-versatile").strip()
        pref_deep_key = getattr(self.settings, "DEEP_MODEL_API_KEY", "").strip()
        deep_api_key = self.resolve_api_key(pref_deep_key)

        if "groq.com" in deep_base_url:
            deep_model_name = self.map_groq_model_name(raw_deep_model, "deep_lane")
        else:
            deep_model_name = raw_deep_model

        return (deep_base_url, deep_model_name, deep_api_key, "deep_lane", "")

    async def forward(
        self,
        request: ChatCompletionRequest,
        decision: RouteDecision
    ):
        base_url, model_name, api_key, route_name, fallback_reason = self.resolve_target(decision)

        if decision.target_route == "deep_lane":
            target_url = "https://api.groq.com/openai/v1/chat/completions"
            target_model = self.map_groq_model_name(self.settings.DEEP_MODEL_NAME.strip(), "deep_lane")
            api_key = self.resolve_api_key(self.settings.DEEP_MODEL_API_KEY.strip())
        else:
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

        try:
            print(f"[DEBUG GROQ] URL: {url} | Model: {clean_payload['model']}")
            response = await self.client.post(url, json=clean_payload, headers=headers)
            
            if response.status_code == 200:
                data = response.json()
                usage = data.get("usage", {})
                prompt_tokens = usage.get("prompt_tokens", 100)
                completion_tokens = usage.get("completion_tokens", 50)
                cost_saved = self.tracker.record_usage(route_name, prompt_tokens, completion_tokens)
                resp_headers["x-estimated-cost-saved"] = f"{cost_saved:.6f}"
                return JSONResponse(content=data, status_code=200, headers=resp_headers)

            print(f"[DEBUG GROQ ERROR]: {response.status_code} - {response.text}")

            # If 404 model_not_found or non-200 error, fallback across all known active Groq models
            fallback_models = ["llama3-8b-8192", "llama-3.3-70b-versatile", "llama-3.1-8b-instant", "llama3-70b-8192", "mixtral-8x7b-32768"]

            for alt_model in fallback_models:
                if alt_model == clean_payload['model']:
                    continue
                fb_payload = clean_payload.copy()
                fb_payload["model"] = alt_model
                print(f"[DEBUG GROQ FALLBACK] URL: {url} | Model: {alt_model}")
                fb_res = await self.client.post(url, json=fb_payload, headers=headers)
                if fb_res.status_code == 200:
                    data = fb_res.json()
                    usage = data.get("usage", {})
                    prompt_tokens = usage.get("prompt_tokens", 100)
                    completion_tokens = usage.get("completion_tokens", 50)
                    cost_saved = self.tracker.record_usage(route_name, prompt_tokens, completion_tokens)
                    resp_headers["x-estimated-cost-saved"] = f"{cost_saved:.6f}"
                    resp_headers["x-semantic-model"] = alt_model
                    resp_headers["X-Semantic-Router-Model"] = alt_model
                    return JSONResponse(content=data, status_code=200, headers=resp_headers)
                else:
                    print(f"[DEBUG GROQ FALLBACK ERROR]: {fb_res.status_code} - {fb_res.text}")

            err_msg = f"[Upstream Error (HTTP {response.status_code})]: {response.text}"

        except httpx.HTTPStatusError as e:
            print(f"[Groq Error {e.response.status_code}]: {e.response.text}")
            err_msg = f"[Upstream Error ({type(e).__name__})]: {str(e)}"
        except httpx.RequestError as e:
            print(f"[Groq Network Error]: {str(e)}")
            err_msg = f"[Upstream Error ({type(e).__name__})]: {str(e)}"
        except Exception as e:
            print(f"[Upstream General Error]: {repr(e)}")
            err_msg = f"[Upstream Error ({type(e).__name__})]: {str(e)}"

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
                    "message": {"role": "assistant", "content": err_msg},
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

        async def event_generator() -> AsyncGenerator[bytes, None]:
            try:
                print(f"[DEBUG GROQ] URL: {url} | Model: {clean_payload['model']}")
                async with self.client.stream("POST", url, json=clean_payload, headers=headers) as response:
                    if response.status_code >= 400:
                        err_bytes = await response.aread()
                        err_text = err_bytes.decode("utf-8", errors="replace")
                        print(f"[DEBUG GROQ ERROR]: {response.status_code} - {err_text}")

                        fallback_models = ["llama3-8b-8192", "llama-3.3-70b-versatile", "llama-3.1-8b-instant", "llama3-70b-8192", "mixtral-8x7b-32768"]

                        for alt_model in fallback_models:
                            if alt_model == clean_payload['model']:
                                continue
                            fb_payload = clean_payload.copy()
                            fb_payload["model"] = alt_model
                            print(f"[DEBUG GROQ FALLBACK] URL: {url} | Model: {alt_model}")
                            async with self.client.stream("POST", url, json=fb_payload, headers=headers) as fb_res:
                                if fb_res.status_code == 200:
                                    async for chunk in fb_res.aiter_bytes():
                                        yield chunk
                                    return
                                else:
                                    fb_err = await fb_res.aread()
                                    print(f"[DEBUG GROQ FALLBACK ERROR]: {fb_res.status_code} - {fb_err.decode('utf-8', errors='replace')}")

                        err_json = {
                            "choices": [{
                                "message": {
                                    "role": "assistant",
                                    "content": f"[Upstream Error (HTTP {response.status_code})]: {err_text}"
                                }
                            }]
                        }
                        yield f"data: {json.dumps(err_json)}\n\n".encode("utf-8")
                        return

                    async for chunk in response.aiter_bytes():
                        yield chunk
            except Exception as e:
                print(f"[Upstream General Error]: {repr(e)}")
                err_json = {"choices": [{"message": {"role": "assistant", "content": f"[Upstream Error ({type(e).__name__})]: {str(e)}"}}]}
                yield f"data: {json.dumps(err_json)}\n\n".encode("utf-8")

        return StreamingResponse(event_generator(), media_type="text/event-stream", headers=response_headers)
