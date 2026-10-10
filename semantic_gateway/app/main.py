import sys
from pathlib import Path
from contextlib import asynccontextmanager
from typing import List, Dict, Any, Optional

from fastapi import FastAPI, Depends, Request, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware

# Add package directory to sys.path so 'app.*' imports resolve cleanly in all serverless contexts
_PKG_DIR = Path(__file__).resolve().parent.parent
if str(_PKG_DIR) not in sys.path:
    sys.path.insert(0, str(_PKG_DIR))

try:
    from app.config import Settings, get_settings
    from app.router.models import ChatCompletionRequest, RouteDecision, MetricSummary
    from app.router.engine import SemanticEngine
    from app.upstream.client import UpstreamClient
    from app.telemetry.tracker import TelemetryTracker
except (ImportError, ModuleNotFoundError):
    from semantic_gateway.app.config import Settings, get_settings
    from semantic_gateway.app.router.models import ChatCompletionRequest, RouteDecision, MetricSummary
    from semantic_gateway.app.router.engine import SemanticEngine
    from semantic_gateway.app.upstream.client import UpstreamClient
    from semantic_gateway.app.telemetry.tracker import TelemetryTracker

# Shared runtime singletons
engine: Optional[SemanticEngine] = None
upstream_client: Optional[UpstreamClient] = None
telemetry_tracker: Optional[TelemetryTracker] = None

# Robust STATIC_DIR resolution checking project root, package root, and CWD
STATIC_DIR = Path(__file__).resolve().parent.parent.parent / "static"
if not (STATIC_DIR / "index.html").exists():
    STATIC_DIR = Path(__file__).resolve().parent.parent / "static"
if not (STATIC_DIR / "index.html").exists():
    STATIC_DIR = Path.cwd() / "static"
STATIC_DIR.mkdir(parents=True, exist_ok=True)


def get_tracker() -> TelemetryTracker:
    global telemetry_tracker
    if telemetry_tracker is None:
        telemetry_tracker = TelemetryTracker(settings=get_settings())
    return telemetry_tracker


def get_engine() -> SemanticEngine:
    global engine
    if engine is None:
        engine = SemanticEngine(settings=get_settings())
        engine.load_routes()
    return engine


def get_client() -> UpstreamClient:
    global upstream_client
    if upstream_client is None:
        upstream_client = UpstreamClient(tracker=get_tracker(), settings=get_settings())
    return upstream_client


@asynccontextmanager
async def lifespan(app: FastAPI):
    """FastAPI Lifespan manager: Initializes router engine & pre-computed vector index during startup."""
    global engine, upstream_client, telemetry_tracker

    settings = get_settings()
    telemetry_tracker = TelemetryTracker(settings=settings)
    upstream_client = UpstreamClient(tracker=telemetry_tracker, settings=settings)

    print(f"[SemanticGateway] Initializing SemanticEngine with '{settings.EMBEDDING_MODEL_NAME}'...")
    engine = SemanticEngine(settings=settings)
    engine.load_routes()
    print(f"[SemanticGateway] Route index built with {len(engine.anchor_metadata)} anchor utterances. Ready!")

    yield

    # Server shutdown
    if upstream_client:
        print("[SemanticGateway] Closing upstream client HTTP connections...")
        await upstream_client.close()


app = FastAPI(
    title="Semantic Gateway",
    description="High-performance, cost-optimizing LLM proxy gateway with dense sentence embedding routing",
    version="0.1.0",
    lifespan=lifespan,
)

# CORS middleware enabling all origins, credentials, methods, and headers
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
)

# Mount static files if directory exists
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/", response_class=HTMLResponse)
async def serve_dashboard():
    """Serve the modern Semantic Gateway Web Dashboard."""
    index_file = STATIC_DIR / "index.html"
    if not index_file.exists():
        cwd_index = Path("static/index.html")
        if cwd_index.exists():
            index_file = cwd_index
    if index_file.exists():
        return HTMLResponse(content=index_file.read_text(encoding="utf-8"))
    return HTMLResponse(
        content="""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Semantic Gateway</title>
    <style>body { background: #090d16; color: #f3f4f6; font-family: sans-serif; padding: 2rem; }</style>
</head>
<body>
    <h1>⚡ Semantic Gateway</h1>
    <p>Static dashboard file not found at <code>static/index.html</code>.</p>
</body>
</html>""",
        status_code=200
    )


@app.get("/health")
@app.get("/api/health")
async def health_check(
    router_engine: SemanticEngine = Depends(get_engine),
    settings: Settings = Depends(get_settings),
):
    """Health check endpoint for routing gateway and model index status."""
    return {
        "status": "healthy",
        "embedding_model": settings.EMBEDDING_MODEL_NAME,
        "indexed_anchors": len(router_engine.anchor_metadata) if router_engine else 0,
        "fast_lane_model": settings.FAST_MODEL_NAME,
        "deep_lane_model": settings.DEEP_MODEL_NAME,
    }


@app.get("/metrics", response_model=MetricSummary)
@app.get("/api/metrics", response_model=MetricSummary)
@app.get("/telemetry/metrics", response_model=MetricSummary)
async def get_metrics(tracker: TelemetryTracker = Depends(get_tracker)):
    """Telemetry metrics endpoint: latency, fast vs deep lane ratio, cumulative USD savings."""
    return tracker.get_summary()


@app.get("/api/telemetry")
@app.get("/telemetry")
async def get_telemetry_analytics(tracker: TelemetryTracker = Depends(get_tracker)):
    """
    Aggregate telemetry analytics endpoint returning live stats and last 15 routed items.
    """
    return tracker.get_telemetry_summary()


@app.get("/dashboard/stats")
@app.get("/api/dashboard/stats")
async def get_dashboard_stats(tracker: TelemetryTracker = Depends(get_tracker)):
    """Dashboard analytics endpoint returning real-time metrics overview."""
    summary = tracker.get_summary()
    return {
        "total_requests": summary.total_requests,
        "fast_lane_count": summary.fast_lane_count,
        "deep_lane_count": summary.deep_lane_count,
        "fast_lane_percentage": summary.fast_lane_percentage,
        "deep_lane_percentage": summary.deep_lane_percentage,
        "avg_classification_latency_ms": summary.avg_classification_latency_ms,
        "estimated_prompt_tokens_processed": summary.estimated_prompt_tokens_processed,
        "estimated_completion_tokens_processed": summary.estimated_completion_tokens_processed,
        "estimated_cumulative_dollar_savings": summary.estimated_cumulative_dollar_savings,
    }


@app.get("/v1/models")
@app.get("/api/v1/models")
async def list_models(settings: Settings = Depends(get_settings)):
    """OpenAI-compatible models listing endpoint."""
    return {
        "object": "list",
        "data": [
            {
                "id": settings.FAST_MODEL_NAME,
                "object": "model",
                "created": 1700000000,
                "owned_by": "semantic-gateway-fast-lane",
            },
            {
                "id": settings.FRONTIER_MODEL_NAME,
                "object": "model",
                "created": 1700000000,
                "owned_by": "semantic-gateway-deep-lane",
            },
            {
                "id": "auto",
                "object": "model",
                "created": 1700000000,
                "owned_by": "semantic-gateway-auto",
            },
        ],
    }


@app.post("/v1/chat/completions")
@app.post("/api/v1/chat/completions")
async def chat_completions(
    request: ChatCompletionRequest,
    router_engine: SemanticEngine = Depends(get_engine),
    client: UpstreamClient = Depends(get_client),
    tracker: TelemetryTracker = Depends(get_tracker),
):
    """
    OpenAI-compatible chat completion proxy endpoint.
    Intercepts user query, classifies complexity in <15ms,
    and forwards request asynchronously to optimal fast_lane or deep_lane LLM target.
    Injects x-semantic-route, x-semantic-model, x-semantic-classification-ms, x-estimated-cost-saved headers.
    """
    decision = None
    try:
        prompt = router_engine.extract_prompt(request.messages)
        decision = router_engine.classify(prompt)

        # Record routing decision telemetry
        tracker.record_routing(
            tier_or_route=decision.target_route,
            latency_ms=decision.classification_time_ms,
            prompt=prompt,
            model_name=decision.selected_model,
            score=decision.similarity_score,
            decision_source=decision.decision_source
        )

        # Forward to target upstream LLM provider
        return await client.forward(request, decision)
    except Exception as e:
        print(f"[SemanticGateway ERROR]: {repr(e)}")
        error_msg = f"[Upstream Error ({type(e).__name__})]: {str(e)}"
        resp_headers = {
            "x-semantic-route": decision.target_route if decision else "deep_lane",
            "x-semantic-model": decision.selected_model if decision else "error",
            "x-semantic-classification-ms": str(decision.classification_time_ms) if decision else "0.0",
            "x-semantic-decision-source": decision.decision_source if decision else "SEMANTIC_EMBEDDING",
            "x-estimated-cost-saved": "0.000000",
            "X-Semantic-Router-Tier": "fast" if decision and decision.target_route == "fast_lane" else "deep",
            "X-Semantic-Router-Model": decision.selected_model if decision else "error",
            "X-Semantic-Router-Score": str(decision.similarity_score) if decision else "0.0",
            "X-Semantic-Router-Latency-MS": str(decision.classification_time_ms) if decision else "0.0",
            "X-Semantic-Router-Decision-Source": decision.decision_source if decision else "SEMANTIC_EMBEDDING"
        }
        return JSONResponse(
            status_code=200,
            headers=resp_headers,
            content={
                "id": "chatcmpl-error-fallback",
                "object": "chat.completion",
                "created": 1700000000,
                "model": decision.selected_model if decision else "error",
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": error_msg
                        },
                        "finish_reason": "stop"
                    }
                ],
                "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
            }
        )
