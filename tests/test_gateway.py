import pytest
from unittest.mock import AsyncMock
from httpx import AsyncClient, Response as HttpxResponse, Request as HttpxRequest, ASGITransport

try:
    from semantic_gateway.app.config import Settings
    from semantic_gateway.app.router.engine import SemanticEngine
    from semantic_gateway.app.router.models import ChatCompletionRequest, ChatMessage
    from semantic_gateway.app.telemetry.tracker import TelemetryTracker
    from semantic_gateway.app.upstream.client import UpstreamClient
    from semantic_gateway.app.main import app, get_engine, get_client, get_tracker, get_settings
except ImportError:
    from app.config import Settings
    from app.router.engine import SemanticEngine
    from app.router.models import ChatCompletionRequest, ChatMessage
    from app.telemetry.tracker import TelemetryTracker
    from app.upstream.client import UpstreamClient
    from app.main import app, get_engine, get_client, get_tracker, get_settings


@pytest.fixture(scope="module")
def settings():
    return Settings(
        SIMILARITY_THRESHOLD=0.45,
        FAST_MODEL_NAME="llama3-8b-8192",
        FRONTIER_MODEL_NAME="llama-3.3-70b-versatile",
        ROUTES_FILE="routes.yaml"
    )


@pytest.fixture(scope="module")
def router_engine(settings):
    engine = SemanticEngine(settings=settings)
    engine.load_routes(settings.ROUTES_FILE)
    engine.classify("warmup init prompt 1")
    engine.classify("warmup init prompt 2")
    return engine


@pytest.fixture(scope="module")
def tracker(settings):
    return TelemetryTracker(settings=settings)


def test_temperature_conversion_routes_to_fast_lane(router_engine):
    """Test that 'what is 25 degree in fahrenheit' strictly routes to fast_lane."""
    prompt = "what is 25 degree in fahrenheit"
    decision = router_engine.classify(prompt)

    assert decision.target_route == "fast_lane"
    assert decision.selected_model == "llama3-8b-8192"
    assert decision.classification_time_ms < 35.0


def test_unit_math_factual_queries_route_to_fast_lane(router_engine):
    """Test everyday math, unit conversions, and factual queries route to fast_lane."""
    queries = [
        "convert 100 celsius to fahrenheit",
        "how many kilometers in 5 miles",
        "calculate 15 percent of 80",
        "what is the capital of Japan?",
        "What is the capital of France?"
    ]
    for q in queries:
        decision = router_engine.classify(q)
        assert decision.target_route == "fast_lane", f"Query '{q}' should route to fast_lane but got '{decision.target_route}'"


def test_byzantine_consensus_routes_to_deep_lane(router_engine):
    """Test that complex system design queries route to deep_lane."""
    prompt = "Design a distributed consensus algorithm with Byzantine fault tolerance in C++"
    decision = router_engine.classify(prompt)

    assert decision.target_route == "deep_lane"
    assert decision.selected_model == "llama-3.3-70b-versatile"
    assert decision.classification_time_ms < 35.0


def test_high_complexity_queries_route_to_deep_lane(router_engine):
    """Test high-complexity tasks (C++ memory allocators, distributed systems, quantum algorithms, smart contracts) resolve to deep_lane."""
    queries = [
        "Design a custom C++ memory allocator with arena allocation",
        "Architect a distributed consensus protocol similar to Raft with leader election",
        "Explain Quantum Field Theory and derive the Dirac equation step by step",
        "Audit this Ethereum smart contract for reentrancy vulnerabilities and flash loan exploits"
    ]
    for q in queries:
        decision = router_engine.classify(q)
        assert decision.target_route == "deep_lane", f"Query '{q}' should route to deep_lane but got '{decision.target_route}'"


def test_telemetry_tracker(settings):
    tr = TelemetryTracker(settings=settings)
    tr.record_routing("fast_lane", 4.2)
    tr.record_routing("deep_lane", 7.8)
    tr.record_usage("fast_lane", 1000, 500)

    summary = tr.get_summary()
    assert summary.total_requests == 2
    assert summary.fast_lane_count == 1
    assert summary.deep_lane_count == 1
    assert summary.fast_lane_percentage == 50.0
    assert summary.deep_lane_percentage == 50.0


@pytest.mark.asyncio
async def test_health_endpoint(router_engine, tracker, settings):
    app.dependency_overrides[get_engine] = lambda: router_engine
    app.dependency_overrides[get_tracker] = lambda: tracker
    app.dependency_overrides[get_settings] = lambda: settings

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        res = await ac.get("/health")
        assert res.status_code == 200
        data = res.json()
        assert data["status"] == "healthy"
        assert data["indexed_anchors"] > 0

    app.dependency_overrides.clear()


@pytest.mark.asyncio
async def test_chat_completions_fast_lane_proxy(router_engine, tracker, settings):
    mock_request = HttpxRequest("POST", "http://localhost:11434/v1/chat/completions")
    mock_response = HttpxResponse(
        status_code=200,
        json={
            "id": "chatcmpl-mock-123",
            "object": "chat.completion",
            "created": 1700000000,
            "model": "gemma2:2b",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "77°F"},
                    "finish_reason": "stop"
                }
            ],
            "usage": {"prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 12}
        },
        request=mock_request
    )

    upstream_client_instance = UpstreamClient(tracker=tracker)
    upstream_client_instance.client.post = AsyncMock(return_value=mock_response)

    app.dependency_overrides[get_engine] = lambda: router_engine
    app.dependency_overrides[get_tracker] = lambda: tracker
    app.dependency_overrides[get_client] = lambda: upstream_client_instance
    app.dependency_overrides[get_settings] = lambda: settings

    payload = {
        "model": "auto",
        "messages": [{"role": "user", "content": "what is 25 degree in fahrenheit"}]
    }

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        res = await ac.post("/v1/chat/completions", json=payload)
        assert res.status_code == 200
        assert res.headers.get("x-semantic-route") == "fast_lane"
        assert res.headers.get("x-semantic-model") in (settings.FAST_MODEL_NAME, "gemma2:2b", "gemma2-9b-it")
        assert res.headers.get("x-semantic-decision-source") in ("HEURISTIC_PREFILTER", "SEMANTIC_EMBEDDING")
        data = res.json()
        assert data["choices"][0]["message"]["content"] == "77°F"

    app.dependency_overrides.clear()


def test_layer1_fast_trivial_conversational_filter(router_engine):
    """Test fast trivial conversational intents route directly to fast_lane via HEURISTIC_PREFILTER."""
    conversational_inputs = [
        "hi", "hii", "hello", "hey", "bye", "byebye", "thanks", "thank you", "ok", "okay",
        "hi!", "hello there", "bye bye"
    ]
    for text in conversational_inputs:
        decision = router_engine.classify(text)
        assert decision.target_route == "fast_lane", f"Input '{text}' failed to route to fast_lane"
        assert decision.decision_source == "HEURISTIC_PREFILTER", f"Input '{text}' was not classified by HEURISTIC_PREFILTER"
        assert decision.classification_time_ms < 5.0, f"Heuristic classification took too long: {decision.classification_time_ms}ms"


def test_layer1_pure_math_arithmetic_filter(router_engine):
    """Test pure arithmetic and simple math expressions route to fast_lane via HEURISTIC_PREFILTER."""
    math_inputs = [
        "8+532",
        "25 * 4",
        "100 / 5",
        "50 - 12",
        "(10 + 20) * 3",
        "15%",
        "what is 15 + 30",
        "calculate 15 percent of 80",
        "solve 100 * 5"
    ]
    for text in math_inputs:
        decision = router_engine.classify(text)
        assert decision.target_route == "fast_lane", f"Input '{text}' failed to route to fast_lane"
        assert decision.decision_source == "HEURISTIC_PREFILTER", f"Input '{text}' was not classified by HEURISTIC_PREFILTER"
        assert decision.classification_time_ms < 5.0, f"Heuristic classification took too long: {decision.classification_time_ms}ms"


def test_layer1_high_complexity_guard(router_engine):
    """Test high complexity keywords and long prompts flag directly for deep_lane via HEURISTIC_PREFILTER."""
    complexity_inputs = [
        "implement a lock-free ring buffer queue in C++",
        "architect a resilient multi-region database replication scheme",
        "byzantine fault tolerance consensus protocol",
        "concurrency primitives and actor systems in Rust",
        "audit this smart contract for reentrancy bugs",
        "write a multi-stage dockerfile for python fastapi",
        "debug this stacktrace: IndexError: list index out of range at line 42"
    ]
    for text in complexity_inputs:
        decision = router_engine.classify(text)
        assert decision.target_route == "deep_lane", f"Input '{text}' failed to route to deep_lane"
        assert decision.decision_source == "HEURISTIC_PREFILTER", f"Input '{text}' was not classified by HEURISTIC_PREFILTER"

    # Token length > 120 words
    long_prompt = "explain the system " * 125
    decision_long = router_engine.classify(long_prompt)
    assert decision_long.target_route == "deep_lane"
    assert decision_long.decision_source == "HEURISTIC_PREFILTER"


def test_layer2_adaptive_semantic_cosine_matching(router_engine):
    """Test queries that bypass Layer 1 use Layer 2 embedding cosine matching."""
    semantic_fast_queries = [
        "What is the capital of France?",
        "convert 10 km to miles",
        "usd to eur",
        "What time zone is London in?",
        "Convert hello world to uppercase."
    ]
    for q in semantic_fast_queries:
        decision = router_engine.classify(q)
        assert decision.target_route == "fast_lane", f"Query '{q}' failed to route to fast_lane"
        assert decision.decision_source == "SEMANTIC_EMBEDDING"

    # High entropy / unmatched query falls back to deep_lane
    unmatched_query = "zk99_unrelated_random_token_string_entropy_abc_123"
    decision_unmatched = router_engine.classify(unmatched_query)
    assert decision_unmatched.target_route == "deep_lane"
    assert decision_unmatched.decision_source == "SEMANTIC_EMBEDDING"


def test_telemetry_tracker_decision_source(settings):
    """Test TelemetryTracker correctly logs decision sources."""
    tr = TelemetryTracker(settings=settings)
    tr.record_routing("fast_lane", 0.5, prompt="hi", decision_source="HEURISTIC_PREFILTER")
    tr.record_routing("fast_lane", 12.0, prompt="What is the capital of France?", decision_source="SEMANTIC_EMBEDDING")
    tr.record_routing("deep_lane", 0.4, prompt="implement raft", decision_source="HEURISTIC_PREFILTER")

    summary = tr.get_summary()
    assert summary.total_requests == 3
    assert summary.fast_lane_count == 2
    assert summary.deep_lane_count == 1
    assert summary.heuristic_prefilter_count == 2
    assert summary.semantic_embedding_count == 1

    telemetry = tr.get_telemetry_summary()
    assert telemetry["heuristic_prefilter_count"] == 2
    assert telemetry["semantic_embedding_count"] == 1
    assert telemetry["recent_history"][0]["decision_source"] == "HEURISTIC_PREFILTER"
    assert telemetry["recent_history"][1]["decision_source"] == "SEMANTIC_EMBEDDING"

