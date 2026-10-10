import time
import threading
from collections import deque
from typing import Deque, List, Dict, Any, Optional
try:
    from app.router.models import MetricSummary
    from app.config import Settings, get_settings
except (ImportError, ModuleNotFoundError):
    from semantic_gateway.app.router.models import MetricSummary
    from semantic_gateway.app.config import Settings, get_settings

_default_tracker: Optional["TelemetryTracker"] = None


class TelemetryTracker:
    """Thread-safe telemetry & metric recorder for SemanticRouter gateway."""

    def __init__(self, settings: Settings):
        global _default_tracker
        self._settings = settings
        self._lock = threading.Lock()
        self.total_requests: int = 0
        self.fast_lane_count: int = 0
        self.deep_lane_count: int = 0
        self.total_prompt_tokens: int = 0
        self.total_completion_tokens: int = 0
        self.cumulative_savings_usd: float = 0.0
        self.total_classification_latency_ms: float = 0.0
        self.heuristic_prefilter_count: int = 0
        self.semantic_embedding_count: int = 0
        self._recent_latencies: Deque[float] = deque(maxlen=500)
        self._recent_history: Deque[Dict[str, Any]] = deque(maxlen=100)
        _default_tracker = self

    def record_routing(
        self,
        tier_or_route: str,
        latency_ms: float,
        prompt: str = "",
        model_name: str = "",
        score: float = 0.0,
        cost_saved: float = 0.0,
        decision_source: str = "SEMANTIC_EMBEDDING"
    ) -> None:
        with self._lock:
            self.total_requests += 1
            is_fast = tier_or_route in ("fast", "fast_lane")
            if is_fast:
                self.fast_lane_count += 1
            else:
                self.deep_lane_count += 1

            if decision_source == "HEURISTIC_PREFILTER":
                self.heuristic_prefilter_count += 1
            else:
                self.semantic_embedding_count += 1

            self.total_classification_latency_ms += latency_ms
            self._recent_latencies.append(latency_ms)

            # Record item in recent history
            history_item = {
                "id": self.total_requests,
                "timestamp": time.strftime("%H:%M:%S"),
                "prompt": prompt,
                "route": "fast_lane" if is_fast else "deep_lane",
                "target_route": "fast_lane" if is_fast else "deep_lane",
                "model": model_name or ("llama-3.1-8b-instant" if is_fast else "llama-3.3-70b-versatile"),
                "selected_model": model_name or ("llama-3.1-8b-instant" if is_fast else "llama-3.3-70b-versatile"),
                "classification_latency_ms": round(latency_ms, 3),
                "similarity_score": round(score, 4),
                "decision_source": decision_source,
                "cost_saved_usd": round(cost_saved, 6),
            }
            self._recent_history.appendleft(history_item)

    def calculate_cost_saved(self, route_name: str, prompt_tokens: int = 100, completion_tokens: int = 50) -> float:
        """Calculate estimated USD savings for routing a request to fast_lane vs deep_lane."""
        if route_name not in ("fast", "fast_lane"):
            return 0.0
        frontier_cost = (
            (prompt_tokens / 1000.0) * self._settings.FRONTIER_MODEL_COST_PER_1K_INPUT
            + (completion_tokens / 1000.0) * self._settings.FRONTIER_MODEL_COST_PER_1K_OUTPUT
        )
        fast_cost = (
            (prompt_tokens / 1000.0) * self._settings.FAST_MODEL_COST_PER_1K_INPUT
            + (completion_tokens / 1000.0) * self._settings.FAST_MODEL_COST_PER_1K_OUTPUT
        )
        return max(0.0, frontier_cost - fast_cost)

    def record_usage(self, route_name: str, prompt_tokens: int, completion_tokens: int) -> float:
        with self._lock:
            self.total_prompt_tokens += prompt_tokens
            self.total_completion_tokens += completion_tokens
            saved = self.calculate_cost_saved(route_name, prompt_tokens, completion_tokens)
            self.cumulative_savings_usd += saved
            if self._recent_history and self._recent_history[0]["cost_saved_usd"] == 0.0:
                self._recent_history[0]["cost_saved_usd"] = round(saved, 6)
            return saved

    def get_summary(self) -> MetricSummary:
        with self._lock:
            total = self.total_requests
            fast_pct = (self.fast_lane_count / total * 100.0) if total > 0 else 0.0
            deep_pct = (self.deep_lane_count / total * 100.0) if total > 0 else 0.0
            avg_lat = (
                (sum(self._recent_latencies) / len(self._recent_latencies))
                if self._recent_latencies
                else 0.0
            )

            # If exact token usage hasn't updated cumulative_savings_usd yet, estimate based on counts
            savings = self.cumulative_savings_usd
            if savings == 0.0 and self.fast_lane_count > 0:
                fast_ratio = (self.fast_lane_count / total) if total > 0 else 0.0
                fast_prompt_tokens = self.total_prompt_tokens * fast_ratio
                fast_completion_tokens = self.total_completion_tokens * fast_ratio
                frontier_cost = (
                    (fast_prompt_tokens / 1000.0) * self._settings.FRONTIER_MODEL_COST_PER_1K_INPUT
                    + (fast_completion_tokens / 1000.0) * self._settings.FRONTIER_MODEL_COST_PER_1K_OUTPUT
                )
                fast_cost = (
                    (fast_prompt_tokens / 1000.0) * self._settings.FAST_MODEL_COST_PER_1K_INPUT
                    + (fast_completion_tokens / 1000.0) * self._settings.FAST_MODEL_COST_PER_1K_OUTPUT
                )
                savings = max(0.0, frontier_cost - fast_cost)

            return MetricSummary(
                total_requests=total,
                fast_lane_count=self.fast_lane_count,
                deep_lane_count=self.deep_lane_count,
                fast_lane_percentage=round(fast_pct, 2),
                deep_lane_percentage=round(deep_pct, 2),
                avg_classification_latency_ms=round(avg_lat, 3),
                estimated_prompt_tokens_processed=self.total_prompt_tokens,
                estimated_completion_tokens_processed=self.total_completion_tokens,
                estimated_cumulative_dollar_savings=round(savings, 6),
                heuristic_prefilter_count=self.heuristic_prefilter_count,
                semantic_embedding_count=self.semantic_embedding_count,
            )

    def get_telemetry_summary(self) -> Dict[str, Any]:
        """Aggregate telemetry analytics summary containing stats and recent routing history (last 15 items)."""
        with self._lock:
            total = self.total_requests
            fast_pct = (self.fast_lane_count / total * 100.0) if total > 0 else 0.0
            avg_lat = (
                (sum(self._recent_latencies) / len(self._recent_latencies))
                if self._recent_latencies
                else 0.0
            )
            savings = self.cumulative_savings_usd
            recent_items = list(self._recent_history)[:15]

            return {
                "total_requests": total,
                "fast_lane_count": self.fast_lane_count,
                "deep_lane_count": self.deep_lane_count,
                "fast_lane_ratio": round(fast_pct, 2),
                "total_cost_saved_usd": round(savings, 6),
                "avg_classification_latency_ms": round(avg_lat, 3),
                "heuristic_prefilter_count": self.heuristic_prefilter_count,
                "semantic_embedding_count": self.semantic_embedding_count,
                "recent_history": recent_items,
            }


def get_telemetry_summary(tracker: Optional[TelemetryTracker] = None) -> Dict[str, Any]:
    """Expose helper function returning aggregate telemetry summary."""
    t = tracker or _default_tracker
    if t is not None:
        return t.get_telemetry_summary()
    return {
        "total_requests": 0,
        "fast_lane_count": 0,
        "deep_lane_count": 0,
        "fast_lane_ratio": 0.0,
        "total_cost_saved_usd": 0.0,
        "avg_classification_latency_ms": 0.0,
        "recent_history": []
    }
