"""Structured JSON query telemetry, live metrics counters, and optional OpenTelemetry."""

from contextlib import contextmanager
from dataclasses import asdict, dataclass
import json
import logging
from threading import Lock
from typing import Any

from opentelemetry import trace

from ..config import settings

telemetry_logger = logging.getLogger("querypilot.telemetry")
if not telemetry_logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(message)s"))
    telemetry_logger.addHandler(handler)
    telemetry_logger.setLevel(logging.INFO)

_tracing_initialized = False
_tracing_lock = Lock()


@dataclass
class QueryTelemetryEvent:
    """Structured telemetry record emitted for every query and worker job."""

    request_id: str
    user_id: int | None
    model_used: str | None
    fallback_triggered: bool
    self_correction_attempts: int
    cache_hit: bool
    retrieved_tables: list[str]
    sql_guard_status: str
    latency_ms: float
    input_tokens: int
    output_tokens: int


class MetricsRegistry:
    """Thread-safe in-memory counters for live API/worker observability."""

    def __init__(self) -> None:
        self._lock = Lock()
        self.reset()

    def reset(self) -> None:
        """Reset all counters to zero (primarily for isolated tests)."""
        with self._lock:
            self.total_queries = 0
            self.cache_hits = 0
            self.self_corrections = 0
            self.fallback_count = 0
            self.guard_rejections = 0
            self.error_count = 0
            self.total_input_tokens = 0
            self.total_output_tokens = 0

    def record(self, event: QueryTelemetryEvent) -> None:
        """Update counters from a completed query telemetry event."""
        with self._lock:
            self.total_queries += 1
            if event.cache_hit:
                self.cache_hits += 1
            if event.self_correction_attempts > 0 and not event.cache_hit:
                self.self_corrections += 1
            if event.fallback_triggered and not event.cache_hit:
                self.fallback_count += 1
            if event.sql_guard_status == "blocked":
                self.guard_rejections += 1
            if event.sql_guard_status in ("blocked", "error"):
                self.error_count += 1
            if not event.cache_hit:
                self.total_input_tokens += max(0, int(event.input_tokens))
                self.total_output_tokens += max(0, int(event.output_tokens))

    def snapshot(self) -> dict[str, Any]:
        """Return current counters and derived rates as a dictionary."""
        with self._lock:
            total = self.total_queries
            cache_rate = round(self.cache_hits / total, 4) if total else 0.0
            correction_rate = round(self.self_corrections / total, 4) if total else 0.0
            return {
                "total_queries": total,
                "cache_hits": self.cache_hits,
                "cache_hit_rate": cache_rate,
                "self_corrections": self.self_corrections,
                "self_correction_rate": correction_rate,
                "fallback_count": self.fallback_count,
                "guard_rejections": self.guard_rejections,
                "error_count": self.error_count,
                "total_input_tokens": self.total_input_tokens,
                "total_output_tokens": self.total_output_tokens,
            }


metrics_registry = MetricsRegistry()


def init_tracing() -> None:
    """Lazily configure OpenTelemetry OTLP exporter only when ENABLE_TRACING=true."""
    global _tracing_initialized
    if not settings.enable_tracing or _tracing_initialized:
        return

    with _tracing_lock:
        if _tracing_initialized:
            return
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor

        resource = Resource.create({"service.name": settings.otel_service_name})
        provider = TracerProvider(resource=resource)
        endpoint = settings.otel_exporter_otlp_endpoint.strip()
        if endpoint:
            from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter

            exporter = OTLPSpanExporter(endpoint=endpoint)
            provider.add_span_processor(BatchSpanProcessor(exporter))
        trace.set_tracer_provider(provider)
        _tracing_initialized = True


@contextmanager
def trace_query_span(name: str, attributes: dict[str, Any] | None = None):
    """Open an OpenTelemetry span when tracing is enabled, or no-op locally."""
    if not settings.enable_tracing:
        yield None
        return

    init_tracing()
    tracer = trace.get_tracer(settings.otel_service_name)
    with tracer.start_as_current_span(name) as span:
        if attributes:
            for key, val in attributes.items():
                if val is not None:
                    span.set_attribute(key, val)
        yield span


def emit_query_telemetry(event: QueryTelemetryEvent) -> None:
    """Record metrics counters and emit structured JSON telemetry log."""
    metrics_registry.record(event)
    telemetry_logger.info(json.dumps(asdict(event), default=str))
