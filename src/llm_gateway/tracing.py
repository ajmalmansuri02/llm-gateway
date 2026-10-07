"""OpenTelemetry tracing.

When ``OTEL_EXPORTER_OTLP_ENDPOINT`` is set, spans are exported over OTLP/HTTP
(to the collector in docker-compose, which forwards them to Jaeger). Otherwise
the API's no-op tracer is used and tracing costs nearly nothing.
"""

from __future__ import annotations

import logging

from opentelemetry import trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor

log = logging.getLogger(__name__)

tracer = trace.get_tracer("llm_gateway")

_configured = False


def setup_tracing(endpoint: str | None, service_name: str) -> TracerProvider | None:
    global _configured
    if not endpoint or _configured:
        return None
    from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter

    provider = TracerProvider(resource=Resource.create({"service.name": service_name}))
    exporter = OTLPSpanExporter(endpoint=f"{endpoint.rstrip('/')}/v1/traces")
    provider.add_span_processor(BatchSpanProcessor(exporter))
    trace.set_tracer_provider(provider)
    _configured = True
    log.info("tracing enabled", extra={"otlp_endpoint": endpoint})
    return provider
