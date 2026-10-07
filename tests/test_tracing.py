"""Tracing: spans for the request, the gateway pipeline and each provider attempt
should land in one trace."""

from __future__ import annotations

import pytest
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from llm_gateway.providers import FakeProvider
from tests.conftest import MakeHarness

_EXPORTER = InMemorySpanExporter()


@pytest.fixture
def spans() -> InMemorySpanExporter:
    provider = trace.get_tracer_provider()
    if not isinstance(provider, TracerProvider):
        provider = TracerProvider()
        provider.add_span_processor(SimpleSpanProcessor(_EXPORTER))
        trace.set_tracer_provider(provider)
    _EXPORTER.clear()
    return _EXPORTER


async def test_request_produces_single_trace_with_provider_attempts(
    make_harness: MakeHarness, spans: InMemorySpanExporter
) -> None:
    h = await make_harness({"primary": FakeProvider("primary", fail_status=503)})
    resp = await h.chat("trace me")
    assert resp.status_code == 200

    finished = spans.get_finished_spans()
    names = [s.name for s in finished]
    assert "gateway.chat" in names
    attempts = [s for s in finished if s.name == "provider.call"]
    # two failed attempts on primary, one success on secondary
    assert [a.attributes["gen_ai.system"] for a in attempts] == ["primary", "primary", "secondary"]
    assert len({s.context.trace_id for s in finished}) == 1


async def test_probes_are_not_traced(
    make_harness: MakeHarness, spans: InMemorySpanExporter
) -> None:
    h = await make_harness()
    await h.client.get("/metrics")
    await h.client.get("/healthz")
    assert spans.get_finished_spans() == ()
