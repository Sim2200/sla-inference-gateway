"""Trace context must flow gateway -> backend: the backend sees a traceparent with the gateway's trace id."""

from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from fastapi.testclient import TestClient

import tracing
from tests.test_gateway_app import make

EXPORTER = InMemorySpanExporter()
tracing.add_exporter(EXPORTER)


def test_traceparent_propagates_to_backend_with_same_trace_id():
    EXPORTER.clear()
    app, fake = make()
    with TestClient(app) as client:
        r = client.post("/predict", content=b"img")
    assert r.status_code == 200
    traceparent = fake.last_headers.get("traceparent")
    assert traceparent, "backend did not receive a traceparent header"
    header_trace_id = traceparent.split("-")[1]

    spans = {s.name: s for s in EXPORTER.get_finished_spans()}
    route = spans["gateway.route"]
    assert format(route.context.trace_id, "032x") == header_trace_id
    assert route.attributes["gateway.tier"] == "accurate"
    assert any(s.kind.name == "CLIENT" for s in spans.values()), "no httpx client span"
    assert any(s.kind.name == "SERVER" for s in spans.values()), "no FastAPI server span"
