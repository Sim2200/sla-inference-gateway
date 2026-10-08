"""OpenTelemetry setup shared by the gateway and the model server.

One tracer provider per process. Where spans go is decided by environment:
  OTEL_EXPORTER_OTLP_ENDPOINT   set -> OTLP/HTTP export (e.g. http://jaeger:4318)
  OTEL_TRACES_EXPORTER=console  -> print spans to stdout
  neither                       -> spans are created but not exported (tests add their own exporter)
Trace context travels between services in the W3C traceparent header: the FastAPI
instrumentation reads it on the way in, the httpx instrumentation writes it on the way out.
"""

from __future__ import annotations

import os

from opentelemetry import trace
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.instrumentation.httpx import HTTPXClientInstrumentor
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import (BatchSpanProcessor, ConsoleSpanExporter, SimpleSpanProcessor,
                                            SpanExporter)

_provider: TracerProvider | None = None


def provider(service_name: str = "unknown") -> TracerProvider:
    """Create the process-wide provider on first use; later calls return the same one."""
    global _provider
    if _provider is None:
        _provider = TracerProvider(resource=Resource.create({"service.name": service_name}))
        endpoint = os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT")
        if endpoint:
            from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
            _provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter(endpoint=endpoint.rstrip("/") + "/v1/traces")))
        elif os.environ.get("OTEL_TRACES_EXPORTER") == "console":
            _provider.add_span_processor(SimpleSpanProcessor(ConsoleSpanExporter()))
        trace.set_tracer_provider(_provider)
    return _provider


def add_exporter(exporter: SpanExporter) -> None:
    """Attach an extra exporter (tests use an in-memory one)."""
    provider().add_span_processor(SimpleSpanProcessor(exporter))


def instrument_app(app, service_name: str) -> None:
    FastAPIInstrumentor.instrument_app(app, tracer_provider=provider(service_name),
                                       excluded_urls="healthz,readyz,metrics")


def instrument_client(client) -> None:
    HTTPXClientInstrumentor().instrument_client(client, tracer_provider=provider())


def tracer(name: str):
    return trace.get_tracer(name, tracer_provider=provider())
