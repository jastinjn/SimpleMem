"""OpenTelemetry tracing → self-hosted Langfuse (OTLP/HTTP).

A thin **context-manager facade** over OpenTelemetry so the rest of the codebase never
imports raw OTel. It owns three things:

1. Lifecycle — :func:`setup_telemetry` (wires an OTLP/HTTP exporter to Langfuse) and
   :func:`shutdown_telemetry` (flushes on exit). Called once from the app lifespan.
2. Span creation — :func:`trace` (root), :func:`span` (child), :func:`generation`
   (LLM child). Each wraps ``tracer.start_as_current_span`` and *yields* the span, so
   OTel's automatic parent/child nesting (which propagates through ``asyncio.gather``)
   is preserved.
3. Langfuse conventions — the ``langfuse.*`` / ``gen_ai.*`` attribute names live here
   and nowhere else.

**API vs SDK split:** instrumentation goes through :func:`trace`/:func:`span`/etc.,
which use only the OTel *API* (``opentelemetry.trace``). The *SDK*
(``opentelemetry.sdk.*``) is imported only in :func:`setup_telemetry`. When tracing is
not enabled, the API returns a no-op tracer, so every helper below is a cheap no-op —
tests and unconfigured deployments pay nothing.

Tracing is enabled only when ``OTEL_ENABLED`` is true *and* the Langfuse host + keys are
set. When enabled, spans carry raw content (dialogue, memory text, retrieved hits).
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from typing import Any

from opentelemetry import trace as otel_trace
from opentelemetry.trace import Span

logger = logging.getLogger(__name__)

_SERVICE_NAME = "marklymem"
_TRACER_NAME = "marklymem"

# Langfuse OTLP path (self-hosted, v3.22.0+). Appended to the configured host.
_LANGFUSE_OTEL_PATH = "/api/public/otel/v1/traces"

_provider: Any = None  # SDK TracerProvider, held for shutdown()
_enabled = False


# --------------------------------------------------------------------------- #
# Lifecycle
# --------------------------------------------------------------------------- #
def setup_telemetry(settings) -> bool:
    """Configure the global tracer provider to export to Langfuse over OTLP/HTTP.

    Returns ``True`` if tracing was enabled, ``False`` (no-op) otherwise. Safe to call
    when disabled or misconfigured — it logs and returns ``False`` rather than raising.
    """
    global _provider, _enabled

    if not settings.OTEL_ENABLED:
        return False
    if not (settings.LANGFUSE_HOST and settings.LANGFUSE_PUBLIC_KEY and settings.LANGFUSE_SECRET_KEY):
        logger.warning(
            "[telemetry] OTEL_ENABLED is true but LANGFUSE_HOST/PUBLIC_KEY/SECRET_KEY "
            "are not all set — tracing disabled."
        )
        return False

    # SDK imports are local so the package has no hard dependency on them at import time.
    from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor

    auth = base64.b64encode(
        f"{settings.LANGFUSE_PUBLIC_KEY}:{settings.LANGFUSE_SECRET_KEY}".encode()
    ).decode()
    endpoint = settings.LANGFUSE_HOST.rstrip("/") + _LANGFUSE_OTEL_PATH
    exporter = OTLPSpanExporter(
        endpoint=endpoint,
        headers={
            "Authorization": f"Basic {auth}",
            "x-langfuse-ingestion-version": "4",
        },
    )
    provider = TracerProvider(resource=Resource.create({"service.name": _SERVICE_NAME}))
    provider.add_span_processor(BatchSpanProcessor(exporter))
    otel_trace.set_tracer_provider(provider)

    _provider = provider
    _enabled = True
    logger.info("[telemetry] tracing enabled → %s", endpoint)
    return True


async def shutdown_telemetry() -> None:
    """Flush and shut down the batch span processor. Call once on app shutdown.

    ``BatchSpanProcessor.shutdown()`` blocks while flushing pending spans over the
    network. Running it in a thread prevents blocking the event loop during teardown.
    """
    global _provider, _enabled
    if _provider is not None:
        provider = _provider
        _provider = None
        _enabled = False
        await asyncio.to_thread(provider.shutdown)


def is_enabled() -> bool:
    return _enabled


# --------------------------------------------------------------------------- #
# Span creation (context managers)
# --------------------------------------------------------------------------- #
@contextmanager
def trace(
    name: str,
    *,
    user_id: str | None = "",
    namespace: str | None = "",
    session_id: str | None = "",
    input: Any = None,
    **attrs: Any,
) -> Iterator[Span]:
    """Open a **root** span for one logical operation and stamp Langfuse trace context.

    ``session_id`` is set as ``langfuse.session.id`` so Langfuse groups every trace
    sharing that id under one Session. ``input`` (when given) is recorded as the trace
    input. Yields the span so callers can attach result attributes on exit.
    """
    tracer = otel_trace.get_tracer(_TRACER_NAME)
    with tracer.start_as_current_span(name) as sp:
        sp.set_attribute("langfuse.trace.name", name)
        if session_id:
            sp.set_attribute("langfuse.session.id", session_id)
        if user_id:
            sp.set_attribute("langfuse.user.id", user_id)
        if namespace:
            sp.set_attribute("langfuse.trace.metadata.namespace", namespace)
        _apply(sp, attrs)
        if input is not None:
            set_input(sp, input)
        yield sp


@contextmanager
def span(name: str, **attrs: Any) -> Iterator[Span]:
    """Open a child span with arbitrary attributes."""
    tracer = otel_trace.get_tracer(_TRACER_NAME)
    with tracer.start_as_current_span(name) as sp:
        _apply(sp, attrs)
        yield sp


@contextmanager
def generation(name: str, *, model: str, **attrs: Any) -> Iterator[Span]:
    """Open a child span typed as a Langfuse *generation* (an LLM call).

    Set token usage on exit with :func:`record_usage` and input/output with
    :func:`set_input` / :func:`set_output`.
    """
    tracer = otel_trace.get_tracer(_TRACER_NAME)
    with tracer.start_as_current_span(name) as sp:
        sp.set_attribute("langfuse.observation.type", "generation")
        sp.set_attribute("gen_ai.request.model", model)
        _apply(sp, attrs)
        yield sp


# --------------------------------------------------------------------------- #
# Attribute / event helpers (operate on the given or current span)
# --------------------------------------------------------------------------- #
def set_input(span_obj: Span, value: Any) -> None:
    span_obj.set_attribute("langfuse.observation.input", _to_json(value))


def set_output(span_obj: Span, value: Any) -> None:
    span_obj.set_attribute("langfuse.observation.output", _to_json(value))


def record_usage(span_obj: Span, *, input_tokens: int | None, output_tokens: int | None) -> None:
    if input_tokens is not None:
        span_obj.set_attribute("gen_ai.usage.input_tokens", int(input_tokens))
    if output_tokens is not None:
        span_obj.set_attribute("gen_ai.usage.output_tokens", int(output_tokens))


def add_event(name: str, **attrs: Any) -> None:
    """Add an event to the currently-active span (no-op if none)."""
    current = otel_trace.get_current_span()
    current.add_event(name, attributes=_clean(attrs))


# --------------------------------------------------------------------------- #
# Internals
# --------------------------------------------------------------------------- #
def _apply(span_obj: Span, attrs: Mapping[str, Any]) -> None:
    for key, value in _clean(attrs).items():
        span_obj.set_attribute(key, value)


def _clean(attrs: Mapping[str, Any]) -> dict[str, Any]:
    """Drop None values; coerce non-primitive attribute values to strings.

    OTel accepts str/bool/int/float (and sequences thereof) as attribute values.
    """
    out: dict[str, Any] = {}
    for key, value in attrs.items():
        if value is None:
            continue
        if isinstance(value, (str, bool, int, float)):
            out[key] = value
        else:
            out[key] = str(value)
    return out


def _to_json(value: Any) -> str:
    if isinstance(value, str):
        return value
    try:
        return json.dumps(value, ensure_ascii=False, default=str)
    except (TypeError, ValueError):
        return str(value)
