#!/usr/bin/env python3
#
# Copyright 2026 Google LLC
# SPDX-License-Identifier: Apache-2.0

"""CLI collector URL (handshake / notify) turns on the Developer UI Traces tab."""

from __future__ import annotations

import threading
from collections.abc import Generator
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any

import pytest
from genkit_otel import OtelInstrumentation
from httpx import ASGITransport, AsyncClient
from opentelemetry import trace as trace_api
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from genkit import ActionKind, Genkit
from genkit._core._action import Action
from genkit._core._environment import GENKIT_ENV
from genkit._core._reflection import create_reflection_asgi_app
from genkit._core._reflection_v2 import ReflectionServerV2
from genkit._core._registry import Registry
from genkit._core._telemetry._instrumentation import (
    instrumentations,
    is_instrumented_by,
    parent_path_context,
    reset_instrumentation,
)
from genkit._core._telemetry._log_exporter import reset_log_export
from genkit._core._telemetry.http import (
    GenkitBuiltinInstrumentation,
    flush_direct_http_instrumentations,
)
from genkit.telemetry import configure_instrumentation


def _hex_id(value: str, length: int) -> bool:
    return len(value) == length and all(c in '0123456789abcdef' for c in value)


def _flush_exporters_in_provider(provider: TracerProvider) -> None:
    active = getattr(provider, '_active_span_processor', None)
    if active is None:
        return
    processors = getattr(active, '_span_processors', [active])
    for proc in processors:
        exp = getattr(proc, 'span_exporter', None) or getattr(proc, 'exporter', None)
        if exp is not None and hasattr(exp, 'force_flush'):
            exp.force_flush()


def _force_flush() -> None:
    flush_direct_http_instrumentations()
    for inst in instrumentations:
        if isinstance(inst, OtelInstrumentation) and inst._tracer_provider is not None:
            inst._tracer_provider.force_flush()
            _flush_exporters_in_provider(inst._tracer_provider)
    provider = trace_api.get_tracer_provider()
    if isinstance(provider, TracerProvider):
        provider.force_flush()
        _flush_exporters_in_provider(provider)


async def _joke() -> str:
    return 'Why did the cat cross the road?'


def _hang_exporter(exporter: InMemorySpanExporter) -> None:
    provider = trace_api.get_tracer_provider()
    assert isinstance(provider, TracerProvider)
    provider.add_span_processor(SimpleSpanProcessor(exporter))


def _handshake_server() -> ReflectionServerV2:
    return ReflectionServerV2(Registry(), 'ws://127.0.0.1:1')


def _start_collector() -> tuple[HTTPServer, list[str]]:
    received: list[str] = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:  # noqa: N802
            n = int(self.headers.get('Content-Length', '0'))
            received.append(self.rfile.read(n).decode())
            self.send_response(200)
            self.end_headers()

        def log_message(self, format: str, *args: Any) -> None:
            return

    server = HTTPServer(('127.0.0.1', 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server, received


@pytest.fixture(autouse=True)
def _isolate_telemetry(monkeypatch: pytest.MonkeyPatch) -> Generator[None, None, None]:
    """Each test starts with no providers, unset collector env, and its own tracer."""
    reset_instrumentation()
    reset_log_export()
    monkeypatch.delenv(GENKIT_ENV, raising=False)
    monkeypatch.delenv('GENKIT_TELEMETRY_SERVER', raising=False)
    monkeypatch.setattr(Genkit, '_start_reflection_background', lambda self: None)
    isolated = TracerProvider()
    monkeypatch.setattr(trace_api, 'get_tracer_provider', lambda: isolated)
    monkeypatch.setattr(trace_api, 'set_tracer_provider', lambda _provider: None)
    path_token = parent_path_context.set('')
    try:
        yield
    finally:
        parent_path_context.reset(path_token)
        reset_instrumentation()
        reset_log_export()
        isolated.shutdown()


@pytest.mark.asyncio
async def test_handshake_url_in_dev_turns_tracing_on(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """UI-only genkit start: handshake URL turns tracing on so the Traces tab fills."""
    monkeypatch.setenv(GENKIT_ENV, 'dev')
    server, posts = _start_collector()
    url = f'http://127.0.0.1:{server.server_address[1]}'
    try:
        Genkit()
        assert not is_instrumented_by(GenkitBuiltinInstrumentation)

        _handshake_server().apply_handshake_telemetry(url)
        action = Action(name='joke', kind=ActionKind.FLOW, fn=_joke)
        result = await action.run()
        _force_flush()

        assert is_instrumented_by(GenkitBuiltinInstrumentation)
        assert not is_instrumented_by(OtelInstrumentation)
        assert _hex_id(result.trace_id, 32)
        assert _hex_id(result.span_id, 16)
        assert posts
    finally:
        server.shutdown()


@pytest.mark.asyncio
async def test_stale_collector_env_does_not_override_handshake_url(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Today's genkit start handshake URL wins over a GENKIT_TELEMETRY_SERVER already in the shell."""
    stale_server, stale_posts = _start_collector()
    live_server, live_posts = _start_collector()
    stale_url = f'http://127.0.0.1:{stale_server.server_address[1]}'
    live_url = f'http://127.0.0.1:{live_server.server_address[1]}'
    try:
        monkeypatch.setenv('GENKIT_TELEMETRY_SERVER', stale_url)

        Genkit()
        _handshake_server().apply_handshake_telemetry(live_url)
        action = Action(name='joke', kind=ActionKind.FLOW, fn=_joke)
        result = await action.run()
        _force_flush()

        assert is_instrumented_by(GenkitBuiltinInstrumentation)
        assert _hex_id(result.trace_id, 32)
        assert live_posts
        assert not stale_posts
    finally:
        stale_server.shutdown()
        live_server.shutdown()


@pytest.mark.asyncio
async def test_handshake_is_noop_when_genkit_start_already_wired_the_collector(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """genkit start -- python: Genkit() already wired the collector; handshake is a no-op."""
    monkeypatch.setenv(GENKIT_ENV, 'dev')
    monkeypatch.setenv('GENKIT_TELEMETRY_SERVER', 'http://127.0.0.1:4033')

    Genkit()
    before = list(instrumentations)
    _handshake_server().apply_handshake_telemetry('http://127.0.0.1:4041')

    assert is_instrumented_by(GenkitBuiltinInstrumentation)
    assert instrumentations == before


@pytest.mark.asyncio
async def test_notify_url_in_dev_turns_tracing_on(monkeypatch: pytest.MonkeyPatch) -> None:
    """Default genkit start POSTs /api/notify; that turns tracing on like v2 handshake."""
    monkeypatch.setenv(GENKIT_ENV, 'dev')
    app = create_reflection_asgi_app(Registry())
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url='http://test') as client:
        response = await client.post('/api/notify', json={'telemetryServerUrl': 'http://127.0.0.1:4041'})
    assert response.status_code == 200

    action = Action(name='joke', kind=ActionKind.FLOW, fn=_joke)
    result = await action.run()

    assert is_instrumented_by(GenkitBuiltinInstrumentation)
    assert _hex_id(result.trace_id, 32)


@pytest.mark.asyncio
async def test_cloud_already_on_still_posts_handshake_traces_to_developer_ui(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Cloud spans are already on; the handshake URL still fills the Developer UI Traces tab."""
    monkeypatch.setenv(GENKIT_ENV, 'dev')
    server, posts = _start_collector()
    url = f'http://127.0.0.1:{server.server_address[1]}'
    try:
        _hang_exporter(InMemorySpanExporter())
        configure_instrumentation(OtelInstrumentation())
        assert is_instrumented_by(OtelInstrumentation)

        _handshake_server().apply_handshake_telemetry(url)
        action = Action(name='joke', kind=ActionKind.FLOW, fn=_joke)
        result = await action.run()
        _force_flush()

        assert is_instrumented_by(GenkitBuiltinInstrumentation)
        assert _hex_id(result.trace_id, 32)
        assert posts
    finally:
        server.shutdown()


@pytest.mark.asyncio
async def test_empty_handshake_url_leaves_trace_ids_empty() -> None:
    """An empty collector URL on notify/handshake does not turn tracing on."""
    _handshake_server().apply_handshake_telemetry('')
    action = Action(name='joke', kind=ActionKind.FLOW, fn=_joke)
    result = await action.run()

    assert result.trace_id == ''
    assert result.span_id == ''
    assert not is_instrumented_by(GenkitBuiltinInstrumentation)


@pytest.mark.asyncio
async def test_second_handshake_does_not_add_another_collector(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A second notify URL does not stack another Developer UI poster."""
    monkeypatch.setenv(GENKIT_ENV, 'dev')
    first = _start_collector()
    second = _start_collector()
    first_server, first_posts = first
    second_server, second_posts = second
    try:
        _handshake_server().apply_handshake_telemetry(f'http://127.0.0.1:{first_server.server_address[1]}')
        _handshake_server().apply_handshake_telemetry(f'http://127.0.0.1:{second_server.server_address[1]}')
        action = Action(name='joke', kind=ActionKind.FLOW, fn=_joke)
        result = await action.run()
        _force_flush()

        builtins = [i for i in instrumentations if isinstance(i, GenkitBuiltinInstrumentation)]
        assert len(builtins) == 1
        assert _hex_id(result.trace_id, 32)
        assert first_posts
        assert not second_posts
    finally:
        first_server.shutdown()
        second_server.shutdown()
