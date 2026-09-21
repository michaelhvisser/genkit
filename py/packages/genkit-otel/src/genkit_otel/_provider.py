# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0

"""Built-in OpenTelemetry Instrumentation provider."""

from __future__ import annotations

import asyncio
import os
import traceback
from collections.abc import Mapping, Sequence
from contextlib import AbstractContextManager
from typing import Any, TypeVar

from opentelemetry import trace as trace_api
from opentelemetry.context import Context
from opentelemetry.instrumentation.logging import LoggingInstrumentor
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SpanExporter
from opentelemetry.trace import Link, NoOpTracer, NoOpTracerProvider, ProxyTracerProvider, Span, SpanKind, StatusCode
from opentelemetry.util import types

from genkit._core._environment import is_dev_environment
from genkit._core._error import GenkitError, GenkitInterrupt
from genkit._core._logger import get_logger
from genkit._core._telemetry._attrs import Attr, State, metadata_key
from genkit._core._telemetry._default_exporter import create_span_processor
from genkit._core._telemetry._path import build_path
from genkit._core._telemetry.instrumentation import (
    SpanMetadata,
    SpanNext,
    configure_instrumentation,
    instrumentations,
    is_instrumented_by,
    parent_path_context,
    start_attributes,
    to_json_attr,
)

logger = get_logger(__name__)

T = TypeVar('T')


class OtelSpanContext:
    """SpanContext backed by an OpenTelemetry span."""

    def __init__(self, span: trace_api.Span) -> None:
        self._span = span
        self._output: object | None = None
        self._output_set = False
        self._state: str | None = None

    @property
    def trace_id(self) -> str:
        ctx = self._span.get_span_context()
        if ctx is None or not ctx.trace_id:
            return ''
        return format(ctx.trace_id, '032x')

    @property
    def span_id(self) -> str:
        ctx = self._span.get_span_context()
        if ctx is None or not ctx.span_id:
            return ''
        return format(ctx.span_id, '016x')

    def set_metadata(self, metadata: Mapping[str, object]) -> None:
        if not self._span.is_recording():
            return
        for key, value in metadata.items():
            try:
                value_string = value if isinstance(value, str) else to_json_attr(value)
            except Exception as e:
                value_string = f'Error encoding metadata: {e}'
            self._span.set_attribute(metadata_key(key), value_string)

    @property
    def output_was_set(self) -> bool:
        return self._output_set

    def set_output(self, value: object) -> None:
        self._output = value
        self._output_set = True
        if self._span.is_recording():
            self._span.set_attribute(Attr.OUTPUT, to_json_attr(value))

    @property
    def state_was_set(self) -> bool:
        return self._state is not None

    def set_state(self, state: str) -> None:
        self._state = state
        if self._span.is_recording():
            self._span.set_attribute(Attr.STATE, state)
            if state == State.ERROR:
                self._span.set_status(StatusCode.ERROR)


class OtelInstrumentation:
    """OpenTelemetry provider for Cloud Trace and a host APM.

    Records each action as a span with ``genkit:*`` attributes. Pass
    ``tracer_provider`` to mint on your provider; Cloud Trace exporters
    hang there too. Omit it to use the process-global provider.

    The Developer UI collector is a separate HTTP poster. Construct this
    yourself for Cloud, or call ``enable_google_cloud_telemetry()``.
    """

    def __init__(self, *, tracer_provider: TracerProvider | None = None) -> None:
        if tracer_provider is not None and not isinstance(tracer_provider, TracerProvider):
            cls = type(tracer_provider)
            raise TypeError(f'OtelInstrumentation expected a TracerProvider, got {cls.__module__}.{cls.__qualname__}')
        self._tracer_provider = tracer_provider
        self._cached_tracer: trace_api.Tracer | None = None

    @property
    def tracer(self) -> trace_api.Tracer:
        if self._cached_tracer is None:
            provider = self._tracer_provider or trace_api.get_tracer_provider()
            self._cached_tracer = provider.get_tracer('genkit-tracer', 'v1')
        return self._cached_tracer

    async def run_in_new_span(
        self,
        metadata: SpanMetadata,
        next: SpanNext[T],
    ) -> T:
        qualified_path = build_path(
            metadata.name,
            parent_path_context.get(),
            metadata.action_type or '',
            metadata.subtype,
        )
        start_attrs = start_attributes(metadata, qualified_path=qualified_path)
        path_token = parent_path_context.set(qualified_path)
        try:
            with self.tracer.start_as_current_span(
                name=metadata.name,
                attributes=start_attrs,
                record_exception=False,
                set_status_on_exception=False,
            ) as span:
                ctx = OtelSpanContext(span)
                try:
                    result = await next(ctx)
                    if not ctx.output_was_set and result is not None:
                        span.set_attribute(Attr.OUTPUT, to_json_attr(result))
                    if not ctx.state_was_set:
                        span.set_attribute(Attr.STATE, State.SUCCESS)
                    return result
                except GenkitInterrupt:
                    span.set_attribute(Attr.STATE, State.SUCCESS)
                    raise
                except (asyncio.CancelledError, KeyboardInterrupt):
                    raise
                except Exception as e:
                    logger.debug(f'Error in run_in_new_span: {e!s}')
                    logger.debug(traceback.format_exc())
                    span.set_attribute(Attr.STATE, State.ERROR)
                    err_text = e.original_message if isinstance(e, GenkitError) else str(e)
                    span.set_attribute(Attr.ERROR, err_text)
                    span.set_status(StatusCode.ERROR, str(e))
                    span.record_exception(e)
                    raise
        finally:
            parent_path_context.reset(path_token)


def init_provider() -> TracerProvider:
    """Init and return the global tracer provider."""
    tracer_provider = trace_api.get_tracer_provider()

    if tracer_provider is None or not isinstance(tracer_provider, TracerProvider):  # pyright: ignore[reportUnnecessaryComparison]
        tracer_provider = TracerProvider()
        trace_api.set_tracer_provider(tracer_provider)
        # Booting a tracer shouldn't rewrite the process log format.
        # pyrefly: ignore[missing-attribute]
        LoggingInstrumentor().instrument(set_logging_format=False)
        logger.debug('Creating a new global tracer provider for telemetry.')

    if not isinstance(tracer_provider, TracerProvider):  # pyright: ignore[reportUnnecessaryIsInstance]
        raise TypeError(
            f'The current trace provider is not an instance of TracerProvider.  It is of type: {type(tracer_provider)}'
        )

    return tracer_provider


def is_placeholder_provider(provider: object) -> bool:
    """True when the global provider is still OTel's unset proxy / no-op."""
    return isinstance(provider, (ProxyTracerProvider, NoOpTracerProvider))


def provider_for_exporters() -> tuple[TracerProvider, bool]:
    """Provider minting Genkit OTel spans, and whether they handed it to us.

    If they already configured ``OtelInstrumentation(tracer_provider=theirs)``,
    exporters have to land on ``theirs`` or Cloud Trace stays empty. Otherwise
    the global provider.
    """
    for inst in instrumentations:
        if not isinstance(inst, OtelInstrumentation):
            continue
        provider = inst._tracer_provider
        if provider is None:
            continue
        if not isinstance(provider, TracerProvider):
            raise TypeError(
                'Cannot attach an exporter: OtelInstrumentation is using '
                f'{type(provider).__name__}, not a TracerProvider.'
            )
        return provider, True
    return init_provider(), False


def add_custom_exporter(exporter: SpanExporter | None, name: str = 'last') -> None:
    """Attach a span exporter to the provider minting Genkit spans.

    If you passed ``tracer_provider=`` to ``OtelInstrumentation``, the
    exporter hangs there. Otherwise the process-global provider. This
    does not turn tracing on. Call ``configure_instrumentation`` so
    spans exist for the exporter to see. Under ``genkit start``,
    ``Genkit()`` still owns the Developer UI HTTP poster.
    """
    if exporter is None:
        logger.warn(f'{name} exporter is None')
        return

    provider, theirs = provider_for_exporters()
    try:
        provider.add_span_processor(create_span_processor(exporter))
        logger.debug(f'{name} exporter added successfully.')
    except Exception:
        logger.error(f'tracing.add_custom_exporter: failed to add exporter {name}')
        logger.exception('Failed to add custom exporter')
        if theirs:
            raise


def maybe_configure_otel_for_exporters() -> None:
    """Turn on OTel when nothing else will mint Genkit spans.

    Skip if you already configured ``OtelInstrumentation``, or if
    ``Genkit()`` under ``genkit start`` will still attach the Developer
    UI HTTP poster (``GENKIT_ENV=dev`` and a collector URL). Call after
    exporters are attached.
    """
    if is_instrumented_by(OtelInstrumentation):
        return
    if is_dev_environment() and os.environ.get('GENKIT_TELEMETRY_SERVER'):
        return
    configure_instrumentation(OtelInstrumentation())


class PluginTracer:
    """Follows the provider minting Genkit spans. No-op when uninstrumented."""

    def inner(self) -> trace_api.Tracer:
        if not is_instrumented_by(OtelInstrumentation):
            return NoOpTracer()
        for inst in instrumentations:
            if isinstance(inst, OtelInstrumentation):
                return inst.tracer
        return NoOpTracer()

    def start_as_current_span(
        self,
        name: str,
        context: Context | None = None,
        kind: SpanKind = SpanKind.INTERNAL,
        attributes: types.Attributes = None,
        links: Sequence[Link] | None = None,
        start_time: int | None = None,
        record_exception: bool = True,
        set_status_on_exception: bool = True,
        end_on_exit: bool = True,
    ) -> AbstractContextManager[Span]:
        # Imagen and Veo open their spans on this name. Keep it a real method
        # so those call sites stay valid even when no provider is minting yet.
        return self.inner().start_as_current_span(
            name,
            context=context,
            kind=kind,
            attributes=attributes,
            links=links,
            start_time=start_time,
            record_exception=record_exception,
            set_status_on_exception=set_status_on_exception,
            end_on_exit=end_on_exit,
        )

    def __getattr__(self, name: str) -> Any:  # noqa: ANN401
        return getattr(self.inner(), name)


tracer = PluginTracer()
