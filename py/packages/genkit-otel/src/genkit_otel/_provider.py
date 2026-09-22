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
import traceback
from collections.abc import Mapping
from typing import TypeVar

from opentelemetry import trace as trace_api
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.trace import StatusCode

from genkit._core._error import GenkitError, GenkitInterrupt
from genkit._core._logger import get_logger
from genkit._core._telemetry._attrs import Attr, State, metadata_key
from genkit._core._telemetry._instrumentation import (
    SpanMetadata,
    SpanNext,
    parent_path_context,
    start_attributes,
    to_json_attr,
)
from genkit._core._telemetry._path import build_path

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

    Records each action as a span with ``genkit:*`` attributes. Omit
    ``tracer_provider`` to mint on the process-global provider — the
    one they already registered, if any. Cloud Trace is
    ``enable_google_cloud_telemetry()``.

    The Developer UI collector is a separate HTTP poster.
    """

    def __init__(self, *, tracer_provider: TracerProvider | None = None) -> None:
        if tracer_provider is not None and not isinstance(tracer_provider, TracerProvider):
            cls = type(tracer_provider)
            raise TypeError(f'OtelInstrumentation expected a TracerProvider, got {cls.__module__}.{cls.__qualname__}')
        self._tracer_provider = tracer_provider
        self._cached_tracer: trace_api.Tracer | None = None

    @property
    def tracer_provider(self) -> TracerProvider | None:
        return self._tracer_provider

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

    def flush(self) -> None:
        provider = self._tracer_provider or trace_api.get_tracer_provider()
        if isinstance(provider, TracerProvider):
            provider.force_flush()
