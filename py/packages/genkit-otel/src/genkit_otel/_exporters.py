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

"""Process tracer boot and the plugin span opener."""

from __future__ import annotations

from collections.abc import Sequence
from contextlib import AbstractContextManager
from typing import Any

from opentelemetry import trace as trace_api
from opentelemetry.context import Context
from opentelemetry.instrumentation.logging import LoggingInstrumentor
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.trace import Link, NoOpTracer, NoOpTracerProvider, ProxyTracerProvider, Span, SpanKind
from opentelemetry.util import types

from genkit._core._logger import get_logger
from genkit._core._telemetry._instrumentation import instrumentations

logger = get_logger(__name__)


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


class PluginTracer:
    """Follows the provider minting Genkit spans. No-op when uninstrumented."""

    def inner(self) -> trace_api.Tracer:
        for inst in instrumentations:
            found = getattr(inst, 'tracer', None)
            if found is not None:
                return found
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
        # A real method so a plugin can open a span before a provider is
        # minting. Uninstrumented, this is a no-op span.
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
