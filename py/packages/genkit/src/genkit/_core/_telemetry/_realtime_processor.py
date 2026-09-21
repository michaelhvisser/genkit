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

"""Realtime span processor for live trace visualization."""

from opentelemetry.context import Context
from opentelemetry.sdk.trace import ReadableSpan, Span
from opentelemetry.sdk.trace.export import SimpleSpanProcessor

from genkit._core._compat import override
from genkit._core._telemetry._instrumentation import suppress_telemetry


class RealtimeSpanProcessor(SimpleSpanProcessor):
    """Exports spans on start (real-time) and on end, unlike SimpleSpanProcessor (end only)."""

    @override
    def on_start(self, span: Span, parent_context: Context | None = None) -> None:
        """Export span immediately so DevUI can show in-progress traces."""
        if suppress_telemetry.get():
            return
        # Transport failures stay in the exporter. generate() should still return.
        self.span_exporter.export([span])

    @override
    def on_end(self, span: ReadableSpan) -> None:
        """Skip export for prompt-playground re-renders and other ignored runs."""
        if suppress_telemetry.get():
            return
        super().on_end(span)

    @override
    def force_flush(self, timeout_millis: int = 30000) -> bool:
        # SimpleSpanProcessor.force_flush is a no-op True. The Dev UI
        # exporter queues POSTs; wait for that queue.
        return bool(self.span_exporter.force_flush(timeout_millis=timeout_millis))
