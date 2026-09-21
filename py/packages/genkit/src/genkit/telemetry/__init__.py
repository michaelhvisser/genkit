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

"""Turn Genkit traces on.

Flows and ``generate()`` already open spans. This package is how those
spans get recorded — Developer UI, Cloud Trace, or your own OpenTelemetry
provider.

``genkit start -- uv run app.py`` records to the Developer UI Traces tab.
A script you run with plain ``uv run`` does not record traces.

To send spans to Cloud Trace or your own provider:

    from genkit.telemetry import configure_instrumentation, OtelInstrumentation

    configure_instrumentation(OtelInstrumentation(tracer_provider=theirs))

``enable_google_cloud_telemetry()`` sets that provider up for you.
"""

from genkit._core._telemetry.instrumentation import configure_instrumentation
from genkit._core._telemetry.otel import OtelInstrumentation

__all__ = [
    'OtelInstrumentation',
    'configure_instrumentation',
]
