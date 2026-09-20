# Copyright 2025 Google LLC
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

"""Telemetry types for exporter plugins.

Application code should call :class:`genkit.Genkit`. An exporter plugin
uses this module to attach spans and redact payloads.

Example:
    from genkit.telemetry import AdjustingTraceExporter, RedactedSpan, add_custom_exporter
"""

from genkit._core._environment import is_dev_environment
from genkit._core._trace._adjusting_exporter import AdjustingTraceExporter, RedactedSpan
from genkit._core._trace._path import to_display_path
from genkit._core._tracing import add_custom_exporter, tracer

__all__ = [
    'AdjustingTraceExporter',
    'RedactedSpan',
    'add_custom_exporter',
    'is_dev_environment',
    'to_display_path',
    'tracer',
]
