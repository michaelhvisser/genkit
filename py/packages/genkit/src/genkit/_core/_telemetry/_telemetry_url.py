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

"""Collector URL checks for Developer UI log and trace POSTs."""

from urllib.parse import urljoin, urlparse


def resolve_telemetry_server_url(*, telemetry_server_url: str, telemetry_server_endpoint: str) -> str:
    """A typo'd collector URL should fail when tracing starts, not as missing Dev UI traces later."""
    url = telemetry_server_url.strip()
    try:
        joined = urljoin(url, telemetry_server_endpoint)
    except ValueError as error:
        raise ValueError(f'invalid telemetry server URL {telemetry_server_url!r}') from error
    parsed = urlparse(joined)
    if parsed.scheme not in ('http', 'https') or not parsed.netloc:
        raise ValueError(f'invalid telemetry server URL {telemetry_server_url!r}')
    return url
