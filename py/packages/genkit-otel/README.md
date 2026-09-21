# genkit-otel

OpenTelemetry backend for [Genkit](https://genkit.dev) traces.

```python
from genkit.telemetry import configure_instrumentation
from genkit_otel import OtelInstrumentation

configure_instrumentation(OtelInstrumentation(tracer_provider=theirs))
```

`genkit start` records to the Developer UI without this package.
`enable_google_cloud_telemetry()` uses this backend for Cloud Trace.
