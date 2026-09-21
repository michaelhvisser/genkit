# genkit-otel

Set up a global tracer provider following the [OpenTelemetry Python SDK](https://opentelemetry.io/docs/languages/python/instrumentation/) documentation, then register Genkit instrumentation:

```python
from genkit.telemetry import configure_instrumentation
from genkit_otel import OtelInstrumentation

configure_instrumentation(OtelInstrumentation())
```

Pass `tracer_provider` to `OtelInstrumentation` to use an explicit provider instead of the process-global provider.

`genkit start` records to the Developer UI without this package.
