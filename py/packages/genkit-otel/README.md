# genkit-otel

If they already set the process tracer, turn Genkit spans on with:

```python
from genkit.telemetry import configure_instrumentation
from genkit_otel import OtelInstrumentation

configure_instrumentation(OtelInstrumentation())
```

Pass `tracer_provider` to mint on that provider instead of the process-global one.

Cloud Trace is `enable_google_cloud_telemetry()` from `genkit_google_cloud`. `genkit start` records to the Developer UI without this package. `pip install genkit` does not install OpenTelemetry.
