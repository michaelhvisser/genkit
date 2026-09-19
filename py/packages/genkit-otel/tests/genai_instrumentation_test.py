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

from __future__ import annotations

import asyncio
from collections.abc import Mapping

import pytest
from genkit_otel import GenAiInstrumentation, OtelInstrumentation
from genkit_otel._gen_ai_attributes import (
    GenAiAttr,
    GenkitAttr,
)
from opentelemetry.trace import StatusCode

from genkit import FinishReason, Part
from genkit._core._error import GenkitInterrupt
from genkit._core._model import OutputConfig
from genkit._core._telemetry._instrumentation import (
    SpanNext,
    configure_instrumentation,
    reset_instrumentation,
    run_in_new_span,
)
from genkit._core._typing import GenerationUsage, Role
from genkit.model import Candidate, Message, ModelRequest, ModelResponse
from genkit.telemetry import SpanMetadata


def _model_request(
    *,
    config: Mapping[str, object] | None = None,
    messages: list[Message] | None = None,
    output: OutputConfig | None = None,
) -> ModelRequest:
    kwargs: dict[str, object] = {
        'messages': messages or [Message(role=Role.USER, content=[Part.from_text('hi')])],
    }
    if config is not None:
        kwargs['config'] = config
    if output is not None:
        kwargs['output'] = output
    return ModelRequest(**kwargs)


def _model_response(
    *,
    finish_reason: FinishReason | None = None,
    message: Message | None = None,
    usage: GenerationUsage | None = None,
) -> ModelResponse:
    return ModelResponse(
        finish_reason=finish_reason or FinishReason.STOP,
        message=message or Message(role=Role.MODEL, content=[Part.from_text('ok')]),
        usage=usage,
    )


async def _run_model(
    instr: GenAiInstrumentation,
    name: str,
    input: object,
    fn: SpanNext[object],
) -> object:
    return await instr.run_in_new_span(
        SpanMetadata(name=name, action_type='model', input=input),
        fn,
    )


@pytest.mark.asyncio
async def test_emits_chat_span_with_request_provider_model(harness) -> None:
    instr = harness.instrumentation()
    await _run_model(
        instr,
        'googleai/gemini-flash-latest',
        _model_request(config={'temperature': 0.5, 'topK': 40, 'maxOutputTokens': 100}),
        lambda span=None: _awaitable(_model_response()),
    )
    span = harness.span_named('chat gemini-flash-latest')
    assert span is not None
    assert harness.attr(span, GenAiAttr.OPERATION_NAME) == 'chat'
    assert harness.attr(span, GenAiAttr.PROVIDER_NAME) == 'gcp.gemini'
    assert harness.attr(span, GenAiAttr.REQUEST_MODEL) == 'gemini-flash-latest'
    assert harness.attr(span, GenAiAttr.REQUEST_TEMPERATURE) == 0.5
    assert harness.attr(span, GenAiAttr.REQUEST_TOP_K) == 40
    assert harness.attr(span, GenAiAttr.REQUEST_MAX_TOKENS) == 100


@pytest.mark.asyncio
async def test_records_usage_and_finish_reason(harness) -> None:
    instr = harness.instrumentation()
    await _run_model(
        instr,
        'openai/gpt-x',
        _model_request(),
        lambda span=None: _awaitable(
            _model_response(
                finish_reason=FinishReason.LENGTH,
                usage=GenerationUsage(input_tokens=10, output_tokens=20),
            )
        ),
    )
    span = harness.span_named('chat gpt-x')
    assert span is not None
    assert harness.attr(span, GenAiAttr.PROVIDER_NAME) == 'openai'
    assert harness.attr(span, GenAiAttr.USAGE_INPUT_TOKENS) == 10
    assert harness.attr(span, GenAiAttr.USAGE_OUTPUT_TOKENS) == 20
    assert list(harness.attr(span, GenAiAttr.RESPONSE_FINISH_REASONS)) == ['length']


@pytest.mark.asyncio
async def test_reports_tool_calls_when_response_has_tool_request(harness) -> None:
    instr = harness.instrumentation()
    await _run_model(
        instr,
        'googleai/gemini-flash-latest',
        _model_request(),
        lambda span=None: _awaitable(
            _model_response(
                message=Message(
                    role=Role.MODEL,
                    content=[Part.from_tool_request('lookup', input={})],
                )
            )
        ),
    )
    span = harness.span_named('chat gemini-flash-latest')
    assert span is not None
    assert list(harness.attr(span, GenAiAttr.RESPONSE_FINISH_REASONS)) == ['tool_calls']


@pytest.mark.asyncio
async def test_records_error_status_and_type_on_throw(harness) -> None:
    instr = harness.instrumentation()

    async def boom(span=None):
        raise RuntimeError('boom')

    with pytest.raises(RuntimeError, match='boom'):
        await _run_model(instr, 'googleai/gemini-flash-latest', _model_request(), boom)

    span = harness.span_named('chat gemini-flash-latest')
    assert span is not None
    assert span.status.status_code == StatusCode.ERROR
    assert harness.attr(span, GenAiAttr.ERROR_TYPE) == 'RuntimeError'


@pytest.mark.asyncio
async def test_does_not_capture_content_by_default(harness) -> None:
    instr = harness.instrumentation()
    await _run_model(
        instr,
        'googleai/gemini-flash-latest',
        _model_request(
            messages=[Message(role=Role.USER, content=[Part.from_text('secret')])],
        ),
        lambda span=None: _awaitable(_model_response()),
    )
    span = harness.span_named('chat gemini-flash-latest')
    assert span is not None
    assert harness.attr(span, GenAiAttr.INPUT_MESSAGES) is None


@pytest.mark.asyncio
async def test_nests_flow_and_model_into_one_trace(harness) -> None:
    instr = harness.instrumentation()

    async def flow_body(span=None):
        return await _run_model(
            instr,
            'googleai/gemini-flash-latest',
            _model_request(),
            lambda inner=None: _awaitable(_model_response()),
        )

    await instr.run_in_new_span(SpanMetadata(name='myFlow', action_type='flow'), flow_body)
    flow = harness.span_named('myFlow')
    model = harness.span_named('chat gemini-flash-latest')
    assert flow is not None and model is not None
    assert harness.attr(flow, GenkitAttr.ACTION_TYPE) == 'flow'
    assert model.context.trace_id == flow.context.trace_id


@pytest.mark.asyncio
async def test_execute_tool_spans_only_when_enabled(harness) -> None:
    off = harness.instrumentation()

    async def sunny(span=None):
        return 'sunny'

    await off.run_in_new_span(SpanMetadata(name='weather', action_type='tool'), sunny)
    assert harness.span_named('execute_tool weather') is None

    on = harness.instrumentation(emit_tool_spans=True)
    await on.run_in_new_span(SpanMetadata(name='weather', action_type='tool'), sunny)
    span = harness.span_named('execute_tool weather')
    assert span is not None
    assert harness.attr(span, GenAiAttr.OPERATION_NAME) == 'execute_tool'
    assert harness.attr(span, GenAiAttr.TOOL_NAME) == 'weather'


@pytest.mark.asyncio
async def test_does_not_capture_action_io_by_default(harness) -> None:
    instr = harness.instrumentation()
    await _run_model(
        instr,
        'googleai/gemini-flash-latest',
        _model_request(),
        lambda span=None: _awaitable(_model_response()),
    )
    span = harness.span_named('chat gemini-flash-latest')
    assert span is not None
    assert harness.attr(span, GenkitAttr.INPUT) is None
    assert harness.attr(span, GenkitAttr.OUTPUT) is None


@pytest.mark.asyncio
async def test_capture_action_io_records_raw_io_on_all_span_types(harness) -> None:
    instr = harness.instrumentation(capture_action_io=True, emit_tool_spans=True)

    await _run_model(
        instr,
        'googleai/gemini-flash-latest',
        _model_request(),
        lambda span=None: _awaitable(_model_response()),
    )
    model = harness.span_named('chat gemini-flash-latest')
    assert model is not None
    assert isinstance(harness.attr(model, GenkitAttr.INPUT), str)
    assert isinstance(harness.attr(model, GenkitAttr.OUTPUT), str)

    async def sunny(span=None):
        return 'sunny'

    await instr.run_in_new_span(
        SpanMetadata(name='weather', action_type='tool', input='Paris'),
        sunny,
    )
    tool = harness.span_named('execute_tool weather')
    assert tool is not None
    assert 'Paris' in harness.attr(tool, GenkitAttr.INPUT)
    assert 'sunny' in harness.attr(tool, GenkitAttr.OUTPUT)
    assert harness.attr(tool, GenAiAttr.INPUT_MESSAGES) is None
    assert harness.attr(tool, GenAiAttr.OUTPUT_MESSAGES) is None

    async def flow_out(span=None):
        return 'out'

    await instr.run_in_new_span(SpanMetadata(name='myFlow', action_type='flow', input='in'), flow_out)
    flow = harness.span_named('myFlow')
    assert flow is not None
    assert 'in' in harness.attr(flow, GenkitAttr.INPUT)
    assert 'out' in harness.attr(flow, GenkitAttr.OUTPUT)
    assert harness.attr(flow, GenAiAttr.INPUT_MESSAGES) is None


@pytest.mark.asyncio
async def test_reports_tool_calls_from_legacy_candidates(harness) -> None:
    instr = harness.instrumentation()
    legacy = ModelResponse(
        finish_reason=FinishReason.STOP,
        candidates=[
            Candidate(
                index=0,
                finish_reason=FinishReason.STOP,
                message=Message(
                    role=Role.MODEL,
                    content=[Part.from_tool_request('lookup', input={})],
                ),
            )
        ],
    )
    await _run_model(
        instr,
        'googleai/gemini-flash-latest',
        _model_request(),
        lambda span=None: _awaitable(legacy),
    )
    span = harness.span_named('chat gemini-flash-latest')
    assert span is not None
    assert list(harness.attr(span, GenAiAttr.RESPONSE_FINISH_REASONS)) == ['tool_calls']


@pytest.mark.asyncio
async def test_classifies_action_subtype_model_as_chat(harness) -> None:
    """Python action leftover: type=action, subtype=model still becomes a chat span."""
    instr = harness.instrumentation()

    async def body(span=None):
        return _model_response()

    await instr.run_in_new_span(
        SpanMetadata(
            name='googleai/gemini-flash-latest',
            action_type='action',
            subtype='model',
            input=_model_request(),
        ),
        body,
    )
    assert harness.span_named('chat gemini-flash-latest') is not None
    assert harness.span_named('googleai/gemini-flash-latest') is None


@pytest.mark.asyncio
async def test_output_type_json_from_request(harness) -> None:
    instr = harness.instrumentation()
    await _run_model(
        instr,
        'googleai/gemini-flash-latest',
        _model_request(output=OutputConfig(format='json')),
        lambda span=None: _awaitable(_model_response()),
    )
    span = harness.span_named('chat gemini-flash-latest')
    assert span is not None
    assert harness.attr(span, GenAiAttr.OUTPUT_TYPE) == 'json'


async def _awaitable(value: object) -> object:
    return value


@pytest.mark.asyncio
async def test_tool_interrupt_sets_interrupted_attribute_and_clean_status(harness) -> None:
    instr = harness.instrumentation(emit_tool_spans=True)

    async def tool_body(span=None):
        raise GenkitInterrupt('tool requires human approval')

    with pytest.raises(GenkitInterrupt):
        await instr.run_in_new_span(
            SpanMetadata(name='transfer_funds', action_type='tool', input={'amount': 100}),
            tool_body,
        )

    span = harness.span_named('execute_tool transfer_funds')
    assert span is not None
    assert span.status.status_code != StatusCode.ERROR
    assert harness.attr(span, 'genkit.interrupted') is True
    assert not any(event.name == 'exception' for event in span.events)


@pytest.mark.asyncio
async def test_model_interrupt_sets_interrupted_attribute_and_clean_status(harness) -> None:
    instr = harness.instrumentation()

    async def model_body(span=None):
        raise GenkitInterrupt('paused for confirmation')

    with pytest.raises(GenkitInterrupt):
        await _run_model(
            instr,
            'googleai/gemini-flash-latest',
            _model_request(),
            model_body,
        )

    span = harness.span_named('chat gemini-flash-latest')
    assert span is not None
    assert span.status.status_code != StatusCode.ERROR
    assert harness.attr(span, 'genkit.interrupted') is True
    assert not any(event.name == 'exception' for event in span.events)


@pytest.mark.asyncio
async def test_generic_span_interrupt_sets_interrupted_attribute_and_clean_status(harness) -> None:
    instr = harness.instrumentation()

    async def flow_body(span=None):
        raise GenkitInterrupt('flow paused')

    with pytest.raises(GenkitInterrupt):
        await instr.run_in_new_span(
            SpanMetadata(name='order_flow', action_type='flow'),
            flow_body,
        )

    span = harness.span_named('order_flow')
    assert span is not None
    assert span.status.status_code != StatusCode.ERROR
    assert harness.attr(span, 'genkit.interrupted') is True
    assert not any(event.name == 'exception' for event in span.events)


@pytest.mark.asyncio
async def test_cancellation_does_not_record_error_status(harness) -> None:
    instr = harness.instrumentation()

    async def cancelled_body(span=None):
        raise asyncio.CancelledError('task cancelled')

    with pytest.raises(asyncio.CancelledError):
        await instr.run_in_new_span(
            SpanMetadata(name='long_task', action_type='flow'),
            cancelled_body,
        )

    span = harness.span_named('long_task')
    assert span is not None
    assert span.status.status_code != StatusCode.ERROR
    assert not any(event.name == 'exception' for event in span.events)


@pytest.mark.asyncio
async def test_stacked_genai_and_otel_interrupt_coexistence(harness) -> None:
    genai_instr = GenAiInstrumentation(
        tracer=harness.tracer_provider.get_tracer('test'),
        emit_tool_spans=True,
    )
    otel_instr = OtelInstrumentation(tracer_provider=harness.tracer_provider)

    configure_instrumentation(genai_instr)
    configure_instrumentation(otel_instr)

    try:

        async def tool_body(span=None):
            raise GenkitInterrupt('approval needed')

        with pytest.raises(GenkitInterrupt):
            await run_in_new_span(
                'transfer_money',
                tool_body,
                action_type='tool',
                input={'amount': 500},
            )

        tool_span = harness.span_named('execute_tool transfer_money')
        native_span = harness.span_named('transfer_money')

        assert tool_span is not None
        assert native_span is not None
        assert tool_span.status.status_code != StatusCode.ERROR
        assert harness.attr(tool_span, 'genkit.interrupted') is True
        assert native_span.status.status_code != StatusCode.ERROR
        assert harness.attr(native_span, 'genkit:state') == 'success'
    finally:
        reset_instrumentation()


@pytest.mark.asyncio
async def test_custom_otel_and_gcp_adjusting_exporter_isolation() -> None:
    """A developer's custom OTel exporter receives unmutated attributes while GCP receives redacted."""
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

    from genkit._core._telemetry._adjusting_exporter import AdjustingTraceExporter

    provider = TracerProvider()
    datadog_exporter = InMemorySpanExporter()
    gcp_inner_exporter = InMemorySpanExporter()
    gcp_adjusting_exporter = AdjustingTraceExporter(
        exporter=gcp_inner_exporter,
        log_input_and_output=False,
    )

    provider.add_span_processor(SimpleSpanProcessor(datadog_exporter))
    provider.add_span_processor(SimpleSpanProcessor(gcp_adjusting_exporter))

    otel_instr = OtelInstrumentation(tracer_provider=provider)
    configure_instrumentation(otel_instr)

    try:

        async def action_body(span=None):
            return {'user': 'sensitive_pii'}

        await run_in_new_span('login', action_body, action_type='action', input={'password': 'secret_password'})

        assert len(datadog_exporter.get_finished_spans()) == 1
        assert len(gcp_inner_exporter.get_finished_spans()) == 1

        datadog_span = datadog_exporter.get_finished_spans()[0]
        gcp_span = gcp_inner_exporter.get_finished_spans()[0]

        assert datadog_span.context is not None and gcp_span.context is not None
        assert datadog_span.context.trace_id == gcp_span.context.trace_id
        assert datadog_span.context.span_id == gcp_span.context.span_id

        assert datadog_span.attributes is not None and gcp_span.attributes is not None
        # GCP received redacted input/output
        assert gcp_span.attributes['genkit/input'] == '<redacted>'
        assert gcp_span.attributes['genkit/output'] == '<redacted>'

        # Datadog received intact input/output without mutation
        assert 'secret_password' in str(datadog_span.attributes['genkit:input'])
        assert 'sensitive_pii' in str(datadog_span.attributes['genkit:output'])
    finally:
        reset_instrumentation()
