from types import SimpleNamespace

import pytest
from test_openrouter_text import _chat_response, _FakeCompletions, _provider_with_completions
from test_structured_conformance import ScriptedText, _request, _result

from generationengine import GenerationClient, GenerationEngineError, TextRequest
from generationengine.client import _aggregate_cost
from generationengine.providers.base import TextGenerationCall
from generationengine.providers.openrouter_text import _usage_fields


@pytest.mark.parametrize(
    "value, expected",
    [
        (0, 0),
        (0.0123, 0.0123),
        (None, None),
        (-1, None),
        (True, None),
        ("0.02", None),
        (float("nan"), None),
        (float("inf"), None),
    ],
)
def test_provider_charge_normalization_preserves_tokens(value, expected):
    fields = _usage_fields(SimpleNamespace(cost=value, prompt_tokens=7, completion_tokens=4))
    assert fields["cost_usd"] == expected
    assert fields["input_tokens"] == 7
    assert fields["pricing_source"] == (
        "provider_reported:openrouter.usage.cost" if expected is not None else None
    )


@pytest.mark.asyncio
async def test_charge_reaches_adapter_and_client_without_catalog_estimate():
    response = _chat_response(
        usage=SimpleNamespace(cost=0.012, prompt_tokens=8, completion_tokens=5)
    )
    provider = _provider_with_completions(_FakeCompletions(response=response))
    client = GenerationClient(text_providers={"openrouter": provider})
    result = await client.generate_text(
        TextRequest(provider="openrouter", model="example/model", user_prompt="hello")
    )
    assert result.observation.cost_usd == 0.012
    assert result.observation.known_cost_usd == 0.012
    assert result.observation.cost_status == "reported"
    assert result.observation.pricing_source == "provider_reported:openrouter.usage.cost"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "second,expected,status,subtotal",
    [(0.02, 0.03, "reported", 0.03), (None, None, "partial", 0.01), (0, 0.01, "reported", 0.01)],
)
async def test_structured_attempt_charge_is_distinct_from_operation_total(
    second, expected, status, subtotal
):
    provider = ScriptedText(
        [
            _result(
                text="bad", parsed=None, cost_usd=0.01, pricing_source="provider_reported:test"
            ),
            _result(cost_usd=second, pricing_source="provider_reported:test"),
        ]
    )
    events = []

    async def capture(event):
        events.append(event)

    result = await GenerationClient(text_provider=provider).generate_structured(
        _request(), on_attempt_completed=capture
    )
    assert [e.cost_usd for e in events] == [0.01, second]
    assert result.observation.cost_usd == expected
    assert result.observation.known_cost_usd == subtotal
    assert result.observation.cost_status == status
    assert result.observation.provider_attempt_count == 2


@pytest.mark.asyncio
async def test_failed_conformance_retains_all_reported_charges():
    provider = ScriptedText(
        [
            _result(text="bad", parsed=None, cost_usd=0.01),
            _result(text="bad", parsed=None, cost_usd=0.02),
        ]
    )
    with pytest.raises(GenerationEngineError) as exc:
        await GenerationClient(text_provider=provider).generate_structured(_request())
    assert exc.value.observation.cost_usd == 0.03
    assert exc.value.observation.cost_status == "reported"


def test_unknown_attempt_prevents_false_complete_total():
    assert _aggregate_cost(
        [SimpleNamespace(cost_usd=0.1, pricing_source="provider_reported:test"), None]
    ) == {
        "cost_usd": None,
        "known_cost_usd": 0.1,
        "cost_status": "partial",
        "pricing_source": "provider_reported:test",
    }


@pytest.mark.asyncio
async def test_stream_cumulative_snapshots_are_not_double_counted():
    chunks = [
        SimpleNamespace(
            id="r",
            model="m",
            usage=SimpleNamespace(cost=v, prompt_tokens=5, completion_tokens=2),
            choices=[],
        )
        for v in [0.01, 0.02]
    ]

    class Completions:
        async def create(self, **kw):
            async def stream():
                for chunk in chunks:
                    yield chunk

            return stream()

    provider = _provider_with_completions(Completions())
    events = [e async for e in provider.stream(TextGenerationCall(model="m", user_prompt="hi"))]
    assert events[-1].observation.cost_usd == 0.02
    assert events[-1].observation.cost_status == "reported"


@pytest.mark.asyncio
async def test_partial_stream_keeps_known_subtotal_without_complete_price():
    class Completions:
        async def create(self, **kw):
            async def stream():
                yield SimpleNamespace(
                    id="r",
                    model="m",
                    usage=SimpleNamespace(cost=0.01, prompt_tokens=5, completion_tokens=2),
                    choices=[],
                )
                raise RuntimeError("synthetic stream error")

            return stream()

    events = [
        e
        async for e in _provider_with_completions(Completions()).stream(
            TextGenerationCall(model="m", user_prompt="hi")
        )
    ]
    assert events[-1].observation.cost_usd is None
    assert events[-1].observation.known_cost_usd == 0.01
    assert events[-1].observation.cost_status == "partial"


@pytest.mark.asyncio
async def test_plain_transport_retry_keeps_reported_charge_from_failed_attempt(monkeypatch):
    from generationengine import FailureCode
    from generationengine import client as module
    from generationengine.providers.errors import ProviderError

    monkeypatch.setattr(module, "BACKOFF_SECONDS", (0,))
    failure = ProviderError.from_code(
        FailureCode.RATE_LIMITED, cost_usd=0.01, pricing_source="provider_reported:test"
    )
    provider = ScriptedText(
        [failure, _result(cost_usd=0.02, pricing_source="provider_reported:test")]
    )
    r = await GenerationClient(text_provider=provider).generate_text(
        TextRequest(user_prompt="hi", provider="openai", model="example/model")
    )
    assert r.observation.cost_usd == 0.03
    assert r.observation.known_cost_usd == 0.03
    assert r.observation.provider_attempt_count == 2


@pytest.mark.asyncio
@pytest.mark.parametrize("fail", [False, True])
async def test_stream_missing_final_cost_preserves_prior_charge_as_partial(fail):
    class Completions:
        async def create(self, **kw):
            async def stream():
                for cost in [0.01, None]:
                    yield SimpleNamespace(
                        id="r",
                        model="m",
                        usage=SimpleNamespace(cost=cost, prompt_tokens=6, completion_tokens=3),
                        choices=[],
                    )
                if fail:
                    raise RuntimeError("synthetic")

            return stream()

    events = [
        e
        async for e in _provider_with_completions(Completions()).stream(
            TextGenerationCall(model="m", user_prompt="hi")
        )
    ]
    assert events[-1].observation.cost_usd is None
    assert events[-1].observation.known_cost_usd == 0.01
    assert events[-1].observation.cost_status == "partial"
