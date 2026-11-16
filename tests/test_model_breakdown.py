"""Per-model token (and cost) breakdown, for every provider and across subagents."""

from types import SimpleNamespace as NS

from omniharness import Agent, Reporter, RunResult, Usage
from omniharness.providers.openai_provider import OpenAIProvider
from omniharness.reporting import format_summary


def _openai_run(model, inp=100, out=20, cached=40, reasoning=5):
    response = NS(output=[NS(type="message", content=[NS(text="ok")])],
                  usage=NS(input_tokens=inp, output_tokens=out,
                           input_tokens_details=NS(cached_tokens=cached),
                           output_tokens_details=NS(reasoning_tokens=reasoning)))

    class Responses:
        with_raw_response = None

        async def create(self, **params):
            return NS(parse=lambda: response, headers={})

    responses = Responses()
    responses.with_raw_response = responses
    provider = OpenAIProvider(client=NS(responses=responses), api="responses")
    return Agent(provider=provider, model=model, reporter=Reporter()).run_sync("hi")


def test_openai_records_its_model_usage():
    result = _openai_run("my-local-model")
    assert result.model_usage == {"my-local-model": {
        "inputTokens": 60, "cacheReadInputTokens": 40, "cacheCreationInputTokens": 0,
        "outputTokens": 20, "reasoningTokens": 5, "costUSD": None}}  # no price known
    priced = _openai_run("gpt-6-luna")
    assert priced.model_usage["gpt-6-luna"]["costUSD"] == priced.total_cost_usd > 0


def test_breakdown_across_subagents():
    lead = RunResult(success=True, model="claude-sonnet-5-5", total_cost_usd=0.05,
                     model_usage={  # the Claude SDK's own shape
                         "claude-sonnet-5-5": {"inputTokens": 1000, "outputTokens": 200,
                                               "cacheReadInputTokens": 500,
                                               "cacheCreationInputTokens": 100,
                                               "costUSD": 0.0499},
                         "claude-haiku-5-5": {"inputTokens": 10, "outputTokens": 1,
                                              "costUSD": 0.0001}})
    local = _openai_run("qwen3-8b")
    local2 = _openai_run("qwen3-8b", inp=200, cached=0, reasoning=0)
    # A provider that reports no breakdown: its totals count for its model.
    custom = RunResult(success=True, model="mine", total_cost_usd=0.01,
                       usage=Usage(input_tokens=7, output_tokens=3))
    lead.subagents = [local, local2, custom]

    breakdown = lead.model_breakdown()
    assert list(breakdown) == ["claude-sonnet-5-5", "claude-haiku-5-5", "qwen3-8b", "mine"]
    qwen = breakdown["qwen3-8b"]
    assert (qwen["inputTokens"], qwen["cacheReadInputTokens"], qwen["outputTokens"],
            qwen["reasoningTokens"], qwen["runs"], qwen["costUSD"]) == (260, 40, 40, 5, 2, None)
    assert breakdown["mine"]["inputTokens"] == 7 and breakdown["mine"]["costUSD"] == 0.01
    assert lead.model_breakdown(include_subagents=False).keys() == {
        "claude-sonnet-5-5", "claude-haiku-5-5"}

    summary = format_summary(lead)
    assert "Per-model breakdown (incl. subagents):" in summary
    assert "  qwen3-8b, 2 runs:" in summary
    section = summary[summary.index("  qwen3-8b"):]
    assert "Output: 40 tokens (5 reasoning)" in section and "Cost:   unknown" in section
    assert "Cost:   $0.0499" in summary
