"""Provider loops, exercised against fake model backends (no network)."""

import json
from types import SimpleNamespace as NS

import pytest

from omniharness import Agent, Reporter, tool
from omniharness.providers.openai_provider import OpenAIProvider


@tool
def add(a: int, b: int) -> int:
    """Add two integers."""
    return a + b


class RecordingReporter(Reporter):
    def __init__(self):
        self.events = []

    def on_text(self, text):
        self.events.append(("text", text))

    def on_tool_call(self, name, args):
        self.events.append(("call", name, args))

    def on_tool_result(self, name, output, is_error=False):
        self.events.append(("result", name, output))


# ── OpenAI ─────────────────────────────────────────────────────────────────


def _usage(inp, out, cached=0, reasoning=0):
    return NS(input_tokens=inp, output_tokens=out,
              input_tokens_details=NS(cached_tokens=cached),
              output_tokens_details=NS(reasoning_tokens=reasoning))


class FakeResponses:
    """Scripted responses.with_raw_response.create()."""

    def __init__(self, script, headers=None):
        self.script = list(script)
        self.calls = []
        self.headers = headers or {}
        self.with_raw_response = self

    async def create(self, **params):
        self.calls.append(json.loads(json.dumps(params, default=str)))
        response = self.script.pop(0)
        return NS(parse=lambda: response, headers=self.headers)


def _openai_agent(script, headers=None, **kwargs):
    responses = FakeResponses(script, headers)
    provider = OpenAIProvider(client=NS(responses=responses), **kwargs)
    reporter = RecordingReporter()
    agent = Agent(provider=provider, model="gpt-5", tools=[add],
                  system_prompt="sys", reporter=reporter, max_turns=3)
    return agent, responses, reporter


async def test_openai_tool_loop():
    script = [
        NS(output=[
            NS(type="message", content=[NS(text="Adding.")]),
            NS(type="function_call", call_id="c1", name="add", arguments='{"a": 2, "b": 3}'),
        ], usage=_usage(100, 10, cached=40, reasoning=5)),
        NS(output=[NS(type="message", content=[NS(text="It is 5.")])],
           usage=_usage(200, 5)),
    ]
    agent, responses, reporter = _openai_agent(script)
    result = await agent.run("2+3?")

    assert result.success and result.final_text == "It is 5." and result.num_turns == 2
    assert reporter.events == [
        ("text", "Adding."), ("call", "add", {"a": 2, "b": 3}), ("result", "add", "5"),
        ("text", "It is 5."),
    ]
    first, second = responses.calls
    assert first["instructions"] == "sys"
    assert first["tools"][0]["name"] == "add"
    # History is carried client-side, tool output included.
    assert second["input"][-1] == {"type": "function_call_output", "call_id": "c1", "output": "5"}
    u = result.usage
    assert (u.input_tokens, u.cache_read_input_tokens, u.output_tokens, u.reasoning_tokens) == (
        260, 40, 15, 5)
    # gpt-5 table: 260*1.25 + 40*0.125 + 15*10 per 1M.
    assert result.total_cost_usd == pytest.approx((260 * 1.25 + 40 * 0.125 + 150) / 1e6)


async def test_openai_max_turns_and_gateway_cost():
    call = NS(output=[NS(type="function_call", call_id="c", name="add",
                         arguments='{"a": 1, "b": 1}')], usage=_usage(1, 1))
    agent, _, _ = _openai_agent([call] * 3, headers={"x-litellm-response-cost": "0.01"})
    result = await agent.run("loop")
    assert not result.success and result.subtype == "error_max_turns"
    assert result.total_cost_usd == pytest.approx(0.03)


async def test_openai_unknown_price_note():
    script = [NS(output=[NS(type="message", content=[NS(text="hi")])], usage=_usage(1, 1))]
    responses = FakeResponses(script)
    agent = Agent(provider=OpenAIProvider(client=NS(responses=responses)), model="mystery",
                  reporter=Reporter())
    result = await agent.run("hi")
    assert result.total_cost_usd is None and "Cost: unknown" in result.notes[0]
    assert "tools" not in responses.calls[0]


# ── Claude ─────────────────────────────────────────────────────────────────


async def test_claude_provider_maps_tools_and_messages(monkeypatch):
    sdk = pytest.importorskip("claude_agent_sdk")
    seen = {}

    async def fake_query(prompt, options):
        seen["options"] = options
        # Call the custom tool through the in-process MCP server handler.
        server = options.mcp_servers["harness"]
        seen["server"] = server
        yield sdk.AssistantMessage(content=[
            sdk.TextBlock(text="Let me add."),
            sdk.ToolUseBlock(id="t1", name="mcp__harness__add", input={"a": 1, "b": 2}),
        ], model="m")
        yield sdk.UserMessage(content=[sdk.ToolResultBlock(tool_use_id="t1", content="3")])
        yield sdk.ResultMessage(
            subtype="success", duration_ms=1, duration_api_ms=1, is_error=False, num_turns=2,
            session_id="s1", total_cost_usd=0.5, result="3",
            usage={"input_tokens": 10, "cache_read_input_tokens": 5, "output_tokens": 2},
        )

    monkeypatch.setattr(sdk, "query", fake_query)
    reporter = RecordingReporter()
    agent = Agent(provider="claude", model="claude-sonnet-5", tools=["Read", "Bash", add],
                  reporter=reporter, effort="high")
    result = await agent.run("1+2")

    opts = seen["options"]
    assert opts.tools == ["Read", "Bash"]
    assert opts.allowed_tools == ["Read", "Bash", "mcp__harness__add"]
    assert opts.effort == "high"
    assert seen["server"]["type"] == "sdk"
    assert reporter.events == [
        ("text", "Let me add."), ("call", "add", {"a": 1, "b": 2}), ("result", "add", "3"),
    ]
    assert result.success and result.final_text == "3" and result.session_id == "s1"
    assert result.usage.total_input_tokens == 15 and result.total_cost_usd == 0.5
