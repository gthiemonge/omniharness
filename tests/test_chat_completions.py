"""The openai provider over /v1/chat/completions, and api="auto" detection (fakes only)."""

import json
from types import SimpleNamespace as NS

import pytest

from omniharness import Agent, Reporter, cli, load_profiles, tool
from omniharness.providers.openai_provider import OpenAIProvider


@tool
def add(a: int, b: int) -> int:
    """Add two integers."""
    return a + b


class HTTPError(Exception):
    def __init__(self, status_code):
        super().__init__(f"HTTP {status_code}")
        self.status_code = status_code


def _usage(prompt, completion, cached=0, reasoning=0):
    return NS(prompt_tokens=prompt, completion_tokens=completion,
              prompt_tokens_details=NS(cached_tokens=cached),
              completion_tokens_details=NS(reasoning_tokens=reasoning))


def _reply(content=None, calls=(), usage=None):
    tool_calls = [NS(id=cid, function=NS(name=name, arguments=args))
                  for cid, name, args in calls] or None
    return NS(choices=[NS(message=NS(content=content, tool_calls=tool_calls))],
              usage=usage or _usage(1, 1))


class Endpoint:
    """Scripted create() for one endpoint, recording the requests."""

    def __init__(self, script=(), error=None):
        self.script = list(script)
        self.error = error
        self.calls = []
        self.with_raw_response = self

    async def create(self, **params):
        self.calls.append(json.loads(json.dumps(params, default=str)))
        if self.error:
            raise self.error
        response = self.script.pop(0)
        return NS(parse=lambda: response, headers={})


def _client(chat=(), responses=None):
    return NS(chat=NS(completions=Endpoint(chat)), responses=responses or Endpoint())


def _agent(client, api="chat", **kwargs):
    return Agent(provider=OpenAIProvider(client=client, api=api), model="gpt-6.1-sol",
                 tools=[add], system_prompt="sys", reporter=Reporter(), **kwargs)


async def test_chat_tool_loop():
    client = _client([
        _reply("Adding.", [("c1", "add", '{"a": 2, "b": 3}')], _usage(100, 10, cached=40)),
        _reply("It is 5.", usage=_usage(200, 5, reasoning=2)),
    ])
    result = await _agent(client).run("2+3?")

    assert result.success and result.final_text == "It is 5." and result.num_turns == 2
    first, second = client.chat.completions.calls
    assert first["messages"] == [{"role": "system", "content": "sys"},
                                 {"role": "user", "content": "2+3?"}]
    assert first["tools"] == [{"type": "function", "function": {
        "name": "add", "description": "Add two integers.",
        "parameters": add.parameters}}]
    assert second["messages"][2:] == [
        {"role": "assistant", "content": "Adding.", "tool_calls": [
            {"id": "c1", "type": "function",
             "function": {"name": "add", "arguments": '{"a": 2, "b": 3}'}}]},
        {"role": "tool", "tool_call_id": "c1", "content": "5"},
    ]
    u = result.usage
    assert (u.input_tokens, u.cache_read_input_tokens, u.output_tokens, u.reasoning_tokens) == (
        260, 40, 15, 2)
    assert result.total_cost_usd is not None  # gpt-6.1-sol is in the price table
    assert not client.responses.calls


async def test_missing_tool_call_ids_are_generated():
    client = _client([_reply(None, [(None, "add", '{"a": 1, "b": 1}')]), _reply("2")])
    await _agent(client).run("1+1?")
    assistant, tool_result = client.chat.completions.calls[1]["messages"][2:]
    assert assistant["content"] is None
    assert assistant["tool_calls"][0]["id"] == tool_result["tool_call_id"] == "call_1_0"


async def test_effort_and_options():
    client = _client([_reply("ok")])
    provider = OpenAIProvider(client=client, api="chat", options={"temperature": 0})
    await Agent(provider=provider, model="m", effort="low", reporter=Reporter()).run("hi")
    params = client.chat.completions.calls[0]
    assert params["reasoning_effort"] == "low" and params["temperature"] == 0
    assert "tools" not in params


# ── api="auto" ─────────────────────────────────────────────────────────────


async def test_auto_falls_back_to_chat_and_remembers():
    responses = Endpoint(error=HTTPError(404))
    client = _client([_reply("first"), _reply("second")], responses)
    agent = _agent(client, api="auto")

    first = await agent.run("hi")
    assert first.success and first.final_text == "first"
    assert "used /v1/chat/completions" in first.notes[0]
    second = await agent.run("again")
    assert second.final_text == "second" and not second.notes
    assert len(responses.calls) == 1  # probed once, then remembered


async def test_auto_stays_on_responses_when_it_works():
    responses = Endpoint([NS(output=[NS(type="message", content=[NS(text="ok")])],
                             usage=None)])
    client = _client(responses=responses)
    result = await _agent(client, api="auto").run("hi")
    assert result.final_text == "ok" and not client.chat.completions.calls


@pytest.mark.parametrize(("api", "status"), [("auto", 401), ("auto", 500), ("responses", 404)])
async def test_other_errors_are_not_hidden(api, status):
    client = _client(responses=Endpoint(error=HTTPError(status)))
    with pytest.raises(HTTPError):
        await _agent(client, api=api).run("hi")
    assert not client.chat.completions.calls


def test_invalid_api():
    with pytest.raises(ValueError, match="api must be one of"):
        OpenAIProvider(api="completions")


# ── configuration ──────────────────────────────────────────────────────────


def test_preset_api_field(tmp_path, monkeypatch):
    monkeypatch.delenv("AGENT_PROFILES", raising=False)
    path = tmp_path / "p.toml"
    path.write_text('[profiles.lm]\nquick = { provider = "openai", model = "m", '
                    'base_url = "http://localhost:1234/v1", api = "chat" }\n')
    agent = Agent(profiles=load_profiles(path), profile="lm", preset="quick",
                  reporter=Reporter())
    assert agent.provider.api == "chat"


def test_cli_openai_api(monkeypatch):
    for var in ("AGENT_PROVIDER", "AGENT_MODEL", "AGENT_PRESET", "AGENT_PROFILES",
                "AGENT_PROFILE"):
        monkeypatch.delenv(var, raising=False)
    seen = {}

    class FakeAgent:
        def __init__(self, **kwargs):
            seen.update(kwargs)

        def run_sync(self, prompt):
            from omniharness import RunResult
            return RunResult(success=True)

    monkeypatch.setattr(cli, "Agent", FakeAgent)
    assert cli.main(["--provider", "openai", "--openai-api", "chat", "-q", "hi"]) == 0
    assert seen["provider_options"] == {"openai": {"api": "chat"}}
