"""Keeping the openai provider's requests inside a context window (fake servers)."""

import json
from types import SimpleNamespace as NS

import pytest

from omniharness import Agent, Reporter, cli, load_profiles, tool
from omniharness.providers.openai_provider import OpenAIProvider, _Window


@tool
def big(n: int) -> str:
    """Return n characters."""
    return "x" * n


class HTTPError(Exception):
    def __init__(self, status_code, message):
        super().__init__(message)
        self.status_code = status_code


class Model:
    """Calls big(n) for `steps` turns, then answers; works for both wire formats."""

    def __init__(self, api, steps, n, fail_on=None):
        self.api, self.steps, self.n, self.fail_on = api, steps, n, fail_on
        self.requests = []
        self.with_raw_response = self
        self.turn = 0

    async def create(self, **params):
        self.requests.append(json.loads(json.dumps(params)))
        if self.fail_on and len(self.requests) == self.fail_on:
            raise HTTPError(400, "the request exceeds the available context size")
        self.turn += 1
        call = self.turn <= self.steps
        args = json.dumps({"n": self.n})
        if self.api == "responses":
            output = ([NS(type="function_call", call_id=f"c{self.turn}", name="big",
                          arguments=args)] if call
                      else [NS(type="message", content=[NS(text="done")])])
            response = NS(output=output, usage=None)
        else:
            calls = [NS(id=f"c{self.turn}", function=NS(name="big", arguments=args))]
            message = NS(content=None if call else "done", tool_calls=calls if call else None)
            response = NS(choices=[NS(message=message)], usage=None)
        return NS(parse=lambda: response, headers={})


def _run(api, steps, n, context_window, fail_on=None, max_turns=100):
    model = Model(api, steps, n, fail_on)
    client = (NS(responses=model) if api == "responses"
              else NS(chat=NS(completions=model)))
    provider = OpenAIProvider(client=client, api=api, context_window=context_window)
    # A priced model, so the only notes are about trimming.
    agent = Agent(provider=provider, model="gpt-6-luna", tools=[big], max_turns=max_turns,
                  system_prompt="sys", reporter=Reporter())
    return agent.run_sync("go"), model


def _outputs(api, request):
    items = request["input"] if api == "responses" else request["messages"]
    key, match = (("output", "function_call_output") if api == "responses"
                  else ("content", "tool"))
    return [i[key] for i in items if i.get("type") == match or i.get("role") == match]


def _pairs_ok(api, request):
    items = request["input"] if api == "responses" else request["messages"]
    if api == "responses":
        calls = {i["call_id"] for i in items if i.get("type") == "function_call"}
        results = {i["call_id"] for i in items if i.get("type") == "function_call_output"}
    else:
        calls = {c["id"] for i in items for c in i.get("tool_calls") or []}
        results = {i["tool_call_id"] for i in items if i.get("role") == "tool"}
    return calls == results


APIS = ["responses", "chat"]


def test_window_math():
    w = _Window(32768)
    assert w.budget() == int((32768 - 8192) * 0.95)
    assert w.tool_output_chars() == int(w.budget() * 3.5 / 4)
    w.calibrate(40_000, 10_000)
    assert w.chars_per_token == 4.0
    w.calibrate(100, 10_000)  # absurd ratios are clamped
    assert w.chars_per_token == 1.5


@pytest.mark.parametrize("api", APIS)
def test_no_window_no_trimming(api):
    result, model = _run(api, steps=3, n=50_000, context_window=None)
    assert result.success and not result.notes
    assert [len(o) for o in _outputs(api, model.requests[-1])] == [50_000] * 3


@pytest.mark.parametrize("api", APIS)
def test_tool_outputs_are_capped(api):
    result, model = _run(api, steps=1, n=50_000, context_window=4000)
    cap = _Window(4000).tool_output_chars()
    (output,) = _outputs(api, model.requests[-1])
    assert output.startswith("x" * cap) and "truncated to fit the context window" in output
    assert len(output) < cap + 100
    assert "1 tool output(s) capped" in result.notes[0]


@pytest.mark.parametrize("api", APIS)
def test_older_outputs_are_shortened_recent_ones_kept(api):
    result, model = _run(api, steps=6, n=4000, context_window=8000)
    assert result.success
    last = model.requests[-1]
    outputs = _outputs(api, last)
    assert len(outputs) == 6 and _pairs_ok(api, last)
    assert [len(o) for o in outputs[-2:]] == [4000, 4000]  # the last two turns whole
    # Only as many as needed are shortened, oldest first.
    shortened = ["removed to fit the context window" in o for o in outputs]
    assert any(shortened) and shortened == sorted(shortened, reverse=True)
    assert "0 tool output(s) capped" in result.notes[0]
    assert "older one(s) shortened, 0 turn(s) removed" in result.notes[0]
    assert all(_Window(8000).fits(r) for r in model.requests)


@pytest.mark.parametrize("api", APIS)
def test_oldest_turns_are_dropped_whole(api):
    result, model = _run(api, steps=60, n=4000, context_window=4000)
    assert result.success
    last = model.requests[-1]
    items = last["input"] if api == "responses" else last["messages"]
    assert _pairs_ok(api, last)
    assert any("were removed to fit the context window" in str(i.get("content"))
               for i in items)
    first_user = next(i for i in items if i.get("role") == "user")
    assert first_user["content"] == "go"  # the task itself is never dropped
    assert "turn(s) removed" in result.notes[0] and "0 turn(s)" not in result.notes[0]
    assert all(_Window(4000).fits(r) for r in model.requests)


@pytest.mark.parametrize("api", APIS)
def test_context_error_trims_harder_and_retries(api):
    # A huge window, so the estimate never trims; the server says otherwise.
    result, model = _run(api, steps=4, n=3000, context_window=1_000_000, fail_on=4)
    assert result.success
    failed, retried = model.requests[3], model.requests[4]
    assert len(_outputs(api, retried)) < len(_outputs(api, failed))
    assert _pairs_ok(api, retried)
    assert "Trimmed to fit" in result.notes[0]


def test_other_400s_are_not_retried():
    class Bad(Model):
        async def create(self, **params):
            raise HTTPError(400, "invalid tool schema")

    model = Bad("responses", 1, 1)
    provider = OpenAIProvider(client=NS(responses=model), api="responses", context_window=8000)
    with pytest.raises(HTTPError, match="invalid tool schema"):
        Agent(provider=provider, model="m", reporter=Reporter()).run_sync("go")


# ── configuration ──────────────────────────────────────────────────────────


def test_preset_context_window(tmp_path, monkeypatch):
    monkeypatch.delenv("AGENT_PROFILES", raising=False)
    path = tmp_path / "p.toml"
    path.write_text('[profiles.local]\nquick = { provider = "openai", model = "m", '
                    'base_url = "http://localhost:8080/v1", context_window = 32768 }\n')
    agent = Agent(profiles=load_profiles(path), profile="local", preset="quick",
                  reporter=Reporter())
    assert agent.provider.context_window == 32768


def test_cli_context_window(monkeypatch):
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
    assert cli.main(["--provider", "openai", "--context-window", "16384", "-q", "hi"]) == 0
    assert seen["provider_options"] == {"openai": {"context_window": 16384}}
