"""model="auto": the d1 router and its use by Agent (fake d1 server, fake LLM backend)."""

import logging
import os
import stat
import sys
from pathlib import Path
from types import SimpleNamespace as NS

import fake_d1
import pytest

from omniharness import Agent, D1Router, FixedRouter, Reporter, resolve_model
from omniharness.providers.openai_provider import OpenAIProvider
from omniharness.routing import clamp


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    for var in ("AGENT_PROVIDER", "AGENT_MODEL", "AGENT_PRESET", "D1_URL", "D1_MODEL",
                "LLAMA_SERVER"):
        monkeypatch.delenv(var, raising=False)
    fake_d1.Handler.requests = []


@pytest.fixture
def d1_url():
    server = fake_d1.serve()
    yield f"http://127.0.0.1:{server.server_port}"
    server.shutdown()


# ── router ─────────────────────────────────────────────────────────────────


def test_tier_cutoffs():
    router = D1Router(url="http://unused", method="complexity")
    assert [router.preset_for(x) for x in (0, 1.49, 1.5, 3.49, 3.5, 4)] == [
        "quick", "quick", "standard", "standard", "thorough", "thorough"]
    with pytest.raises(ValueError, match="no upper bound"):
        D1Router(url="x", tiers=[(2.0, "quick")])
    with pytest.raises(ValueError, match="unknown method"):
        D1Router(url="x", method="vibes")


@pytest.mark.parametrize(("preset", "allowed", "expected"), [
    ("thorough", None, "thorough"),
    ("thorough", ["quick", "standard"], "standard"),
    ("quick", ["standard", "thorough"], "standard"),
    ("standard", ["quick", "thorough"], "thorough"),  # tie: the more capable
    ("standard", ["quick"], "quick"),
])
def test_clamp(preset, allowed, expected):
    assert clamp(preset, ["quick", "standard", "thorough"], allowed) == expected


@pytest.mark.parametrize(("label", "preset"), [
    ("fast", "quick"), ("capable", "standard"), ("expert", "thorough"),
])
async def test_d1_choice(d1_url, label, preset):
    decision = await D1Router(d1_url).route(f"Do something [pick:{label}]", "be terse")
    assert (decision.source, decision.method, decision.preset, decision.label) == (
        "d1", "choice", preset, label)
    assert decision.probabilities[preset] == 0.9 and set(decision.probabilities) == {
        "quick", "standard", "thorough"}
    sent = fake_d1.Handler.requests[-1]
    assert sent["state"].startswith("User prompt:\n") and "be terse" not in sent["state"]
    question = sent["questions"]["route"]
    assert question["type"] == "choice" and list(question["criteria"]) == [
        "fast", "capable", "expert"]


async def test_d1_choice_is_clamped_but_sees_every_option(d1_url):
    decision = await D1Router(d1_url).route("[pick:expert]", allowed=["quick", "standard"])
    assert (decision.choice, decision.preset, decision.clamped) == ("thorough", "standard", True)
    assert len(fake_d1.Handler.requests[-1]["questions"]["route"]["criteria"]) == 3


async def test_custom_choices(d1_url):
    router = D1Router(d1_url, choices={"cheap": ("small", "easy stuff"),
                                       "smart": ("big", "hard stuff")})
    assert router.tier_order == ["cheap", "smart"]
    assert (await router.route("[pick:big]")).preset == "smart"


@pytest.mark.parametrize(("level", "preset", "name"), [
    (0, "quick", "trivial"), (2, "standard", "moderate"), (4, "thorough", "very complex"),
])
async def test_d1_complexity(d1_url, level, preset, name):
    decision = await D1Router(d1_url, method="complexity").route(f"x [level:{level}]")
    assert (decision.method, decision.preset, decision.label) == ("complexity", preset, name)
    assert fake_d1.Handler.requests[-1]["questions"]["route"]["type"] == "score"


async def test_system_prompt_and_long_prompt_truncation(d1_url):
    router = D1Router(d1_url, include_system_prompt=True)
    await router.route("x" * 50_000, "be terse")
    state = fake_d1.Handler.requests[-1]["state"]
    assert state.startswith("Agent instructions:\nbe terse") and "[...]" in state
    assert len(state) < 25_000


async def test_http_error_falls_back(d1_url, caplog):
    decision = await D1Router(d1_url, fallback_preset="quick").route("[fail]")
    assert (decision.source, decision.preset) == ("fallback", "quick")
    assert "HTTP 500" in decision.reason and "HTTP 500" in caplog.text


async def test_unreachable_server_falls_back():
    router = D1Router("http://127.0.0.1:9", request_timeout=2)
    decision = await router.route("hi")
    assert decision.source == "fallback" and decision.preset == "standard"
    # The agent's own fallback wins; otherwise the router's, clamped to what's allowed.
    assert (await router.route("hi", fallback="quick")).preset == "quick"
    assert (await router.route("hi", allowed=["quick"])).preset == "quick"


async def test_missing_llama_server_falls_back_once(monkeypatch, caplog):
    monkeypatch.setenv("PATH", "/nonexistent")
    router = D1Router()
    with caplog.at_level(logging.WARNING):
        first = await router.route("a")
        second = await router.route("b")
    assert first.source == second.source == "fallback"
    assert "llama-server not found" in first.reason
    assert caplog.text.count("llama-server not found") == 1


@pytest.fixture
def fake_llama(tmp_path, monkeypatch):
    """A fake llama-server (our fake d1 under the venv's python) and a model in an HF cache."""
    binary = tmp_path / "llama-server"
    binary.write_text(f'#!/bin/sh\nexec {sys.executable} {Path(fake_d1.__file__)} "$@"\n')
    binary.chmod(binary.stat().st_mode | stat.S_IEXEC)
    snapshot = tmp_path / "hub" / "models--LiquidAI--d1-2B-GGUF" / "snapshots" / "abc"
    snapshot.mkdir(parents=True)
    (snapshot / "d1-2B-Q8_0.gguf").write_text("")
    monkeypatch.setenv("HF_HUB_CACHE", str(tmp_path / "hub"))
    return NS(binary=str(binary), model=str(snapshot / "d1-2B-Q8_0.gguf"))


async def test_autostart_runs_llama_server(fake_llama, monkeypatch):
    # Lots of non-UTF-8 log output: must neither block the server nor break decoding.
    monkeypatch.setenv("FAKE_D1_NOISE_KB", "256")
    router = D1Router(llama_server=fake_llama.binary, startup_timeout=20)
    try:
        decision = await router.route("Plan a migration [pick:capable]")
        assert (decision.source, decision.preset) == ("d1", "standard")
        assert router.url.startswith("http://127.0.0.1:")
        process, log_path = router._process, router.log_path
        assert "-ub" in process.args and fake_llama.model in process.args
        assert os.path.getsize(log_path) > 200_000
    finally:
        router.close()
    assert process.poll() is not None
    assert not os.path.exists(log_path)


async def test_llama_server_crash_reports_its_log(fake_llama, monkeypatch):
    monkeypatch.setenv("FAKE_D1_CRASH", "1")
    router = D1Router(llama_server=fake_llama.binary, startup_timeout=20)
    decision = await router.route("hi")
    assert decision.source == "fallback"
    assert "exited with code 1" in decision.reason
    assert "failed to load model" in decision.reason and router.log_path in decision.reason
    assert os.path.exists(router.log_path)  # kept for debugging
    os.unlink(router.log_path)


async def test_too_old_llama_server_gets_a_hint(fake_llama, monkeypatch):
    monkeypatch.setenv("FAKE_D1_CRASH", "arch")
    router = D1Router(llama_server=fake_llama.binary, startup_timeout=20)
    decision = await router.route("hi")
    assert fake_llama.binary in decision.reason
    assert "too old for d1-2B" in decision.reason and "LLAMA_SERVER" in decision.reason
    os.unlink(router.log_path)


# ── config ─────────────────────────────────────────────────────────────────


def test_resolve_auto(monkeypatch):
    assert resolve_model(model="auto").model == "auto"
    assert resolve_model(preset="auto").model == "auto"
    monkeypatch.setenv("AGENT_PRESET", "auto")
    assert resolve_model().model == "auto"
    assert resolve_model(preset="quick").model == "claude-haiku-4-5"


# ── Agent ──────────────────────────────────────────────────────────────────


class EchoResponses:
    """Fake Responses API: answers with the model it was called with."""

    def __init__(self):
        self.with_raw_response = self
        self.models = []

    async def create(self, **params):
        self.models.append(params["model"])
        response = NS(output=[NS(type="message", content=[NS(text=params["model"])])],
                      usage=NS(input_tokens=1, output_tokens=1, input_tokens_details=None,
                               output_tokens_details=None))
        return NS(parse=lambda: response, headers={})


def _auto_agent(router, **kwargs):
    responses = EchoResponses()
    agent = Agent(provider=OpenAIProvider(client=NS(responses=responses)), model="auto",
                  router=router, reporter=Reporter(), **kwargs)
    return agent, responses


async def test_agent_routes_each_run(d1_url):
    agent, responses = _auto_agent(D1Router(d1_url))
    assert agent.auto and agent.model == "auto"

    easy = await agent.run("What is 2+2? [pick:fast]")
    hard = await agent.run("Design a consensus protocol [pick:expert]")

    assert responses.models == ["gpt-5-mini", "gpt-5"]
    assert easy.final_text == "gpt-5-mini" and easy.route.preset == "quick"
    assert easy.route.model == "gpt-5-mini" and easy.route.source == "d1"
    assert hard.route.preset == "thorough"


async def test_agent_auto_presets_limit_and_fallback(d1_url):
    agent, responses = _auto_agent(D1Router(d1_url), auto_presets=["quick", "standard"],
                                   auto_fallback="quick")
    assert agent.auto_presets == ["quick", "standard"]
    result = await agent.run("Design a consensus protocol [pick:expert]")
    assert (result.route.choice, result.route.preset) == ("thorough", "standard")
    # d1 down: this agent's fallback, not the router's "standard".
    down, responses = _auto_agent(D1Router("http://127.0.0.1:9", request_timeout=2),
                                  auto_presets=["quick", "standard"], auto_fallback="quick")
    result = await down.run("anything")
    assert result.route.source == "fallback" and responses.models == ["gpt-5-mini"]


def test_agent_auto_settings_are_checked():
    router = D1Router("http://unused")
    with pytest.raises(ValueError, match="unusable"):
        _auto_agent(router, auto_presets=["quick", "turbo"])
    with pytest.raises(ValueError, match="auto_fallback 'thorough' is not in"):
        _auto_agent(router, auto_presets=["quick", "standard"], auto_fallback="thorough")
    with pytest.raises(ValueError, match="need model='auto'"):
        Agent(provider=OpenAIProvider(client=object()), model="gpt-5", auto_presets=["quick"])
    # Presets the provider lacks are left out.
    agent, _ = _auto_agent(router, presets={"openai": {"quick": {"model": "m"}}})
    assert agent.auto_presets == ["quick"]


async def test_max_turns_from_routed_preset_unless_explicit():
    agent, _ = _auto_agent(FixedRouter("quick"))
    assert (await agent._select_model("x"))[1] == 30
    agent, _ = _auto_agent(FixedRouter("quick"), max_turns=7)
    assert (await agent._select_model("x"))[1] == 7


async def test_fallback_is_noted_on_result():
    agent, responses = _auto_agent(D1Router("http://127.0.0.1:9", request_timeout=2))
    result = await agent.run("hi")
    assert responses.models == ["gpt-5"]
    assert result.success and result.route.source == "fallback"
    assert "fell back to preset 'standard'" in result.notes[0]


async def test_auto_subagent_routes_per_task(d1_url):
    worker, responses = _auto_agent(D1Router(d1_url))
    tool = worker.as_tool("worker", "Does work.")
    await tool.ainvoke({"task": "list files [pick:fast]"}, None)
    await tool.ainvoke({"task": "audit the architecture [pick:capable]"}, None)
    assert responses.models == ["gpt-5-mini", "gpt-5"]


async def test_explicit_model_never_routes():
    class Boom(FixedRouter):
        async def route(self, prompt, system_prompt=""):
            raise AssertionError("routed")

    responses = EchoResponses()
    agent = Agent(provider=OpenAIProvider(client=NS(responses=responses)), model="gpt-5",
                  router=Boom("quick"), reporter=Reporter())
    result = await agent.run("hi")
    assert not agent.auto and result.route is None and responses.models == ["gpt-5"]
