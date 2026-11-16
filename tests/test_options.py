"""AgentOptions: the omniharness options, reusable in scripts (no model calls)."""

import argparse

import pytest

from omniharness import Agent, AgentOptions, D1Router, Reporter


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    for var in ("AGENT_PROVIDER", "AGENT_MODEL", "AGENT_PRESET", "AGENT_PROFILES",
                "AGENT_PROFILE", "AGENT_EFFORT", "D1_URL"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "test")


def _options(*agents):
    parser = argparse.ArgumentParser()
    parser.add_argument("repo")
    options = AgentOptions(parser)
    for name, kwargs in agents:
        options.add_agent(name, **kwargs)
    return parser, options


def test_single_agent():
    parser, options = _options((None, {}))
    args = parser.parse_args(["src", "--model", "auto", "--auto-presets", "quick,standard",
                              "--effort", "low", "--max-turns", "7"])
    kwargs = options.agent_kwargs(args)
    assert (kwargs["model"], kwargs["effort"], kwargs["max_turns"]) == ("auto", "low", 7)
    assert kwargs["auto_presets"] == ["quick", "standard"]
    assert kwargs["strict_options"] is True and kwargs["router"] is None
    assert "claude" in kwargs["profiles"]
    assert args.repo == "src"  # the script's own arguments are untouched


def test_prefixed_agents_and_shared_resources():
    parser, options = _options(("lead", {"preset": "standard"}),
                               ("worker", {"preset": "quick"}))
    args = parser.parse_args(["x", "--worker-provider", "openai", "--lead-effort", "high",
                              "--d1-url", "http://127.0.0.1:8080"])
    lead, worker = options.agent_kwargs(args, "lead"), options.agent_kwargs(args, "worker")
    assert (lead["preset"], lead["provider"], lead["effort"]) == ("standard", None, "high")
    assert (worker["preset"], worker["provider"], worker["effort"]) == ("quick", "openai", None)
    # One profiles table and one d1 server for every agent.
    assert lead["profiles"] is worker["profiles"]
    assert isinstance(lead["router"], D1Router) and lead["router"] is worker["router"]
    assert lead["strict_options"] is worker["strict_options"] is False


def test_endpoint_options():
    parser, options = _options((None, {}))
    kwargs = options.agent_kwargs(parser.parse_args(["x", "--base-url", "http://h/v1"]))
    assert kwargs["provider_options"] == {"base_url": "http://h/v1"}
    args = parser.parse_args(["x", "--provider", "openai", "--base-url", "http://h/v1",
                              "--openai-api", "chat", "--context-window", "8192"])
    assert options.agent_kwargs(args)["provider_options"] == {
        "openai": {"api": "chat", "context_window": 8192, "base_url": "http://h/v1"}}
    with pytest.raises(ValueError, match="needs --provider"):
        options.agent_kwargs(parser.parse_args(["x", "--base-url", "u", "--openai-org", "o"]))


def test_misuse():
    _, options = _options(("worker", {}))
    with pytest.raises(ValueError, match="already added"):
        options.add_agent("worker")
    with pytest.raises(ValueError, match="add_agent"):
        options.agent_kwargs(argparse.Namespace(), "lead")


def test_agents_build_from_the_options():
    # A Claude lead and OpenAI workers share --openai-org: the lead ignores it.
    parser, options = _options(("lead", {}), ("worker", {"preset": "quick"}))
    args = parser.parse_args(["x", "--worker-provider", "openai", "--openai-org", "org-1"])
    lead = Agent(**options.agent_kwargs(args, "lead"), reporter=Reporter())
    worker = Agent(**options.agent_kwargs(args, "worker"), reporter=Reporter())
    assert (lead.provider.name, worker.provider.name) == ("claude", "openai")
    assert worker.provider.organization == "org-1"


def test_strict_options_on_agent():
    options = {"openai": {"organization": "o"}}
    with pytest.raises(ValueError, match="provider_options for"):
        Agent(provider="claude", provider_options=options, reporter=Reporter())
    agent = Agent(provider="claude", provider_options=options, strict_options=False,
                  reporter=Reporter())
    assert agent.provider.name == "claude"
