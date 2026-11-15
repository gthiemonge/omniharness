"""Profiles: presets of one set, each with its own provider (no model calls)."""

import asyncio

import pytest

from omniharness import Agent, FixedRouter, Reporter, cli, load_profiles, resolve_model
from omniharness.config import format_profiles

PROFILES = """
[profiles.hybrid]
quick = { provider = "openai", model = "qwen3-8b", base_url = "http://localhost:8080/v1" }
standard = { provider = "claude", model = "claude-sonnet-5-5" }
thorough = { provider = "claude", model = "claude-opus-5-5" }

[profiles.offline]
quick = { provider = "openai", model = "qwen3-8b", base_url = "http://localhost:8080/v1" }
standard = { provider = "openai", model = "qwen3-32b", base_url = "http://gpu-box:8000/v1" }
thorough = { provider = "openai", model = "qwen3-32b", base_url = "http://gpu-box:8000/v1" }
"""


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    for var in ("AGENT_PROVIDER", "AGENT_MODEL", "AGENT_PRESET", "AGENT_PROFILES",
                "AGENT_PROFILE", "AGENT_EFFORT", "OPENAI_BASE_URL"):
        monkeypatch.delenv(var, raising=False)


@pytest.fixture
def presets(tmp_path):
    path = tmp_path / "profiles.toml"
    path.write_text(PROFILES)
    return load_profiles(path)


def _agent(presets, **kwargs):
    return Agent(profiles=presets, reporter=Reporter(), **kwargs)


# ── loading ────────────────────────────────────────────────────────────────


def test_profiles_load_next_to_the_builtin_ones(presets):
    assert list(presets) == ["claude", "openai", "hybrid", "offline"]
    assert presets["hybrid"]["thorough"]["provider"] == "claude"
    assert presets["openai"]["quick"]["provider"] == "openai"
    listing = format_profiles(presets)
    assert "openai/qwen3-8b" in listing and "@ http://localhost:8080/v1" in listing
    assert "claude/claude-opus-5-5" in listing  # provider shown when it isn't the profile's


@pytest.mark.parametrize(("text", "error"), [
    ('[profiles.mine]\nquick = { model = "m" }\n', "unknown provider 'mine'"),
    ('[profiles.mine]\nquick = { provider = "gemini", model = "m" }\n', "unknown provider"),
    ('[openai.quick]\nprovider = "claude"\n', "can't set provider"),
    ('[profiles.mine]\nquick = { provider = "claude", model = "m", api_key_env = "K" }\n',
     "unknown field"),
])
def test_invalid_profiles(tmp_path, text, error):
    path = tmp_path / "p.toml"
    path.write_text(text)
    with pytest.raises(ValueError, match=error):
        load_profiles(path)


def test_overriding_one_field_keeps_the_presets_provider(tmp_path, presets):
    path = tmp_path / "more.toml"
    path.write_text('[profiles.hybrid.thorough]\nmodel = "claude-fable-5-2"\n')
    merged = load_profiles(path, base=presets)
    assert merged["hybrid"]["thorough"] == {"provider": "claude", "model": "claude-fable-5-2"}


# ── resolution ─────────────────────────────────────────────────────────────


def test_each_preset_brings_its_provider(presets):
    quick = _agent(presets, profile="hybrid", preset="quick")
    assert (quick.provider.name, quick.model, quick.provider.base_url) == (
        "openai", "qwen3-8b", "http://localhost:8080/v1")
    thorough = _agent(presets, profile="hybrid", preset="thorough")
    assert (thorough.provider.name, thorough.model) == ("claude", "claude-opus-5-5")


def test_agent_profile_env(presets, monkeypatch):
    monkeypatch.setenv("AGENT_PROFILE", "hybrid")
    assert _agent(presets, preset="quick").model == "qwen3-8b"
    # A profile wins over AGENT_PROVIDER, which only picks a profile when none is set.
    monkeypatch.setenv("AGENT_PROVIDER", "openai")
    assert _agent(presets, preset="thorough").provider.name == "claude"


def test_explicit_model_uses_the_profiles_default_provider(presets):
    agent = _agent(presets, profile="offline", model="qwen3-coder")
    assert (agent.provider.name, agent.model) == ("openai", "qwen3-coder")


def test_explicit_provider_conflicting_with_the_preset(presets):
    with pytest.raises(ValueError, match="uses provider 'claude', not 'openai'"):
        resolve_model("openai", "thorough", profile="hybrid", profiles=presets)


def test_old_style_tables_still_work():
    agent = _agent({"openai": {"quick": {"model": "m"}}}, provider="openai", preset="quick")
    assert (agent.provider.name, agent.model) == ("openai", "m")


# ── auto ───────────────────────────────────────────────────────────────────


def test_auto_routes_across_providers(presets):
    picks = {}
    for preset in ("quick", "standard", "thorough"):
        agent = _agent(presets, profile="hybrid", model="auto", router=FixedRouter(preset))
        model, _, provider, decision, _ = asyncio.run(agent._select_model("x"))
        picks[preset] = (provider.name, model, decision.model)
    assert picks == {
        "quick": ("openai", "qwen3-8b", "qwen3-8b"),
        "standard": ("claude", "claude-sonnet-5-5", "claude-sonnet-5-5"),
        "thorough": ("claude", "claude-opus-5-5", "claude-opus-5-5"),
    }


def test_offline_profile_builds_no_cloud_provider(presets):
    agent = _agent(presets, profile="offline", model="auto", router=FixedRouter("quick"))
    built = {(p.name, p.base_url) for p in agent._providers.values()}
    assert built == {("openai", "http://localhost:8080/v1"), ("openai", "http://gpu-box:8000/v1")}


# ── provider_options ───────────────────────────────────────────────────────


def test_flat_options_are_ambiguous_on_a_mixed_profile(presets):
    with pytest.raises(ValueError, match="ambiguous"):
        _agent(presets, profile="hybrid", model="auto", router=FixedRouter("quick"),
               provider_options={"base_url": "http://x/v1"})


def test_keyed_options_go_to_their_provider(presets):
    agent = _agent(presets, profile="hybrid", model="auto", router=FixedRouter("quick"),
                   provider_options={"openai": {"price_input": 0.0, "price_output": 0.0},
                                     "claude": {"base_url": "https://gw/anthropic"}})
    _, _, quick, _, _ = asyncio.run(agent._select_model("x"))
    assert (quick.price_input, quick.base_url) == (0.0, "http://localhost:8080/v1")
    agent.router = FixedRouter("standard")
    _, _, standard, _, _ = asyncio.run(agent._select_model("x"))
    assert standard.base_url == "https://gw/anthropic"


def test_options_for_an_unused_provider_are_rejected(presets):
    with pytest.raises(ValueError, match=r"provider_options for \['claude'\]"):
        _agent(presets, profile="offline", preset="quick",
               provider_options={"claude": {"base_url": "x"}})


# ── CLI ────────────────────────────────────────────────────────────────────


def test_cli_profile(tmp_path, monkeypatch):
    path = tmp_path / "p.toml"
    path.write_text(PROFILES)
    seen = {}

    class FakeAgent:
        def __init__(self, **kwargs):
            seen.update(kwargs)

        def run_sync(self, prompt):
            from omniharness import RunResult
            return RunResult(success=True)

    monkeypatch.setattr(cli, "Agent", FakeAgent)
    assert cli.main(["--profiles", str(path), "--profile", "hybrid", "--model", "auto",
                     "-q", "hi"]) == 0
    assert seen["profile"] == "hybrid" and "hybrid" in seen["profiles"]


# ── effort ─────────────────────────────────────────────────────────────────

EFFORT_PROFILES = """
[profiles.tiers]
quick = { provider = "openai", model = "small", effort = "low" }
standard = { provider = "openai", model = "mid" }
thorough = { provider = "claude", model = "claude-opus-5-5", effort = "xhigh" }
"""


@pytest.fixture
def effort_profiles(tmp_path):
    path = tmp_path / "effort.toml"
    path.write_text(EFFORT_PROFILES)
    return load_profiles(path)


def test_effort_precedence(effort_profiles, monkeypatch):
    def effort(**kwargs):
        return _agent(effort_profiles, profile="tiers", **kwargs).effort

    assert effort(preset="quick") == "low"
    assert effort(preset="standard") is None  # provider default
    monkeypatch.setenv("AGENT_EFFORT", "medium")
    assert effort(preset="quick") == "low"  # the preset wins over AGENT_EFFORT
    assert effort(preset="standard") == "medium"  # AGENT_EFFORT fills the gap
    assert effort(preset="quick", effort="max") == "max"  # an explicit effort wins


def test_auto_takes_the_routed_presets_effort(effort_profiles):
    efforts = {}
    for preset in ("quick", "standard", "thorough"):
        agent = _agent(effort_profiles, profile="tiers", model="auto",
                       router=FixedRouter(preset))
        *_, decision, effort = asyncio.run(agent._select_model("x"))
        assert decision.effort == effort
        efforts[preset] = effort
    assert efforts == {"quick": "low", "standard": None, "thorough": "xhigh"}


async def test_effort_reaches_the_request():
    from types import SimpleNamespace as NS

    from omniharness.providers.openai_provider import OpenAIProvider

    class Responses:
        def __init__(self):
            self.with_raw_response, self.params = self, None

        async def create(self, **params):
            self.params = params
            response = NS(output=[NS(type="message", content=[NS(text="ok")])], usage=None)
            return NS(parse=lambda: response, headers={})

    responses = Responses()
    agent = Agent(provider=OpenAIProvider(client=NS(responses=responses), api="responses"),
                  profiles={"openai": {"quick": {"model": "m", "effort": "low"}}},
                  preset="quick", reporter=Reporter())
    await agent.run("hi")
    assert responses.params["reasoning"] == {"effort": "low"}


def test_effort_is_shown(effort_profiles):
    from omniharness import RouteDecision
    from omniharness.reporting import format_route

    assert "effort=xhigh" in format_profiles(effort_profiles)
    line = format_route(RouteDecision(preset="thorough", source="fixed",
                                      model="claude-opus-5-5", effort="xhigh"))
    assert line.startswith("thorough -> claude-opus-5-5 (effort xhigh)")
