"""Profiles files, per-preset endpoints, and which provider/endpoint a run uses."""

import asyncio
import json
from pathlib import Path

import pytest

from omniharness import (
    DEFAULT_PROFILES,
    Agent,
    FixedRouter,
    Reporter,
    cli,
    load_profiles,
    resolve_model,
)

HOSTED = """
[openai.quick]
model = "Qwen/Qwen3-32B"
base_url = "http://vllm.internal:8000/v1"
api_key_env = "VLLM_KEY"
price_input = 0.1
price_output = 0.3

[openai.standard]
model = "gpt-6.1-sol"

[openai.coder]
model = "qwen3-coder"
max_turns = 60
base_url = "http://vllm.internal:8001/v1"
"""


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    for var in ("AGENT_PROVIDER", "AGENT_MODEL", "AGENT_PRESET", "AGENT_PROFILES",
                "OPENAI_BASE_URL", "VLLM_KEY"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-real-openai-key")


@pytest.fixture
def hosted(tmp_path):
    path = tmp_path / "profiles.toml"
    path.write_text(HOSTED)
    return path


def _agent(**kwargs):
    return Agent(provider="openai", reporter=Reporter(), **kwargs)


# ── loading ────────────────────────────────────────────────────────────────


def test_file_replaces_the_builtin_profiles(hosted):
    profiles = load_profiles(hosted)
    assert list(profiles) == ["openai"]  # no "claude" profile: the file doesn't define one
    assert list(profiles["openai"]) == ["quick", "standard", "coder"]  # no built-in "thorough"
    quick = profiles["openai"]["quick"]
    assert quick["model"] == "Qwen/Qwen3-32B" and "max_turns" not in quick  # nothing merged
    assert DEFAULT_PROFILES["openai"]["quick"]["model"] == "gpt-6-luna"  # not mutated


def test_merging_over_the_builtin_profiles_on_request(hosted):
    profiles = load_profiles(hosted, base=DEFAULT_PROFILES)
    assert list(profiles) == ["claude", "openai"]
    assert profiles["openai"]["quick"]["max_turns"] == 30  # kept from the built-in preset
    assert profiles["openai"]["thorough"] == DEFAULT_PROFILES["openai"]["thorough"]


def test_default_profile_is_the_files_first_when_claude_is_missing(hosted):
    # No --profile / --provider, and the file has no "claude" profile.
    agent = Agent(profiles=load_profiles(hosted), preset="coder", reporter=Reporter())
    assert (agent.profile, agent.model) == ("openai", "qwen3-coder")


def test_empty_file(tmp_path):
    path = tmp_path / "p.toml"
    path.write_text("# nothing\n")
    with pytest.raises(ValueError, match="defines no profile"):
        load_profiles(path)


def test_json_file(tmp_path):
    path = tmp_path / "p.json"
    path.write_text(json.dumps({"openai": {"quick": {"model": "m", "base_url": "http://h"}}}))
    assert load_profiles(path)["openai"]["quick"]["base_url"] == "http://h"


@pytest.mark.parametrize(("text", "error"), [
    ('[openai.quick]\nmodel = "m"\nmodle = "x"\n', "unknown field"),
    ('[openai.new]\nmax_turns = 3\n', "has no model"),
    ('[openai.quick]\napi_key = "sk-..."\n', "api_key_env"),
    ('[claude.quick]\nmodel = "m"\napi_key_env = "K"\n', "unknown field"),  # claude: base_url
])
def test_invalid_files(tmp_path, text, error):
    path = tmp_path / "p.toml"
    path.write_text(text)
    with pytest.raises(ValueError, match=error):
        load_profiles(path)


# ── endpoints ──────────────────────────────────────────────────────────────


def test_preset_endpoint_configures_the_provider(hosted, monkeypatch):
    monkeypatch.setenv("VLLM_KEY", "vllm-secret")
    agent = _agent(preset="quick", profiles=load_profiles(hosted))
    p = agent.provider
    assert (agent.model, p.base_url, p.api_key) == (
        "Qwen/Qwen3-32B", "http://vllm.internal:8000/v1", "vllm-secret")
    assert (p.price_input, p.price_output) == (0.1, 0.3)


def test_openai_key_never_sent_to_another_host(hosted):
    agent = _agent(preset="coder", profiles=load_profiles(hosted))
    assert agent.provider.base_url == "http://vllm.internal:8001/v1"
    assert agent.provider.api_key == "EMPTY"


def test_missing_api_key_env_is_an_error(hosted):
    with pytest.raises(ValueError, match="VLLM_KEY is not set"):
        _agent(preset="quick", profiles=load_profiles(hosted))


def test_presets_without_endpoint_use_the_usual_settings(hosted, monkeypatch):
    monkeypatch.setenv("OPENAI_BASE_URL", "https://gw/v1")
    agent = _agent(preset="standard", profiles=load_profiles(hosted))
    assert (agent.provider.base_url, agent.provider.api_key) == ("https://gw/v1", None)


def test_explicit_provider_options_win(hosted):
    agent = _agent(preset="coder", profiles=load_profiles(hosted),
                   provider_options={"base_url": "http://override/v1"})
    assert agent.provider.base_url == "http://override/v1"


def test_endpoint_preset_with_a_provider_instance_is_an_error(hosted):
    from omniharness.providers.openai_provider import OpenAIProvider
    with pytest.raises(ValueError, match="pass the provider by name"):
        Agent(provider=OpenAIProvider(client=object()), preset="coder",
              profiles=load_profiles(hosted), reporter=Reporter())


def test_agent_presets_env(hosted, monkeypatch):
    monkeypatch.setenv("AGENT_PROFILES", str(hosted))
    assert _agent(preset="coder").model == "qwen3-coder"
    assert resolve_model("openai", "coder").model == "qwen3-coder"


# ── auto ───────────────────────────────────────────────────────────────────


def test_auto_switches_endpoint_with_the_routed_preset(hosted, monkeypatch):
    monkeypatch.setenv("VLLM_KEY", "vllm-secret")
    presets = load_profiles(hosted)
    picks = {}
    for preset in ("quick", "standard"):
        agent = _agent(model="auto", profiles=presets, router=FixedRouter(preset))
        model, _, provider, _, _ = asyncio.run(agent._select_model("x"))
        picks[preset] = (model, provider.base_url)
    assert picks == {"quick": ("Qwen/Qwen3-32B", "http://vllm.internal:8000/v1"),
                     "standard": ("gpt-6.1-sol", None)}


def test_agent_model_env_auto_routes_to_real_models(monkeypatch):
    # Regression: AGENT_MODEL=auto used to make every routed run use model "auto".
    monkeypatch.setenv("AGENT_MODEL", "auto")
    agent = _agent(router=FixedRouter("quick"))
    assert agent.auto
    model, max_turns, _, decision, _ = asyncio.run(agent._select_model("x"))
    assert (model, max_turns, decision.model) == ("gpt-6-luna", 30, "gpt-6-luna")


# ── CLI ────────────────────────────────────────────────────────────────────


def test_cli_list_profiles_shows_the_file(hosted, capsys):
    assert cli.main(["--list-profiles", "--profiles", str(hosted)]) == 0
    out = capsys.readouterr().out
    assert "Qwen/Qwen3-32B" in out and "@ http://vllm.internal:8000/v1" in out
    assert "coder" in out


def test_cli_bad_presets_file(tmp_path, capsys):
    path = tmp_path / "p.toml"
    path.write_text('[openai.quick]\nmodel = "m"\nmodle = "x"\n')
    assert cli.main(["--profiles", str(path), "hi"]) == 1
    assert "unknown field" in capsys.readouterr().err


SAMPLES = sorted((Path(__file__).resolve().parent.parent / "examples" / "profiles").glob("*.toml"))


@pytest.mark.parametrize("path", SAMPLES, ids=lambda p: p.name)
def test_sample_presets_files(path, monkeypatch):
    """Every sample loads, and every profile it defines builds all its providers."""
    for var in ("VLLM_API_KEY", "VLLM_KEY", "GATEWAY_KEY"):
        monkeypatch.setenv(var, "test-key")
    presets = load_profiles(path)
    for profile in presets:
        agent = Agent(profiles=presets, profile=profile, model="auto",
                      router=FixedRouter("quick"), reporter=Reporter())
        assert agent.auto_presets
