"""Presets files, per-preset endpoints, and which provider/endpoint a run uses."""

import asyncio
import json

import pytest

from omniharness import Agent, FixedRouter, Reporter, cli
from omniharness import DEFAULT_PRESETS, load_presets, resolve_model

HOSTED = """
[openai.quick]
model = "Qwen/Qwen3-32B"
base_url = "http://vllm.internal:8000/v1"
api_key_env = "VLLM_KEY"
price_input = 0.1
price_output = 0.3

[openai.coder]
model = "qwen3-coder"
max_turns = 60
base_url = "http://vllm.internal:8001/v1"
"""


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    for var in ("AGENT_PROVIDER", "AGENT_MODEL", "AGENT_PRESET", "AGENT_PRESETS",
                "OPENAI_BASE_URL", "VLLM_KEY"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-real-openai-key")


@pytest.fixture
def hosted(tmp_path):
    path = tmp_path / "presets.toml"
    path.write_text(HOSTED)
    return path


def _agent(**kwargs):
    return Agent(provider="openai", reporter=Reporter(), **kwargs)


# ── loading ────────────────────────────────────────────────────────────────


def test_file_merges_over_defaults(hosted):
    presets = load_presets(hosted)
    quick = presets["openai"]["quick"]
    assert quick["model"] == "Qwen/Qwen3-32B"
    assert quick["max_turns"] == 30  # kept from the default preset
    assert presets["openai"]["coder"]["max_turns"] == 60
    assert presets["openai"]["standard"] == DEFAULT_PRESETS["openai"]["standard"]
    assert presets["claude"] == DEFAULT_PRESETS["claude"]
    assert DEFAULT_PRESETS["openai"]["quick"]["model"] == "gpt-6-luna"  # not mutated
    assert load_presets(hosted, base={})["openai"].keys() == {"quick", "coder"}


def test_json_file(tmp_path):
    path = tmp_path / "p.json"
    path.write_text(json.dumps({"openai": {"quick": {"model": "m", "base_url": "http://h"}}}))
    assert load_presets(path)["openai"]["quick"]["base_url"] == "http://h"


@pytest.mark.parametrize(("text", "error"), [
    ('[openai.quick]\nmodle = "x"\n', "unknown field"),
    ('[openai.new]\nmax_turns = 3\n', "has no model"),
    ('[openai.quick]\napi_key = "sk-..."\n', "api_key_env"),
    ('[claude.quick]\napi_key_env = "K"\n', "unknown field"),  # claude: base_url only
])
def test_invalid_files(tmp_path, text, error):
    path = tmp_path / "p.toml"
    path.write_text(text)
    with pytest.raises(ValueError, match=error):
        load_presets(path)


# ── endpoints ──────────────────────────────────────────────────────────────


def test_preset_endpoint_configures_the_provider(hosted, monkeypatch):
    monkeypatch.setenv("VLLM_KEY", "vllm-secret")
    agent = _agent(preset="quick", presets=load_presets(hosted))
    p = agent.provider
    assert (agent.model, p.base_url, p.api_key) == (
        "Qwen/Qwen3-32B", "http://vllm.internal:8000/v1", "vllm-secret")
    assert (p.price_input, p.price_output) == (0.1, 0.3)


def test_openai_key_never_sent_to_another_host(hosted):
    agent = _agent(preset="coder", presets=load_presets(hosted))
    assert agent.provider.base_url == "http://vllm.internal:8001/v1"
    assert agent.provider.api_key == "EMPTY"


def test_missing_api_key_env_is_an_error(hosted):
    with pytest.raises(ValueError, match="VLLM_KEY is not set"):
        _agent(preset="quick", presets=load_presets(hosted))


def test_presets_without_endpoint_use_the_usual_settings(hosted, monkeypatch):
    monkeypatch.setenv("OPENAI_BASE_URL", "https://gw/v1")
    agent = _agent(preset="standard", presets=load_presets(hosted))
    assert (agent.provider.base_url, agent.provider.api_key) == ("https://gw/v1", None)


def test_explicit_provider_options_win(hosted):
    agent = _agent(preset="coder", presets=load_presets(hosted),
                   provider_options={"base_url": "http://override/v1"})
    assert agent.provider.base_url == "http://override/v1"


def test_endpoint_preset_with_a_provider_instance_is_an_error(hosted):
    from omniharness.providers.openai_provider import OpenAIProvider
    with pytest.raises(ValueError, match="pass the provider by name"):
        Agent(provider=OpenAIProvider(client=object()), preset="coder",
              presets=load_presets(hosted), reporter=Reporter())


def test_agent_presets_env(hosted, monkeypatch):
    monkeypatch.setenv("AGENT_PRESETS", str(hosted))
    assert _agent(preset="coder").model == "qwen3-coder"
    assert resolve_model("openai", "coder").model == "qwen3-coder"


# ── auto ───────────────────────────────────────────────────────────────────


def test_auto_switches_endpoint_with_the_routed_preset(hosted, monkeypatch):
    monkeypatch.setenv("VLLM_KEY", "vllm-secret")
    presets = load_presets(hosted)
    picks = {}
    for preset in ("quick", "standard"):
        agent = _agent(model="auto", presets=presets, router=FixedRouter(preset))
        model, _, provider, _ = asyncio.run(agent._select_model("x"))
        picks[preset] = (model, provider.base_url)
    assert picks == {"quick": ("Qwen/Qwen3-32B", "http://vllm.internal:8000/v1"),
                     "standard": ("gpt-6.1-sol", None)}


def test_agent_model_env_auto_routes_to_real_models(monkeypatch):
    # Regression: AGENT_MODEL=auto used to make every routed run use model "auto".
    monkeypatch.setenv("AGENT_MODEL", "auto")
    agent = _agent(router=FixedRouter("quick"))
    assert agent.auto
    model, max_turns, _, decision = asyncio.run(agent._select_model("x"))
    assert (model, max_turns, decision.model) == ("gpt-6-luna", 30, "gpt-6-luna")


# ── CLI ────────────────────────────────────────────────────────────────────


def test_cli_list_presets_shows_the_file(hosted, capsys):
    assert cli.main(["--list-presets", "--presets", str(hosted)]) == 0
    out = capsys.readouterr().out
    assert "Qwen/Qwen3-32B" in out and "@ http://vllm.internal:8000/v1" in out
    assert "coder" in out


def test_cli_bad_presets_file(tmp_path, capsys):
    path = tmp_path / "p.toml"
    path.write_text('[openai.quick]\nmodle = "x"\n')
    assert cli.main(["--presets", str(path), "hi"]) == 1
    assert "unknown field" in capsys.readouterr().err


def test_example_presets_file_is_valid():
    from pathlib import Path
    example = Path(__file__).resolve().parent.parent / "examples" / "presets.example.toml"
    presets = load_presets(example)
    assert presets["openai"]["coder"]["base_url"] == "http://vllm.internal:8001/v1"
