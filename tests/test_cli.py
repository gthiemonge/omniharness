"""CLI flags reach the right Agent / provider / router arguments (no model calls)."""

import pytest

from omniharness import RunResult, cli


@pytest.fixture
def captured(monkeypatch):
    """Replace Agent with a recorder; returns the kwargs of the last Agent built."""
    for var in ("AGENT_PROVIDER", "AGENT_MODEL", "AGENT_PRESET", "AGENT_PROFILES",
                "AGENT_PROFILE", "AGENT_EFFORT"):
        monkeypatch.delenv(var, raising=False)
    calls = {}

    class FakeAgent:
        def __init__(self, **kwargs):
            calls.clear()
            calls.update(kwargs)

        def run_sync(self, prompt):
            calls["prompt"] = prompt
            return RunResult(success=True, final_text="ok")

    monkeypatch.setattr(cli, "Agent", FakeAgent)
    return calls


def test_base_url_goes_to_either_provider(captured):
    # Without --provider: for the only provider in play (the Agent decides).
    assert cli.main(["hi", "-q", "--base-url", "https://gw/anthropic"]) == 0
    assert captured["provider_options"] == {"base_url": "https://gw/anthropic"}
    assert cli.main(["hi", "-q", "--provider", "openai", "--base-url", "http://h/v1"]) == 0
    assert captured["provider_options"] == {"openai": {"base_url": "http://h/v1"}}


def test_openai_flags(captured):
    argv = ["hi", "-q", "--provider", "openai", "--openai-org", "org-1",
            "--openai-price-input", "0.5", "--openai-price-output", "2"]
    assert cli.main(argv) == 0
    assert captured["provider_options"] == {"openai": {"organization": "org-1",
                                                       "price_input": 0.5, "price_output": 2.0}}
    with pytest.raises(SystemExit):  # --base-url can't tell which provider it's for
        cli.main(["hi", "--openai-org", "org-1", "--base-url", "http://h/v1"])


def test_openai_flags_rejected_when_no_preset_uses_openai(monkeypatch, capsys):
    # The real Agent: it fails while being built, before any model call.
    for var in ("AGENT_PROVIDER", "AGENT_MODEL", "AGENT_PRESET", "AGENT_PROFILES",
                "AGENT_PROFILE"):
        monkeypatch.delenv(var, raising=False)
    assert cli.main(["hi", "--openai-org", "org-1"]) == 1
    assert "provider_options for ['openai']" in capsys.readouterr().err


def test_openai_flags_follow_agent_provider_env(captured, monkeypatch):
    monkeypatch.setenv("AGENT_PROVIDER", "openai")
    assert cli.main(["hi", "-q", "--openai-org", "org-1"]) == 0


def test_d1_flags_build_a_router(captured, tmp_path):
    assert cli.main(["hi", "-q", "--model", "auto"]) == 0
    assert captured["router"] is None  # the process-wide default router
    model = tmp_path / "d1.gguf"
    argv = ["hi", "-q", "--model", "auto", "--llama-server", "/opt/llama-server",
            "--d1-model", str(model), "--auto-presets", "quick,standard",
            "--auto-fallback", "quick"]
    assert cli.main(argv) == 0
    router = captured["router"]
    assert (router.llama_server, router.model_path) == ("/opt/llama-server", str(model))
    assert captured["auto_presets"] == ["quick", "standard"]
    assert captured["auto_fallback"] == "quick"
    assert cli.main(["hi", "-q", "--d1-url", "http://127.0.0.1:8080/"]) == 0
    assert captured["router"].url == "http://127.0.0.1:8080"


def test_env_vars_all_have_flags():
    """Every env var the package reads has a CLI flag (OPENAI_API_KEY excepted on purpose)."""
    help_text = cli.build_parser().format_help()
    for var in ("AGENT_PROVIDER", "AGENT_MODEL", "AGENT_PRESET", "AGENT_PROFILES",
                "AGENT_PROFILE", "AGENT_EFFORT", "ANTHROPIC_BASE_URL", "OPENAI_BASE_URL", "OPENAI_ORG", "OPENAI_PRICE_INPUT",
                "OPENAI_PRICE_OUTPUT", "D1_URL", "LLAMA_SERVER", "D1_MODEL"):
        assert var in help_text, var
