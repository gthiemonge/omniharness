import pytest

from omniharness import load_skill, resolve_model


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    for var in ("AGENT_PROVIDER", "AGENT_MODEL", "AGENT_PRESET"):
        monkeypatch.delenv(var, raising=False)


def test_resolve_defaults():
    cfg = resolve_model()
    assert (cfg.provider, cfg.model, cfg.max_turns) == ("claude", "claude-sonnet-5-5", 50)


def test_resolve_precedence(monkeypatch):
    monkeypatch.setenv("AGENT_PROVIDER", "openai")
    monkeypatch.setenv("AGENT_PRESET", "quick")
    assert resolve_model().model == "gpt-6-luna"
    assert resolve_model(preset="standard").model == "gpt-6.1-sol"
    monkeypatch.setenv("AGENT_MODEL", "my-model")
    assert resolve_model(preset="standard").model == "my-model"
    assert resolve_model(model="explicit").model == "explicit"


def test_resolve_custom_presets_and_errors():
    presets = {"claude": {"fast": {"model": "m", "max_turns": 5}}}
    cfg = resolve_model(preset="fast", presets=presets)
    assert (cfg.model, cfg.max_turns) == ("m", 5)
    with pytest.raises(ValueError, match="Unknown preset"):
        resolve_model(preset="nope", presets=presets)
    with pytest.raises(ValueError, match="No presets"):
        resolve_model(provider="other", presets=presets)


def test_load_skill(tmp_path):
    (tmp_path / "SKILL.md").write_text(
        "---\nname: demo\ndescription: \"A demo\"\n---\n\nRun $ARGUMENTS in $SKILL_DIR.\n"
    )
    skill = load_skill(tmp_path)
    assert skill.name == "demo"
    assert skill.description == "A demo"
    assert skill.body == "Run $ARGUMENTS in $SKILL_DIR.\n"
    assert skill.render(ARGUMENTS="42", SKILL_DIR="/s") == "Run 42 in /s.\n"
