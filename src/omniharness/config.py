"""
Model presets and provider/model resolution.

Each application can pass its own presets; DEFAULT_PRESETS is a starting point.
Resolution order (first match wins):

  provider: explicit argument > {PREFIX}PROVIDER env > default_provider
  model:    explicit model > {PREFIX}MODEL env > preset (explicit >
            {PREFIX}PRESET env > default_preset)

A model or preset of "auto" resolves to model "auto": the Agent then asks a
router to pick a preset for each run (see routing.py).

With the default prefix "AGENT_", e.g.:
  AGENT_PROVIDER=openai AGENT_PRESET=thorough python my_agent.py
"""

import os
from dataclasses import dataclass

# Presets per provider. Model availability depends on your plan/account and
# endpoint: adjust these, or pass your own to resolve_model().
DEFAULT_PRESETS: dict[str, dict[str, dict]] = {
    "claude": {
        "quick": {"model": "claude-haiku-4-5", "max_turns": 30,
                  "description": "Fast and cheap"},
        "standard": {"model": "claude-sonnet-5", "max_turns": 50,
                     "description": "Balanced default"},
        "thorough": {"model": "claude-opus-5-5", "max_turns": 75,
                     "description": "Deepest reasoning, highest cost"},
    },
    "openai": {
        "quick": {"model": "gpt-5-mini", "max_turns": 30,
                  "description": "Fast and cheap"},
        "standard": {"model": "gpt-5", "max_turns": 50,
                     "description": "Balanced default"},
        "thorough": {"model": "gpt-5", "max_turns": 75,
                     "description": "Deepest reasoning, highest cost"},
    },
}

DEFAULT_PROVIDER = "claude"
DEFAULT_PRESET = "standard"
DEFAULT_MAX_TURNS = 50

# Model / preset value meaning "pick per run with a router" (see routing.py).
AUTO = "auto"


@dataclass
class ModelConfig:
    provider: str
    model: str
    max_turns: int
    # Human-readable explanation of where the model came from.
    source: str


def resolve_model(
    provider: str | None = None,
    preset: str | None = None,
    model: str | None = None,
    *,
    presets: dict[str, dict[str, dict]] | None = None,
    default_provider: str = DEFAULT_PROVIDER,
    default_preset: str = DEFAULT_PRESET,
    env_prefix: str = "AGENT_",
) -> ModelConfig:
    presets = presets if presets is not None else DEFAULT_PRESETS
    provider_name = (
        provider or os.getenv(f"{env_prefix}PROVIDER") or default_provider
    ).lower()

    if model:
        return ModelConfig(provider_name, model, DEFAULT_MAX_TURNS, "explicit model")
    env_model = os.getenv(f"{env_prefix}MODEL")
    if env_model:
        return ModelConfig(provider_name, env_model, DEFAULT_MAX_TURNS,
                           f"{env_prefix}MODEL={env_model}")

    provider_presets = presets.get(provider_name)
    if not provider_presets:
        raise ValueError(
            f"No presets for provider '{provider_name}' and no model given. "
            f"Providers with presets: {', '.join(presets)}"
        )
    preset_name = preset or os.getenv(f"{env_prefix}PRESET") or default_preset
    if preset_name == AUTO:
        return ModelConfig(provider_name, AUTO, DEFAULT_MAX_TURNS,
                           f"preset '{AUTO}' (provider '{provider_name}')")
    if preset_name not in provider_presets:
        raise ValueError(
            f"Unknown preset '{preset_name}' for provider '{provider_name}'. "
            f"Available: {', '.join(provider_presets)}"
        )
    p = provider_presets[preset_name]
    return ModelConfig(
        provider_name,
        p["model"],
        p.get("max_turns", DEFAULT_MAX_TURNS),
        f"preset '{preset_name}' (provider '{provider_name}')",
    )


def format_presets(presets: dict[str, dict[str, dict]] | None = None) -> str:
    presets = presets if presets is not None else DEFAULT_PRESETS
    lines = ["Available presets (by provider):"]
    for provider_name, provider_presets in presets.items():
        lines.append(f"\n{provider_name}:")
        for name, cfg in provider_presets.items():
            lines.append(
                f"  {name:10s} {cfg['model']:22s} max_turns={cfg.get('max_turns', '-')}"
                f"  {cfg.get('description', '')}"
            )
    return "\n".join(lines)
