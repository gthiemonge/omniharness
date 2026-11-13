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

Your own presets, e.g. models hosted behind an OpenAI-compatible endpoint, go
in a TOML or JSON file loaded with load_presets(), --presets or
{PREFIX}PRESETS. The file is merged over DEFAULT_PRESETS field by field:

    [openai.quick]                           # override part of a default preset
    model = "Qwen/Qwen3-32B"
    base_url = "http://vllm.internal:8000/v1"
    api_key_env = "VLLM_API_KEY"             # the variable holding the key
    price_input = 0.10                       # USD per 1M tokens, for cost reports
    price_output = 0.30

    [openai.coder]                           # or add a preset
    model = "qwen3-coder"
    max_turns = 60
    base_url = "http://vllm.internal:8001/v1"

A preset with its own base_url gets only the key named by its api_key_env, never
OPENAI_API_KEY: your OpenAI key is not sent to another host unless you say so.
Without api_key_env it sends the placeholder "EMPTY", as vLLM expects.
"""

import json
import os
from dataclasses import dataclass, field
from pathlib import Path

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


# Preset fields, besides the model, that configure the provider for that preset.
ENDPOINT_FIELDS: dict[str, set[str]] = {
    "claude": {"base_url"},
    "openai": {"base_url", "api_key_env", "organization", "price_input", "price_output"},
}
PRESET_FIELDS = {"model", "max_turns", "description"}


@dataclass
class ModelConfig:
    provider: str
    model: str
    max_turns: int
    # Human-readable explanation of where the model came from.
    source: str
    # The preset's endpoint fields (base_url, api_key_env, ...), if any.
    endpoint: dict = field(default_factory=dict)


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
    presets = presets if presets is not None else get_presets(env_prefix)
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
    return preset_config(provider_name, preset_name, presets)


def preset_config(provider: str, preset: str, presets: dict[str, dict[str, dict]]
                  ) -> ModelConfig:
    """The ModelConfig of one preset, ignoring {PREFIX}MODEL (used for routed presets)."""
    p = presets[provider][preset]
    return ModelConfig(
        provider,
        p["model"],
        p.get("max_turns", DEFAULT_MAX_TURNS),
        f"preset '{preset}' (provider '{provider}')",
        {k: v for k, v in p.items() if k not in PRESET_FIELDS},
    )


def endpoint_kwargs(provider: str, endpoint: dict) -> dict:
    """Provider constructor kwargs for a preset's endpoint fields."""
    kwargs = {k: v for k, v in endpoint.items() if k != "api_key_env"}
    if provider == "openai" and ("base_url" in endpoint or "api_key_env" in endpoint):
        var = endpoint.get("api_key_env")
        if var:
            key = os.getenv(var)
            if not key:
                raise ValueError(f"{var} is not set (api_key_env of this preset)")
            kwargs["api_key"] = key
        else:
            # Never fall back to OPENAI_API_KEY for another host.
            kwargs["api_key"] = "EMPTY"
    return kwargs


# ── presets files ──────────────────────────────────────────────────────────


def load_presets(path: str | Path, base: dict[str, dict[str, dict]] | None = None
                 ) -> dict[str, dict[str, dict]]:
    """Read a TOML or JSON presets file and merge it over `base` (default: DEFAULT_PRESETS).

    Pass base={} to use the file alone.
    """
    path = Path(path).expanduser()
    text = path.read_text()
    if path.suffix == ".json":
        data = json.loads(text)
    else:
        try:
            import tomllib
        except ImportError:  # Python 3.10
            try:
                import tomli as tomllib
            except ImportError:
                raise ValueError(f"{path}: TOML needs Python 3.11+ or `pip install tomli`; "
                                 "or use a .json file") from None
        data = tomllib.loads(text)
    _check_presets(data, str(path))
    return merge_presets(DEFAULT_PRESETS if base is None else base, data)


def merge_presets(base: dict[str, dict[str, dict]], overrides: dict[str, dict[str, dict]]
                  ) -> dict[str, dict[str, dict]]:
    """A new table: `overrides` merged over `base`, field by field."""
    merged = {prov: {name: dict(cfg) for name, cfg in table.items()}
              for prov, table in base.items()}
    for prov, table in overrides.items():
        target = merged.setdefault(prov, {})
        for name, cfg in table.items():
            target.setdefault(name, {}).update(cfg)
    for prov, table in merged.items():
        for name, cfg in table.items():
            if "model" not in cfg:
                raise ValueError(f"preset {prov}.{name} has no model")
    return merged


def get_presets(env_prefix: str = "AGENT_") -> dict[str, dict[str, dict]]:
    """DEFAULT_PRESETS, merged with the file named by {env_prefix}PRESETS if set."""
    path = os.getenv(f"{env_prefix}PRESETS")
    return load_presets(path) if path else DEFAULT_PRESETS


def _check_presets(data: object, where: str) -> None:
    # Bad file content is a ValueError like every other presets-file error.
    if not isinstance(data, dict):
        raise ValueError(f"{where}: expected a table of providers")  # noqa: TRY004
    for prov, table in data.items():
        if not isinstance(table, dict):
            raise ValueError(f"{where}: [{prov}] must be a table of presets")  # noqa: TRY004
        allowed = PRESET_FIELDS | ENDPOINT_FIELDS.get(prov, ENDPOINT_FIELDS["openai"])
        for name, cfg in table.items():
            if not isinstance(cfg, dict):
                raise ValueError(f"{where}: [{prov}.{name}] must be a table")  # noqa: TRY004
            unknown = sorted(set(cfg) - allowed)
            if unknown:
                raise ValueError(f"{where}: [{prov}.{name}] unknown field(s) {unknown}; "
                                 f"allowed: {sorted(allowed)}")
            if "api_key" in cfg:
                raise ValueError(f"{where}: [{prov}.{name}] put keys in the environment "
                                 "and name the variable with api_key_env")


def format_presets(presets: dict[str, dict[str, dict]] | None = None) -> str:
    presets = presets if presets is not None else DEFAULT_PRESETS
    lines = ["Available presets (by provider):"]
    for provider_name, provider_presets in presets.items():
        lines.append(f"\n{provider_name}:")
        for name, cfg in provider_presets.items():
            line = (f"  {name:10s} {cfg['model']:22s} max_turns={cfg.get('max_turns', '-')}"
                    f"  {cfg.get('description', '')}")
            if cfg.get("base_url"):
                line = f"{line.rstrip()}  @ {cfg['base_url']}"
            lines.append(line)
    return "\n".join(lines)
