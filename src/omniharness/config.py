"""
Profiles, presets and model resolution.

A profile is a set of presets (tiers such as quick / standard / thorough), and
each preset names its provider, model and endpoint. The built-in profiles,
"claude" and "openai", each use one provider; a profile of your own can mix
them, e.g. a local model for quick and Claude for thorough. A profile whose
presets are all local models runs offline.

Resolution order (first match wins), with the default prefix "AGENT_":

  profile:  explicit profile > {PREFIX}PROFILE > explicit provider >
            {PREFIX}PROVIDER > default_provider
  model:    explicit model > {PREFIX}MODEL > the profile's preset (explicit >
            {PREFIX}PRESET > default_preset), which brings its provider

A model or preset of "auto" resolves to model "auto": the Agent then asks a
router to pick a preset of the profile for each run (see routing.py).

    AGENT_PROVIDER=openai AGENT_PRESET=thorough python my_agent.py
    AGENT_PROFILE=hybrid AGENT_MODEL=auto python my_agent.py

Your own profiles and presets go in a profiles file (TOML or JSON) loaded with
load_profiles(), --profiles or {PREFIX}PROFILES, merged over DEFAULT_PROFILES field
by field:

    [openai.quick]                           # a preset of the "openai" profile
    model = "Qwen/Qwen3-32B"
    base_url = "http://vllm.internal:8000/v1"
    api_key_env = "VLLM_API_KEY"             # the variable holding the key
    price_input = 0.10                       # USD per 1M tokens, for cost reports
    price_output = 0.30

    [profiles.hybrid]                        # a profile mixing providers
    quick = { provider = "openai", model = "qwen3-8b", base_url = "http://localhost:8080/v1" }
    standard = { provider = "claude", model = "claude-sonnet-5-5" }
    thorough = { provider = "claude", model = "claude-opus-5-5" }

A preset with its own base_url gets only the key named by its api_key_env, never
OPENAI_API_KEY: your OpenAI key is not sent to another host unless you say so.
Without api_key_env it sends the placeholder "EMPTY", as vLLM expects.
"""

import json
import os
from dataclasses import dataclass, field
from pathlib import Path

# Built-in profiles. A profile is a set of presets (tiers); each tier names its
# provider. These two use one provider each, and are what --provider selects.
# Model availability depends on your plan/account and endpoint: adjust these
# with a profiles file (load_profiles), or pass your own table.
DEFAULT_PROFILES: dict[str, dict[str, dict]] = {
    "claude": {
        "quick": {"provider": "claude", "model": "claude-haiku-5-5", "max_turns": 30,
                  "description": "Fast and cheap"},
        "standard": {"provider": "claude", "model": "claude-sonnet-5-5", "max_turns": 50,
                     "description": "Balanced default"},
        "thorough": {"provider": "claude", "model": "claude-opus-5-5", "max_turns": 75,
                     "description": "Deepest reasoning, highest cost"},
    },
    "openai": {
        "quick": {"provider": "openai", "model": "gpt-6-luna", "max_turns": 30,
                  "description": "Fast and cheap"},
        "standard": {"provider": "openai", "model": "gpt-6.1-sol", "max_turns": 50,
                     "description": "Balanced default"},
        "thorough": {"provider": "openai", "model": "gpt-6-astra", "max_turns": 75,
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
    "openai": {"base_url", "api_key_env", "organization", "price_input", "price_output",
               "api", "context_window"},
}
PRESET_FIELDS = {"provider", "model", "max_turns", "description", "effort"}


@dataclass
class ModelConfig:
    provider: str
    model: str
    max_turns: int
    # Human-readable explanation of where the model came from.
    source: str
    # The preset's endpoint fields (base_url, api_key_env, ...), if any.
    endpoint: dict = field(default_factory=dict)
    # The profile the presets come from.
    profile: str | None = None
    # The preset's reasoning effort, if it sets one.
    effort: str | None = None


def resolve_model(
    provider: str | None = None,
    preset: str | None = None,
    model: str | None = None,
    *,
    profile: str | None = None,
    profiles: dict[str, dict[str, dict]] | None = None,
    default_provider: str = DEFAULT_PROVIDER,
    default_preset: str = DEFAULT_PRESET,
    env_prefix: str = "AGENT_",
) -> ModelConfig:
    """Resolve the profile, then the model (see the module docstring for the order)."""
    profiles = normalize_profiles(profiles if profiles is not None else get_profiles(env_prefix))
    arg_provider = provider.lower() if provider else None
    env_provider = (os.getenv(f"{env_prefix}PROVIDER") or "").lower() or None
    profile_name = (profile or os.getenv(f"{env_prefix}PROFILE") or arg_provider
                    or env_provider or default_provider)
    table = profiles.get(profile_name)

    def run_provider() -> str:
        """The provider for an explicit model, or before "auto" picks a preset."""
        if arg_provider:
            return arg_provider
        if table:
            tier = table.get(default_preset) or next(iter(table.values()))
            return tier["provider"]
        return env_provider or default_provider

    if model:
        return ModelConfig(run_provider(), model, DEFAULT_MAX_TURNS, "explicit model",
                           profile=profile_name)
    env_model = os.getenv(f"{env_prefix}MODEL")
    if env_model:
        return ModelConfig(run_provider(), env_model, DEFAULT_MAX_TURNS,
                           f"{env_prefix}MODEL={env_model}", profile=profile_name)

    if not table:
        raise ValueError(f"No profile '{profile_name}' and no model given. "
                         f"Profiles: {', '.join(profiles)}")
    preset_name = preset or os.getenv(f"{env_prefix}PRESET") or default_preset
    if preset_name == AUTO:
        return ModelConfig(run_provider(), AUTO, DEFAULT_MAX_TURNS,
                           f"preset '{AUTO}' (profile '{profile_name}')", profile=profile_name)
    if preset_name not in table:
        raise ValueError(f"Unknown preset '{preset_name}' in profile '{profile_name}'. "
                         f"Available: {', '.join(table)}")
    config = preset_config(profile_name, preset_name, profiles)
    if arg_provider and config.provider != arg_provider:
        raise ValueError(f"{config.source} uses provider '{config.provider}', "
                         f"not '{arg_provider}'")
    return config


def preset_config(profile: str, preset: str, profiles: dict[str, dict[str, dict]]
                  ) -> ModelConfig:
    """The ModelConfig of one preset, ignoring {PREFIX}MODEL (used for routed presets)."""
    p = profiles[profile][preset]
    provider = p.get("provider", profile)
    source = f"preset '{preset}' (profile '{profile}')"
    if provider != profile:
        source = f"preset '{preset}' (profile '{profile}', provider '{provider}')"
    return ModelConfig(
        provider,
        p["model"],
        p.get("max_turns", DEFAULT_MAX_TURNS),
        source,
        {k: v for k, v in p.items() if k not in PRESET_FIELDS},
        profile,
        p.get("effort"),
    )


def profile_providers(profiles: dict[str, dict[str, dict]], profile: str) -> list[str]:
    """The providers a profile's presets use, in order of first appearance."""
    seen: list[str] = []
    for cfg in profiles.get(profile, {}).values():
        prov = cfg.get("provider", profile)
        if prov not in seen:
            seen.append(prov)
    return seen


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


# ── profiles files ─────────────────────────────────────────────────────────


def load_profiles(path: str | Path, base: dict[str, dict[str, dict]] | None = None
                 ) -> dict[str, dict[str, dict]]:
    """Read a TOML or JSON profiles file and merge it over `base` (default: DEFAULT_PROFILES).

    The file holds provider tables ([openai.quick] ...: presets of the profile
    of that name, on that provider) and/or a profiles table
    ([profiles.hybrid.quick] ...: each preset names its provider). Pass
    base={} to use the file alone.
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
    overrides = _file_to_profiles(data, str(path))
    merged = merge_profiles(DEFAULT_PROFILES if base is None else base, overrides)
    _check_fields(merged, str(path))
    return merged


def merge_profiles(base: dict[str, dict[str, dict]], overrides: dict[str, dict[str, dict]]
                  ) -> dict[str, dict[str, dict]]:
    """A new table: `overrides` merged over `base`, field by field.

    A preset without a provider gets its profile's name as provider.
    """
    merged = {prof: {name: dict(cfg) for name, cfg in table.items()}
              for prof, table in base.items()}
    for prof, table in overrides.items():
        target = merged.setdefault(prof, {})
        for name, cfg in table.items():
            target.setdefault(name, {}).update(cfg)
    merged = normalize_profiles(merged)
    for prof, table in merged.items():
        for name, cfg in table.items():
            if "model" not in cfg:
                raise ValueError(f"preset {prof}.{name} has no model")
    return merged


def normalize_profiles(profiles: dict[str, dict[str, dict]]) -> dict[str, dict[str, dict]]:
    """The table with a provider on every preset (its profile's name if missing).

    Accepts the pre-profile shape {provider: {preset: {...}}} unchanged.
    """
    if all("provider" in cfg for table in profiles.values() for cfg in table.values()):
        return profiles
    return {prof: {name: {"provider": prof, **cfg} for name, cfg in table.items()}
            for prof, table in profiles.items()}


def get_profiles(env_prefix: str = "AGENT_") -> dict[str, dict[str, dict]]:
    """DEFAULT_PROFILES, merged with the file named by {env_prefix}PROFILES if set."""
    path = os.getenv(f"{env_prefix}PROFILES")
    return load_profiles(path) if path else DEFAULT_PROFILES


def _file_to_profiles(data: object, where: str) -> dict[str, dict[str, dict]]:
    """A profiles file as {profile: {preset: cfg}}, with shape errors reported."""
    # Bad file content is a ValueError like every other profiles-file error.
    if not isinstance(data, dict):
        raise ValueError(f"{where}: expected tables of presets")  # noqa: TRY004
    profiles: dict[str, dict[str, dict]] = {}
    for key, table in data.items():
        if not isinstance(table, dict):
            raise ValueError(f"{where}: [{key}] must be a table")  # noqa: TRY004
        if key == "profiles":
            for prof, tiers in table.items():
                if not isinstance(tiers, dict):
                    raise ValueError(f"{where}: [profiles.{prof}] must be a table "  # noqa: TRY004
                                     "of presets")
                for name, cfg in tiers.items():
                    _check_preset_shape(cfg, f"profiles.{prof}.{name}", where)
                profiles.setdefault(prof, {}).update(tiers)
        else:
            # [openai.quick]: a preset of profile "openai", on provider "openai".
            for name, cfg in table.items():
                _check_preset_shape(cfg, f"{key}.{name}", where)
                if "provider" in cfg:
                    raise ValueError(f"{where}: [{key}.{name}] can't set provider: use "
                                     f"[profiles.<name>.{name}] for a preset on another "
                                     "provider")
                profiles.setdefault(key, {})[name] = {**cfg, "provider": key}
    return profiles


def _check_preset_shape(cfg: object, label: str, where: str) -> None:
    if not isinstance(cfg, dict):
        raise ValueError(f"{where}: [{label}] must be a table")  # noqa: TRY004
    if "api_key" in cfg:
        raise ValueError(f"{where}: [{label}] put keys in the environment "
                         "and name the variable with api_key_env")


def _check_fields(profiles: dict[str, dict[str, dict]], where: str) -> None:
    """Every preset names a known provider and only uses fields it understands."""
    from .providers import available_providers  # here: providers import config's users

    known = available_providers()
    for prof, table in profiles.items():
        for name, cfg in table.items():
            prov = cfg["provider"]
            if prov not in known:
                raise ValueError(f"{where}: preset {prof}.{name} has unknown provider "
                                 f"'{prov}' (presets under [profiles] must set provider; "
                                 f"known: {', '.join(known)})")
            allowed = PRESET_FIELDS | ENDPOINT_FIELDS.get(prov, ENDPOINT_FIELDS["openai"])
            unknown = sorted(set(cfg) - allowed)
            if unknown:
                raise ValueError(f"{where}: preset {prof}.{name} (provider '{prov}') has "
                                 f"unknown field(s) {unknown}; allowed: {sorted(allowed)}")


def format_profiles(profiles: dict[str, dict[str, dict]] | None = None) -> str:
    profiles = normalize_profiles(profiles if profiles is not None else DEFAULT_PROFILES)
    lines = ["Available profiles and their presets:"]
    for prof, table in profiles.items():
        lines.append(f"\n{prof}:")
        for name, cfg in table.items():
            model = cfg["model"]
            if cfg["provider"] != prof:
                model = f"{cfg['provider']}/{model}"
            line = (f"  {name:10s} {model:28s} max_turns={cfg.get('max_turns', DEFAULT_MAX_TURNS)}"
                    f"  {cfg.get('description', '')}")
            if cfg.get("effort"):
                line = f"{line.rstrip()}  effort={cfg['effort']}"
            if cfg.get("base_url"):
                line = f"{line.rstrip()}  @ {cfg['base_url']}"
            lines.append(line)
    return "\n".join(lines)
