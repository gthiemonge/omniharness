# Sample profiles files

Each file works as is with `--profiles` (or `AGENT_PROFILES`, or
`load_profiles()` in Python), merged over the built-in profiles. Copy one and
change the hosts, models and key variables to yours.

| File | Profile(s) | Setup |
|------|-----------|-------|
| [`local-llama-server.toml`](local-llama-server.toml) | `local` | Offline: every tier on local llama-server instances |
| [`ollama.toml`](ollama.toml) | `ollama` | Offline: every tier on Ollama |
| [`vllm.toml`](vllm.toml) | `vllm` | A vLLM cluster on your network, with an API key and per-tier prices |
| [`hybrid.toml`](hybrid.toml) | `hybrid`, `hybrid-openai`, `claude-openai` | Local models for cheap tiers, Claude or OpenAI for hard ones; or both clouds mixed |
| [`gateway.toml`](gateway.toml) | `openai`, `claude` (overridden) | Both providers through a company LLM gateway; `--provider` keeps working |
| [`claude-top-tier.toml`](claude-top-tier.toml) | `claude` (overridden) | Claude Fable 5.2 for thorough work, plus a hand-picked `max` preset |
| [`reference.toml`](reference.toml) | `openai` (overridden), `hybrid` | Every field, annotated |

```bash
omniharness --profiles examples/profiles/hybrid.toml --list-profiles
omniharness --profiles examples/profiles/hybrid.toml --profile hybrid --model auto "..."
AGENT_PROFILES=examples/profiles/local-llama-server.toml AGENT_PROFILE=local \
    python examples/06_subagents.py . --worker-model auto
```

- **One file at a time:** `--profiles` takes a single file, but a file can hold
  several profiles, like `hybrid.toml`.
- **Local servers** need an OpenAI-compatible API: `/v1/responses` or
  `/v1/chat/completions`. By default the harness tries the first and falls
  back to the second; `api = "chat"` on a preset skips the probe. llama.cpp
  needs `--jinja` for tool calls.
- **Models for agents:** pick models trained for tool calls (Qwen3, ...).
  Small ones (under ~8B) often stop before a multi-step task is done.
- **Prices of 0** make local runs report $0 instead of "cost unknown".
- **Keys** are never written in these files: `api_key_env` names the variable
  that holds one. A preset with its own `base_url` never gets `OPENAI_API_KEY`.
- **`--model auto`** routes among `quick`, `standard` and `thorough`; other
  presets (`max`, `coder`) are picked by hand with `--preset`.
