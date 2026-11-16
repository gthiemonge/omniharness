# omniharness

A small, provider-agnostic harness for tool-using AI agents, extracted from
[openstack-review](https://github.com/gthiemonge/openstack-review-claude-skill).

You write the agent once: a system prompt, some tools, and a working directory.
The harness runs it on either backend:

| Provider | Backend | Agent loop | Built-in tools | Your Python tools |
|----------|---------|------------|----------------|-------------------|
| `claude` | [Claude Agent SDK](https://pypi.org/project/claude-agent-sdk/) (Anthropic API, Vertex AI, Bedrock) | SDK | SDK-native `Read`/`Write`/`Bash`/`Grep`/`Glob`/`WebFetch` | in-process MCP server |
| `openai` | OpenAI, or any compatible endpoint (Azure, litellm, vLLM, llama.cpp, Ollama, LM Studio, ...), over the Responses or Chat Completions API | harness | Python versions with the same names | called directly |

Both providers return the same `RunResult`: success, final text, normalized
token usage, and cost.

## Install

```bash
pip install -e '.[claude]'      # Claude only
pip install -e '.[openai]'      # OpenAI only
pip install -e '.[all,dev]'     # everything + pytest/ruff
```

Credentials are the same as each SDK's:

```bash
export ANTHROPIC_API_KEY=...              # or an existing Claude Code login
# Vertex AI: CLAUDE_CODE_USE_VERTEX=1 ANTHROPIC_VERTEX_PROJECT_ID=... CLOUD_ML_REGION=...
export OPENAI_API_KEY=...
export ANTHROPIC_BASE_URL=https://my-gateway/anthropic   # optional, custom endpoint
export OPENAI_BASE_URL=https://my-gateway/v1   # optional, OpenAI-compatible endpoint
```

## Quick start

```python
from omniharness import Agent, READ_ONLY_TOOLS, tool

@tool
def lookup_ticket(ticket_id: str) -> dict:
    """Fetch a ticket from the tracker.

    Args:
        ticket_id: e.g. 'OSPRH-1234'.
    """
    return {"id": ticket_id, "status": "open"}

agent = Agent(
    provider="claude",              # or "openai"; default: $AGENT_PROVIDER, then claude
    preset="standard",              # or model="claude-sonnet-5-5"
    system_prompt="You are a support engineer.",
    tools=[*READ_ONLY_TOOLS, lookup_ticket],
    work_dir="~/src/myproject",
)
result = agent.run_sync("Is OSPRH-1234 caused by the code in this repo?")
print(result.final_text, result.total_cost_usd)
```

From async code, use `await agent.run(prompt)`.

## Command line

```bash
omniharness "What does this project do?" --tools Read,Grep,Glob
omniharness --provider openai --model gpt-6-luna -q "Name 3 prime numbers"
omniharness --skill examples/skills/changelog --work-dir ~/src/octavia --output-dir out "since 14.0.0"
git diff | omniharness - --system "Review this diff for bugs." --json
omniharness --list-profiles
```

## Examples

All examples run as-is. Switch provider with `AGENT_PROVIDER=openai`.

| File | Shows |
|------|-------|
| [`01_hello.py`](examples/01_hello.py) | Smallest agent: no tools, one question |
| [`02_code_explorer.py`](examples/02_code_explorer.py) | Read-only Q&A over a local repo (`Read`/`Grep`/`Glob`) |
| [`03_custom_tools.py`](examples/03_custom_tools.py) | `@tool` functions (sync, typed, with `Literal`), `ToolContext` injection |
| [`04_skill_runner.py`](examples/04_skill_runner.py) + [`skills/changelog`](examples/skills/changelog/SKILL.md) | Running a Claude-style `SKILL.md` with helper scripts on any provider (the openstack-review pattern) |
| [`05_structured_output.py`](examples/05_structured_output.py) | Reliable structured output through a `submit_result` tool, plus a custom JSONL `Reporter` |
| [`06_subagents.py`](examples/06_subagents.py) | Mid-tier orchestrator delegating to low-tier `explorer` / `runner` subagents in parallel |
| [`07_auto_model.py`](examples/07_auto_model.py) | `model="auto"`: a local decision model (d1-2B) picks the model per run; `--rate-only` to check routing |

```bash
python examples/02_code_explorer.py ~/src/octavia "How are amphora health checks scheduled?"
python examples/04_skill_runner.py examples/skills/changelog ~/src/octavia "since 14.0.0"
python examples/05_structured_output.py ~/src/octavia
python examples/06_subagents.py ~/src/octavia
python examples/07_auto_model.py --rate-only
```

## Configuration

For each setting, an explicit argument wins over the environment, which wins
over the default. Every environment variable the harness reads has an argument
and a CLI flag, except `OPENAI_API_KEY`: a key on the command line would show in
`ps` and in shell history. The `AGENT_*` names match openstack-review's.

| Setting | Argument | CLI | Environment | Default |
|---------|----------|-----|-------------|---------|
| profile | `Agent(profile=)` | `--profile` | `AGENT_PROFILE` | the provider's profile |
| provider | `Agent(provider=)` | `--provider` | `AGENT_PROVIDER` | `claude` |
| model | `Agent(model=)` | `--model` | `AGENT_MODEL` | preset's model |
| preset | `Agent(preset=)` | `--preset` | `AGENT_PRESET` | `standard` |
| effort | `Agent(effort=)` | `--effort` | `AGENT_EFFORT` | the preset's `effort`, else provider default |
| endpoint (claude) | `provider_options={"base_url": ...}` | `--base-url` | `ANTHROPIC_BASE_URL` | Anthropic API |
| endpoint (openai) | `provider_options={"base_url": ...}` | `--base-url` | `OPENAI_BASE_URL` | api.openai.com |
| OpenAI organization | `provider_options={"organization": ...}` | `--openai-org` | `OPENAI_ORG` | none |
| context window (openai) | `provider_options={"context_window": N}`, preset `context_window` | `--context-window` | none | no trimming |
| OpenAI prices (USD / 1M tokens) | `provider_options={"price_input": ..., "price_output": ...}` | `--openai-price-input/-output` | `OPENAI_PRICE_INPUT/OUTPUT` | `pricing.PRICING` table |
| OpenAI API key | `provider_options={"api_key": ...}` | none | `OPENAI_API_KEY` | required |
| profiles file | `profiles=load_profiles(path)` | `--profiles` | `AGENT_PROFILES` | built-in profiles |
| d1 server (`auto`) | `D1Router(url=)` | `--d1-url` | `D1_URL` | start one |
| llama-server binary (`auto`) | `D1Router(llama_server=)` | `--llama-server` | `LLAMA_SERVER` | `PATH` |
| d1 model file (`auto`) | `D1Router(model_path=)` | `--d1-model` | `D1_MODEL` | Hugging Face cache (`HF_HUB_CACHE` / `HF_HOME`) |

The Claude SDK reads its own variables (`ANTHROPIC_API_KEY`,
`CLAUDE_CODE_USE_VERTEX`, ...) directly. To set them for one agent, use
`provider_options={"options": {"env": {...}}}`.

### Profiles, presets and hosted models

A **preset** is one tier (`quick`, `standard`, `thorough`): a provider, a
model and optionally an endpoint. A **profile** is a named set of presets. You
pick a profile once (`--profile`, or `--provider` for the built-in ones), and
`--preset` or `--model auto` picks a preset inside it for each run.

The built-in profiles, `claude` and `openai`, live in
`config.DEFAULT_PROFILES`. To change them, or to add models hosted behind an
OpenAI-compatible endpoint (vLLM, llama.cpp, TGI, a gateway, ...), write a
profiles file in TOML or JSON. It is merged over the built-in profiles field
by field. Each preset can bring its own endpoint:

```toml
[openai.quick]                       # the cheap tier, on a self-hosted model
model = "Qwen/Qwen3-32B"
base_url = "http://vllm.internal:8000/v1"
api_key_env = "VLLM_API_KEY"         # the variable that holds the key
price_input = 0.10                   # USD per 1M tokens, for cost reports
price_output = 0.30

[openai.coder]                       # a new preset
model = "qwen3-coder"
base_url = "http://vllm.internal:8001/v1"
```

```bash
omniharness --profiles my-profiles.toml --list-profiles
omniharness --profiles my-profiles.toml --provider openai --preset quick "..."
export AGENT_PROFILES=my-profiles.toml        # for every agent in the process
```

```python
from omniharness import Agent, load_profiles
agent = Agent(provider="openai", preset="quick", profiles=load_profiles("my-profiles.toml"))
```

- **Fields:** `model`, `max_turns`, `description` and `effort`, plus the
  preset's endpoint. For `openai` that's `base_url`, `api_key_env`, `organization`,
  `price_input` and `price_output`; for `claude`, `base_url` only. Anything
  else is rejected, as is a literal `api_key`.
- **Keys:** a preset with its own `base_url` never gets `OPENAI_API_KEY`, so
  your OpenAI key isn't sent to another host. It gets the key from the
  variable named by `api_key_env`, or the placeholder `EMPTY` (what vLLM
  expects) if there's no such field.
- **Explicit endpoint options win:** `--base-url` / `provider_options` beat
  the preset's endpoint.
- **With `model="auto"`,** the endpoint follows the routed preset: a task
  routed to `quick` goes to the hosted model, one routed to `standard` goes to
  OpenAI.
- **Python:** `load_profiles(path)` returns the merged table;
  `load_profiles(path, base={})` uses the file alone. `Agent(profiles=...)` takes
  a whole table.
- **Responses or Chat Completions:** the openai provider speaks both
  `/v1/responses` and `/v1/chat/completions`. With the default `api = "auto"`,
  it tries Responses first; if a server answers 404, 405 or 501, that endpoint
  switches to Chat Completions for the rest of the process, and the run's
  `RunResult.notes` says so. Set `api = "chat"` or `api = "responses"` on a
  preset (or `--openai-api`) to skip the probe. Tested with llama.cpp v0.8.0's
  `llama-server --jinja` serving Qwen2.5-1.5B-Instruct over both APIs, and
  behind a proxy that only offers Chat Completions: answers, tool calls, key
  checking (a wrong key gets a 401) and cost reporting.

### Profiles: mixing providers and local models

In a profile, each preset names its own provider. The built-in profiles use
one provider each; `--provider openai` selects the `openai` profile. Your own
profiles can mix providers, so any tier can be a local model, Claude or
OpenAI:

```toml
[profiles.hybrid]                    # a local model for quick, Claude for the rest
quick    = { provider = "openai", model = "qwen3-8b", base_url = "http://localhost:8080/v1" }
standard = { provider = "claude", model = "claude-sonnet-5-5" }
thorough = { provider = "claude", model = "claude-opus-5-5" }

[profiles.offline]                   # local models only: nothing leaves your network
quick    = { provider = "openai", model = "qwen3-8b",  base_url = "http://localhost:8080/v1" }
standard = { provider = "openai", model = "qwen3-32b", base_url = "http://gpu-box:8000/v1" }
thorough = { provider = "openai", model = "qwen3-32b", base_url = "http://gpu-box:8000/v1" }
```

```bash
omniharness --profiles my-profiles.toml --profile hybrid --preset quick "..."
omniharness --profiles my-profiles.toml --profile hybrid --model auto "..."
AGENT_PROFILES=my-profiles.toml AGENT_PROFILE=offline python examples/06_subagents.py . --worker-model auto
```

- **Selection:** `--profile` / `AGENT_PROFILE`, else the profile named by
  `--provider` / `AGENT_PROVIDER`, else `claude`. A preset brings its
  provider, model and endpoint. An explicit `--model` runs on the profile's
  `standard` preset's provider, unless `--provider` says otherwise.
- **Effort per preset:** `effort = "low"` / `"xhigh"` / ... on a preset is
  used for its runs; with `auto`, each run takes the routed preset's effort.
  An explicit `effort=` / `--effort` wins; a preset's effort wins over
  `AGENT_EFFORT`, which only fills in for presets without one (unlike
  `AGENT_MODEL`, which overrides presets). Values go to the provider as is:
  Claude takes `low` to `max`, OpenAI its reasoning levels; leave it unset for
  servers that reject it.
- **With `model="auto"`,** d1 picks the tier and the tier picks the provider.
  With `hybrid`, a lookup goes to the local model and a design question to
  Claude Opus. `auto_presets` limits the tiers as usual.
- **Offline by design:** if every preset of the profile is local, an agent
  using it never creates a cloud client, not even before its first run.
- **Provider options on a mixed profile** are keyed by provider:
  `provider_options={"openai": {...}, "claude": {...}}`. The flat form is an
  error when it would be ambiguous, and options for a provider the profile
  doesn't use are rejected. On the CLI, `--base-url` needs `--provider` then.
- **File syntax:** `[profiles.<name>.<preset>]` or inline tables, as above;
  each new preset must set `provider`. `[openai.<preset>]` is shorthand for a
  preset of the `openai` profile on the `openai` provider.

### Context windows

Local models often have a small context window (8K to 32K tokens), and an agent
fills it fast: every tool output stays in the conversation, which the openai
provider resends each turn. Set the model's window with `context_window =
32768` on a preset (or `--context-window`), and the provider keeps each request
inside it, in steps:

1. **Every tool output is capped** at a quarter of the budget (the window
   minus room for the reply). Built-in tools cut their own output; custom
   tools' outputs are cut by the provider.
2. **Older tool outputs are shortened** to their first 500 characters, oldest
   first, until the request fits; the last two turns stay whole.
3. **The oldest turns are dropped whole** if that isn't enough: a tool call
   always goes with its result, the task itself is never dropped, and a note
   tells the model that earlier steps were removed.
4. **If the server still says the request is too long,** the provider trims
   harder and retries once.

Token counts are estimated from the request's size, then calibrated on the
token counts the server reports. Each run's `RunResult.notes` says what was
trimmed. Without `context_window`, nothing is trimmed (OpenAI's own models have
large windows). The Claude provider doesn't need it: the Claude Agent SDK
manages its own context.

Tested with llama.cpp's `llama-server -c 4096` and a tool returning a 26 KB
file: without `context_window` the server rejected the request (15,274 tokens
for a 4,096-token window); with it, the output was capped and the run
succeeded.

See [`examples/profiles/`](examples/profiles/) for sample files: offline with
llama-server or Ollama, a vLLM cluster, hybrid local and cloud, a company
gateway, and an annotated reference.

For OpenAI, cost comes from a gateway's `x-litellm-response-cost` header when
present. Otherwise it is computed from the pricing table; add models with
`register_pricing("my-model", input=0.6, output=2.2)`.

Provider-specific knobs pass through `provider_options`:

```python
Agent(provider="claude", provider_options={"permission_mode": "acceptEdits",
                                           "options": {"setting_sources": []}})
Agent(provider="openai", provider_options={"base_url": "http://localhost:11434/v1",
                                           "options": {"temperature": 0}})
```

## Tools

- **Built-ins**: `Read`, `Write`, `Bash`, `Grep`, `Glob`, `WebFetch`, grouped as
  `READ_ONLY_TOOLS`, `CODING_TOOLS`, `ALL_TOOLS`. You can also pass them by
  name: `tools=["Read", "Bash"]`. On `claude` the SDK's own implementations
  run. On `openai` the Python versions in `tools/builtin.py` run.
- **`@tool`**: builds the JSON schema from type hints (`str`, `int`, `float`,
  `bool`, `list[...]`, `dict`, `Literal[...]`, `X | None`) and the docstring's
  `Args:` section. Sync tools run in a thread and async tools are awaited.
  Exceptions go back to the model as `Error ...` text.
- **`ToolContext`**: annotate a parameter with it to receive `work_dir`, `env`
  and an `extra` dict. The model never sees that parameter.
- **Big integers**: pass them as `str`. On the claude provider, tool arguments
  go through the JavaScript Claude Code CLI, which rounds JSON numbers above
  2^53.

Tools run with the permissions of your process. `Write` and `Bash` are not
sandboxed on the openai provider, so run untrusted tasks in a container.

## Subagents

A subagent is an `Agent` exposed to another agent as a tool. The parent sends
it a self-contained `task` and gets back its final answer:

```python
worker = Agent(preset="quick", tools=READ_ONLY_TOOLS, work_dir=repo,
               system_prompt="Answer the task concisely, citing path:line.",
               reporter=ConsoleReporter(prefix="    │ worker: ", show_summary=False))

lead = Agent(preset="standard", work_dir=repo,
             tools=["Glob", worker.as_tool("explorer", "Delegate a code-reading question.")])
result = lead.run_sync("Write an onboarding guide for this repo.")

result.subagents              # RunResult of each direct subagent call
result.all_runs()             # this run + every nested subagent run, flattened
result.cumulative_cost_usd    # parent + all subagents
result.cumulative_usage       # same for tokens
```

- Each tier picks its own model, or its own provider: a Claude lead can drive
  workers on an OpenAI-compatible endpoint.
- File contents and command output stay in the cheap workers' context. The
  lead only sees their summaries.
- Calls issued in the same turn run concurrently on both providers.
- Subagent runs attach to the run that called them, even with several runs in
  flight. Tracking uses a context variable, not state on the `Agent`.
- On Claude only, the SDK's native subagents (`ClaudeAgentOptions.agents`, via
  `provider_options={"options": {"agents": ...}}`) are an alternative, but
  they don't port to other providers.

## Automatic model selection

With `model="auto"` (or `preset="auto"`, `--model auto`, `AGENT_MODEL=auto`), a
router picks a preset for **each run**, from the prompt. A subagent with
`model="auto"` is therefore routed per delegated task.

The default router, `D1Router`, asks Liquid AI's
[d1-2B](https://huggingface.co/LiquidAI/d1-2B) decision model *"Which
assistant should handle this task?"*, with one described option per preset.
d1 runs locally in llama-server and answers in under a second on CPU (about
0.7–0.9 s measured).

| d1 option | for | preset | claude | openai |
|---|---|---|---|---|
| fast | lookups, listing, running commands and reporting results, short or mechanical tasks | `quick` | claude-haiku-5-5 | gpt-6-luna |
| capable | explaining, writing or changing code, comparing options, multi-step analysis | `standard` | claude-sonnet-5-5 | gpt-6.1-sol |
| expert | open-ended design, subtle concurrency or correctness reasoning, deep expertise | `thorough` | claude-opus-5-5 | gpt-6-astra |

Each agent sets its own limits; all agents share one router and one server:

```python
from omniharness import READ_ONLY_TOOLS, Agent

lead = Agent(model="auto")                               # any preset
worker = Agent(model="auto", tools=READ_ONLY_TOOLS,
               auto_presets=["quick", "standard"],       # never Opus
               auto_fallback="quick")                    # if d1 is down
result = worker.run_sync("Design a rate limiter for a public API")
result.route   # RouteDecision(choice="thorough", preset="standard", label="expert", ...)
```

d1 always sees every option. Its pick is then moved to the nearest preset the
agent allows, so a task d1 calls "expert" goes to a worker's most capable
preset, not its cheapest. On the CLI the same settings are
`--model auto --auto-presets quick,standard --auto-fallback quick`.

Setup:

```bash
hf download LiquidAI/d1-2B-GGUF d1-2B-Q8_0.gguf   # ~1.7 GB, into the HF cache
# llama-server (llama.cpp v0.8.0+, for lfm3 and /v1/systemone) on PATH, or:
export LLAMA_SERVER=/path/to/llama.cpp/build/bin/llama-server
# or reuse a running server instead of starting one:
export D1_URL=http://127.0.0.1:8080
```

Older llama.cpp builds, such as Fedora's `llama-cpp` package, fail with
`unknown model architecture: 'lfm3'`. The router recognizes this error and
says so.

**Measured accuracy.** `evals/routing.jsonl` holds 34 prompts labelled with
the tier they should get: chat prompts, and subagent tasks in the long,
self-contained style an orchestrator writes. Labelling policy:

- quick: facts and lookups, finding or listing things, running commands and
  reporting their output, trivial snippets, mechanical edits
- standard: explaining code or concepts, writing or changing non-trivial code,
  debugging, comparisons, multi-step investigation
- thorough: open-ended design, subtle concurrency, correctness or security
  reasoning, proofs, large migrations

With d1-2B Q8_0 on llama.cpp v0.8.0:

| method | correct | too expensive | too cheap |
|---|---|---|---|
| `choice` (default) | 29/34 (85%) | 5 | 0 |
| `complexity` | 20/34 (59%) | 9 | 5 |

The `choice` misses were a trivial function and a rename sent to `standard`,
plus a database comparison, a derivation and one vague one-liner sent to
`thorough`. The `complexity` method (`D1Router(method="complexity")`, a
5-level complexity score with cutoffs) rates nearly every task that needs tools as
"moderate". It sent 9 of the 13 quick prompts to `standard`.

Re-check after changing the descriptions, labels or model:

```bash
python scripts/routing_eval.py -v                      # both methods, every prompt
LLAMA_SERVER=... pytest -m d1                          # fails below 80% or >1 too cheap
python examples/07_auto_model.py --rate-only "..."     # one-off prompts
```

- **Descriptions**: pass your own as `D1Router(choices={preset: (label,
  description), ...})`, cheapest first. Rewording them to repeat the
  labelling policy scored the same (29/34) but sent the security audit to
  `standard`, so the defaults were kept.
- **Server**: without `D1_URL`, the router starts llama-server the first time
  it's needed. It runs on a free local port, is shared by all `auto` agents in
  the process (`get_default_router()`), and stops at exit or on
  `router.close()`. The model comes from `model_path=`, then `$D1_MODEL`, then
  the Hugging Face cache. llama-server's output goes to a temporary log, which
  is kept if startup fails.
- **Failures**: if d1 can't be started or doesn't answer, the run still goes
  ahead on the agent's `auto_fallback`. Without one, it uses the router's
  `fallback_preset` (`standard`), moved into the agent's allowed range. The
  router logs a warning once and the run's `RunResult.notes` says so.
- **What d1 sees**: the task prompt only, or the agent's instructions too with
  `include_system_prompt=True`. Long prompts are cut to fit d1's single-batch
  input.
- **Other routers**: subclass `Router` (`tier_order` and an async `route()`
  returning a `RouteDecision`). `FixedRouter("quick")` is handy in tests.

## Container

`Containerfile` builds an image with omniharness, both provider SDKs, and a
portable llama-server (llama.cpp v0.8.0, built with every CPU variant) for
`model="auto"`. The container is also a sandbox: `Bash` and `Write` only reach
what you mount.

```bash
podman build -t omniharness .

# Once: download the d1 model (1.7 GB) into a volume, with the image's hf CLI
podman volume create agent-hf
podman run --rm --userns=keep-id -v agent-hf:/home/agent/.cache/huggingface:Z \
    --entrypoint hf omniharness download LiquidAI/d1-2B-GGUF d1-2B-Q8_0.gguf

# Run an agent on the current directory
podman run --rm -it --userns=keep-id -e ANTHROPIC_API_KEY \
    -v agent-hf:/home/agent/.cache/huggingface:Z -v "$PWD:/work:Z" \
    omniharness --model auto --tools Read,Grep,Glob "What does this project do?"
```

- **Your project** is mounted on `/work`, the working directory. With
  `--userns=keep-id`, files the agent writes belong to you on the host.
- **Credentials** are passed by name with `-e`, so their values don't appear
  on the command line: `-e ANTHROPIC_API_KEY`, `-e OPENAI_API_KEY`,
  `-e OPENAI_BASE_URL`, and so on. The image doesn't contain your Claude Code
  login. Mounting `~/.claude` would provide it, but it would also bring your
  Claude Code settings and hooks into every agent.
- **Hosted models:** mount a profiles file and point to it, e.g.
  `-v ./profiles.toml:/profiles.toml:Z,ro -e AGENT_PROFILES=/profiles.toml -e VLLM_API_KEY`.
  To reach a server on the host itself, use `http://host.containers.internal:PORT/v1`
  as its `base_url`.
- **The examples and the routing eval** are in `/opt/omniharness`:

  ```bash
  podman run --rm -it --userns=keep-id -e ANTHROPIC_API_KEY \
      -v agent-hf:/home/agent/.cache/huggingface:Z -v "$PWD:/work:Z" \
      --entrypoint python omniharness \
      /opt/omniharness/examples/06_subagents.py /work --worker-model auto
  ```

- **d1 runs inside the container.** The router starts llama-server the first
  time it's needed (a few seconds) and stops it when the run ends. To share
  one d1 server across many runs, start a llama-server yourself and pass
  `-e D1_URL=http://host.containers.internal:8080`.
- **Build options:** `--build-arg LLAMA_CPP_REF=v0.9.0` uses another
  llama.cpp release (v0.8.0 or later), and `--build-arg FEDORA_VERSION=...`
  changes the base image.

## Progress reporting

`ConsoleReporter` (the default) prints assistant text, a one-line summary of
each tool call, abbreviated tool results, and a final usage/cost summary.
Use `prefix=` to label a subagent's lines and `show_summary=False` to skip its
per-run summary. Subclass `Reporter` (`on_start`, `on_text`, `on_tool_call`,
`on_tool_result`, `on_result`) to log elsewhere. `Reporter()` itself is silent.

## Adding a provider

```python
from omniharness import Provider, RunRequest, RunResult, register_provider

class MyProvider(Provider):
    name = "mine"
    async def run(self, request: RunRequest, reporter) -> RunResult: ...

register_provider("mine", lambda **kw: MyProvider(**kw))
Agent(provider="mine", model="my-model")
```

## Migrating openstack-review

The `providers/` package, `print_usage_summary` and the OpenAI pricing logic
map to this library. `run_review()` in `review_agent.py` becomes:

```python
from omniharness import ALL_TOOLS, Agent, register_pricing, resolve_model
from .review_config import OPENAI_PRICING, PRESETS

for name, rates in OPENAI_PRICING.items():
    register_pricing(name, **rates)

cfg = resolve_model(provider, preset, model_override, profiles=PRESETS)
print(f"Using {cfg.provider} / {cfg.model} (from {cfg.source})")
provider_options = {}
if cfg.provider == "openai":
    provider_options = {"base_url": openai_base_url,
                        "price_input": openai_price_input,
                        "price_output": openai_price_output}

agent = Agent(
    provider=cfg.provider, model=cfg.model,
    max_turns=max_turns_override or cfg.max_turns,
    system_prompt=build_system_prompt(review_number, output_dir, gerrit_host),
    tools=ALL_TOOLS,
    work_dir=work_dir,
    add_dirs=[output_dir, str(SKILL_DIR)],
    env={"SKILL_DIR": str(SKILL_DIR)},
    effort=effort_override,
    provider_options=provider_options,
)
result = await agent.run(prompt)
if report_path.exists():
    print(f"Report saved to: {report_path}")
```

Behavior changes compared with the original code:

- On `claude`, the harness also sets `ClaudeAgentOptions.tools` to the
  requested built-ins. The original only set `allowed_tools`, which
  auto-approved those tools without restricting the rest.
- The `openai` provider uses `AsyncOpenAI`, so it no longer blocks the event
  loop, and it reports cached and reasoning tokens.
- `RunResult.usage` is a `Usage` dataclass instead of the SDK's raw dict.

## Development

```bash
pip install -e '.[all,dev]'
pytest          # unit tests, no network: providers run against fake backends
ruff check src tests examples
```

## Layout

```
src/omniharness/
  agent.py            Agent: high-level entry point
  config.py           presets + provider/model resolution
  result.py           RunRequest, RunResult, Usage
  reporting.py        Reporter, ConsoleReporter, format_summary
  pricing.py          OpenAI-style token pricing
  skill.py            SKILL.md loader
  routing.py          model="auto": Router, D1Router (local d1-2B decision model)
  cli.py              omniharness command
  providers/          claude.py (Agent SDK), openai_provider.py (Responses loop)
  tools/              base.py (@tool, ToolRegistry), builtin.py (Read/Bash/...)
evals/routing.jsonl   labelled prompts for model="auto"
scripts/              routing_eval.py: check routing against them with the real d1
Containerfile         image with omniharness and llama-server (see Container)
```
