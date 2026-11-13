# omniharness

A small, provider-agnostic harness for tool-using AI agents, extracted from
[openstack-review](https://github.com/gthiemonge/openstack-review-claude-skill).

You write the agent once: a system prompt, some tools, and a working directory.
The harness runs it on either backend:

| Provider | Backend | Agent loop | Built-in tools | Your Python tools |
|----------|---------|------------|----------------|-------------------|
| `claude` | [Claude Agent SDK](https://pypi.org/project/claude-agent-sdk/) (Anthropic API, Vertex AI, Bedrock) | SDK | SDK-native `Read`/`Write`/`Bash`/`Grep`/`Glob`/`WebFetch` | in-process MCP server |
| `openai` | OpenAI Responses API, or any compatible endpoint (Azure, litellm, vLLM, Ollama, ...) | harness | Python versions with the same names | called directly |

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
    preset="standard",              # or model="claude-sonnet-5"
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
omniharness --provider openai --model gpt-5-mini -q "Name 3 prime numbers"
omniharness --skill examples/skills/changelog --work-dir ~/src/octavia --output-dir out "since 14.0.0"
git diff | omniharness - --system "Review this diff for bugs." --json
omniharness --list-presets
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

Model resolution, first match wins. The env var names match openstack-review's.

| Setting | Explicit | Environment | Fallback |
|---------|----------|-------------|----------|
| provider | `provider=` / `--provider` | `AGENT_PROVIDER` | `claude` |
| model | `model=` / `--model` | `AGENT_MODEL` | preset's model |
| preset | `preset=` / `--preset` | `AGENT_PRESET` | `standard` |
| effort | `effort=` / `--effort` | `AGENT_EFFORT` | provider default |
| OpenAI endpoint | `provider_options={"base_url": ...}` / `--base-url` | `OPENAI_BASE_URL` | api.openai.com |
| OpenAI prices (USD / 1M tokens) | `provider_options={"price_input": ..., "price_output": ...}` | `OPENAI_PRICE_INPUT/OUTPUT` | `pricing.PRICING` table |

Presets (`quick` / `standard` / `thorough`) live in `config.DEFAULT_PRESETS`.
Pass your own table with `Agent(presets=...)`. Its shape is the same as
openstack-review's `PRESETS`.

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
| fast | lookups, listing, running commands and reporting results, short or mechanical tasks | `quick` | claude-haiku-4-5 | gpt-5-mini |
| capable | explaining, writing or changing code, comparing options, multi-step analysis | `standard` | claude-sonnet-5 | gpt-5 |
| expert | open-ended design, subtle concurrency or correctness reasoning, deep expertise | `thorough` | claude-opus-5-5 | gpt-5 |

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

cfg = resolve_model(provider, preset, model_override, presets=PRESETS)
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
```
