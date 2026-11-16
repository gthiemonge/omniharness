#!/usr/bin/env python3
"""
Orchestrator + subagents, each tier on its own model.

A mid-tier orchestrator plans the work and writes the answer. It cannot read
files itself: it delegates to cheap low-tier workers, exposed to it as tools
with Agent.as_tool():

  explorer  reads/greps the code and reports findings with path:line citations
  runner    runs shell commands (tests, linters, git) and summarizes the output

Benefits: the bulk of the tokens (file contents, command output) goes through
the cheap model, the orchestrator's context only holds the summaries, and
independent subtasks run in parallel when the orchestrator issues several
calls at once.

    python examples/06_subagents.py ~/src/octavia
    python examples/06_subagents.py . "How would I add a third provider? List the touch points."
    AGENT_PROVIDER=openai python examples/06_subagents.py .
    # Mix providers: Claude orchestrator, OpenAI(-compatible) workers
    python examples/06_subagents.py . --worker-provider openai --worker-preset quick
    # Pick exact models per tier (overrides the presets)
    python examples/06_subagents.py . --orchestrator-model claude-opus-5-5 --worker-model claude-sonnet-5-5
    # Let the local d1 decision model pick each worker task's model, between
    # quick and standard (see 07_auto_model.py)
    python examples/06_subagents.py . --worker-model auto
    # Workers on a local profile, the orchestrator on Claude
    python examples/06_subagents.py . --profiles examples/profiles/hybrid.toml \
        --worker-profile hybrid --worker-model auto

The options come from omniharness.AgentOptions: every omniharness model
option, once per agent (--orchestrator-*, --worker-*), plus the shared ones
(--profiles, --base-url, --d1-url, ...). See --help.

Model selection, per tier: --*-model if given, else the --*-preset's model for
the tier's profile. AGENT_MODEL is ignored here, since it would put both tiers
on the same model.
"""

import argparse
import os
import sys

from omniharness import READ_ONLY_TOOLS, Agent, AgentOptions, ConsoleReporter, RunResult

DEFAULT_TASK = (
    "Write a one-page onboarding guide for this repository: what it does, how "
    "the code is organized, the main flow through the code, and how to run the "
    "tests (and whether they currently pass)."
)

ORCHESTRATOR_PROMPT = """\
You are the lead engineer coordinating a small team of junior assistants to
answer a question about the repository in your working directory.

You cannot read files or run commands yourself. Delegate:
  - explorer: reads and searches the code, answers a focused question.
  - runner:   runs shell commands and reports the important parts of the output.

How to delegate well:
  - Split the work into independent, focused subtasks and issue them in the
    SAME turn so they run in parallel (e.g. one explorer per package).
  - Each task must be self-contained: the assistants cannot see this
    conversation. Say what to look at, what to report, and the format.
  - Ask for concise answers with path:line citations, not file dumps.
  - Use Glob first to see the layout and plan the split.
  - Follow up with more targeted subtasks if an answer is vague or conflicting.

When you have enough, write the final answer yourself in markdown.
"""

EXPLORER_PROMPT = """\
You are a code-reading assistant. Answer the task you are given about the
repository in your working directory using Glob/Grep/Read. Be efficient: grep
before reading, and read only the relevant parts of files.
Reply with a concise, factual report (at most ~300 words) citing path:line for
every claim. Do not speculate; say so when something is not found.
"""

RUNNER_PROMPT = """\
You are a build/test assistant. Run the shell commands needed for the task you
are given, in the repository in your working directory. Never modify tracked
files or install system packages; if a command needs missing dependencies,
report that instead of working around it.
Reply with: the exact commands you ran, whether they succeeded, and only the
relevant lines of output (failures, counts, versions), at most ~200 words.
"""


class WorkerReporter(ConsoleReporter):
    """Prints a worker's progress with a prefix, and records each of its runs."""

    def __init__(self, name: str, runs: list[RunResult]):
        super().__init__(prefix=f"    │ {name}: ", show_tool_results=False, show_summary=False)
        self.runs = runs

    def on_result(self, result: RunResult) -> None:
        self.runs.append(result)


def _cost(runs: list[RunResult]) -> float | None:
    costs = [r.total_cost_usd for r in runs]
    return None if None in costs else sum(costs)


def print_costs(result: RunResult, worker_runs: dict[str, list[RunResult]]) -> None:
    """Cost of the orchestrator, of each worker, of all workers, and in total."""

    def row(label: str, runs: list[RunResult] | None, tokens: int, cost: float | None) -> str:
        n = "" if runs is None else f"{len(runs)} run{'s' if len(runs) != 1 else ''}"
        models = "" if runs is None else ", ".join(sorted({r.model for r in runs}))
        price = "unknown" if cost is None else f"${cost:.4f}"
        return f"  {label:13s} {n:>7s} {tokens:>11,} tok {price:>9s}   {models}".rstrip()

    def runs_row(label: str, runs: list[RunResult]) -> str:
        return row(label, runs, sum(r.usage.total_tokens for r in runs), _cost(runs))

    all_workers = [r for runs in worker_runs.values() for r in runs]
    print("\nCost:")
    print(runs_row("orchestrator", [result]))
    for name, runs in worker_runs.items():
        print(runs_row(name, runs))
    print(runs_row("workers", all_workers))
    print(row("total", None, result.cumulative_usage.total_tokens, result.cumulative_cost_usd))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("repo", help="Repository to work on")
    parser.add_argument("task", nargs="?", default=DEFAULT_TASK)
    options = AgentOptions(parser)
    options.add_agent("orchestrator", preset="standard")
    options.add_agent("worker", preset="quick")
    args = parser.parse_args()

    # AGENT_MODEL outranks presets in the harness, so it would collapse both
    # tiers onto one model. Drop it; per-tier flags replace it.
    env_model = os.environ.pop("AGENT_MODEL", None)
    if env_model:
        print(f"Warning: ignoring AGENT_MODEL={env_model}: this example picks a model "
              "per tier. Use --orchestrator-model / --worker-model instead.",
              file=sys.stderr)

    try:
        lead_options = options.agent_kwargs(args, "orchestrator")
        worker_options = options.agent_kwargs(args, "worker")
    except ValueError as e:
        parser.error(str(e))
    # Turn caps: the --*-max-turns options, else these.
    lead_turns = lead_options.pop("max_turns") or 20
    worker_turns = worker_options.pop("max_turns")

    # With --worker-model auto, d1 picks each task's model, but by default a
    # worker never goes above "standard" and falls back to "quick" if d1 is
    # unavailable (--worker-auto-presets / --worker-auto-fallback change that).
    if worker_options["model"] == "auto":
        worker_options["auto_presets"] = worker_options["auto_presets"] or ["quick", "standard"]
        worker_options["auto_fallback"] = worker_options["auto_fallback"] or "quick"

    # Each worker's runs, recorded by its reporter, for the cost breakdown.
    worker_runs: dict[str, list[RunResult]] = {"explorer": [], "runner": []}

    def worker(name: str, system_prompt: str, tools: list, max_turns: int) -> Agent:
        return Agent(
            **worker_options,
            system_prompt=system_prompt,
            tools=tools,
            work_dir=args.repo,
            max_turns=worker_turns or max_turns,
            # Workers may run concurrently: prefix their lines and keep them short.
            reporter=WorkerReporter(name, worker_runs[name]),
        )

    explorer = worker("explorer", EXPLORER_PROMPT, READ_ONLY_TOOLS, max_turns=25)
    runner = worker("runner", RUNNER_PROMPT, ["Bash"], max_turns=15)

    orchestrator = Agent(
        **lead_options,
        system_prompt=ORCHESTRATOR_PROMPT,
        tools=[
            "Glob",
            explorer.as_tool(
                "explorer",
                "Delegate a focused code-reading question to an assistant that can "
                "Glob/Grep/Read the repository. Returns a short report with "
                "path:line citations.",
            ),
            runner.as_tool(
                "runner",
                "Delegate running shell commands (tests, linters, git log, ...) in "
                "the repository to an assistant. Returns the commands run and the "
                "relevant output.",
            ),
        ],
        work_dir=args.repo,
        max_turns=lead_turns,
        # The cost table below replaces the generic summary.
        reporter=ConsoleReporter(max_result_lines=6, show_summary=False),
    )

    print(f"Orchestrator: {orchestrator.model_config.provider}/{orchestrator.model} "
          f"({orchestrator.model_config.source})")
    workers = f"{explorer.model_config.provider}/{explorer.model}"
    if explorer.auto:
        workers = (f"auto in profile '{explorer.profile}', among {explorer.auto_presets}, "
                   f"fallback {explorer.auto_fallback}")
    print(f"Workers:      {workers} ({explorer.model_config.source})\n")
    result = orchestrator.run_sync(args.task)
    if not result.success:
        print(f"\nOrchestrator stopped: {result.subtype}")
    print_costs(result, worker_runs)
    return 0 if result.success else 1


if __name__ == "__main__":
    sys.exit(main())
