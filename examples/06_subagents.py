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
    python examples/06_subagents.py . --orchestrator-model claude-opus-5-5 --worker-model claude-sonnet-5
    # Let the local d1 decision model pick each worker task's model, between
    # quick and standard (see 07_auto_model.py)
    python examples/06_subagents.py . --worker-model auto

Model selection, per tier: --*-model if given, else the --*-preset's model for
the tier's provider. AGENT_MODEL is ignored here, since it would put both tiers
on the same model.
"""

import argparse
import os
import sys

from omniharness import READ_ONLY_TOOLS, Agent, ConsoleReporter

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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("repo", help="Repository to work on")
    parser.add_argument("task", nargs="?", default=DEFAULT_TASK)
    parser.add_argument("--orchestrator-preset", default="standard")
    parser.add_argument("--worker-preset", default="quick")
    parser.add_argument("--orchestrator-model", help="Exact orchestrator model "
                        "(overrides --orchestrator-preset)")
    parser.add_argument("--worker-model", help="Exact worker model (overrides --worker-preset)")
    parser.add_argument("--worker-provider", help="Provider for the workers "
                        "(default: same as the orchestrator, $AGENT_PROVIDER or claude)")
    args = parser.parse_args()

    # AGENT_MODEL outranks presets in the harness, so it would collapse both
    # tiers onto one model. Drop it; per-tier flags replace it.
    env_model = os.environ.pop("AGENT_MODEL", None)
    if env_model:
        print(f"Warning: ignoring AGENT_MODEL={env_model}: this example picks a model "
              "per tier. Use --orchestrator-model / --worker-model instead.",
              file=sys.stderr)

    # With --worker-model auto, d1 picks each task's model, but a worker never
    # goes above "standard" and falls back to "quick" if d1 is unavailable.
    worker_auto = {}
    if args.worker_model == "auto":
        worker_auto = {"auto_presets": ["quick", "standard"], "auto_fallback": "quick"}

    def worker(name: str, system_prompt: str, tools: list, max_turns: int) -> Agent:
        return Agent(
            provider=args.worker_provider,
            model=args.worker_model,
            preset=args.worker_preset,
            **worker_auto,
            system_prompt=system_prompt,
            tools=tools,
            work_dir=args.repo,
            max_turns=max_turns,
            # Workers may run concurrently: prefix their lines and keep them
            # short; the orchestrator's summary includes their usage and cost.
            reporter=ConsoleReporter(prefix=f"    │ {name}: ", show_tool_results=False,
                                     show_summary=False),
        )

    explorer = worker("explorer", EXPLORER_PROMPT, READ_ONLY_TOOLS, max_turns=25)
    runner = worker("runner", RUNNER_PROMPT, ["Bash"], max_turns=15)

    orchestrator = Agent(
        model=args.orchestrator_model,
        preset=args.orchestrator_preset,
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
        max_turns=20,
        reporter=ConsoleReporter(max_result_lines=6),
    )

    print(f"Orchestrator: {orchestrator.model_config.provider}/{orchestrator.model} "
          f"({orchestrator.model_config.source})")
    workers = f"{explorer.model_config.provider}/{explorer.model}"
    if explorer.auto:
        workers += f" among {explorer.auto_presets}, fallback {explorer.auto_fallback}"
    print(f"Workers:      {workers} ({explorer.model_config.source})\n")
    result = orchestrator.run_sync(args.task)
    return 0 if result.success else 1


if __name__ == "__main__":
    sys.exit(main())
