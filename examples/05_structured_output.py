#!/usr/bin/env python3
"""
Structured output that works on every provider: give the agent a
`submit_result` tool and capture what it passes. The tool schema enforces the
shape; no JSON parsing of free text needed.

Also shows a custom Reporter that logs every event as JSON lines.

    python examples/05_structured_output.py ~/src/myrepo
"""

import json
import sys
from typing import Literal

from omniharness import READ_ONLY_TOOLS, Agent, Reporter, tool

captured = {}


@tool
def submit_result(
    language: str,
    build_system: str,
    has_tests: bool,
    test_command: str,
    maturity: Literal["prototype", "beta", "production"],
    concerns: list[str],
) -> str:
    """Submit the final assessment. Call exactly once, when you are done.

    Args:
        language: Main programming language.
        build_system: e.g. setuptools, poetry, cargo, make.
        has_tests: Whether the project has an automated test suite.
        test_command: Command to run the tests, or '' if none.
        maturity: Overall maturity estimate.
        concerns: Up to 5 short maintainability concerns.
    """
    captured.update(locals())
    return "Result recorded. You are done."


class JsonlReporter(Reporter):
    def __init__(self, path):
        self.f = open(path, "w")

    def _log(self, **event):
        self.f.write(json.dumps(event, default=str) + "\n")
        self.f.flush()

    def on_text(self, text):
        self._log(type="text", text=text)

    def on_tool_call(self, name, args):
        self._log(type="tool_call", name=name, args=args)
        print(f"  -> {name}", file=sys.stderr)

    def on_tool_result(self, name, output, is_error=False):
        self._log(type="tool_result", name=name, is_error=is_error, chars=len(output))


repo = sys.argv[1] if len(sys.argv) > 1 else "."
agent = Agent(
    preset="quick",
    system_prompt="Assess the repository in your working directory. Explore briefly "
                  "(at most ~10 tool calls), then call submit_result.",
    tools=[*READ_ONLY_TOOLS, submit_result],
    work_dir=repo,
    max_turns=20,
    reporter=JsonlReporter("events.jsonl"),
)
result = agent.run_sync("Assess this repository.")

if not captured:
    sys.exit(f"Agent finished without submitting a result ({result.subtype}).")
print(json.dumps(captured, indent=2))
print(f"(events logged to events.jsonl, cost: {result.total_cost_usd})", file=sys.stderr)
