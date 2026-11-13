#!/usr/bin/env python3
"""
Read-only code explorer: answers questions about a local repository using the
Read / Grep / Glob tools. It cannot modify anything.

    python examples/02_code_explorer.py ~/src/octavia "How are amphora health checks scheduled?"
    AGENT_PROVIDER=openai python examples/02_code_explorer.py . "What does this project do?"
"""

import argparse
import sys

from omniharness import READ_ONLY_TOOLS, Agent

SYSTEM_PROMPT = """\
You are a senior engineer answering questions about the codebase in your
working directory. Explore with Glob/Grep/Read before answering; never guess.
Cite the files and line numbers you rely on as path:line.
Before each tool call, write one short sentence explaining why.
Finish with a concise answer under a '## Answer' heading.
"""

parser = argparse.ArgumentParser()
parser.add_argument("repo", help="Path to the repository to explore")
parser.add_argument("question")
parser.add_argument("--preset", default="standard")
args = parser.parse_args()

agent = Agent(
    preset=args.preset,
    system_prompt=SYSTEM_PROMPT,
    tools=READ_ONLY_TOOLS,
    work_dir=args.repo,
    max_turns=30,
)
result = agent.run_sync(args.question)
sys.exit(0 if result.success else 1)
