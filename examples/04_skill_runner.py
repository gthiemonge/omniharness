#!/usr/bin/env python3
"""
Run a Claude-style skill (SKILL.md) on either provider — the pattern used by
openstack-review. The skill's directory is exposed to the agent as $SKILL_DIR
so it can call its helper scripts.

    python examples/04_skill_runner.py examples/skills/changelog ~/src/myrepo "since v1.2.0"
    AGENT_PROVIDER=openai python examples/04_skill_runner.py examples/skills/changelog .
"""

import argparse
import os
import sys

from omniharness import ALL_TOOLS, Agent, load_skill

parser = argparse.ArgumentParser()
parser.add_argument("skill", help="Skill directory or SKILL.md")
parser.add_argument("work_dir", help="Directory the skill operates on")
parser.add_argument("arguments", nargs="?", default="", help="Value for $ARGUMENTS")
parser.add_argument("--output-dir", default="out")
args = parser.parse_args()

skill = load_skill(args.skill)
output_dir = os.path.abspath(args.output_dir)
os.makedirs(output_dir, exist_ok=True)

system_prompt = f"""\
You are an autonomous agent executing the '{skill.name}' skill.
The skill's helper files are in: {skill.path}
Write every output file to: {output_dir}
Before each tool call, write one short sentence explaining why.

---

{skill.render(ARGUMENTS=args.arguments, SKILL_DIR=str(skill.path), OUTPUT_DIR=output_dir)}
"""

agent = Agent(
    system_prompt=system_prompt,
    tools=ALL_TOOLS,
    work_dir=args.work_dir,
    add_dirs=[output_dir, str(skill.path)],
    env={"SKILL_DIR": str(skill.path), "OUTPUT_DIR": output_dir},
)
result = agent.run_sync(
    f"Run the {skill.name} skill. Arguments: {args.arguments or '(none)'}"
)
sys.exit(0 if result.success else 1)
