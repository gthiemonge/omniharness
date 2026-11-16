"""
omniharness command line.

    omniharness "Summarize this repo" --tools Read,Grep,Glob
    omniharness --skill examples/skills/changelog --work-dir ~/src/proj "last 20 commits"
    echo "What is 2+2?" | omniharness - --provider openai --model gpt-6-luna
    omniharness --model auto --llama-server ~/llama.cpp/build/bin/llama-server "..."

Every environment variable the harness reads has a flag here, except
OPENAI_API_KEY: a key on the command line would show in `ps` and shell history.
"""

import argparse
import json
import os
import sys
from dataclasses import asdict

from .agent import Agent
from .config import format_profiles
from .options import AgentOptions
from .reporting import Reporter
from .skill import load_skill
from .tools.builtin import BUILTIN_TOOLS


def _parser() -> tuple[argparse.ArgumentParser, AgentOptions]:
    p = argparse.ArgumentParser(
        prog="omniharness",
        description="Run a tool-using AI agent with Claude or OpenAI.",
        epilog=format_profiles(),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("prompt", nargs="?", help="Task for the agent ('-' reads stdin).")
    p.add_argument("--tools", default="",
                   help=f"Comma-separated built-in tools ({','.join(BUILTIN_TOOLS)}), "
                        "or 'all'. Default: none (with --skill: all).")
    p.add_argument("--work-dir", default=os.getcwd(), help="Directory the tools operate in.")
    p.add_argument("--add-dir", action="append", default=[],
                   help="Extra directory the agent may access (repeatable).")
    sp = p.add_mutually_exclusive_group()
    sp.add_argument("--system", help="System prompt text.")
    sp.add_argument("--system-file", help="Read the system prompt from a file.")
    sp.add_argument("--skill", help="SKILL.md (or its directory) to use as the system prompt.")
    p.add_argument("--output-dir", default=os.getcwd(),
                   help="Where a --skill writes its outputs (exposed as $OUTPUT_DIR).")
    p.add_argument("--quiet", "-q", action="store_true",
                   help="Only print the final answer.")
    p.add_argument("--json", action="store_true", help="Print the result as JSON.")
    p.add_argument("--list-profiles", action="store_true",
                   help="List the profiles in effect (with --profiles / AGENT_PROFILES) and exit.")
    options = AgentOptions(p, title="models and providers")
    options.add_agent()
    return p, options


def build_parser() -> argparse.ArgumentParser:
    return _parser()[0]


def main(argv=None) -> int:
    parser, options = _parser()
    args = parser.parse_args(argv)
    try:
        profiles = options.profiles(args)
    except (OSError, ValueError) as exc:
        print(f"Error: profiles: {exc}", file=sys.stderr)
        return 1

    if args.list_profiles:
        print(format_profiles(profiles))
        return 0

    prompt = args.prompt
    if prompt == "-" or (prompt is None and not sys.stdin.isatty()):
        prompt = sys.stdin.read()
    if not prompt or not prompt.strip():
        parser.error("a prompt is required")

    system_prompt = args.system or ""
    add_dirs = list(args.add_dir)
    env = {}
    tools_arg = args.tools
    if args.system_file:
        with open(args.system_file) as f:
            system_prompt = f.read()
    if args.skill:
        skill = load_skill(args.skill)
        output_dir = os.path.abspath(args.output_dir)
        os.makedirs(output_dir, exist_ok=True)
        system_prompt = skill.render(ARGUMENTS=prompt, SKILL_DIR=str(skill.path),
                                     OUTPUT_DIR=output_dir)
        add_dirs += [str(skill.path), output_dir]
        env.update(SKILL_DIR=str(skill.path), OUTPUT_DIR=output_dir)
        tools_arg = tools_arg or "all"

    if tools_arg == "all":
        tools = list(BUILTIN_TOOLS)
    else:
        tools = [t.strip() for t in tools_arg.split(",") if t.strip()]

    try:
        agent_kwargs = options.agent_kwargs(args)
    except ValueError as exc:
        parser.error(str(exc))

    quiet = args.quiet or args.json
    try:
        agent = Agent(
            **agent_kwargs,
            system_prompt=system_prompt,
            tools=tools,
            work_dir=args.work_dir,
            add_dirs=add_dirs,
            env=env,
            reporter=Reporter() if quiet else None,
        )
        result = agent.run_sync(prompt)
    except Exception as exc:  # noqa: BLE001
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    if args.json:
        print(json.dumps(asdict(result), indent=2, default=str))
    elif args.quiet:
        print(result.final_text)
    return 0 if result.success else 2


if __name__ == "__main__":
    sys.exit(main())
