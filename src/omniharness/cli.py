"""
omniharness command line.

    omniharness "Summarize this repo" --tools Read,Grep,Glob
    omniharness --skill examples/skills/changelog --work-dir ~/src/proj "last 20 commits"
    echo "What is 2+2?" | omniharness - --provider openai --model gpt-5-mini
"""

import argparse
import json
import os
import sys
from dataclasses import asdict

from .agent import Agent
from .config import format_presets
from .providers import available_providers
from .reporting import Reporter
from .routing import D1Router
from .skill import load_skill
from .tools.builtin import BUILTIN_TOOLS


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="omniharness",
        description="Run a tool-using AI agent with Claude or OpenAI.",
        epilog=format_presets(),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("prompt", nargs="?", help="Task for the agent ('-' reads stdin).")
    p.add_argument("--provider", choices=available_providers(),
                   help="Provider (env AGENT_PROVIDER, default claude).")
    p.add_argument("--model", help="Model id, or 'auto' to let the local d1 decision model "
                   "pick a preset from the prompt (env AGENT_MODEL; overrides --preset).")
    p.add_argument("--preset", help="Model preset: quick, standard, thorough (env AGENT_PRESET).")
    p.add_argument("--effort", help="Reasoning effort: low, medium, high (env AGENT_EFFORT).")
    p.add_argument("--max-turns", type=int, help="Maximum agent turns.")
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
    p.add_argument("--auto-presets", help="With --model auto: comma-separated presets "
                   "allowed, e.g. quick,standard.")
    p.add_argument("--auto-fallback", help="With --model auto: preset when d1 is unavailable.")
    p.add_argument("--d1-url", help="Running llama-server with d1-2B for --model auto "
                   "(env D1_URL; default: start one).")
    p.add_argument("--base-url", help="OpenAI-compatible endpoint (env OPENAI_BASE_URL).")
    p.add_argument("--quiet", "-q", action="store_true",
                   help="Only print the final answer.")
    p.add_argument("--json", action="store_true", help="Print the result as JSON.")
    p.add_argument("--list-presets", action="store_true", help="List presets and exit.")
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)

    if args.list_presets:
        print(format_presets())
        return 0

    prompt = args.prompt
    if prompt == "-" or (prompt is None and not sys.stdin.isatty()):
        prompt = sys.stdin.read()
    if not prompt or not prompt.strip():
        build_parser().error("a prompt is required")

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

    provider_options = {}
    if args.base_url:
        if (args.provider or os.getenv("AGENT_PROVIDER")) != "openai":
            build_parser().error("--base-url requires --provider openai")
        provider_options["base_url"] = args.base_url

    quiet = args.quiet or args.json
    try:
        agent = Agent(
            provider=args.provider,
            model=args.model,
            preset=args.preset,
            system_prompt=system_prompt,
            tools=tools,
            work_dir=args.work_dir,
            max_turns=args.max_turns,
            effort=args.effort,
            add_dirs=add_dirs,
            env=env,
            reporter=Reporter() if quiet else None,
            provider_options=provider_options,
            router=D1Router(args.d1_url) if args.d1_url else None,
            auto_presets=args.auto_presets.split(",") if args.auto_presets else None,
            auto_fallback=args.auto_fallback,
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
