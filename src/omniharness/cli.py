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
from .config import DEFAULT_PROVIDER, format_presets, get_presets, load_presets
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
    p.add_argument("--presets", metavar="FILE", help="TOML or JSON file of presets, merged "
                   "over the built-in ones; a preset can set its own base_url, "
                   "api_key_env, ... (env AGENT_PRESETS).")
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
    p.add_argument("--llama-server", help="llama-server binary (llama.cpp v0.8.0+) to start "
                   "for --model auto (env LLAMA_SERVER; default: from PATH).")
    p.add_argument("--d1-model", help="d1-2B GGUF file for --model auto (env D1_MODEL; "
                   "default: from the Hugging Face cache).")
    p.add_argument("--base-url", help="Custom API endpoint for the provider (env "
                   "ANTHROPIC_BASE_URL for claude, OPENAI_BASE_URL for openai).")
    p.add_argument("--openai-org", help="OpenAI organization id (env OPENAI_ORG).")
    p.add_argument("--openai-price-input", type=float, help="OpenAI input price, USD per "
                   "1M tokens, for cost reporting (env OPENAI_PRICE_INPUT).")
    p.add_argument("--openai-price-output", type=float, help="OpenAI output price, USD per "
                   "1M tokens, for cost reporting (env OPENAI_PRICE_OUTPUT).")
    p.add_argument("--quiet", "-q", action="store_true",
                   help="Only print the final answer.")
    p.add_argument("--json", action="store_true", help="Print the result as JSON.")
    p.add_argument("--list-presets", action="store_true",
                   help="List the presets in effect (with --presets / AGENT_PRESETS) and exit.")
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    try:
        presets = load_presets(args.presets) if args.presets else get_presets()
    except (OSError, ValueError) as exc:
        print(f"Error: presets: {exc}", file=sys.stderr)
        return 1

    if args.list_presets:
        print(format_presets(presets))
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

    provider = (args.provider or os.getenv("AGENT_PROVIDER") or DEFAULT_PROVIDER).lower()
    provider_options = {}
    if args.base_url:
        provider_options["base_url"] = args.base_url
    openai_only = {"organization": args.openai_org, "price_input": args.openai_price_input,
                   "price_output": args.openai_price_output}
    openai_only = {k: v for k, v in openai_only.items() if v is not None}
    if openai_only:
        if provider != "openai":
            build_parser().error("--openai-org / --openai-price-* need --provider openai")
        provider_options.update(openai_only)

    router = None
    if args.d1_url or args.llama_server or args.d1_model:
        router = D1Router(args.d1_url, llama_server=args.llama_server,
                          model_path=args.d1_model)

    quiet = args.quiet or args.json
    try:
        agent = Agent(
            provider=args.provider,
            model=args.model,
            preset=args.preset,
            presets=presets,
            system_prompt=system_prompt,
            tools=tools,
            work_dir=args.work_dir,
            max_turns=args.max_turns,
            effort=args.effort,
            add_dirs=add_dirs,
            env=env,
            reporter=Reporter() if quiet else None,
            provider_options=provider_options,
            router=router,
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
