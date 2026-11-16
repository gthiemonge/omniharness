"""
The omniharness command-line options, reusable in your own scripts.

    import argparse
    from omniharness import Agent, AgentOptions

    parser = argparse.ArgumentParser()
    parser.add_argument("repo")
    options = AgentOptions(parser)          # --profiles, --base-url, --d1-url, ...
    options.add_agent()                     # --provider, --model, --preset, ...
    args = parser.parse_args()
    agent = Agent(**options.agent_kwargs(args), tools=[...], system_prompt="...")

Several agents get their own options with a name, as prefix:

    options.add_agent("lead", preset="standard")    # --lead-model, --lead-preset, ...
    options.add_agent("worker", preset="quick")     # --worker-model, ...
    lead = Agent(**options.agent_kwargs(args, "lead"), ...)
    worker = Agent(**options.agent_kwargs(args, "worker"), ...)

Shared options (profiles file, endpoints, context window, d1 server) are added
once and apply to every agent; they load one profiles file and start at most
one d1 server for all of them.
"""

import argparse
from typing import Any

from .config import get_profiles, load_profiles
from .providers import available_providers
from .routing import D1Router


class AgentOptions:
    def __init__(self, parser: argparse.ArgumentParser,
                 title: str = "models and providers (shared by all agents)"):
        self.parser = parser
        self._agents: dict[str, str] = {}  # name -> dest prefix
        self._profiles: Any = None
        self._router: Any = None
        self._router_built = False

        g = parser.add_argument_group(title)
        g.add_argument("--profiles", metavar="FILE",
                       help="TOML or JSON file of profiles, merged over the built-in ones; "
                            "a preset can set its own base_url, api_key_env, ... "
                            "(env AGENT_PROFILES).")
        g.add_argument("--base-url", help="Custom API endpoint for the provider (env "
                       "ANTHROPIC_BASE_URL for claude, OPENAI_BASE_URL for openai).")
        g.add_argument("--openai-org", help="OpenAI organization id (env OPENAI_ORG).")
        g.add_argument("--openai-api", choices=["auto", "responses", "chat"],
                       help="Wire format of the openai provider: responses (/v1/responses), "
                            "chat (/v1/chat/completions), or auto: responses, falling back "
                            "to chat when the server lacks it (default).")
        g.add_argument("--context-window", type=int, metavar="TOKENS",
                       help="Context size of the openai provider's model: tool outputs are "
                            "capped and older history trimmed to fit (default: no "
                            "trimming).")
        g.add_argument("--openai-price-input", type=float, help="OpenAI input price, USD "
                       "per 1M tokens, for cost reporting (env OPENAI_PRICE_INPUT).")
        g.add_argument("--openai-price-output", type=float, help="OpenAI output price, USD "
                       "per 1M tokens, for cost reporting (env OPENAI_PRICE_OUTPUT).")
        g.add_argument("--d1-url", help="Running llama-server with d1-2B for model auto "
                       "(env D1_URL; default: start one).")
        g.add_argument("--llama-server", help="llama-server binary (llama.cpp v0.8.0+) to "
                       "start for model auto (env LLAMA_SERVER; default: from PATH).")
        g.add_argument("--d1-model", help="d1-2B GGUF file for model auto (env D1_MODEL; "
                       "default: from the Hugging Face cache).")

    def add_agent(self, name: str | None = None, *, preset: str | None = None,
                  model: str | None = None, title: str | None = None) -> None:
        """Add one agent's options; with a name, they're prefixed: --<name>-model, ...

        preset / model: defaults for this agent (else AGENT_PRESET / AGENT_MODEL, then
        the profile's default preset).
        """
        key = name or ""
        if key in self._agents:
            raise ValueError(f"agent options '{key}' already added")
        flag = f"--{name}-" if name else "--"
        dest = f"{name.replace('-', '_')}_" if name else ""
        self._agents[key] = dest
        who = f" of the {name}" if name else ""
        g = self.parser.add_argument_group(title or (f"{name} agent" if name else "agent"))
        g.add_argument(f"{flag}provider", dest=f"{dest}provider",
                       choices=available_providers(),
                       help=f"Provider{who} (env AGENT_PROVIDER, default claude).")
        g.add_argument(f"{flag}profile", dest=f"{dest}profile",
                       help=f"Profile{who}: a set of presets, each with its provider, e.g. "
                            "'hybrid' from a profiles file (env AGENT_PROFILE; default: "
                            "the provider's own profile).")
        g.add_argument(f"{flag}model", dest=f"{dest}model", default=model,
                       help=f"Model id{who}, or 'auto' to let the local d1 decision model "
                            "pick a preset per task (env AGENT_MODEL; overrides the preset)"
                            + (f"; default: {model}" if model else "") + ".")
        g.add_argument(f"{flag}preset", dest=f"{dest}preset", default=preset,
                       help=f"Preset{who}: quick, standard, thorough, ... (env AGENT_PRESET"
                            + (f"; default: {preset}" if preset else "") + ").")
        g.add_argument(f"{flag}effort", dest=f"{dest}effort",
                       help=f"Reasoning effort{who}, e.g. low, medium, high (default: the "
                            "preset's effort, then env AGENT_EFFORT).")
        g.add_argument(f"{flag}max-turns", dest=f"{dest}max_turns", type=int,
                       help=f"Maximum turns{who} (default: the preset's).")
        g.add_argument(f"{flag}auto-presets", dest=f"{dest}auto_presets",
                       help=f"With model auto: comma-separated presets{who} may get, "
                            "e.g. quick,standard.")
        g.add_argument(f"{flag}auto-fallback", dest=f"{dest}auto_fallback",
                       help=f"With model auto: preset{who} when d1 is unavailable.")

    def profiles(self, args: argparse.Namespace) -> dict:
        """The profiles table (--profiles merged over the built-in ones), loaded once."""
        if self._profiles is None:
            self._profiles = load_profiles(args.profiles) if args.profiles else get_profiles()
        return self._profiles

    def router(self, args: argparse.Namespace) -> D1Router | None:
        """The d1 router the d1 options describe, shared by all agents; None: the default."""
        if not self._router_built:
            self._router_built = True
            if args.d1_url or args.llama_server or args.d1_model:
                self._router = D1Router(args.d1_url, llama_server=args.llama_server,
                                        model_path=args.d1_model)
        return self._router

    def agent_kwargs(self, args: argparse.Namespace, name: str | None = None) -> dict[str, Any]:
        """Keyword arguments for Agent(...) from the parsed options of one agent.

        Raises ValueError on options that contradict each other.
        """
        key = name or ""
        if key not in self._agents:
            raise ValueError(f"no agent options '{key}': call add_agent({name!r}) first")
        dest = self._agents[key]

        def get(option: str) -> Any:
            return getattr(args, f"{dest}{option}")

        provider = get("provider")
        # OpenAI-only options are keyed to "openai". --base-url goes to the
        # agent's --provider when given, else to the only provider in play
        # (the Agent rejects it if that's ambiguous).
        provider_options: dict[str, Any] = {}
        openai_only = {"organization": args.openai_org, "price_input": args.openai_price_input,
                       "price_output": args.openai_price_output, "api": args.openai_api,
                       "context_window": args.context_window}
        openai_only = {k: v for k, v in openai_only.items() if v is not None}
        if openai_only:
            provider_options["openai"] = openai_only
        if args.base_url:
            if provider:
                provider_options.setdefault(provider, {})["base_url"] = args.base_url
            elif provider_options:
                flag = f"--{name}-provider" if name else "--provider"
                raise ValueError(f"--base-url with --openai-* options needs {flag}")
            else:
                provider_options["base_url"] = args.base_url

        auto_presets = get("auto_presets")
        return {
            "provider": provider,
            "profile": get("profile"),
            "model": get("model"),
            "preset": get("preset"),
            "effort": get("effort"),
            "max_turns": get("max_turns"),
            "profiles": self.profiles(args),
            "provider_options": provider_options,
            # Shared options may concern only some of several agents.
            "strict_options": len(self._agents) == 1,
            "router": self.router(args),
            "auto_presets": auto_presets.split(",") if auto_presets else None,
            "auto_fallback": get("auto_fallback"),
        }
