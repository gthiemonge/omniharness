"""
High-level entry point.

    from omniharness import Agent, READ_ONLY_TOOLS

    agent = Agent(provider="claude", system_prompt="You are a code explorer.",
                  tools=READ_ONLY_TOOLS, work_dir="~/src/myproject")
    result = agent.run_sync("Where is the configuration loaded?")
    print(result.final_text)
"""

import asyncio
import contextvars
import os
from collections.abc import Sequence
from typing import Any

from .config import (
    AUTO,
    ModelConfig,
    endpoint_kwargs,
    get_profiles,
    normalize_profiles,
    preset_config,
    profile_providers,
    resolve_model,
)
from .providers import Provider, available_providers, get_provider
from .reporting import ConsoleReporter, Reporter
from .result import RouteDecision, RunRequest, RunResult
from .routing import Router, get_default_router
from .tools.base import Tool
from .tools.builtin import get_builtin

ToolSpec = Tool | str

# Results of subagent runs started while the current Agent.run() is active.
# A contextvar (rather than state on the Agent) keeps concurrent runs apart:
# asyncio tasks and worker threads inherit the context of the run that
# started them.
_subagent_results: contextvars.ContextVar[list[RunResult] | None] = contextvars.ContextVar(
    "omniharness_subagent_results", default=None
)


class Agent:
    def __init__(
        self,
        *,
        provider: str | Provider | None = None,
        model: str | None = None,
        preset: str | None = None,
        profile: str | None = None,
        system_prompt: str = "",
        tools: Sequence[ToolSpec] = (),
        work_dir: str | None = None,
        max_turns: int | None = None,
        effort: str | None = None,
        add_dirs: Sequence[str] = (),
        env: dict[str, str] | None = None,
        reporter: Reporter | None = None,
        profiles: dict[str, dict[str, dict]] | None = None,
        provider_options: dict[str, Any] | None = None,
        router: Router | None = None,
        auto_presets: Sequence[str] | None = None,
        auto_fallback: str | None = None,
    ):
        """
        Args:
            provider: "claude", "openai", a registered name, or a Provider
                instance. Selects the profile of that name when no profile is
                given. Defaults to AGENT_PROVIDER, then "claude".
            model: Model id. Defaults to AGENT_MODEL, then the preset's model.
            preset: Preset name ("quick", "standard", "thorough", ...).
            profile: The set of presets to use, e.g. "hybrid" from a profiles
                file; each preset brings its provider. Defaults to
                AGENT_PROFILE, then the provider's own profile.
            system_prompt: Instructions for the model.
            tools: Tool objects and/or built-in tool names ("Read", "Bash", ...).
            work_dir: Directory tools operate in. Defaults to the current one.
            max_turns: Turn cap. Defaults to the preset's value.
            effort: Reasoning effort ("low", "medium", "high", ...), passed to
                the provider as is. Defaults to the preset's effort (the routed
                preset's with "auto"), then AGENT_EFFORT, then the provider
                default.
            add_dirs: Extra directories the agent may access (claude provider).
            env: Extra environment variables for the agent's tools.
            reporter: Progress sink. Defaults to ConsoleReporter().
            profiles: Your own profiles table (see config.DEFAULT_PROFILES and
                load_profiles()). Default: DEFAULT_PROFILES merged with the file
                named by AGENT_PROFILES, if set.
            provider_options: kwargs for the provider constructors, e.g.
                {"base_url": ...}, or keyed by provider when the profile mixes
                them: {"openai": {...}, "claude": {...}}. They win over the
                presets' endpoints.
            router: Picks a preset per run when the model or preset is "auto".
                Defaults to the process-wide D1Router (routing.py).
            auto_presets: With "auto", the presets this agent may get, e.g.
                ["quick", "standard"] for a worker. The router's pick is clamped
                to the nearest one. Default: every preset the router knows.
            auto_fallback: With "auto", the preset when routing fails. Default:
                the router's fallback_preset, clamped to auto_presets.
        """
        provider_name = provider.name if isinstance(provider, Provider) else provider
        self.profiles = normalize_profiles(profiles if profiles is not None else get_profiles())
        self.model_config: ModelConfig = resolve_model(
            provider_name, preset, model, profile=profile, profiles=self.profiles
        )
        self.profile = self.model_config.profile
        # A Provider instance is used as is; names are built per provider and
        # endpoint, since presets can differ in both (see _provider_for).
        self._fixed_provider = provider if isinstance(provider, Provider) else None
        self._provider_options = self._split_options(provider_options, provider_name)
        self._providers: dict[str, Provider] = {}
        # With "auto" each run picks its preset, and so its provider: none is
        # built up front (an offline profile must not create a cloud client).
        self.provider: Provider | None = (
            self._fixed_provider if self.auto else self._provider_for(self.model_config))

        self.system_prompt = system_prompt
        self.tools: list[Tool] = [get_builtin(t) if isinstance(t, str) else t for t in tools]
        self.work_dir = os.path.abspath(os.path.expanduser(work_dir or os.getcwd()))
        self.router = router
        # With "auto", an explicit max_turns wins; otherwise the routed preset's.
        self._explicit_max_turns = max_turns
        self.auto_presets: list[str] | None = None
        self.auto_fallback = auto_fallback
        if self.model_config.model == AUTO:
            self._check_auto(auto_presets, auto_fallback)
        elif auto_presets is not None or auto_fallback is not None:
            raise ValueError("auto_presets / auto_fallback need model='auto' or preset='auto'")
        self.max_turns = max_turns or self.model_config.max_turns
        # An explicit effort wins; else each run's preset, then AGENT_EFFORT.
        self._explicit_effort = effort
        self.effort = self._effort_for(self.model_config)
        self.add_dirs = [os.path.abspath(os.path.expanduser(d)) for d in add_dirs]
        self.env = dict(env or {})
        self.reporter = reporter if reporter is not None else ConsoleReporter()

    @property
    def model(self) -> str:
        """The configured model id, or "auto" when it is picked per run."""
        return self.model_config.model

    @property
    def auto(self) -> bool:
        return self.model_config.model == AUTO

    def _split_options(self, options: dict[str, Any] | None, provider_name: str | None
                       ) -> dict[str, dict[str, Any]]:
        """provider_options as {provider: kwargs}, checked against the providers in play."""
        in_play = (profile_providers(self.profiles, self.profile) if self.auto
                   else [self.model_config.provider])
        if not options:
            return {}
        names = set(available_providers())
        if all(k in names and isinstance(v, dict) for k, v in options.items()):
            unused = [k for k in options if k not in in_play]
            if unused:
                raise ValueError(f"provider_options for {unused}, but this agent only uses "
                                 f"{in_play} (profile '{self.profile}')")
            return {k: dict(v) for k, v in options.items()}
        target = provider_name or (in_play[0] if len(in_play) == 1 else None)
        if target is None:
            raise ValueError(f"provider_options are ambiguous: profile '{self.profile}' uses "
                             f"{in_play}; key them by provider, e.g. {{'{in_play[0]}': {{...}}}}")
        return {target: dict(options)}

    def _check_auto(self, auto_presets: Sequence[str] | None, auto_fallback: str | None) -> None:
        """Fail at construction, not at the first run, on presets that can't be used."""
        router = self.router or get_default_router()
        table = self.profiles.get(self.profile, {})
        usable = [p for p in router.tier_order if p in table]
        if auto_presets is None:
            allowed = usable
        else:
            allowed = [p for p in router.tier_order if p in auto_presets]
            unknown = [p for p in auto_presets if p not in usable]
            if unknown:
                raise ValueError(f"auto_presets {unknown} unusable: the router picks among "
                                 f"{router.tier_order}, profile '{self.profile}' has "
                                 f"{list(table)}")
        if not allowed:
            raise ValueError(f"no preset both the router ({router.tier_order}) and "
                             f"profile '{self.profile}' ({list(table)}) know")
        if auto_fallback is not None and auto_fallback not in allowed:
            raise ValueError(f"auto_fallback '{auto_fallback}' is not in {allowed}")
        self.auto_presets = allowed
        # Build each allowed preset's provider now: a missing key fails here.
        for name in allowed:
            self._provider_for(preset_config(self.profile, name, self.profiles))

    def _provider_for(self, config: ModelConfig) -> Provider:
        """The provider for a model config: its preset's endpoint, then explicit options."""
        if self._fixed_provider is not None:
            if config.provider != self._fixed_provider.name:
                raise ValueError(f"{config.source} uses provider '{config.provider}': pass "
                                 "providers by name for profiles that mix them")
            if config.endpoint:
                raise ValueError(
                    f"{config.source} sets an endpoint ({', '.join(config.endpoint)}): pass "
                    "the provider by name so the harness can configure it")
            return self._fixed_provider
        key = repr((config.provider, sorted(config.endpoint.items())))
        if key not in self._providers:
            # Explicit provider_options (e.g. --base-url) win over the preset's.
            kwargs = {**endpoint_kwargs(config.provider, config.endpoint),
                      **self._provider_options.get(config.provider, {})}
            self._providers[key] = get_provider(config.provider, **kwargs)
        return self._providers[key]

    def _effort_for(self, config: ModelConfig) -> str | None:
        return self._explicit_effort or config.effort or os.getenv("AGENT_EFFORT") or None

    async def _select_model(self, prompt: str
                            ) -> tuple[str, int, Provider, RouteDecision | None, str | None]:
        """The model, turn cap, provider, routing decision and effort of one run."""
        if not self.auto:
            return self.model, self.max_turns, self.provider, None, self.effort
        router = self.router or get_default_router()
        decision = await router.route(prompt, self.system_prompt,
                                      allowed=self.auto_presets, fallback=self.auto_fallback)
        # The preset directly, not resolve_model(): AGENT_MODEL may be "auto" itself.
        config = preset_config(self.profile, decision.preset, self.profiles)
        decision.model = config.model
        effort = self._effort_for(config)
        decision.effort = effort
        return (config.model, self._explicit_max_turns or config.max_turns,
                self._provider_for(config), decision, effort)

    async def run(self, prompt: str) -> RunResult:
        """Run one task to completion."""
        model, max_turns, provider, decision, effort = await self._select_model(prompt)
        if decision is not None:
            self.reporter.on_route(decision)
        request = RunRequest(
            model=model,
            prompt=prompt,
            system_prompt=self.system_prompt,
            tools=self.tools,
            work_dir=self.work_dir,
            max_turns=max_turns,
            effort=effort,
            add_dirs=self.add_dirs,
            env=self.env,
        )
        parent_results = _subagent_results.get()
        children: list[RunResult] = []
        token = _subagent_results.set(children)
        try:
            self.reporter.on_start(provider.name, model)
            result = await provider.run(request, self.reporter)
        finally:
            _subagent_results.reset(token)
        result.subagents = children
        result.route = decision
        if decision is not None and decision.source == "fallback":
            result.notes.append(f"Auto model fell back to preset '{decision.preset}': "
                                f"{decision.reason}")
        if parent_results is not None:
            parent_results.append(result)
        self.reporter.on_result(result)
        return result

    def as_tool(self, name: str, description: str) -> Tool:
        """Expose this agent as a tool, so another agent can delegate to it.

        The calling agent sends a self-contained `task` and receives this
        agent's final answer. The subagent's run is attached to the caller's
        RunResult.subagents, so cumulative_usage / cumulative_cost_usd cover it.
        """

        async def run_subagent(task: str) -> str:
            result = await self.run(task)
            if not result.success:
                return (f"Error: subagent '{name}' stopped ({result.subtype}). "
                        f"Partial output:\n{result.final_text}")
            return result.final_text or "(no output)"

        return Tool(
            name=name,
            description=description,
            parameters={
                "type": "object",
                "properties": {
                    "task": {
                        "type": "string",
                        "description": "Complete, self-contained instructions. The "
                                       "subagent cannot see your conversation.",
                    },
                },
                "required": ["task"],
                "additionalProperties": False,
            },
            func=run_subagent,
        )

    def run_sync(self, prompt: str) -> RunResult:
        """Blocking wrapper around run() for scripts."""
        return asyncio.run(self.run(prompt))
