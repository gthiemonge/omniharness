"""
High-level entry point.

    from omniharness import Agent, READ_ONLY_TOOLS

    agent = Agent(provider="claude", system_prompt="You are a code explorer.",
                  tools=READ_ONLY_TOOLS, work_dir="~/src/myproject")
    result = agent.run_sync("Where is the configuration loaded?")
    print(result.final_text)
"""

import asyncio
import os
from collections.abc import Sequence
from typing import Any

from .config import ModelConfig, resolve_model
from .providers import Provider, get_provider
from .reporting import ConsoleReporter, Reporter
from .result import RunRequest, RunResult
from .tools.base import Tool
from .tools.builtin import get_builtin

ToolSpec = Tool | str


class Agent:
    def __init__(
        self,
        *,
        provider: str | Provider | None = None,
        model: str | None = None,
        preset: str | None = None,
        system_prompt: str = "",
        tools: Sequence[ToolSpec] = (),
        work_dir: str | None = None,
        max_turns: int | None = None,
        effort: str | None = None,
        add_dirs: Sequence[str] = (),
        env: dict[str, str] | None = None,
        reporter: Reporter | None = None,
        presets: dict[str, dict[str, dict]] | None = None,
        provider_options: dict[str, Any] | None = None,
    ):
        """
        Args:
            provider: "claude", "openai", a registered name, or a Provider
                instance. Defaults to AGENT_PROVIDER, then "claude".
            model: Model id. Defaults to AGENT_MODEL, then the preset's model.
            preset: Preset name ("quick", "standard", "thorough", ...).
            system_prompt: Instructions for the model.
            tools: Tool objects and/or built-in tool names ("Read", "Bash", ...).
            work_dir: Directory tools operate in. Defaults to the current one.
            max_turns: Turn cap. Defaults to the preset's value.
            effort: Reasoning effort ("low", "medium", "high"). Defaults to
                AGENT_EFFORT, then the provider default.
            add_dirs: Extra directories the agent may access (claude provider).
            env: Extra environment variables for the agent's tools.
            reporter: Progress sink. Defaults to ConsoleReporter().
            presets: Your own preset table (see config.DEFAULT_PRESETS).
            provider_options: kwargs for the provider constructor when
                `provider` is a name (e.g. {"base_url": ...} for openai).
        """
        provider_name = provider.name if isinstance(provider, Provider) else provider
        self.model_config: ModelConfig = resolve_model(
            provider_name, preset, model, presets=presets
        )
        if isinstance(provider, Provider):
            self.provider = provider
        else:
            self.provider = get_provider(self.model_config.provider, **(provider_options or {}))

        self.system_prompt = system_prompt
        self.tools: list[Tool] = [get_builtin(t) if isinstance(t, str) else t for t in tools]
        self.work_dir = os.path.abspath(os.path.expanduser(work_dir or os.getcwd()))
        self.max_turns = max_turns or self.model_config.max_turns
        self.effort = effort or os.getenv("AGENT_EFFORT") or None
        self.add_dirs = [os.path.abspath(os.path.expanduser(d)) for d in add_dirs]
        self.env = dict(env or {})
        self.reporter = reporter if reporter is not None else ConsoleReporter()

    @property
    def model(self) -> str:
        return self.model_config.model

    async def run(self, prompt: str) -> RunResult:
        """Run one task to completion."""
        request = RunRequest(
            model=self.model,
            prompt=prompt,
            system_prompt=self.system_prompt,
            tools=self.tools,
            work_dir=self.work_dir,
            max_turns=self.max_turns,
            effort=self.effort,
            add_dirs=self.add_dirs,
            env=self.env,
        )
        self.reporter.on_start(self.provider.name, self.model)
        result = await self.provider.run(request, self.reporter)
        self.reporter.on_result(result)
        return result

    def run_sync(self, prompt: str) -> RunResult:
        """Blocking wrapper around run() for scripts."""
        return asyncio.run(self.run(prompt))
