"""
omniharness: a small, provider-agnostic harness for tool-using AI agents.

Write your agent once (system prompt + tools), run it on the Claude Agent SDK
or on any OpenAI Responses-compatible endpoint.
"""

from .agent import Agent
from .config import DEFAULT_PRESETS, ModelConfig, format_presets, resolve_model
from .pricing import register_pricing
from .providers import Provider, get_provider, register_provider
from .reporting import ConsoleReporter, Reporter, format_summary, print_usage_summary
from .result import RunRequest, RunResult, Usage
from .skill import Skill, load_skill
from .tools import (
    ALL_TOOLS,
    BUILTIN_TOOLS,
    CODING_TOOLS,
    READ_ONLY_TOOLS,
    Tool,
    ToolContext,
    ToolRegistry,
    tool,
)

__version__ = "0.1.0"

__all__ = [
    "ALL_TOOLS",
    "BUILTIN_TOOLS",
    "CODING_TOOLS",
    "DEFAULT_PRESETS",
    "READ_ONLY_TOOLS",
    "Agent",
    "ConsoleReporter",
    "ModelConfig",
    "Provider",
    "Reporter",
    "RunRequest",
    "RunResult",
    "Skill",
    "Tool",
    "ToolContext",
    "ToolRegistry",
    "Usage",
    "format_presets",
    "format_summary",
    "get_provider",
    "load_skill",
    "print_usage_summary",
    "register_pricing",
    "register_provider",
    "resolve_model",
    "tool",
]
