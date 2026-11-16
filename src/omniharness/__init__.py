"""
omniharness: a small, provider-agnostic harness for tool-using AI agents.

Write your agent once (system prompt + tools), run it on the Claude Agent SDK
or on any OpenAI Responses-compatible endpoint.
"""

from .agent import Agent
from .config import (
    AUTO,
    DEFAULT_PROFILES,
    ModelConfig,
    format_profiles,
    load_profiles,
    merge_profiles,
    resolve_model,
)
from .options import AgentOptions
from .pricing import register_pricing
from .providers import Provider, get_provider, register_provider
from .reporting import ConsoleReporter, Reporter, format_summary, print_usage_summary
from .result import RouteDecision, RunRequest, RunResult, Usage
from .routing import D1Router, FixedRouter, Router, get_default_router, set_default_router
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
    "AUTO",
    "BUILTIN_TOOLS",
    "CODING_TOOLS",
    "DEFAULT_PROFILES",
    "READ_ONLY_TOOLS",
    "Agent",
    "AgentOptions",
    "ConsoleReporter",
    "D1Router",
    "FixedRouter",
    "ModelConfig",
    "Provider",
    "Reporter",
    "RouteDecision",
    "Router",
    "RunRequest",
    "RunResult",
    "Skill",
    "Tool",
    "ToolContext",
    "ToolRegistry",
    "Usage",
    "format_profiles",
    "format_summary",
    "get_default_router",
    "get_provider",
    "load_profiles",
    "load_skill",
    "merge_profiles",
    "print_usage_summary",
    "register_pricing",
    "register_provider",
    "resolve_model",
    "set_default_router",
    "tool",
]
