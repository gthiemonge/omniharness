"""
Provider-neutral run request and result types.
"""

from dataclasses import dataclass, field
from typing import Any

from .tools.base import Tool


@dataclass
class RunRequest:
    """Everything a provider needs to run one agent task."""

    model: str
    prompt: str
    system_prompt: str = ""
    tools: list[Tool] = field(default_factory=list)
    work_dir: str = "."
    max_turns: int = 50
    # Reasoning effort ("low", "medium", "high", ...). None = provider default.
    effort: str | None = None
    # Extra directories the agent may access (Claude SDK sandbox; ignored by
    # providers that run tools locally without restriction).
    add_dirs: list[str] = field(default_factory=list)
    # Extra environment variables for the agent's tools / subprocesses.
    env: dict[str, str] = field(default_factory=dict)


@dataclass
class Usage:
    """Token usage, normalized across providers."""

    # Uncached input tokens.
    input_tokens: int = 0
    cache_read_input_tokens: int = 0
    cache_creation_input_tokens: int = 0
    output_tokens: int = 0
    # Part of output_tokens spent on reasoning/thinking, when reported.
    reasoning_tokens: int = 0

    @property
    def total_input_tokens(self) -> int:
        return self.input_tokens + self.cache_read_input_tokens + self.cache_creation_input_tokens

    @property
    def total_tokens(self) -> int:
        return self.total_input_tokens + self.output_tokens


@dataclass
class RunResult:
    """Outcome of an agent run, normalized across providers."""

    success: bool
    # "success", "error_max_turns", "error", ...
    subtype: str = "success"
    # The model's final answer (last assistant message).
    final_text: str = ""
    usage: Usage = field(default_factory=Usage)
    # Per-model breakdown, as reported by the provider (Claude SDK only).
    model_usage: dict[str, Any] | None = None
    # Total cost in USD, when reported by the provider or computable.
    total_cost_usd: float | None = None
    # Session identifier for resuming (Claude only).
    session_id: str | None = None
    num_turns: int = 0
    provider: str = ""
    model: str = ""
    # Extra free-form notes to print.
    notes: list[str] = field(default_factory=list)
