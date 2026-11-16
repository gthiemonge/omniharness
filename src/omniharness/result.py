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

    def __add__(self, other: "Usage") -> "Usage":
        return Usage(
            input_tokens=self.input_tokens + other.input_tokens,
            cache_read_input_tokens=self.cache_read_input_tokens + other.cache_read_input_tokens,
            cache_creation_input_tokens=(
                self.cache_creation_input_tokens + other.cache_creation_input_tokens
            ),
            output_tokens=self.output_tokens + other.output_tokens,
            reasoning_tokens=self.reasoning_tokens + other.reasoning_tokens,
        )


@dataclass
class RouteDecision:
    """How a model="auto" agent picked its preset for one run."""

    # The preset used: the router's choice, clamped to the agent's allowed presets.
    preset: str
    # "d1", "fallback", "fixed", ...
    source: str
    # The router's own pick, before clamping (None on fallback).
    choice: str | None = None
    # "choice" or "complexity" for d1.
    method: str | None = None
    # d1's answer: the option label ("fast", ...) or complexity level ("moderate", ...).
    label: str | None = None
    # Mean complexity level (0 = trivial ... 4 = very complex), "complexity" only.
    complexity: float | None = None
    confidence: float | None = None
    # Per preset ("choice") or per level "0".."4" ("complexity").
    probabilities: dict[str, float] | None = None
    elapsed_ms: float | None = None
    # Why routing fell back, when it did.
    reason: str | None = None
    # Filled in by the Agent once the preset is resolved.
    model: str = ""
    effort: str | None = None

    @property
    def clamped(self) -> bool:
        return self.choice is not None and self.choice != self.preset


@dataclass
class RunResult:
    """Outcome of an agent run, normalized across providers."""

    success: bool
    # "success", "error_max_turns", "error", ...
    subtype: str = "success"
    # The model's final answer (last assistant message).
    final_text: str = ""
    usage: Usage = field(default_factory=Usage)
    # Per-model breakdown: {model: {"inputTokens", "cacheReadInputTokens",
    # "cacheCreationInputTokens", "outputTokens", "costUSD", ...}} (the Claude
    # SDK's shape; costUSD None when unknown). See model_breakdown().
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
    # Set when the model was chosen automatically (model="auto").
    route: RouteDecision | None = None
    # Runs of subagents this agent called (see Agent.as_tool), in completion order.
    subagents: list["RunResult"] = field(default_factory=list)

    def all_runs(self) -> list["RunResult"]:
        """This run followed by every nested subagent run."""
        runs = [self]
        for sub in self.subagents:
            runs += sub.all_runs()
        return runs

    @property
    def cumulative_usage(self) -> Usage:
        """Token usage of this run plus all subagent runs."""
        total = Usage()
        for run in self.all_runs():
            total = total + run.usage
        return total

    @property
    def cumulative_cost_usd(self) -> float | None:
        """Cost of this run plus all subagent runs; None if any cost is unknown."""
        costs = [run.total_cost_usd for run in self.all_runs()]
        return None if None in costs else sum(costs)

    def model_breakdown(self, include_subagents: bool = True) -> dict[str, dict[str, Any]]:
        """Tokens and cost per model, summed over this run (and its subagents).

        Each entry: inputTokens (uncached), cacheReadInputTokens,
        cacheCreationInputTokens, outputTokens, reasoningTokens, costUSD (None
        when any part is unknown) and runs (how many runs used the model).
        """
        totals: dict[str, dict[str, Any]] = {}
        for run in self.all_runs() if include_subagents else [self]:
            per_model = run.model_usage or {run.model or "?": _usage_entry(run)}
            for model, stats in per_model.items():
                if not isinstance(stats, dict):
                    continue
                entry = totals.setdefault(model, {
                    "inputTokens": 0, "cacheReadInputTokens": 0,
                    "cacheCreationInputTokens": 0, "outputTokens": 0,
                    "reasoningTokens": 0, "costUSD": 0.0, "runs": 0})
                for key in ("inputTokens", "cacheReadInputTokens",
                            "cacheCreationInputTokens", "outputTokens", "reasoningTokens"):
                    entry[key] += stats.get(key) or 0
                cost = stats.get("costUSD")
                entry["costUSD"] = (None if cost is None or entry["costUSD"] is None
                                    else entry["costUSD"] + cost)
                entry["runs"] += 1
        return totals


def _usage_entry(run: "RunResult") -> dict[str, Any]:
    """A model_usage entry from a run's totals (providers without a breakdown)."""
    u = run.usage
    return {"inputTokens": u.input_tokens, "cacheReadInputTokens": u.cache_read_input_tokens,
            "cacheCreationInputTokens": u.cache_creation_input_tokens,
            "outputTokens": u.output_tokens, "reasoningTokens": u.reasoning_tokens,
            "costUSD": run.total_cost_usd}
