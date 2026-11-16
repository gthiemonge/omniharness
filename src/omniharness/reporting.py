"""
Progress reporting.

Providers emit events (assistant text, tool calls, tool results) to a Reporter.
ConsoleReporter prints a compact live transcript; subclass Reporter to send
events elsewhere (logs, a UI, a test recorder, ...).
"""

import sys
from typing import Any, TextIO

from .result import RouteDecision, RunResult


class Reporter:
    """Base reporter: ignores every event. Override what you need."""

    def on_route(self, decision: RouteDecision) -> None:
        """A model="auto" agent picked its model for the coming run."""

    def on_start(self, provider: str, model: str) -> None:
        pass

    def on_text(self, text: str) -> None:
        pass

    def on_tool_call(self, name: str, args: dict[str, Any]) -> None:
        pass

    def on_tool_result(self, name: str, output: str, is_error: bool = False) -> None:
        pass

    def on_result(self, result: RunResult) -> None:
        pass


# Which argument best summarizes a call to each built-in tool.
_TOOL_LABEL_KEYS = {
    "Read": "file_path",
    "Write": "file_path",
    "Grep": "pattern",
    "Glob": "pattern",
    "WebFetch": "url",
}


def tool_label(name: str, args: dict[str, Any]) -> str:
    """One-line summary of a tool call, e.g. '[Bash] Run the unit tests'."""
    if name == "Bash":
        label = args.get("description") or (args.get("command") or "")[:120]
        return f"[Bash] {label}"
    key = _TOOL_LABEL_KEYS.get(name)
    if key:
        return f"[{name}] {args.get(key, '?')}"
    if args:
        preview = ", ".join(f"{k}={v!r}" for k, v in args.items())
        return f"[{name}] {preview[:120]}"
    return f"[{name}]"


class ConsoleReporter(Reporter):
    """Print the agent's text, tool calls and (abbreviated) tool results."""

    def __init__(self, stream: TextIO | None = None, show_tool_results: bool = True,
                 max_result_lines: int = 20, prefix: str = "", show_summary: bool = True):
        """
        Args:
            prefix: Prepended to every line, e.g. "[explorer] " to tell
                concurrent subagents apart.
            show_summary: Print the usage/cost summary when a run ends.
        """
        self.stream = stream or sys.stdout
        self.show_tool_results = show_tool_results
        self.max_result_lines = max_result_lines
        self.prefix = prefix
        self.show_summary = show_summary

    def _print(self, text: str = "") -> None:
        lines = text.split("\n") if self.prefix else [text]
        print("\n".join(self.prefix + line for line in lines), file=self.stream, flush=True)

    def on_route(self, decision: RouteDecision) -> None:
        self._print(f"Auto model: {format_route(decision)}")

    def on_start(self, provider: str, model: str) -> None:
        self._print(f"Provider: {provider}  Model: {model}")
        self._print("-" * 60)

    def on_text(self, text: str) -> None:
        self._print(text)

    def on_tool_call(self, name: str, args: dict[str, Any]) -> None:
        self._print(f"  {tool_label(name, args)}")

    def on_tool_result(self, name: str, output: str, is_error: bool = False) -> None:
        if not self.show_tool_results or not output:
            return
        lines = output.strip().split("\n")
        if len(lines) > self.max_result_lines:
            head = self.max_result_lines // 2
            lines = lines[:head] + [f"... ({len(lines) - head} more lines)"]
        for line in lines:
            self._print(f"    {line}")
        self._print()

    def on_result(self, result: RunResult) -> None:
        if self.show_summary:
            self._print(format_summary(result))


def format_route(decision: RouteDecision) -> str:
    """e.g. "quick -> claude-haiku-5-5 (d1: fast 0.93, 712 ms)"."""
    head = f"{decision.preset} -> {decision.model or '?'}"
    if decision.effort:
        head += f" (effort {decision.effort})"
    if decision.source == "fallback":
        return f"{head} (fallback: {decision.reason})"
    if decision.label is None:
        detail = decision.source
    elif decision.method == "complexity":
        detail = f"{decision.source}: {decision.label}, complexity {decision.complexity:.2f}"
    else:
        p = (decision.probabilities or {}).get(decision.choice)
        detail = f"{decision.source}: {decision.label}" + (f" {p:.2f}" if p is not None else "")
    if decision.clamped:
        detail += f", {decision.choice} not allowed"
    if decision.elapsed_ms is not None:
        detail += f", {decision.elapsed_ms:.0f} ms"
    return f"{head} ({detail})"


def format_summary(result: RunResult) -> str:
    """Human-readable outcome + usage summary for any provider."""
    out = ["-" * 60]

    if not result.success:
        if result.subtype == "error_max_turns":
            msg = "Hit turn limit before finishing."
            if result.session_id:
                msg += f" Resume session {result.session_id} to continue."
            out.append(msg)
        else:
            out.append(f"Stopped: {result.subtype}")
    else:
        out.append(f"✓ Done in {result.num_turns} turn(s).")

    u = result.usage
    if u.total_tokens:
        out.append("")
        out.append("Token usage:")
        out.append(f"  Input:  {u.input_tokens:,} new")
        if u.cache_read_input_tokens:
            out.append(f"          {u.cache_read_input_tokens:,} cache read")
        if u.cache_creation_input_tokens:
            out.append(f"          {u.cache_creation_input_tokens:,} cache created")
        if u.cache_read_input_tokens or u.cache_creation_input_tokens:
            out.append(f"          {u.total_input_tokens:,} total input")
        line = f"  Output: {u.output_tokens:,} tokens"
        if u.reasoning_tokens:
            line += f" ({u.reasoning_tokens:,} reasoning)"
        out.append(line)
        out.append(f"  Total:  {u.total_tokens:,} tokens")

    breakdown = result.model_breakdown()
    if any(s["inputTokens"] or s["outputTokens"] for s in breakdown.values()):
        out.append("")
        out.append("Per-model breakdown" + (" (incl. subagents):" if result.subagents
                                            else ":"))
        for model_name, stats in breakdown.items():
            runs = f", {stats['runs']} runs" if stats["runs"] > 1 else ""
            out.append(f"  {model_name}{runs}:")
            out.append(
                f"    Input:  {stats['inputTokens']:,} new / "
                f"{stats['cacheReadInputTokens']:,} cache read / "
                f"{stats['cacheCreationInputTokens']:,} cache created"
            )
            line = f"    Output: {stats['outputTokens']:,} tokens"
            if stats["reasoningTokens"]:
                line += f" ({stats['reasoningTokens']:,} reasoning)"
            out.append(line)
            cost = stats["costUSD"]
            out.append("    Cost:   " + ("unknown" if cost is None else f"${cost:.4f}"))

    if result.total_cost_usd is not None:
        out.append("")
        out.append(f"Total cost: ${result.total_cost_usd:.4f}")

    if result.subagents:
        out.append("")
        out.append(f"Subagents: {len(result.all_runs()) - 1} run(s)")
        cumulative = result.cumulative_cost_usd
        out.append(f"  Total tokens incl. subagents: {result.cumulative_usage.total_tokens:,}")
        if cumulative is not None:
            out.append(f"  Total cost incl. subagents:   ${cumulative:.4f}")

    out.extend(result.notes)
    return "\n".join(out)


def print_usage_summary(result: RunResult, stream: TextIO | None = None) -> None:
    print(format_summary(result), file=stream or sys.stdout)
