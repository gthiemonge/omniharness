#!/usr/bin/env python3
"""
Custom Python tools. The @tool decorator builds the JSON schema from the type
hints and the docstring's Args section; the same tools work with both
providers (Claude gets them through an in-process MCP server).

    python examples/03_custom_tools.py
    python examples/03_custom_tools.py "Is 2**61-1 prime? Note the answer."
"""

import ast
import operator
import sys
from datetime import datetime
from pathlib import Path
from typing import Literal
from zoneinfo import ZoneInfo

from omniharness import Agent, ToolContext, tool


@tool
def current_time(timezone: str = "UTC") -> str:
    """Return the current date and time in an IANA timezone.

    Args:
        timezone: IANA timezone name, e.g. 'Europe/Paris'.
    """
    return datetime.now(ZoneInfo(timezone)).isoformat(timespec="seconds")


_OPS = {
    ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
    ast.Div: operator.truediv, ast.FloorDiv: operator.floordiv, ast.Mod: operator.mod,
    ast.Pow: operator.pow, ast.USub: operator.neg,
}


def _eval(node):
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_eval(node.left), _eval(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_eval(node.operand))
    raise ValueError("only numbers and + - * / // % ** are allowed")


@tool
def calculate(expression: str) -> str:
    """Evaluate an arithmetic expression exactly (arbitrary-precision integers).

    Args:
        expression: e.g. '2**64 - 1' or '(3 + 4) * 12 / 7'.
    """
    return str(_eval(ast.parse(expression, mode="eval").body))


@tool
def is_prime(n: str) -> bool:
    """Deterministic Miller-Rabin primality test for n < 3.3e24.

    Args:
        n: The integer to test, as a decimal string.
    """
    # Big integers travel as strings: JSON numbers above 2**53 get rounded on
    # the way (the Claude Code CLI is JavaScript).
    n = int(n)
    if n < 2:
        return False
    bases = (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41)
    for p in bases:
        if n % p == 0:
            return n == p
    d, s = n - 1, 0
    while d % 2 == 0:
        d, s = d // 2, s + 1
    for a in bases:
        x = pow(a, d, n)
        if x in (1, n - 1):
            continue
        for _ in range(s - 1):
            x = pow(x, 2, n)
            if x == n - 1:
                break
        else:
            return False
    return True


@tool
def save_note(ctx: ToolContext, title: str, text: str,
              mode: Literal["append", "overwrite"] = "append") -> str:
    """Save a note to notes.md in the working directory.

    Args:
        title: Short heading for the note.
        text: Note body (markdown).
        mode: Append to the file or replace it.
    """
    # ctx is injected by the harness and hidden from the model.
    path = Path(ctx.work_dir) / "notes.md"
    with open(path, "a" if mode == "append" else "w") as f:
        f.write(f"## {title}\n\n{text}\n\n")
    return f"Saved to {path}"


prompt = " ".join(sys.argv[1:]) or (
    "What time is it in Tokyo and in Paris? Then compute 2**89 - 1, check "
    "whether it is prime, and save a note summarizing all of it."
)

agent = Agent(
    preset="quick",
    system_prompt="Use your tools for every fact or computation; never compute in your head.",
    tools=[current_time, calculate, is_prime, save_note],
)
result = agent.run_sync(prompt)
sys.exit(0 if result.success else 1)
