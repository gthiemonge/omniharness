"""
Provider abstraction.

A provider drives a model through an agentic loop: given a RunRequest (prompt,
system prompt, tools, working directory, ...) it lets the model call tools until
it produces a final answer or hits the turn cap, reporting progress to a
Reporter, and returns a RunResult.

Built-in providers:
  - ClaudeProvider  (providers/claude.py)          -> Claude Agent SDK
  - OpenAIProvider  (providers/openai_provider.py) -> OpenAI Responses API
"""

from abc import ABC, abstractmethod

from ..reporting import Reporter
from ..result import RunRequest, RunResult


class Provider(ABC):
    """Base class every provider implements."""

    name: str = "base"

    @abstractmethod
    async def run(self, request: RunRequest, reporter: Reporter) -> RunResult:
        """Run one task end-to-end and return a RunResult."""
        raise NotImplementedError
