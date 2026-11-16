"""
OpenAI provider: a hand-written agentic loop on an OpenAI-compatible API.

OpenAI has no agent SDK with built-in filesystem/bash tools, so this module:
  1. sends the task with the tools' JSON schemas,
  2. executes any function calls the model requests (locally, in Python),
  3. feeds the outputs back and continues until the model stops calling tools
     or the turn cap is hit.

Two wire formats ("dialects") drive the same loop:
  responses  /v1/responses: OpenAI, llama.cpp v0.8.0+, vLLM, litellm, ...
  chat       /v1/chat/completions: the most widely implemented one (Ollama,
             LM Studio, TGI, proxies, ...)
With api="auto" (the default) the first request goes to /v1/responses; if the
server doesn't have it (404 / 405 / 501), this endpoint switches to
/v1/chat/completions for good.

The endpoint comes from base_url: explicit argument -> OPENAI_BASE_URL ->
official API default.
"""

import asyncio
import json
import logging
import os
import re
from dataclasses import dataclass, field
from typing import Any

from .. import pricing
from ..reporting import Reporter
from ..result import RunRequest, RunResult, Usage
from ..tools.base import ToolContext, ToolRegistry
from .base import Provider

log = logging.getLogger("omniharness.openai")

# Response headers OpenAI-compatible gateways use to report per-request cost in
# USD. litellm uses the first; others are accepted defensively.
_COST_HEADERS = ("x-litellm-response-cost", "x-response-cost", "x-cost-usd")

API_MODES = ("auto", "responses", "chat")
# Context window trimming: the last turns keep their tool outputs whole; older
# ones keep this many characters.
_KEEP_RECENT = 2
_ELIDED_HEAD = 500
# How servers say a request is too long (llama.cpp, vLLM, OpenAI, ...).
_CONTEXT_ERROR = re.compile(r"context|too long|maximum.*length|too many tokens", re.IGNORECASE)
# HTTP statuses meaning "this server has no /v1/responses".
_NO_RESPONSES_API = (404, 405, 501)


@dataclass
class _ToolCall:
    id: str
    name: str
    arguments: str


@dataclass
class _Turn:
    """One model response, independent of the wire format."""

    texts: list[str]
    calls: list[_ToolCall]
    # (input incl. cached, cached, output, reasoning) tokens.
    usage: tuple[int, int, int, int]


@dataclass
class _History:
    """The conversation: the initial items, then one list of items per turn."""

    prefix: list[dict[str, Any]]
    turns: list[list[dict[str, Any]]] = field(default_factory=list)
    capped: int = 0
    elided: int = 0
    dropped: int = 0

    def items(self) -> list[dict[str, Any]]:
        items = list(self.prefix)
        if self.dropped:
            items.append({"role": "user", "content": (
                f"[{self.dropped} earlier step(s) of this task, with their tool calls and "
                "results, were removed to fit the context window.]")})
        for turn in self.turns:
            items.extend({k: v for k, v in item.items() if k != "_elided"} for item in turn)
        return items


class _Window:
    """A context window in tokens, with a token estimate calibrated by the server."""

    def __init__(self, tokens: int):
        self.tokens = tokens
        # Characters of request JSON per token; refined from each response's usage.
        self.chars_per_token = 3.5

    def budget(self) -> int:
        """Tokens the request may use: the window minus room for the reply."""
        reserve = min(self.tokens // 4, 8192)
        return int((self.tokens - reserve) * 0.95)

    def fits(self, params: dict[str, Any]) -> bool:
        return _size(params) / self.chars_per_token <= self.budget()

    def tool_output_chars(self) -> int:
        """The cap for one tool output: a quarter of the budget."""
        return max(1000, int(self.budget() * self.chars_per_token / 4))

    def calibrate(self, request_chars: int, input_tokens: int) -> None:
        if input_tokens > 0:
            self.chars_per_token = min(8.0, max(1.5, request_chars / input_tokens))


def _size(params: dict[str, Any]) -> int:
    return len(json.dumps(params, default=str))


class OpenAIProvider(Provider):
    name = "openai"

    def __init__(
        self,
        base_url: str | None = None,
        api_key: str | None = None,
        price_input: float | None = None,
        price_output: float | None = None,
        client: Any = None,
        options: dict[str, Any] | None = None,
        organization: str | None = None,
        api: str = "auto",
        context_window: int | None = None,
    ):
        """
        Args:
            base_url: Custom endpoint. Defaults to OPENAI_BASE_URL, then the
                official API.
            api_key: Defaults to OPENAI_API_KEY.
            price_input / price_output: USD per 1M tokens, overriding the
                pricing table. Defaults to OPENAI_PRICE_INPUT/OUTPUT.
            client: A pre-built openai.AsyncOpenAI (or compatible) client.
            options: Extra fields merged into every request
                (e.g. {"temperature": 0}).
            organization: OpenAI organization id. Defaults to OPENAI_ORG.
            api: "responses", "chat" (Chat Completions), or "auto": Responses,
                falling back to Chat Completions when the server lacks it.
            context_window: The model's context size in tokens. When set, tool
                outputs are capped and older history is shortened, then
                dropped, so requests fit. None: no trimming.
        """
        if api not in API_MODES:
            raise ValueError(f"api must be one of {', '.join(API_MODES)}, not '{api}'")
        self.base_url = base_url or os.getenv("OPENAI_BASE_URL") or None
        self.api_key = api_key
        self.price_input = _first_float(price_input, os.getenv("OPENAI_PRICE_INPUT"))
        self.price_output = _first_float(price_output, os.getenv("OPENAI_PRICE_OUTPUT"))
        self.client = client
        self.options = options or {}
        self.organization = organization or os.getenv("OPENAI_ORG") or None
        self.api = api
        self.context_window = int(context_window) if context_window else None
        # What "auto" settled on for this endpoint, once known.
        self._detected_api: str | None = None

    def _make_client(self):
        try:
            from openai import AsyncOpenAI
        except ImportError as e:
            raise RuntimeError(
                "The 'openai' package is required for the openai provider. "
                "Install it with: pip install 'omniharness[openai]'"
            ) from e

        api_key = self.api_key or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is not set.")
        kwargs: dict[str, Any] = {"api_key": api_key}
        if self.base_url:
            kwargs["base_url"] = self.base_url
        if self.organization:
            kwargs["organization"] = self.organization
        return AsyncOpenAI(**kwargs)

    async def run(self, request: RunRequest, reporter: Reporter) -> RunResult:
        client = self.client or self._make_client()
        registry = ToolRegistry(request.tools)
        window = _Window(self.context_window) if self.context_window else None
        ctx = ToolContext(work_dir=request.work_dir, env=dict(request.env))
        if window:
            ctx.max_output_chars = min(ctx.max_output_chars, window.tool_output_chars())

        result = RunResult(success=False, provider=self.name, model=request.model)
        usage = Usage()
        # Cost reported by a gateway is authoritative when present.
        api_cost = 0.0
        api_cost_seen = False

        api = self._detected_api or ("responses" if self.api == "auto" else self.api)
        dialect = _DIALECTS[api]
        # The full conversation is kept client-side and resent every turn:
        # server-side state (previous_response_id) is OpenAI-specific.
        history = _History(dialect.start(request))

        for turn in range(1, request.max_turns + 1):
            params = self._fit(request, registry, history, dialect, window)
            try:
                raw = await dialect.create(client, params)
            except Exception as e:  # only handled below, else re-raised
                status = getattr(e, "status_code", None)
                if (self.api == "auto" and api == "responses" and turn == 1
                        and status in _NO_RESPONSES_API):
                    api, dialect = "chat", _DIALECTS["chat"]
                    self._detected_api = "chat"
                    where = self.base_url or "the endpoint"
                    log.info("%s has no /v1/responses (%s): using /v1/chat/completions",
                             where, status)
                    result.notes.append(f"{where} has no /v1/responses: used "
                                        "/v1/chat/completions.")
                    history = _History(dialect.start(request))
                    params = self._fit(request, registry, history, dialect, window)
                elif window and status == 400 and _CONTEXT_ERROR.search(str(e)):
                    # The estimate was off: trim harder and try once more.
                    log.info("context window exceeded (%s): trimming harder", e)
                    params = self._fit(request, registry, history, dialect, window,
                                       force=True)
                else:
                    raise
                raw = await dialect.create(client, params)
            if self.api == "auto":
                self._detected_api = api
            response = raw.parse()
            result.num_turns = turn

            turn_cost = _header_cost(raw)
            if turn_cost is not None:
                api_cost += turn_cost
                api_cost_seen = True

            out = dialect.read(response, turn)
            _add_usage(usage, out.usage)
            if window:
                window.calibrate(_size(params), out.usage[0])
            for text in out.texts:
                result.final_text = text
                reporter.on_text(text)
            items: list[dict[str, Any]] = []
            dialect.record_assistant(items, out)

            if not out.calls:
                history.turns.append(items)
                result.success = True
                result.subtype = "success"
                break

            # Run this turn's calls concurrently (e.g. a fan-out to several
            # subagents); outputs go back in call order.
            async def execute(call: _ToolCall, history: _History = history) -> str:
                try:
                    args = json.loads(call.arguments or "{}")
                except json.JSONDecodeError:
                    args = {}
                reporter.on_tool_call(call.name, args)
                output = await registry.dispatch(call.name, args, ctx)
                reporter.on_tool_result(call.name, output, output.startswith("Error"))
                # Custom tools don't know the cap the built-in ones follow.
                if window and len(output) > ctx.max_output_chars:
                    history.capped += 1
                    cut = len(output) - ctx.max_output_chars
                    output = (output[:ctx.max_output_chars]
                              + f"\n... [truncated to fit the context window, {cut} more chars]")
                return output

            outputs = await asyncio.gather(*(execute(call) for call in out.calls))
            dialect.record_results(items, out.calls, outputs)
            history.turns.append(items)
        else:
            result.subtype = "error_max_turns"

        if history.capped or history.elided or history.dropped:
            result.notes.append(
                f"Trimmed to fit the {self.context_window:,}-token context window: "
                f"{history.capped} tool output(s) capped, {history.elided} older one(s) "
                f"shortened, {history.dropped} turn(s) removed.")
        result.usage = usage
        if api_cost_seen:
            result.total_cost_usd = api_cost
        else:
            rates = self._resolve_rates(request.model)
            if rates is not None:
                result.total_cost_usd = pricing.compute_cost(usage, rates)
            else:
                result.notes.append(
                    f"Cost: unknown — no pricing for model '{request.model}' "
                    "(see omniharness.pricing.register_pricing)."
                )
        # One model per run here; same shape as the Claude SDK's breakdown.
        result.model_usage = {request.model: {
            "inputTokens": usage.input_tokens,
            "cacheReadInputTokens": usage.cache_read_input_tokens,
            "cacheCreationInputTokens": usage.cache_creation_input_tokens,
            "outputTokens": usage.output_tokens,
            "reasoningTokens": usage.reasoning_tokens,
            "costUSD": result.total_cost_usd,
        }}
        return result

    def _params(self, request: RunRequest, registry: ToolRegistry, history: "_History",
                dialect: "_Dialect") -> dict[str, Any]:
        params = dialect.params(request, history.items(), registry)
        params.update(self.options)
        return params

    def _fit(self, request: RunRequest, registry: ToolRegistry, history: "_History",
             dialect: "_Dialect", window: "_Window | None", force: bool = False
             ) -> dict[str, Any]:
        """The request params, after trimming the history to the context window.

        In order, until it fits: shorten the tool outputs of older turns (the
        last KEEP_RECENT stay whole), then drop the oldest turns whole (a tool
        call never loses its result). force: the server said it doesn't fit,
        whatever the estimate; shorten all but the last turn and drop half.
        """
        params = self._params(request, registry, history, dialect)
        if window is None or (not force and window.fits(params)):
            return params
        keep = 1 if force else _KEEP_RECENT
        for turn in history.turns[:max(len(history.turns) - keep, 0)]:
            for i, item in enumerate(turn):
                key = dialect.output_key(item)
                if key and not item.get("_elided") and len(item[key]) > _ELIDED_HEAD:
                    turn[i] = {**item, key: (
                        item[key][:_ELIDED_HEAD] + f"\n... [{len(item[key]) - _ELIDED_HEAD} "
                        "chars removed to fit the context window]"), "_elided": True}
                    history.elided += 1
            params = self._params(request, registry, history, dialect)
            if not force and window.fits(params):
                return params
        target = len(history.turns) // 2 if force else 1
        while len(history.turns) > target and (force or not window.fits(params)):
            history.turns.pop(0)
            history.dropped += 1
            params = self._params(request, registry, history, dialect)
        return params

    def _resolve_rates(self, model: str) -> dict[str, float] | None:
        if self.price_input is not None and self.price_output is not None:
            return {"input": self.price_input, "output": self.price_output}
        return pricing.get_pricing(model)


# ── wire formats ───────────────────────────────────────────────────────────


class _Dialect:
    def start(self, request: RunRequest) -> list[dict[str, Any]]:
        raise NotImplementedError

    def params(self, request: RunRequest, history: list, registry: ToolRegistry) -> dict:
        raise NotImplementedError

    async def create(self, client: Any, params: dict) -> Any:
        """The raw response (with_raw_response: gives access to cost headers)."""
        raise NotImplementedError

    def read(self, response: Any, turn: int) -> _Turn:
        raise NotImplementedError

    def record_assistant(self, history: list, out: _Turn) -> None:
        raise NotImplementedError

    def record_results(self, history: list, calls: list[_ToolCall], outputs: list[str]) -> None:
        raise NotImplementedError

    def output_key(self, item: dict[str, Any]) -> str | None:
        """The key holding a tool output in this history item, if it is one."""
        raise NotImplementedError


class _Responses(_Dialect):
    """/v1/responses. History items are rebuilt without server ids so any
    backend accepts them."""

    def start(self, request):
        return [{"role": "user", "content": request.prompt}]

    def params(self, request, history, registry):
        params: dict[str, Any] = {"model": request.model, "input": history}
        if request.system_prompt:
            params["instructions"] = request.system_prompt
        if len(registry):
            params["tools"] = registry.openai_schemas()
        if request.effort is not None:
            params["reasoning"] = {"effort": request.effort}
        return params

    async def create(self, client, params):
        return await client.responses.with_raw_response.create(**params)

    def read(self, response, turn):
        texts, calls = [], []
        for item in response.output:
            itype = getattr(item, "type", None)
            if itype == "message":
                text = "".join(getattr(block, "text", "") or ""
                               for block in (getattr(item, "content", None) or []))
                if text:
                    texts.append(text)
            elif itype == "function_call":
                calls.append(_ToolCall(item.call_id, item.name, item.arguments or ""))
        u = getattr(response, "usage", None)
        inp_details = getattr(u, "input_tokens_details", None)
        out_details = getattr(u, "output_tokens_details", None)
        return _Turn(texts, calls, (
            getattr(u, "input_tokens", 0) or 0,
            getattr(inp_details, "cached_tokens", 0) or 0,
            getattr(u, "output_tokens", 0) or 0,
            getattr(out_details, "reasoning_tokens", 0) or 0,
        ))

    def record_assistant(self, history, out):
        history.extend({"role": "assistant", "content": text} for text in out.texts)
        history.extend({"type": "function_call", "call_id": c.id, "name": c.name,
                        "arguments": c.arguments} for c in out.calls)

    def record_results(self, history, calls, outputs):
        history.extend({"type": "function_call_output", "call_id": c.id, "output": o}
                       for c, o in zip(calls, outputs, strict=True))

    def output_key(self, item):
        return "output" if item.get("type") == "function_call_output" else None


class _Chat(_Dialect):
    """/v1/chat/completions."""

    def start(self, request):
        messages = []
        if request.system_prompt:
            messages.append({"role": "system", "content": request.system_prompt})
        messages.append({"role": "user", "content": request.prompt})
        return messages

    def params(self, request, history, registry):
        params: dict[str, Any] = {"model": request.model, "messages": history}
        if len(registry):
            params["tools"] = [
                {"type": "function", "function": {"name": t["name"],
                                                  "description": t["description"],
                                                  "parameters": t["parameters"]}}
                for t in registry.openai_schemas()
            ]
        if request.effort is not None:
            params["reasoning_effort"] = request.effort
        return params

    async def create(self, client, params):
        return await client.chat.completions.with_raw_response.create(**params)

    def read(self, response, turn):
        message = response.choices[0].message
        text = getattr(message, "content", None) or ""
        calls = []
        for i, tc in enumerate(getattr(message, "tool_calls", None) or []):
            fn = tc.function
            # Some local servers omit the id; the result must reference one.
            calls.append(_ToolCall(getattr(tc, "id", None) or f"call_{turn}_{i}",
                                   fn.name, fn.arguments or ""))
        u = getattr(response, "usage", None)
        inp_details = getattr(u, "prompt_tokens_details", None)
        out_details = getattr(u, "completion_tokens_details", None)
        return _Turn([text] if text else [], calls, (
            getattr(u, "prompt_tokens", 0) or 0,
            getattr(inp_details, "cached_tokens", 0) or 0,
            getattr(u, "completion_tokens", 0) or 0,
            getattr(out_details, "reasoning_tokens", 0) or 0,
        ))

    def record_assistant(self, history, out):
        message: dict[str, Any] = {"role": "assistant", "content": "\n".join(out.texts) or None}
        if out.calls:
            message["tool_calls"] = [
                {"id": c.id, "type": "function",
                 "function": {"name": c.name, "arguments": c.arguments or "{}"}}
                for c in out.calls
            ]
        history.append(message)

    def record_results(self, history, calls, outputs):
        history.extend({"role": "tool", "tool_call_id": c.id, "content": o}
                       for c, o in zip(calls, outputs, strict=True))

    def output_key(self, item):
        return "content" if item.get("role") == "tool" else None


_DIALECTS: dict[str, _Dialect] = {"responses": _Responses(), "chat": _Chat()}


# ── helpers ────────────────────────────────────────────────────────────────


def _add_usage(total: Usage, usage: tuple[int, int, int, int]) -> None:
    inp, cached, out, reasoning = usage
    # OpenAI's input counts include the cached tokens.
    total.input_tokens += inp - cached
    total.cache_read_input_tokens += cached
    total.output_tokens += out
    total.reasoning_tokens += reasoning


def _header_cost(raw: Any) -> float | None:
    headers = getattr(raw, "headers", None)
    if not headers:
        return None
    for key in _COST_HEADERS:
        value = headers.get(key)
        if value not in (None, ""):
            try:
                return float(value)
            except (TypeError, ValueError):
                continue
    return None


def _first_float(*values: Any) -> float | None:
    for v in values:
        if v is None or v == "":
            continue
        try:
            return float(v)
        except (TypeError, ValueError):
            continue
    return None
