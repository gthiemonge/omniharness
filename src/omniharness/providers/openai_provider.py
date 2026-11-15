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
from dataclasses import dataclass
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
        ctx = ToolContext(work_dir=request.work_dir, env=dict(request.env))

        result = RunResult(success=False, provider=self.name, model=request.model)
        usage = Usage()
        # Cost reported by a gateway is authoritative when present.
        api_cost = 0.0
        api_cost_seen = False

        api = self._detected_api or ("responses" if self.api == "auto" else self.api)
        dialect = _DIALECTS[api]
        # The full conversation is kept client-side and resent every turn:
        # server-side state (previous_response_id) is OpenAI-specific.
        history = dialect.start(request)

        for turn in range(1, request.max_turns + 1):
            try:
                raw = await dialect.create(client, self._params(request, registry, history,
                                                                 dialect))
            except Exception as e:  # only switched on, else re-raised
                if not (self.api == "auto" and api == "responses" and turn == 1
                        and getattr(e, "status_code", None) in _NO_RESPONSES_API):
                    raise
                api, dialect = "chat", _DIALECTS["chat"]
                self._detected_api = "chat"
                where = self.base_url or "the endpoint"
                log.info("%s has no /v1/responses (%s): using /v1/chat/completions",
                         where, e.status_code)
                result.notes.append(f"{where} has no /v1/responses: used "
                                    "/v1/chat/completions.")
                history = dialect.start(request)
                raw = await dialect.create(client, self._params(request, registry, history,
                                                                 dialect))
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
            for text in out.texts:
                result.final_text = text
                reporter.on_text(text)
            dialect.record_assistant(history, out)

            if not out.calls:
                result.success = True
                result.subtype = "success"
                break

            # Run this turn's calls concurrently (e.g. a fan-out to several
            # subagents); outputs go back in call order.
            async def execute(call: _ToolCall) -> str:
                try:
                    args = json.loads(call.arguments or "{}")
                except json.JSONDecodeError:
                    args = {}
                reporter.on_tool_call(call.name, args)
                output = await registry.dispatch(call.name, args, ctx)
                reporter.on_tool_result(call.name, output, output.startswith("Error"))
                return output

            outputs = await asyncio.gather(*(execute(call) for call in out.calls))
            dialect.record_results(history, out.calls, outputs)
        else:
            result.subtype = "error_max_turns"

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
        return result

    def _params(self, request: RunRequest, registry: ToolRegistry, history: list,
                dialect: "_Dialect") -> dict[str, Any]:
        params = dialect.params(request, history, registry)
        params.update(self.options)
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
