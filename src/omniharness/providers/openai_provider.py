"""
OpenAI provider: a hand-written agentic loop on the OpenAI Responses API.

OpenAI has no agent SDK with built-in filesystem/bash tools, so this module:
  1. sends the task with the tools' JSON schemas,
  2. executes any function calls the model requests (locally, in Python),
  3. feeds the outputs back and continues until the model stops calling tools
     or the turn cap is hit.

Works with any OpenAI-compatible endpoint that implements /v1/responses
(Azure, litellm, vLLM, Ollama, ...) via base_url:
  explicit argument -> OPENAI_BASE_URL -> official API default.
"""

import json
import os
from typing import Any

from .. import pricing
from ..reporting import Reporter
from ..result import RunRequest, RunResult, Usage
from ..tools.base import ToolContext, ToolRegistry
from .base import Provider

# Response headers OpenAI-compatible gateways use to report per-request cost in
# USD. litellm uses the first; others are accepted defensively.
_COST_HEADERS = ("x-litellm-response-cost", "x-response-cost", "x-cost-usd")


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
    ):
        """
        Args:
            base_url: Custom endpoint. Defaults to OPENAI_BASE_URL, then the
                official API.
            api_key: Defaults to OPENAI_API_KEY.
            price_input / price_output: USD per 1M tokens, overriding the
                pricing table. Defaults to OPENAI_PRICE_INPUT/OUTPUT.
            client: A pre-built openai.AsyncOpenAI (or compatible) client.
            options: Extra fields merged into every responses.create() call
                (e.g. {"temperature": 0}).
        """
        self.base_url = base_url or os.getenv("OPENAI_BASE_URL") or None
        self.api_key = api_key
        self.price_input = _first_float(price_input, os.getenv("OPENAI_PRICE_INPUT"))
        self.price_output = _first_float(price_output, os.getenv("OPENAI_PRICE_OUTPUT"))
        self.client = client
        self.options = options or {}

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
        if os.getenv("OPENAI_ORG"):
            kwargs["organization"] = os.getenv("OPENAI_ORG")
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

        # We keep the full conversation client-side and resend it every turn
        # instead of chaining with previous_response_id: server-side response
        # storage is OpenAI-specific, and OpenAI-compatible backends (litellm,
        # vLLM, ...) typically don't persist responses.
        input_items: list[dict[str, Any]] = [{"role": "user", "content": request.prompt}]

        for turn in range(1, request.max_turns + 1):
            params: dict[str, Any] = {"model": request.model, "input": input_items}
            if request.system_prompt:
                params["instructions"] = request.system_prompt
            if len(registry):
                params["tools"] = registry.openai_schemas()
            if request.effort is not None:
                params["reasoning"] = {"effort": request.effort}
            params.update(self.options)

            # with_raw_response gives access to headers (gateway cost).
            raw = await client.responses.with_raw_response.create(**params)
            response = raw.parse()
            result.num_turns = turn

            turn_cost = _header_cost(raw)
            if turn_cost is not None:
                api_cost += turn_cost
                api_cost_seen = True
            _add_usage(usage, getattr(response, "usage", None))

            # Record assistant outputs into our history (rebuilding minimal
            # items without server ids so any backend accepts them) and collect
            # the tool calls.
            tool_calls = []
            for item in response.output:
                itype = getattr(item, "type", None)
                if itype == "message":
                    text = _message_text(item)
                    if text:
                        result.final_text = text
                        reporter.on_text(text)
                        input_items.append({"role": "assistant", "content": text})
                elif itype == "function_call":
                    tool_calls.append(item)
                    input_items.append({
                        "type": "function_call",
                        "call_id": item.call_id,
                        "name": item.name,
                        "arguments": item.arguments,
                    })

            if not tool_calls:
                result.success = True
                result.subtype = "success"
                break

            for call in tool_calls:
                try:
                    args = json.loads(call.arguments or "{}")
                except json.JSONDecodeError:
                    args = {}
                reporter.on_tool_call(call.name, args)
                output = await registry.dispatch(call.name, args, ctx)
                reporter.on_tool_result(call.name, output, output.startswith("Error"))
                input_items.append({
                    "type": "function_call_output",
                    "call_id": call.call_id,
                    "output": output,
                })
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

    def _resolve_rates(self, model: str) -> dict[str, float] | None:
        if self.price_input is not None and self.price_output is not None:
            return {"input": self.price_input, "output": self.price_output}
        return pricing.get_pricing(model)


def _add_usage(total: Usage, usage: Any) -> None:
    if usage is None:
        return
    inp = getattr(usage, "input_tokens", 0) or 0
    in_details = getattr(usage, "input_tokens_details", None)
    cached = (getattr(in_details, "cached_tokens", 0) or 0) if in_details else 0
    out_details = getattr(usage, "output_tokens_details", None)
    # OpenAI's input_tokens includes the cached ones.
    total.input_tokens += inp - cached
    total.cache_read_input_tokens += cached
    total.output_tokens += getattr(usage, "output_tokens", 0) or 0
    if out_details:
        total.reasoning_tokens += getattr(out_details, "reasoning_tokens", 0) or 0


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


def _message_text(item: Any) -> str:
    return "".join(
        getattr(block, "text", "") or "" for block in (getattr(item, "content", None) or [])
    )
