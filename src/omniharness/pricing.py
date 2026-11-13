"""
Token pricing for providers that do not report cost themselves.

The Claude Agent SDK returns the cost of a run. The OpenAI Responses API does
not, so the openai provider computes it from token usage with this table
(unless an OpenAI-compatible gateway such as litellm reports it in a response
header, which takes precedence).

Rates are USD per 1M tokens. They are EXAMPLES — adjust them for your
account/endpoint, either with register_pricing() or at runtime via the
OPENAI_PRICE_INPUT / OPENAI_PRICE_OUTPUT environment variables.
"""


from .result import Usage

PRICING: dict[str, dict[str, float]] = {
    "gpt-5": {"input": 1.25, "output": 10.00, "cached_input": 0.125},
    "gpt-5-mini": {"input": 0.25, "output": 2.00, "cached_input": 0.025},
}


def register_pricing(model: str, input: float, output: float,
                     cached_input: float | None = None) -> None:
    """Add or replace the rates (USD per 1M tokens) for a model."""
    rates = {"input": input, "output": output}
    if cached_input is not None:
        rates["cached_input"] = cached_input
    PRICING[model] = rates


def get_pricing(model: str) -> dict[str, float] | None:
    return PRICING.get(model)


def compute_cost(usage: Usage, rates: dict[str, float]) -> float:
    cached_rate = rates.get("cached_input", rates["input"])
    return (
        (usage.input_tokens + usage.cache_creation_input_tokens) * rates["input"]
        + usage.cache_read_input_tokens * cached_rate
        + usage.output_tokens * rates["output"]
    ) / 1_000_000
