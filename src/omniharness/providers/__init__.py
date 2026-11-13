"""
Provider factory.

Providers are imported lazily so that a user who only has one provider's SDK
installed can still use that provider without the other's dependency.
Register your own with register_provider().
"""

from collections.abc import Callable

from .base import Provider

__all__ = ["Provider", "available_providers", "get_provider", "register_provider"]


def _claude(**kwargs) -> Provider:
    from .claude import ClaudeProvider

    return ClaudeProvider(**kwargs)


def _openai(**kwargs) -> Provider:
    from .openai_provider import OpenAIProvider

    return OpenAIProvider(**kwargs)


_FACTORIES: dict[str, Callable[..., Provider]] = {
    "claude": _claude,
    "openai": _openai,
}


def register_provider(name: str, factory: Callable[..., Provider]) -> None:
    """Make a provider available by name (factory receives get_provider kwargs)."""
    _FACTORIES[name.lower()] = factory


def available_providers():
    return list(_FACTORIES)


def get_provider(name: str, **kwargs) -> Provider:
    """Return a provider instance, e.g. get_provider("openai", base_url=...)."""
    factory = _FACTORIES.get((name or "").lower())
    if factory is None:
        raise ValueError(
            f"Unknown provider '{name}'. Available: {', '.join(_FACTORIES)}"
        )
    return factory(**kwargs)
