"""Context helpers for configuring agent tool rendering behaviour."""
from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar, Token
from typing import Generator, Literal

RenderMode = Literal["component", "html"]

# The default mode favours component payloads for API consumers.
_render_mode: ContextVar[RenderMode] = ContextVar("render_mode", default="component")


def get_render_mode() -> RenderMode:
    """Return the current render mode preference for tool outputs."""

    return _render_mode.get()


def set_render_mode(mode: RenderMode) -> Token[RenderMode]:
    """Set the render mode preference and return a token for resetting it."""

    return _render_mode.set(mode)


def reset_render_mode(token: Token[RenderMode]) -> None:
    """Reset the render mode preference back to a previous state."""

    _render_mode.reset(token)


@contextmanager
def use_render_mode(mode: RenderMode) -> Generator[None, None, None]:
    """Context manager that temporarily overrides the render mode preference."""

    token = set_render_mode(mode)
    try:
        yield
    finally:
        reset_render_mode(token)


__all__ = ["RenderMode", "get_render_mode", "reset_render_mode", "set_render_mode", "use_render_mode"]
