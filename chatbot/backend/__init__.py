"""Backend package exposing the FastAPI application."""
from __future__ import annotations

from typing import Any

__all__ = ["app"]


def __getattr__(name: str) -> Any:  # pragma: no cover - executed indirectly
    if name == "app":
        from .main import app as fastapi_app

        return fastapi_app
    raise AttributeError(f"module 'backend' has no attribute {name!r}")
