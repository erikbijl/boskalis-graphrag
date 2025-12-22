"""Utility helpers for extracting UI renderables from agent responses."""
from __future__ import annotations

import json
from typing import Any

__all__ = ["extract_renderables", "extract_renderables_from_content"]


def extract_renderables_from_content(content: Any) -> list[dict[str, Any]]:
    """Recursively collect renderable payloads embedded in tool outputs."""

    renderables: list[dict[str, Any]] = []

    # Handle JSON strings (common in tool message content)
    if isinstance(content, str):
        try:
            parsed = json.loads(content)
            renderables.extend(extract_renderables_from_content(parsed))
        except (json.JSONDecodeError, TypeError):
            # Not JSON, ignore
            pass
        return renderables

    if isinstance(content, dict):
        if isinstance(content.get("render_hint"), dict):
            renderables.append(content["render_hint"])

        blocks = content.get("content_blocks")
        if isinstance(blocks, list):
            for block in blocks:
                if isinstance(block, dict):
                    renderables.append(block)

        block = content.get("content_block")
        if isinstance(block, dict):
            renderables.append(block)

        if isinstance(content.get("content_type"), str):
            renderables.append(content)

        for key, value in content.items():
            if key in {"render_hint", "content_blocks", "content_block"}:
                continue
            renderables.extend(extract_renderables_from_content(value))

    elif isinstance(content, list):
        for item in content:
            renderables.extend(extract_renderables_from_content(item))

    return renderables


def extract_renderables(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return a flattened list of renderable payloads from serialised messages."""

    discovered: list[dict[str, Any]] = []
    for message in messages:
        content = message.get("content")
        if content is None:
            continue
        discovered.extend(extract_renderables_from_content(content))

    return discovered
