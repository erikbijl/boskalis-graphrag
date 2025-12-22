"""Helpers for constructing structured UI content blocks from tool outputs."""
from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from typing import Any

__all__ = ["tabular_content"]


def _coerce_mapping(value: Any) -> dict[str, Any]:
    """Return a dictionary representation for the provided value."""

    if isinstance(value, Mapping):
        return dict(value)

    return {"value": value}


def _normalise_records(
    records: Sequence[Any] | Iterable[Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[Any], list[str | None]]:
    """Return both the original and UI-friendly representations of records."""

    original: list[dict[str, Any]] = []
    normalised: list[dict[str, Any]] = []
    header_keys: list[Any] = []
    flattened_sources: list[str | None] = []

    for entry in records:
        entry_dict = _coerce_mapping(entry)
        original.append(entry_dict)

        candidate = entry_dict
        flattened_key: str | None = None

        if len(entry_dict) == 1:
            key, value = next(iter(entry_dict.items()))
            if isinstance(value, Mapping):
                candidate = dict(value)
                flattened_key = str(key)

        normalised.append(candidate)
        flattened_sources.append(flattened_key)

        for key in candidate.keys():
            if key not in header_keys:
                header_keys.append(key)

    return original, normalised, header_keys, flattened_sources


def tabular_content(
    records: Sequence[Any] | Iterable[Any],
    *,
    title: str | None = None,
    summary: str | None = None,
    description: str | None = None,
    context: Mapping[str, Any] | None = None,
    empty_state: str | None = "No results were returned for this query.",
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Create a standardised table content block for UI consumption."""

    original, normalised, header_keys, flattened_sources = _normalise_records(records)

    headers = [str(key) for key in header_keys]
    rows = [
        [record.get(key) for key in header_keys]
        for record in normalised
    ]

    block_context: dict[str, Any] = {"records": normalised}
    if any(flattened_sources):
        block_context["flattened_from"] = flattened_sources
        block_context["original_records"] = original
    elif original != normalised:
        block_context["original_records"] = original

    if context:
        block_context.update(dict(context))

    content_block: dict[str, Any] = {
        "content_type": "table",
        "title": title,
        "headers": headers,
        "rows": rows,
        "context": block_context,
    }

    if summary:
        content_block["summary"] = summary
    if description:
        content_block["description"] = description
    if empty_state and not normalised:
        content_block["empty_state"] = empty_state

    return original, content_block
