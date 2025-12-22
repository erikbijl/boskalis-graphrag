"""Helpers for configuring the Neo4j Cypher MCP server."""
from __future__ import annotations

import json
import os
from typing import Any

from mcp import ClientSession, StdioServerParameters
from mcp.types import CallToolResult


def get_neo4j_cypher_server_parameters() -> StdioServerParameters:
    """Return the configured server parameters for the Neo4j Cypher MCP server."""

    return StdioServerParameters(
        command="uvx",
        args=["mcp-neo4j-cypher@0.5.2", "--transport", "stdio"],
        env={
            "NEO4J_URI": os.getenv("NEO4J_URI"),
            "NEO4J_USERNAME": os.getenv("NEO4J_USERNAME"),
            "NEO4J_PASSWORD": os.getenv("NEO4J_PASSWORD"),
            "NEO4J_DATABASE": os.getenv("NEO4J_DATABASE"),
        },
    )


async def fetch_neo4j_schema(session: ClientSession) -> str | None:
    """Fetch the current Neo4j schema from the MCP server."""

    result = await session.call_tool("get_neo4j_schema")
    if result.isError:
        raise RuntimeError("get_neo4j_schema returned an error result.")

    return _schema_text_from_result(result)


def _schema_text_from_result(result: CallToolResult) -> str | None:
    """Extract a textual schema description from a tool result."""

    text_parts = []
    for block in result.content:
        text = _clean_text(block)
        if text:
            text_parts.append(text)

    if text_parts:
        return "\n\n".join(text_parts)

    structured = result.structuredContent
    if isinstance(structured, dict):
        schema_candidate = structured.get("schema")
        candidate_text = _stringify_structured(schema_candidate)
        if candidate_text:
            return candidate_text
        return _stringify_structured(structured)

    return _stringify_structured(structured)


def _clean_text(block: Any) -> str | None:
    """Return normalized text content from a result block."""

    block_type = getattr(block, "type", None)
    block_text = getattr(block, "text", None)
    if block_type == "text" and isinstance(block_text, str):
        stripped = block_text.strip()
        return stripped or None
    return None


def _stringify_structured(value: Any) -> str | None:
    """Render a structured schema payload into a human-readable string."""

    if value is None:
        return None
    if isinstance(value, str):
        stripped = value.strip()
        return stripped or None
    if isinstance(value, dict):
        try:
            return json.dumps(value, indent=2, sort_keys=True)
        except TypeError:
            return str(value)
    if isinstance(value, (list, tuple)):
        try:
            return json.dumps(value, indent=2)
        except TypeError:
            return str(value)
    return str(value)


__all__ = ["get_neo4j_cypher_server_parameters", "fetch_neo4j_schema"]
