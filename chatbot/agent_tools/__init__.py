"""Utilities for loading and configuring tools available to the agent."""
from __future__ import annotations

from typing import Iterable

from langchain_core.tools import StructuredTool
from langchain_mcp_adapters.tools import load_mcp_tools
from mcp import ClientSession

from .search_on_name import (
    NAME_SEARCH_TOOL,
    NameSearchInput,
    search_on_name,
)

from .search_in_document import (
    DOCUMENT_SEARCH_TOOL,
    search_in_documents,
    DocumentSearchInput,
)

from .neo4j_cypher_server import get_neo4j_cypher_server_parameters


_ALLOWED_MCP_TOOL_NAMES = frozenset({"get_neo4j_schema", "read_neo4j_cypher"})


def _filter_allowed_mcp_tools(tools: Iterable[StructuredTool]) -> list[StructuredTool]:
    return [tool for tool in tools if tool.name in _ALLOWED_MCP_TOOL_NAMES]


async def load_agent_tools(session: ClientSession) -> list[StructuredTool]:
    """Load all tools available to the agent."""

    mcp_tools = await load_mcp_tools(session)
    allowed_tools = _filter_allowed_mcp_tools(mcp_tools)
    allowed_tools.append(NAME_SEARCH_TOOL)
    allowed_tools.append(DOCUMENT_SEARCH_TOOL)

    return allowed_tools


__all__ = [
    "get_neo4j_cypher_server_parameters",
    "load_agent_tools",
    "NAME_SEARCH_TOOL",
    "search_on_name",
    "NameSearchInput",
    "DOCUMENT_SEARCH_TOOL",
    "search_in_documents",
    "DocumentSearchInput",
]
