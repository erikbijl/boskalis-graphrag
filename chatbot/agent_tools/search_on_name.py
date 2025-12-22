"""Structured tool for finding nodes in the database based on names. It uses a full text index."""
from __future__ import annotations

import os
from typing import Any

from langchain_core.tools import StructuredTool
from neo4j import GraphDatabase, RoutingControl
from pydantic import BaseModel, Field


class NameSearchInput(BaseModel):
    name: str = Field(
        ...,
        description="The name to look for in the database",
    )

    limit: int = Field(
        default=10,
        description="The maximum number of nodes to return. ",
        ge=1,
    )

def search_on_name(
    name: str, limit: int = 10
) -> list[dict[str, Any]]:
    """
    Search the database using a full text index on names 
    """

    query = """
        CALL db.index.fulltext.queryNodes("full_text_name", $name) YIELD node, score
        RETURN labels(node) as labels, node.name as name
        LIMIT $limit
    """

    driver = GraphDatabase.driver(
        os.getenv("NEO4J_URI"),
        auth=(os.getenv("NEO4J_USERNAME"), os.getenv("NEO4J_PASSWORD")),
    )

    results = driver.execute_query(
        query,
        parameters_={"name": name,  "limit": limit},
        database_=os.getenv("NEO4J_DATABASE"),
        routing_=RoutingControl.READ,
        result_transformer_=lambda r: r.data(),
    )
    return results


NAME_SEARCH_TOOL = StructuredTool.from_function(
    func=search_on_name,
    name="search_on_name",
    description="Search the database using a full text index on names",
    args_schema=NameSearchInput,
    return_direct=False,
)

__all__ = [
    "NAME_SEARCH_TOOL",
    "search_on_name",
    "NameSearchInput",
]
