"""Structured tool for searching in documents. It uses a vector index."""
from __future__ import annotations

import os

from langchain_core.tools import StructuredTool
from neo4j import GraphDatabase, RoutingControl
from pydantic import BaseModel, Field
from langchain_openai import OpenAIEmbeddings
from json import loads, dumps


class DocumentSearchInput(BaseModel):
    message: str = Field(
        ...,
        description="The search message to look for in the database",
    )

    limit: int = Field(
        default=10,
        description="The maximum number of nodes to return. ",
        ge=1,
    )

def search_in_documents(
    message: str, limit: int = 10
) -> list[dict[str]]:
    """
    Search the database on using a vector index on document chunks.
    """

    driver = GraphDatabase.driver(
        os.getenv("NEO4J_URI"),
        auth=(os.getenv("NEO4J_USERNAME"), os.getenv("NEO4J_PASSWORD")),
    )

    embedding_model = OpenAIEmbeddings(
        model=os.getenv("EMBEDDINGS_MODEL"),
        openai_api_key=os.getenv("OPENAI_API_KEY")
    )

    message_vector = embedding_model.embed_query(message)
    similarity_query = """ 
        CALL db.index.vector.queryNodes("chunk-embeddings", $nn, $message_vector) YIELD node, score
        WITH node as chunk, score ORDER BY score DESC
        MATCH (d:Document)<-[:PART_OF]-(chunk)
        RETURN score, d.name as doc_name, d.url as doc_url, chunk.id as chunk_id, chunk.page as page, chunk.text as text
    """
    results_df = driver.execute_query(
        similarity_query,
        database_=os.getenv("NEO4J_DATABASE"),
        routing_=RoutingControl.READ,
        message_vector=message_vector,
        nn = limit, 
        result_transformer_= lambda r: r.to_df()
    )
    results = dumps(loads(results_df.to_json(orient="records")), indent=2)

    return results


DOCUMENT_SEARCH_TOOL = StructuredTool.from_function(
    func=search_in_documents,
    name="search_documents",
    description="Search the database on using a vector index on document chunks.",
    args_schema=DocumentSearchInput,
    return_direct=False,
)


__all__ = [
    "DOCUMENT_SEARCH_TOOL",
    "search_in_documents",
    "DocumentSearchInput",
]
