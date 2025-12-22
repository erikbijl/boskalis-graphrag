"""Shared utilities for building and running the Boskalis GraphRAG agent."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Sequence

from langchain_core.messages import AnyMessage
from langchain_core.messages.utils import count_tokens_approximately, trim_messages
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.prebuilt import create_react_agent
from langgraph.prebuilt.chat_agent_executor import AgentState
from mcp import ClientSession

from agent_tools import load_agent_tools
from agent_tools.neo4j_cypher_server import fetch_neo4j_schema

CONFIG: dict[str, dict[str, str]] = {"configurable": {"thread_id": "1"}}

SYSTEM_PROMPT = """
You are a Neo4j expert that knows how to write Cypher queries to address questions on Boskalis projects and equipment.
Your job is to trace the graph with projects, equipment, documents and more to find the relevant information for user questions. 

You have the ability to different tools while answering the questions. Please prefer using specific tools and read/write Cypher when needed.
If using a tool that does NOT require writing a Cypher query, you do not need the database schema.

As a Cypher expert, when writing queries:
* You must always ensure you have the data model schema to inform your queries
* If an error is returned from the database, you may refactor your query or ask the user to provide additional information
* If an empty result is returned, use your best judgement to determine if the query is correct.

As a well respected expert:
* Ensure that you provide detailed responses with citations to the underlying data
* Please give you final answer in markdown styling to improve readability in the chatbot. 
"""


@dataclass(slots=True)
class AgentInitialization:
    """Container for resources produced while building the agent."""

    agent: Any
    schema_text: str | None
    schema_error: Exception | None


def build_system_prompt(schema: str | None) -> str:
    """Return the system prompt augmented with the retrieved database schema."""

    if schema:
        return f"{SYSTEM_PROMPT}\n\nDatabase schema:\n{schema}"
    return SYSTEM_PROMPT


def pre_model_hook(state: AgentState) -> dict[str, list[AnyMessage]]:
    """Prepare the message history prior to each LLM invocation."""

    trimmed_messages = trim_messages(
        state["messages"],
        strategy="last",
        token_counter=count_tokens_approximately,
        max_tokens=30_000,
        start_on="human",
        end_on=("human", "tool"),
        include_system=True,
    )
    return {"llm_input_messages": trimmed_messages}


async def initialize_agent(session: ClientSession) -> AgentInitialization:
    """Initialise the ReAct agent using the provided MCP session."""

    allowed_tools = await load_agent_tools(session)

    schema_text: str | None = None
    schema_error: Exception | None = None
    try:
        schema_text = await fetch_neo4j_schema(session)
    except Exception as exc:  # noqa: BLE001 - propagate failure information to the caller
        schema_error = exc

    agent = create_react_agent(
        "openai:gpt-4.1",
        allowed_tools,
        pre_model_hook=pre_model_hook,
        checkpointer=InMemorySaver(),
        prompt=build_system_prompt(schema_text),
    )

    return AgentInitialization(agent=agent, schema_text=schema_text, schema_error=schema_error)


def _extract_text_from_content(content: Any) -> str | None:
    """Attempt to convert message content into human readable text."""

    if isinstance(content, str):
        return content

    if isinstance(content, Iterable) and not isinstance(content, (bytes, bytearray)):
        text_chunks: list[str] = []
        for item in content:
            if isinstance(item, str):
                text_chunks.append(item)
            elif isinstance(item, dict):
                # LangChain structured messages encode text using {"type": "text", "text": "..."}
                if item.get("type") == "text" and "text" in item:
                    text_chunks.append(str(item["text"]))
            elif hasattr(item, "model_dump"):
                dumped = item.model_dump()
                text_value = dumped.get("text") if isinstance(dumped, dict) else None
                if text_value:
                    text_chunks.append(str(text_value))
        if text_chunks:
            return "\n".join(text_chunks)
        return None

    if hasattr(content, "model_dump"):
        dumped = content.model_dump()
        if isinstance(dumped, dict) and "text" in dumped:
            return str(dumped["text"])

    return str(content) if content is not None else None


def extract_ai_response(messages: Sequence[AnyMessage]) -> str | None:
    """Return the last AI message's textual content from a conversation trace."""

    for message in reversed(messages):
        if getattr(message, "type", None) == "ai":
            return _extract_text_from_content(message.content)
    return None


__all__ = [
    "AgentInitialization",
    "CONFIG",
    "SYSTEM_PROMPT",
    "build_system_prompt",
    "extract_ai_response",
    "initialize_agent",
    "pre_model_hook",
]
