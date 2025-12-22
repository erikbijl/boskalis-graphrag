"""FastAPI backend for interacting with the Boskalis GraphRAG agent."""
from __future__ import annotations

import asyncio
import json
import logging
from contextlib import asynccontextmanager
from typing import Any, AsyncGenerator

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from langchain_core.messages import AnyMessage
from mcp import ClientSession
from mcp.client.stdio import stdio_client
from pydantic import BaseModel, Field

from agent_runtime import CONFIG, AgentInitialization, extract_ai_response, initialize_agent
from agent_tools import get_neo4j_cypher_server_parameters
from agent_tools.context import RenderMode, use_render_mode
from .renderables import extract_renderables as _extract_renderables

logger = logging.getLogger(__name__)

if load_dotenv():
    logger.info("Loaded .env file")
else:
    logger.info("No .env file found")


class QuestionRequest(BaseModel):
    """Request payload for the /ask endpoint."""

    question: str = Field(..., description="User question for the agent.")
    response_mode: RenderMode = Field(
        default="component",
        alias="responseMode",
        description=(
            "Preferred rendering format for tool outputs. Use 'component' for structured API "
            "clients and 'html' for inline chat experiences."
        ),
    )

    model_config = {
        "populate_by_name": True,
        "json_schema_extra": {
            "example": {
                "question": "Which projects did Boskalis execute?",
                "responseMode": "component",
            }
        },
    }


class ToolExecution(BaseModel):
    """Information about a tool that was executed during agent processing."""
    
    name: str
    description: str
    input_args: dict[str, Any] = Field(default_factory=dict)
    success: bool = True
    error_message: str | None = None


class AgentAnswer(BaseModel):
    """Response payload returned by the /ask endpoint."""

    answer: str
    messages: list[dict[str, Any]]
    renderables: list[dict[str, Any]] = Field(
        default_factory=list,
        description=(
            "UI-friendly payloads extracted from tool responses, such as HTML snippets "
            "or tabular data."
        ),
    )
    tools_used: list[ToolExecution] = Field(
        default_factory=list,
        description="List of tools that were executed during the agent's reasoning process.",
    )
    reasoning_steps: list[str] = Field(
        default_factory=list,
        description="Key reasoning steps taken by the LLM to arrive at the answer.",
    )


def _encode_event(payload: dict[str, Any]) -> bytes:
    """Serialise a streaming event to a newline terminated JSON payload."""

    return f"{json.dumps(payload, default=str)}\n".encode()


def _message_text(message: AnyMessage) -> str | None:
    """Extract readable text from a LangChain message instance."""

    content = getattr(message, "content", None)
    if isinstance(content, str):
        text = content.strip()
        return text or None

    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                if item.strip():
                    parts.append(item.strip())
            elif isinstance(item, dict):
                if item.get("type") == "text" and isinstance(item.get("text"), str):
                    text_value = str(item["text"]).strip()
                    if text_value:
                        parts.append(text_value)
        if parts:
            return "\n".join(parts)
        return None

    if isinstance(content, dict):
        text_value = content.get("text")
        if isinstance(text_value, str):
            stripped = text_value.strip()
            return stripped or None

    if content is None:
        return None

    text = str(content).strip()
    return text or None


def _coerce_tool_args(raw_args: Any) -> dict[str, Any]:
    """Convert tool arguments into a JSON serialisable dictionary."""

    if isinstance(raw_args, dict):
        return raw_args

    if isinstance(raw_args, str):
        try:
            parsed = json.loads(raw_args)
        except json.JSONDecodeError:
            return {"input": raw_args}
        return parsed if isinstance(parsed, dict) else {"input": raw_args}

    if raw_args is None:
        return {}

    return {"input": raw_args}


def _summarise_tool_args(args: dict[str, Any]) -> str | None:
    """Create a short human readable summary of tool arguments."""

    if not args:
        return None

    summary_parts = [f"{key}={value}" for key, value in args.items()]
    summary = ", ".join(summary_parts)
    return summary if len(summary) <= 160 else f"{summary[:157]}..."


def _summarise_tool_content(content: Any) -> str | None:
    """Return a compact summary of tool output for streaming updates."""

    if content is None:
        return None

    if isinstance(content, str):
        text = content.strip()
        if not text:
            return None
        return f"{text[:197]}..." if len(text) > 200 else text

    try:
        serialised = json.dumps(content, default=str)
    except TypeError:
        serialised = str(content)

    serialised = serialised.strip()
    if not serialised:
        return None
    return f"{serialised[:197]}..." if len(serialised) > 200 else serialised


def _detect_tool_error(content: Any) -> tuple[bool, str | None]:
    """Determine whether a tool response represents an error."""

    summary = _summarise_tool_content(content)
    if not summary:
        return True, None

    lowered = summary.lower()
    if any(keyword in lowered for keyword in ("error", "exception", "failed", "traceback")):
        return False, summary

    return True, None


def _extract_tool_call_details(tool_call: dict[str, Any]) -> tuple[str, dict[str, Any], str | None]:
    """Normalise tool call payloads from the agent stream."""

    if "function" in tool_call and isinstance(tool_call["function"], dict):
        function_data = tool_call["function"]
        name = function_data.get("name", "unknown")
        args = function_data.get("arguments", {})
    else:
        name = tool_call.get("name", "unknown")
        args = tool_call.get("args", {})

    return str(name or "unknown"), _coerce_tool_args(args), tool_call.get("id")


class ConversationStreamTracker:
    """Track conversation state while streaming events to the client."""

    def __init__(self) -> None:
        self.raw_messages: list[AnyMessage] = []
        self.serialised_messages: list[dict[str, Any]] = []
        self.tool_executions: list[ToolExecution] = []
        self._tools_by_call_id: dict[str, ToolExecution] = {}
        self.reasoning_steps: list[str] = []

    def add_message(self, message: AnyMessage) -> dict[str, Any]:
        self.raw_messages.append(message)
        serialised = _serialise_message(message)
        self.serialised_messages.append(serialised)
        return serialised

    def register_tool_call(self, tool_call: dict[str, Any]) -> tuple[ToolExecution, str | None]:
        name, args, call_id = _extract_tool_call_details(tool_call)
        execution = ToolExecution(
            name=name,
            description=f"Called {name} tool",
            input_args=args,
            success=True,
        )
        self.tool_executions.append(execution)
        if call_id:
            self._tools_by_call_id[str(call_id)] = execution
        return execution, str(call_id) if call_id is not None else None

    def update_tool_result(
        self,
        call_id: str | None,
        *,
        name: str | None,
        content: Any,
    ) -> ToolExecution:
        key = str(call_id) if call_id is not None else None
        execution = self._tools_by_call_id.get(key) if key else None

        if execution is None and call_id is None and self.tool_executions:
            execution = self.tool_executions[-1]

        if execution is None:
            tool_name = name or "unknown"
            execution = ToolExecution(
                name=tool_name,
                description=f"Called {tool_name} tool",
                input_args={},
                success=True,
            )
            self.tool_executions.append(execution)
            if key:
                self._tools_by_call_id[key] = execution

        if name and execution.name == "unknown":
            execution.name = name
            execution.description = f"Called {name} tool"

        success, error_message = _detect_tool_error(content)
        execution.success = success
        execution.error_message = error_message
        return execution

    def add_reasoning(self, step: str) -> None:
        clean = step.strip()
        if clean and (not self.reasoning_steps or self.reasoning_steps[-1] != clean):
            self.reasoning_steps.append(clean)

    def final_answer(self) -> AgentAnswer:
        answer_text = extract_ai_response(self.raw_messages)
        if not answer_text:
            raise ValueError("Agent did not return a textual response.")

        steps = [step for step in self.reasoning_steps if step]
        if steps and steps[-1].strip() == answer_text.strip():
            steps = steps[:-1]

        renderables = _extract_renderables(self.serialised_messages)
        return AgentAnswer(
            answer=answer_text,
            messages=self.serialised_messages,
            renderables=renderables,
            tools_used=self.tool_executions,
            reasoning_steps=steps,
        )


def _serialise_content(content: Any) -> Any:
    """Convert message content into JSON serialisable structures."""

    if isinstance(content, (str, int, float, bool)) or content is None:
        return content

    if isinstance(content, bytes):
        return content.decode()

    if isinstance(content, list):
        serialised_items: list[Any] = []
        for item in content:
            if isinstance(item, dict):
                serialised_items.append(item)
            elif hasattr(item, "model_dump"):
                serialised_items.append(item.model_dump())
            else:
                serialised_items.append(str(item))
        return serialised_items

    if isinstance(content, dict):
        return content

    if hasattr(content, "model_dump"):
        return content.model_dump()

    return str(content)


def _serialise_message(message: AnyMessage) -> dict[str, Any]:
    """Convert a LangChain message into a JSON serialisable dictionary."""

    payload: dict[str, Any] = {
        "type": getattr(message, "type", None),
        "content": _serialise_content(message.content),
    }

    for attr in ("name", "id", "tool_call_id", "role"):
        value = getattr(message, attr, None)
        if value is not None:
            payload[attr] = value

    # Include tool_calls for AI messages
    if hasattr(message, "tool_calls") and message.tool_calls:
        payload["tool_calls"] = [
            {
                "name": getattr(tc, "name", "unknown"),
                "id": getattr(tc, "id", None),
                "args": getattr(tc, "args", {})
            } for tc in message.tool_calls
        ]

    if getattr(message, "additional_kwargs", None):
        payload["additional_kwargs"] = message.additional_kwargs
        
        # Also extract tool_calls from additional_kwargs to top level for easier access
        if "tool_calls" in message.additional_kwargs:
            payload["tool_calls"] = message.additional_kwargs["tool_calls"]

    if getattr(message, "response_metadata", None):
        payload["response_metadata"] = message.response_metadata

    if getattr(message, "usage_metadata", None):
        payload["usage_metadata"] = message.usage_metadata

    return payload


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage the lifecycle of the MCP session and agent."""

    server_parameters = get_neo4j_cypher_server_parameters()
    async with stdio_client(server_parameters) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            logger.info("Connected to MCP server.")

            try:
                init: AgentInitialization = await initialize_agent(session)
            except Exception:  # noqa: BLE001 - surface initialisation issues loudly
                logger.exception("Failed to initialise the agent")
                raise

            app.state.agent = init.agent
            app.state.agent_lock = asyncio.Lock()
            app.state.schema_text = init.schema_text
            app.state.schema_error = init.schema_error
            app.state.session = session

            if init.schema_error:
                logger.warning(
                    "Unable to retrieve the Neo4j schema during startup: %s", init.schema_error
                )
            elif init.schema_text:
                logger.info("Loaded Neo4j schema and added it to the system prompt.")
            else:
                logger.info("Neo4j schema request returned no data during startup.")

            yield

            # Cleanup happens automatically when the context managers exit.
            # No explicit teardown required.


app = FastAPI(
    title="Boskalis GraphRAG Agent API",
    description="FastAPI backend for interacting with the Boskalis agent.",
    version="0.1.0",
    lifespan=lifespan,
)


@app.get("/health")
async def health() -> dict[str, Any]:
    """Return readiness information for the backend."""

    schema_error = getattr(app.state, "schema_error", None)
    return {
        "status": "ok" if getattr(app.state, "agent", None) is not None else "initialising",
        "schema_loaded": schema_error is None,
        "schema_error": str(schema_error) if schema_error else None,
    }


@app.post("/ask")
async def ask_question(payload: QuestionRequest) -> StreamingResponse:
    """Stream the agent's reasoning steps and final answer back to the client."""

    question = payload.question.strip()
    if not question:
        raise HTTPException(status_code=422, detail="Question must not be empty.")

    agent = getattr(app.state, "agent", None)
    agent_lock: asyncio.Lock | None = getattr(app.state, "agent_lock", None)
    if agent is None or agent_lock is None:
        raise HTTPException(status_code=503, detail="Agent is not initialised yet.")

    tracker = ConversationStreamTracker()
    initial_message = "Thinking…"
    tracker.add_reasoning(initial_message)

    async def event_stream() -> AsyncGenerator[bytes, None]:
        yield _encode_event({"type": "thinking", "message": initial_message})

        try:
            with use_render_mode(payload.response_mode):
                async with agent_lock:
                    async for chunk in agent.astream(
                        {"messages": question},
                        config=CONFIG,
                        stream_mode="updates",
                    ):
                        for node, update in chunk.items():
                            if node == "pre_model_hook":
                                continue

                            messages = update.get("messages", [])
                            for raw in messages:
                                if isinstance(raw, tuple) or not hasattr(raw, "type"):
                                    continue

                                serialised = tracker.add_message(raw)
                                message_type = serialised.get("type")

                                if message_type == "ai":
                                    text = _message_text(raw)
                                    tool_calls_raw = serialised.get("tool_calls") or []
                                    tool_calls = (
                                        tool_calls_raw
                                        if isinstance(tool_calls_raw, list)
                                        else [tool_calls_raw]
                                    )

                                    if tool_calls:
                                        if text:
                                            tracker.add_reasoning(text)
                                            yield _encode_event(
                                                {"type": "reasoning", "message": text}
                                            )

                                        for tool_call in tool_calls:
                                            if not isinstance(tool_call, dict):
                                                continue

                                            execution, call_id = tracker.register_tool_call(tool_call)
                                            args_summary = _summarise_tool_args(execution.input_args)
                                            if args_summary:
                                                step_message = (
                                                    f"I am using tool {execution.name} to {args_summary}."
                                                )
                                            else:
                                                step_message = f"I am using tool {execution.name}."

                                            tracker.add_reasoning(step_message)
                                            yield _encode_event(
                                                {
                                                    "type": "tool_start",
                                                    "tool_name": execution.name,
                                                    "call_id": call_id,
                                                    "message": step_message,
                                                    "args": execution.input_args,
                                                }
                                            )
                                    elif text:
                                        yield _encode_event(
                                            {"type": "answer", "message": text}
                                        )

                                elif message_type == "tool":
                                    tool_call_id = serialised.get("tool_call_id")
                                    tool_name = str(serialised.get("name") or "unknown")
                                    content = serialised.get("content")

                                    execution = tracker.update_tool_result(
                                        tool_call_id,
                                        name=tool_name,
                                        content=content,
                                    )

                                    summary = _summarise_tool_content(content)
                                    if execution.success:
                                        if summary:
                                            step_message = (
                                                f"I have found {summary} from {execution.name}."
                                            )
                                        else:
                                            step_message = (
                                                f"I have received results from {execution.name}."
                                            )
                                    else:
                                        step_message = (
                                            f"{execution.name} reported an error: "
                                            f"{execution.error_message or summary or 'unknown error'}"
                                        )

                                    tracker.add_reasoning(step_message)
                                    yield _encode_event(
                                        {
                                            "type": "tool_end",
                                            "tool_name": execution.name,
                                            "call_id": str(tool_call_id)
                                            if tool_call_id is not None
                                            else None,
                                            "message": step_message,
                                            "success": execution.success,
                                            "error": execution.error_message,
                                        }
                                    )

            answer = tracker.final_answer()
            yield _encode_event({"type": "final", "answer": answer.model_dump()})
        except Exception as exc:  # noqa: BLE001 - stream errors back to the client
            logger.exception("Error while streaming agent response")
            yield _encode_event({"type": "error", "message": str(exc)})

    return StreamingResponse(event_stream(), media_type="application/x-ndjson")


__all__ = ["app"]
