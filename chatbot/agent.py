"""CLI entrypoint for interacting with the Boskalis GraphRAG ReAct agent."""
from __future__ import annotations

import asyncio

from dotenv import load_dotenv

from mcp import ClientSession
from mcp.client.stdio import stdio_client

from agent_runtime import AgentInitialization, CONFIG, initialize_agent
from agent_tools import get_neo4j_cypher_server_parameters

if load_dotenv():
    print("Loaded .env file")
else:
    print("No .env file found")


async def print_astream(async_stream, output_messages_key: str = "llm_input_messages") -> None:
    """Print the stream of messages from the agent."""

    async for chunk in async_stream:
        for node, update in chunk.items():
            print(f"Update from node: {node}")
            messages_key = output_messages_key if node == "pre_model_hook" else "messages"
            for message in update[messages_key]:
                if isinstance(message, tuple):
                    print(message)
                else:
                    message.pretty_print()

        print("\n\n")


async def _prepare_agent(session: ClientSession) -> AgentInitialization:
    """Build the agent and report any schema loading issues to the CLI."""

    init = await initialize_agent(session)

    if init.schema_error:
        print(
            "Unable to retrieve the Neo4j schema before starting the agent. "
            "The agent may request it during the conversation instead."
        )
        print(f"Schema retrieval error: {init.schema_error}")
    elif init.schema_text:
        print("Loaded Neo4j schema and added it to the system prompt.")
    else:
        print(
            "Received an empty schema response. The agent may request the "
            "schema during the conversation."
        )

    return init


async def main() -> None:
    """Main function to run the agent."""

    server_parameters = get_neo4j_cypher_server_parameters()
    async with stdio_client(server_parameters) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            init = await _prepare_agent(session)
            agent = init.agent

            print(
                "\n===================================== Chat =====================================\n"
            )

            while True:
                user_input = input("> ")
                if user_input.lower() in {"exit", "quit", "q"}:
                    break

                await print_astream(
                    agent.astream({"messages": user_input}, config=CONFIG, stream_mode="updates")
                )


if __name__ == "__main__":
    asyncio.run(main())
