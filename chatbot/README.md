# Supply Chain Graph Agent Demo

This repository showcases how to build a ReAct-style AI agent that reasons over a Neo4j supply chain graph and augments its answers with specialised analytical tools. The project combines the [LangGraph](https://langchain-ai.github.io/langgraph/) agent framework, the [Neo4j Cypher MCP server](https://github.com/neo4j-contrib/mcp-neo4j/tree/main/servers/mcp-neo4j-cypher) for graph access, and a suite of local Python utilities tailored to pharmaceutical supply chain analysis.

It is derived from the [neo4j pharma supply chain demo](https://github.com/neo4j-product-examples/demo-supply_chain) combined with [Quickly build a react agent with LangGraph and MCP](https://medium.com/neo4j/quickly-build-a-react-agent-with-langgraph-and-mcp-828757e3bd69) [GitRepo](https://github.com/neo4j-field/text2cypher-react-agent-example)



## Key capabilities

- **Supply-chain aware reasoning** – GPT-4.1 is guided by a system prompt that teaches it the pharmaceutical product flow model so it can draft precise Cypher queries.
- **Enhanced search capabilities** – Case-insensitive full-text indexing with Lucene enables fuzzy matching, partial searches, and typo tolerance for drug names and product descriptions.

The full text search has been applied to only one tool for demonstrativive purposes - the trace_supply_chain tool. Using full text index enables use of lucene analyzers, which have the benefit of increased performance, with more robust results returns as the input pattern of the parameters from the user question are case insensitive and can take advantage of lucene fuzzy matching in the query. 

- **Hybrid tool stack** – The agent can call the Neo4j MCP tools for arbitrary graph queries and a catalogue of local Python tools for targeted analytics such as supplier concentration, logistics anomalies, or dependency risk.
- **Actionable responses** – Tool outputs are converted into natural language answers with contextual guidance and citations so teams can immediately act on the findings.
- **Enhanced transparency** – The chat interface includes a collapsible "Tools & Reasoning" section that shows which tools were executed and the LLM's reasoning steps, providing full visibility into the agent's decision-making process.
- **Ready for automation** – The code includes an interactive CLI and a FastAPI backend, making it easy to embed the agent into larger workflows or user interfaces.

## Supply chain knowledge graph

The sample Neo4j database models the pharmaceutical manufacturing lifecycle.

- `Suppliers` provide raw materials (`RM`) that feed into drug product (`DP`) and finished good (`FG`) nodes via `SUPPLIES_RM` and `PRODUCT_FLOW` relationships.
- `Distributor` nodes capture downstream distribution partners through `DISTRIBUTED_BY` links.
- Product descriptions, geographies, form factors, and strength metadata allow the agent to reason about batch genealogy, supplier concentration, and logistics paths.

To explore the exact schema the agent retrieves, run the `get_neo4j_schema` MCP tool or consult the generated schema snippet printed when the CLI starts.

## Tooling overview

The agent can mix and match Neo4j MCP tools with local analytics utilities. A detailed breakdown of each tool, its inputs, and example prompts is provided in [`docs/USAGE_GUIDE.md`](docs/USAGE_GUIDE.md).

### Repository layout

```
├── agent.py                  # CLI entry point for the LangGraph agent
├── setup_index.py            # Script to set up full-text search indexes
├── agent_runtime/            # Agent configuration, prompt, and orchestration helpers
├── agent_tools/              # Local analytical tools exposed to the agent
│   ├── setup_fulltext_index.py  # Full-text index management utilities
│   └── trace_supply_chain.py    # Enhanced with full-text search capabilities
├── backend/                  # FastAPI service that wraps the agent for HTTP use
├── tests/                    # Unit tests for tools and runtime helpers
├── README.md                 # High-level project documentation (this file)
├── docs/USAGE_GUIDE.md       # Extended documentation covering tools and workflows
└── docs/FULLTEXT_INDEX_GUIDE.md  # Full-text search setup and usage guide
```

## Quick start

### Prerequisites

- Python 3.10 or later
- Access to a Neo4j instance loaded with the sample pharmaceutical supply chain dataset
- An OpenAI API key with access to the `gpt-4.1` model
- Either the [`uv` package manager](https://docs.astral.sh/uv/getting-started/installation/) (recommended) or `pip`

### Clone and install

#### Option 1: using uv (recommended)

```bash
# Clone the repository
git clone https://github.com/neo4j-field/text2cypher-react-agent-chat-pharma-graph-demo.git
cd text2cypher-react-agent-chat-pharma-graph-demo

# Install dependencies (includes dev tools such as pytest and ruff)
uv sync --dev
```

#### Option 2: using pip

```bash
# Clone the repository
git clone https://github.com/neo4j-field/text2cypher-react-agent-chat-pharma-graph-demo.git
cd text2cypher-react-agent-chat-pharma-graph-demo

# Create a virtual environment
python3 -m venv .venv
source .venv/bin/activate  # On Windows use: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Configuration

Copy the example environment file and update it with your credentials.

```bash
cp .env.example .env
```

Set the following variables inside `.env`:

```dotenv
OPENAI_API_KEY=your_openai_api_key
NEO4J_URI=neo4j+s://your-instance.databases.neo4j.io
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=your_password
NEO4J_DATABASE=neo4j
```

### Running the CLI agent

```bash
# Using uv
uv run python -m agent

# Or using a standard Python environment
python -m agent
```

Type your question at the `>` prompt. Enter `exit`, `quit`, or `q` to leave the chat. The agent will stream tool decisions and intermediate reasoning as it works.

### Running the FastAPI service

```bash
# Using uv
uv run uvicorn backend.main:app --reload

# Or using pip
uvicorn backend.main:app --reload
```

The service exposes:

- `GET /health` – Reports readiness and whether the Neo4j schema was preloaded.
- `POST /ask` – Accepts `{ "question": "..." }` and returns the agent's answer plus the reasoning trace.

### Running the chat frontend

The repository now includes a single-page app for an interactive chat experience.

```bash
cd frontend
npm install
npm run dev
```

By default the dev server proxies API requests to `http://localhost:8000`. Point it at a different backend by exporting `VITE_AGENT_API` before starting Vite, for example `VITE_AGENT_API=https://demo.example.com`. Responses surfaced through the `/ask` endpoint now include a `renderables` array so HTML artefacts (like the interactive Leaflet map) and tabular tool outputs render inline in the chat history.

## Example supply chain workflows

- **Trace a critical product's journey** – “Trace the supply path for Diliprostzolast Tablet 50mg and highlight each supplier and distributor involved.” The agent can combine `trace_supply_path` and `distributors_for_product` to produce a step-by-step lineage.
- **Spot single-supplier risks** – “List raw materials used in Nabitegrpultide with only one approved supplier.” The `find_single_supplier_risks` tool surfaces concentrations and links back to the graph records for validation.
- **Investigate logistics inefficiencies** – “We are seeing high shipping costs for Calciiarottecarin. Are there cyclic or cross-border loops in its distribution route?” The `logistics_optimization` tool inspects product flow paths and flags loops that create cost leakage.
- **Quantify dependency on shared APIs** – “Which APIs appear in five or more downstream drug products?” The `api_dependency_risk` tool highlights APIs that represent systemic exposure.
- **Visualise distribution footprints** – After retrieving distributors or manufacturing sites, call `plot_cities_on_map` to produce an interactive HTML map for quick geographic sense-checking.

More prompts and suggested tool sequences are documented in [`docs/USAGE_GUIDE.md`](docs/USAGE_GUIDE.md).

## 🔍 Enhanced Search Capabilities

This agent now includes advanced full-text search capabilities for pharmaceutical product descriptions:

### Case-Insensitive Search
- **Analyzer**: Uses `standard-no-stop-words` for true case-insensitive matching
- **Wildcard Support**: Supports Lucene wildcard syntax (`*`, `?`)
- **Fuzzy Matching**: Supports approximate matches with `~` operator

### Search Examples
```bash
# Case-insensitive wildcard searches
dilip*          # Finds "Diliprostzolast" products
ASPIRIN*        # Finds aspirin-based medications  
tablet          # Finds all tablet formulations
10mg            # Finds all 10mg dosage products

# Exact phrase matching
"Diliprostzolast Tablet"

# Fuzzy matching for typos
asprin~         # Finds "aspirin" despite typo
```

### Performance Comparison
- **Traditional LIKE queries**: `~500ms` for partial matching across 50k+ nodes
- **Full-text index queries**: `~5-15ms` for complex searches with scoring

## Testing and quality checks

```bash
# Run the automated test suite
pytest

# Test case-insensitive search functionality
python test_case_insensitive_search.py

# Format and lint the codebase
make format
```

The test suite exercises the tool layer and runtime helpers, ensuring that Cypher queries are shaped correctly and helper utilities behave as expected.

The `test_case_insensitive_search.py` script specifically validates that the full-text search works consistently across different case patterns (e.g., `dilip*`, `Dilip*`, `DILIP*` all return identical results).

## Troubleshooting

- **Neo4j connection issues** – Confirm the credentials in `.env`, ensure your Aura or self-hosted instance is reachable, and verify that the supply chain dataset has been loaded.
- **Schema retrieval warnings** – If the CLI prints a schema loading warning, the agent will request the schema lazily during conversation. Check that `uvx` is on your `PATH` so the MCP server can be launched.
- **OpenAI errors** – Validate the API key, confirm model access, and monitor usage limits for throttling.
- **Missing tool dependencies** – Some tools (such as `plot_cities_on_map`) require optional packages like `geopy`. Install them with `uv sync --all-extras --dev` or `pip install geopy` if prompted.

## License

This project is provided for educational and demonstration purposes.
