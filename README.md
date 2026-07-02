# Code Review Agent

> A multi-agent pipeline that analyzes Python code, identifies issues, suggests improvements, and evaluates the quality of its own suggestions — built with MCP, RAG, and LLM-as-Judge evaluation.

---

## What This Is

An automated code review system where specialized agents work in sequence:

**Read code → Analyze issues → Enrich with context → Generate fixes → Evaluate improvements**

Each agent has a single responsibility and communicates structured data to the next. All code analysis tools are exposed through a single MCP server that agents discover dynamically at runtime.

```
Code Input (file path or raw string)
        │
        ▼
┌───────────────────────────┐
│      Analyzer Agent       │  ← MCP tools: read_code, detect_syntax_errors,
│  Static analysis + AST    │    extract_code_structure
│  parsing                  │  ← Local tool: submit_analysis (structured output)
└────────────┬──────────────┘
             │ structured JSON
             ▼
┌───────────────────────────┐
│      Enricher Agent       │  ← MCP tools: knowledge_search (RAG)
│  Enrich findings with     │  ← Local tool: submit_enrichment (structured output)
│  best-practice context    │
└────────────┬──────────────┘
             │ structured JSON
             ▼
┌───────────────────────────┐
│      Optimizer Agent      │  ← MCP tools: generate_fix, knowledge_search (RAG)
│  Generate concrete fix    │
│  suggestions per finding  │
└────────────┬──────────────┘
             │ structured JSON
             ▼
┌───────────────────────────┐
│      Evaluator Agent      │  ← LLM-as-Judge
│  Score improvements on    │
│  multiple dimensions      │
│  Output: human-readable   │
│  report                   │
└───────────────────────────┘
```

---

## Component Status

| Component | Status | Description |
|---|---|---|
| MCP Server (`mcp_server.py`) | ✅ Done | All code analysis tools, STDIO transport |
| Analyzer Agent | ✅ Done | Connects to MCP, runs analysis, structured output via local tool |
| ChromaDB + `knowledge_search` | ✅ Done | RAG knowledge base populated and ready |
| Enricher Agent | ✅ Done | Enriches findings with RAG context; batched to prevent context overflow |
| Orchestrator | ✅ Done | Linear pipeline driver; owns pipeline state; builds per-agent input contracts |
| Optimizer Agent | ✅ Done | Category-aware fix routing; Security/Logic/Maintainability processed individually, Style grouped by rule code |
| Evaluator Agent | ✅ Done | LLM-as-Judge: judges each fix scoped to its anchor lines; derives one of 6 statuses (APPROVED / INCORRECT / INCOMPLETE / NONCOMPLIANT / NO_FIX / NOT_EVALUATED) in Python; produces markdown report |
| Sandbox Executor | 🔲 Planned | Isolated execution to verify generated fixes (post-publish) |

---

## Architecture

### MCP Server

A single MCP server (`code-review-mcp`) exposes all tools. Agents discover tools dynamically at runtime via STDIO — no hardcoded tool registrations on the agent side.

| Tool                        | Purpose | Implementation |
|-----------------------------|---|---|
| `read_code`                 | Read code from file path or raw string | `os.path.isfile()` routing, JSON output |
| `detect_syntax_errors`      | Static analysis: code quality + security | ruff (E,F,W,C90,B rules) + bandit via subprocess |
| `extract_code_structure`    | Extract functions, classes, imports | `ast.parse()` + `ast.walk()` |
| `knowledge_search`          | RAG search for best-practice context | ChromaDB cosine similarity, metadata category filter |
| `generate_fix_suggestion`   | Extract enclosing function source for a finding line | AST walk, falls back to surrounding lines on syntax error or module-level code |

### Agent Design

Each agent runs an agentic tool-call loop:

1. Receive input from the previous agent (or user)
2. Call tools iteratively until all data is collected
3. Submit structured output via a local tool with a strict schema
4. Output is passed as input to the next agent

Agents use two types of tools simultaneously:
- **MCP tools** — executed remotely on the MCP server via `session.call_tool()`
- **Local tools** — executed in-process for structured output validation

The agent doesn't know which tools are remote and which are local. Routing happens transparently in the orchestrator.   
Tool whitelisting — each agent only sees the tools it needs. Whitelists are defined in config.py and applied at runtime after list_tools(). This prevents agents from calling tools outside their responsibility.


### RAG Knowledge Base

ChromaDB stores two types of Python best-practice documents in a single collection (`code_best_practices`):

**Google Python Style Guide (`pyguide`)** — covers naming conventions, imports, type annotations, exceptions, classes, and more. Goes beyond pure style into language patterns not caught by linters.
Source: [google/styleguide](https://github.com/google/styleguide), license CC-BY 3.0.

**example_company Code Standards (`company`)** — a set of fictional internal rules designed to demonstrate that the Enricher Agent retrieves knowledge from the database rather than relying on pre-trained knowledge. Rules include required function naming prefixes for database operations, mandatory `# REASON:` comments, a custom exception hierarchy, and restricted config access patterns.

Setup (run once after cloning):
```bash
python knowledge_base/create_database.py
```

Use `knowledge_base/inspect_database.py` to verify the database contents after setup.

The agent decides when a RAG lookup is needed — not every finding requires one. 
Chunks are filtered by category metadata (Style, Logic, Maintainability, Security) to keep retrieval focused.

### Evaluation

The Evaluator Agent judges each (finding, fix) pair on three criteria:

| Criterion | Verdicts |
|---|---|
| Faithfulness | Does `suggested_code` follow `best_practice_refs`? Does `grounded_in` cite them honestly? |
| Correctness | Is `suggested_code` valid Python that resolves the issue? |
| Completeness | Does the fix address the whole finding? |

Status is derived deterministically in Python from the three verdicts — not by the LLM — via a priority cascade (code problems outrank guideline problems):

| Status | Meaning |
|---|---|
| `APPROVED` | Correct, complete, and guideline-faithful (or no guideline applies) |
| `INCORRECT` | Invalid Python or does not resolve the issue |
| `INCOMPLETE` | Valid fix but only partially resolves the issue |
| `NONCOMPLIANT` | Correct and complete, but violates a retrieved company/style guideline |
| `NO_FIX` | Optimizer produced no fix for this finding |
| `NOT_EVALUATED` | Evaluator hit max iterations or returned malformed output |

The orchestrator's `render_report` layer (`render_report.py`) formats the results into a markdown report — a status-count overview table plus one block per fix (findings, original vs suggested code, verdicts, reasoning). The same functions back both the console output and a future Gradio UI.
---

## Project Structure

```
multi-agent-code-review/
├── agents/
│ ├── __init__.py
│ ├── agent_utils.py                # Shared utilities (MCP tool format conversion, chunking)
│ ├── analyzer_agent.py             # Analyzer agent
│ ├── enricher_agent.py             # Enricher agent (batched)
│ ├── optimizer_agent.py            # Optimizer agent (category-aware routing)
│ └── evaluator_agent.py            # Evaluator agent
├── knowledge_base/
│ ├── create_database.py            # Run once to populate ChromaDB from documents
│ ├── inspect_database.py           # Dev utility to inspect database contents
│ └── documents/
│ ├── pyguide.md                    # Google Python Style Guide (CC-BY 3.0)
│ └── company_rules.md              # example_company internal coding standards
├── tools/
│ ├── __init__.py
│ ├── analyzer_tools.py             # Local submit_analysis tool
│ ├── enricher_tools.py             # Local submit_enrichment tool
│ ├── optimizer_tools.py            # Local submit_optimization tool
│ └── evaluator_tools.py
├── tests/
│ ├── conftest.py
│ ├── test_mcp_tools.py             # MCP tools + knowledge_search tests
│ ├── test_analyzer_tools.py        # submit_analysis tests incl. deduplication
│ ├── test_enricher_tools.py        # submit_enrichment schema validation tests
│ ├── test_rag_retrieval.py
│ ├── test_optimizer_tools.py       # submit_optimization schema validation tests
│ └── test_evaluator_tools.py       # submit_evaluation schema validation tests
├── config.py                       # Global settings, model config, system prompts
├── mcp_server.py                   # MCP server with all code analysis tools
├── orchestrator.py                 # Pipeline driver + owns all user-facing output
├── render_report.py                # Rendering layer: pipeline results → Markdown (console + Gradio)
├── reports/                        # Generated review reports (gitignored)
├── requirements.txt
├── .env
└── .gitignore
```

---

## Testing

Each component has a dedicated test file covering its deterministic logic.
LLM agent behavior is not unit tested — it is non-deterministic and observed via logging instead.

| Test File | What it covers |
|---|---|
| `tests/test_mcp_tools.py` | All MCP tool functions + severity/category helper functions |
| `tests/test_analyzer_tools.py` | `submit_analysis` local tool — schema validation and error handling |
| `tests/test_rag_retrieval.py` | ChromaDB retrieval — one targeted test per company rule, proving RAG context is active |
| `tests/test_enricher_tools.py` | submit_enrichment local tool — schema validation, empty findings, type guards |
| `tests/test_optimizer_tools.py` | `submit_optimization` local tool — schema validation, empty fixes, type guards |
| `tests/test_evaluator_tools.py` | `submit_evaluation` local tool — schema validation, verdict enums, guard tests |

```bash
pytest                   # full suite
pytest tests/test_rag_retrieval.py   # RAG tests only
pytest -k "read_code"    # filter by name
```
---

## Observability
All agents and the MCP server use Python's `logging` module.
Log verbosity is controlled by a single environment variable:

| Level | Output |
|---|---|
| `INFO` (default) | Agent steps, tool calls, finding counts |
| `DEBUG` | Full tool inputs and outputs for tracing the agent loop |

```bash
LOG_LEVEL=DEBUG python agents/analyzer_agent.py   # full trace
python agents/analyzer_agent.py                   # clean output
LOG_LEVEL=DEBUG python agents/enricher_agent.py   # full trace
python agents/enricher_agent.py                   # clean output
```

---

## Getting Started

### Prerequisites

- Python 3.10+
- Anthropic API Key

### Installation

```bash
git clone https://github.com/yourusername/multi-agent-code-review.git
cd multi-agents-code-review

python -m venv .venv
source .venv/bin/activate       # macOS/Linux
.venv\Scripts\activate          # Windows

pip install -r requirements.txt

# Populate the RAG knowledge base (downloads embedding model on first run, ~90 MB)
python knowledge_base/create_database.py
```

### Environment Setup

```bash
cp .env.example .env
```

Add your API key to `.env`:
```
ANTHROPIC_API_KEY=sk-ant-...
```

### Usage

```bash
# Run the full pipeline (recommended)
python orchestrator.py

# Run with a specific file
python orchestrator.py path/to/your_code.py

# Run individual agents standalone
python agents/analyzer_agent.py
python agents/enricher_agent.py
```
---

## Tech Stack

| Technology | Purpose |
|---|---|
| **Anthropic API** | Agent reasoning (Claude Sonnet 4) |
| **MCP (Model Context Protocol)** | Tool discovery and execution via STDIO |
| **ruff** | Static code analysis (code quality) |
| **bandit** | Static code analysis (security) |
| **ChromaDB** | RAG vector database for best practices |
| **Python `ast`** | Code structure extraction |

---

## Author

**Dennis Feyerabend**
March 2026

---

## License

MIT
