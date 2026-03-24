# Code Review Agent

> A multi-agent pipeline that analyzes Python code, identifies issues, suggests improvements, and evaluates the quality of its own suggestions — built with MCP, RAG, and LLM-as-Judge evaluation.

---

## What This Is

An automated code review system where specialized agents work in sequence:

**Read code → Analyze issues → Review & classify → Generate fixes → Evaluate improvements**

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
│      Reviewer Agent       │  ← MCP tools: knowledge_search (RAG), classify_severity
│  Classify findings by     │
│  category + severity      │
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
| ChromaDB + `knowledge_search` | 🔲 In Progress | RAG knowledge base for best practices |
| Reviewer Agent | 🔲 In Progress | Classifies findings using RAG context |
| Optimizer Agent | 🔲 In Progress | Generates fixes grounded in best practices |
| Evaluator Agent | 🔲 In Progress | LLM-as-Judge scoring + final report |
| Sandbox Executor | 🔲 Planned | Isolated execution to verify generated fixes |

---

## Architecture

### MCP Server

A single MCP server (`code-review-mcp`) exposes all tools. Agents discover tools dynamically at runtime via STDIO — no hardcoded tool registrations on the agent side.

| Tool | Purpose | Implementation |
|---|---|---|
| `read_code` | Read code from file path or raw string | `os.path.isfile()` routing, JSON output |
| `detect_syntax_errors` | Static analysis: code quality + security | ruff (E,F,W,C90,B rules) + bandit via subprocess |
| `extract_code_structure` | Extract functions, classes, imports | `ast.parse()` + `ast.walk()` |

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

### RAG Knowledge Base

ChromaDB stores Python best-practice documents:
- PEP8 style guidelines
- OWASP Top 10 for Python
- Common performance anti-patterns

Used by the Reviewer and Optimizer agents via the `knowledge_search` MCP tool. Chunk size: ~300 tokens. The agent decides when a RAG lookup is needed — not every finding requires one.

### Evaluation

The Evaluator agent (LLM-as-Judge) scores the pipeline's output on five dimensions:

| Dimension | What it measures |
|---|---|
| Task Completion | Did every identified issue get a review? |
| Tool Selection | Did agents use the right tools at the right time? |
| Faithfulness | Are suggestions grounded in RAG context? |
| Efficiency | Number of LLM calls per review (target: ≤ 6) |
| Error Recovery | Graceful handling when RAG returns no results |

---

## Project Structure

```
multi-agent-code-review/
├── agent/
│   ├── __init__.py
│   └── analyzer_agent.py       # Analyzer agent with MCP + local tool routing
├── tools/
│   ├── __init__.py
│   └── analyzer_tools.py       # Local submit_analysis tool (schema + executor)
├── tests/
│   └── test_mcp_tools.py       # MCP tool tests
├── config.py                   # Global settings, model config, system prompts
├── mcp_server.py               # MCP server with all code analysis tools
├── requirements.txt
├── .env
└── .gitignore
```

---

## Getting Started

### Prerequisites

- Python 3.10+
- Anthropic API Key

### Installation

```bash
git clone https://github.com/yourusername/multi-agent-code-review.git
cd multi-agent-code-review

python -m venv .venv
source .venv/bin/activate       # macOS/Linux
.venv\Scripts\activate          # Windows

pip install -r requirements.txt
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
# Run with default test code (SQL injection example)
python agent/analyzer_agent.py

# Run with a specific file
python agent/analyzer_agent.py path/to/your_code.py
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
