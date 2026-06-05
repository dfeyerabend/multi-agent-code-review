"""
Shared configuration for the Code Review Agent pipeline.
All agents use these settings. System prompts are defined here
so the agent files stay focused on logic.
"""

import anthropic
from dotenv import load_dotenv

load_dotenv()

client = anthropic.Anthropic()


# === CLAUDE SETTINGS ===
MODEL = "claude-sonnet-4-20250514"
MAX_TOKENS = 4096       # For structured tool output
MAX_ITERATIONS = 10      # Max iterations for every agent

# === PATHS ===
import os
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

MCP_SERVER_PATH = os.path.join(PROJECT_ROOT, "mcp_server.py")   # path to the MCP server script

CHROMA_DB_PATH = os.path.join(PROJECT_ROOT, "knowledge_base", "chroma_db") # path to the persistent ChromaDB storage folder


# === AGENT SPECIFIC SYSTEM PROMPTS ===

ANALYZER_PROMPT = (
    "You are the Analyzer Agent in a code review pipeline. "
    "Your job is to examine Python code using your tools and produce "
    "a structured analysis for the next agent in the chain (the Enricher)."
    "\n\n"

    "## Your Tools\n"
    "You have three tools available via MCP:\n"
    "- `read_code`: Reads a code file from a path or accepts raw code. Always call this first.\n"
    "- `detect_syntax_errors`: Runs ruff (code quality) and bandit (security) on the code.\n"
    "- `extract_code_structure`: Extracts functions, classes, and imports via AST parsing.\n"
    "\n"

    "## Workflow\n"
    "For every input, follow these steps in order:\n"
    "1. Call `read_code` with the user's input to obtain the code string.\n"
    "2. Take the `code` field from the read_code result.\n"
    "3. Call `detect_syntax_errors` with that code string.\n"
    "4. Call `extract_code_structure` with that code string.\n"
    "5. Combine all results into your final JSON output.\n"
    "\n"

    "## Rules\n"
    "- Only analyze Python code. If the input is clearly not Python, "
    "call `submit_analysis` with empty findings and a summary explaining why.\n"
    "- Report ONLY what the tools find. Do NOT invent or hallucinate issues.\n"
    "- If a tool returns no findings, pass an empty list `[]` — never fabricate problems.\n"
    "- Your summary must be factual, not evaluative. You analyze — the Enricher adds context.\n"
)

ENRICHER_PROMPT = (
    "You are the Enricher Agent in a code review pipeline. "
    "You receive structured output from the Analyzer Agent and enrich each finding "
    "with best-practice context from a knowledge base."
    "\n\n"

    "## Your Tools\n"
    "- `knowledge_search`: Searches the ChromaDB knowledge base. Call this once per finding.\n"
    "- `submit_enrichment`: Local tool. Call this once as your final step.\n"
    "\n"

    "## Workflow\n"
    "For each finding in the Analyzer's output:\n"
    "1. Call `knowledge_search` with query='{rule} {message}' and category='{category}'.\n"
    "2. Check the `distance` field of every returned chunk.\n"
    "   - If at least one chunk has distance ≤ 1.0: use those chunks as `best_practice_refs` "
    "and write a `rationale` grounded in their content.\n"
    "   - If all chunks have distance > 1.0: set `best_practice_refs` to [] and note in "
    "`rationale` that no matching best practice was found — reference `doc_url` instead.\n"
    "3. You may upgrade `severity` if the RAG context reveals the issue is more critical "
    "than the linter reported. Keep the original severity otherwise.\n"
    "After processing all findings, call `submit_enrichment` with the enriched list.\n"
    "\n"

    "## Rules\n"
    "- Do NOT invent findings that the Analyzer did not report.\n"
    "- If the Analyzer reported zero findings, call `submit_enrichment` immediately with findings=[].\n"
    "- Always set `rag_used=True` if you called `knowledge_search` at least once, "
    "False otherwise.\n"
    "- Your rationale must be grounded in RAG context or doc_url — never hallucinated.\n"
)

# === AGENT SPECIFIC TOOL LIST ===

ANALYZER_TOOLS = {"read_code", "detect_syntax_errors", "extract_code_structure"}
ENRICHER_TOOLS = {"knowledge_search"}
OPTIMIZER_TOOLS = {"knowledge_search", "generate_fix_suggestion"}

# === LOGGING SETUP ===
import logging

# Read LOG_LEVEL from environment, default to INFO if not set.
# Usage:
#   LOG_LEVEL=DEBUG python agent/analyzer_agent.py   ← full detail
#   python agent/analyzer_agent.py                   ← INFO only (default)
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()

def setup_logging() -> None:
    """
    Configure root logger once for the entire process.
    Used in __main__ of every entrypoint script.
    All module-level loggers inherit this config automatically.
    """
    logging.basicConfig(
        level=LOG_LEVEL,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )
