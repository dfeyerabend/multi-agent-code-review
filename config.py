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


# === AGENT SPECIFIC SYSTEM PROMPTS ===

ANALYZER_PROMPT = (
    "You are the Analyzer Agent in a code review pipeline. "
    "Your job is to examine Python code using your tools and produce "
    "a structured analysis for the next agent in the chain (the Reviewer)."
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
    "- Your summary must be factual, not evaluative. You analyze — the Reviewer judges.\n"
    "- Do NOT suggest fixes. That is the Optimizer's job.\n"
)



