"""
Local tools for the Analyzer Agent.
These are NOT on the MCP server — they run locally in the agents process.
submit_analysis only collects the Analyzer's own judgment (the summary).
All deterministic data (findings, structure) is assembled in Python by
run_analyzer directly from MCP tool outputs — never retyped by the LLM.
"""

import json

# === TOOL SCHEMAS ===
analyzer_local_tools = [
    {
        "name": "submit_analysis",
        "description": (
            "Submit your factual summary after calling all three MCP tools "
            "(read_code, detect_syntax_errors, extract_code_structure). "
            "You MUST call this tool as your final step. "
            "Do NOT respond with plain text — always submit through this tool."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "summary": {
                    "type": "string",
                    "description": "1-2 sentence factual overview of what the analysis found",
                },
            },
            "required": ["summary"],
        },
    }
]

# === HELPER FUNCTIONS ===

def _deduplicate_findings(findings: list) -> list:
    """
    Collapses findings with the same rule code into a single entry.

    Pipeline: called by run_analyzer (agents/analyzer_agent.py) when assembling
    analysis_results from the detect_syntax_errors MCP output, before the
    Enricher receives the findings.

    Args:
        findings: List of finding dicts from ruff or bandit.

    Returns:
        Deduplicated list where each rule code appears once. 'line' keeps the
        first occurrence so downstream agents (Enricher, Optimizer) always
        have a usable line number; 'lines' and 'occurrences' are added
        alongside it for the full picture.
    """
    seen = {}
    for finding in findings:
        rule = finding.get("rule", "unknown")
        if rule not in seen:
            seen[rule] = {**finding, "lines": [finding.get("line")], "occurrences": 1}
        else:
            seen[rule]["lines"].append(finding.get("line"))
            seen[rule]["occurrences"] += 1
    return list(seen.values())

# === TOOL EXECUTION ===
def run_analyzer_tool(name: str, tool_input: dict) -> str:
    """
    Executes local analyzer tools.

    Pipeline: called by run_analyzer (agents/analyzer_agent.py) inside the
    agentic loop whenever the model invokes a tool name in LOCAL_TOOL_NAMES.

    Args:
        name:       Name of the local tool to execute.
        tool_input: Dict of arguments passed by the model.

    Returns:
        JSON string with the validated summary on success, or a structured
        error message naming the failing field.
    """
    if name == "submit_analysis":
        try:
            if "summary" not in tool_input:
                return json.dumps({
                    "status": "error",
                    "message": "Missing required field: 'summary'.",
                })

            if not isinstance(tool_input["summary"], str):
                return json.dumps({
                    "status": "error",
                    "message": f"'summary' must be a string, got {type(tool_input['summary']).__name__}.",
                })

            return json.dumps({
                "status": "success",
                "summary": tool_input["summary"],
            })

        except Exception as e:
            return json.dumps({
                "status": "error",
                "message": f"submit_analysis failed unexpectedly: {str(e)}",
            })

    else:
        return json.dumps({"error": f"Unknown analyzer tool: {name}"})


