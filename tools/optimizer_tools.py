"""
Local tools for the Optimizer Agent.
submit_optimization enforces the output schema so the Evaluator always receives consistent fixes.
"""

import json

# === SCHEMA FRAGMENTS ===

_fix_schema = {
    "type": "object",
    "properties": {
        "finding_rule":    {"type": "string"},   # e.g. "B608" — links the fix back to its finding
        "finding_line":    {"type": "integer"},  # line number from the original finding
        "suggested_code":  {"type": "string"},   # the corrected code snippet
        "explanation":     {"type": "string"},   # why this fix resolves the issue
        "grounded_in":     {
            "type": "array",
            "items": {"type": "string"},         # e.g. ["pyguide §3.10", "company_rules §1.3"]
        },
    },
}

# === TOOL DEFINITION ===

optimizer_local_tools = [
    {
        "name": "submit_optimization",
        "description": (
            "Submit the final fix suggestions after generating a fix for every finding "
            "in the batch. You MUST call this tool as your final step. "
            "Do NOT respond with plain text — always submit through this tool."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "fixes": {
                    "type": "array",
                    "description": (
                        "One fix entry per finding in the batch. "
                        "Empty list if no actionable fix could be generated."
                    ),
                    "items": _fix_schema,
                },
                "summary": {
                    "type": "string",
                    "description": "1-2 sentence overview of the fixes generated in this batch.",
                },
            },
            "required": ["fixes", "summary"],
        },
    }
]

# === TOOL EXECUTION ===

def run_optimizer_tool(name: str, tool_input: dict) -> str:
    """
    Executes local optimizer tools by name.

    Args:
        name:       Name of the tool to execute.
        tool_input: Dict of arguments passed by the agent.

    Returns:
        JSON string with status and validated optimization data, or an error message.
    """
    if name == "submit_optimization":
        try:
            required_fields = ["fixes", "summary"]
            missing = [f for f in required_fields if f not in tool_input]
            if missing:
                return json.dumps({
                    "status": "error",
                    "message": f"Missing required fields: {missing}",
                })

            if not isinstance(tool_input["fixes"], list):
                return json.dumps({
                    "status": "error",
                    "message": "'fixes' must be a list.",
                })

            return json.dumps({
                "status": "success",
                "optimization_results": tool_input,
                "metadata": {
                    "total_fixes": len(tool_input["fixes"]),
                },
            }, indent=2)

        except Exception as e:
            return json.dumps({
                "status": "error",
                "message": f"submit_optimization failed: {str(e)}",
            })

    else:
        return json.dumps({"error": f"Unknown optimizer tool: {name}"})