"""
Local tools for the Optimizer Agent.
submit_optimization enforces the output schema so the Evaluator always receives consistent fixes.
"""

import logging
import json

logger = logging.getLogger(__name__)

# === SCHEMA FRAGMENTS ===

_fix_schema = {
    "type": "object",
    "properties": {
        "index":          {"type": "integer"},  # batch-local position of the finding this fix addresses
        "suggested_code": {"type": "string"},   # the corrected code snippet
        "explanation":    {"type": "string"},   # why this fix resolves the issue
        "grounded_in": {
            "type": "array",
            "items": {"type": "string"},        # e.g. ["pyguide §3.10", "company_rules §1.3"]
        },
    },
}

# === TOOL DEFINITION ===

optimizer_local_tools = [
    {
        "name": "submit_optimization",
        "description": (
            "Submit the final fix suggestions after generating a fix for every finding "
            "in the batch. Reference each finding by its 'index' field only — do NOT "
            "repeat finding_rule or finding_line, those are attached automatically. "
            "You MUST call this tool as your final step. "
            "Do NOT respond with plain text — always submit through this tool."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "fixes": {
                    "type": "array",
                    "description": (
                        "One fix entry per finding in the batch, referenced by 'index'. "
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

    Pipeline: called by agents/optimizer_agent.py inside _run_optimizer_batch
    whenever the model invokes a tool during the optimizer's agentic loop.

    Args:
        name:       Name of the tool to execute.
        tool_input: Dict of arguments passed by the agent.

    Returns:
        JSON string with status and validated optimization data, or an error message.
    """
    try:
        if name != "submit_optimization":
            logger.warning(f"run_optimizer_tool received unknown tool name: {name}")
            return json.dumps({"status": "error", "message": f"Unknown optimizer tool: {name}"})

        if not isinstance(tool_input, dict):
            logger.error(f"submit_optimization received non-dict tool_input: {type(tool_input).__name__}")
            return json.dumps({
                "status": "error",
                "message": f"tool_input must be a dict, got {type(tool_input).__name__}",
            })

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
                "message": f"'fixes' must be a list, got {type(tool_input['fixes']).__name__}",
            })

        if not isinstance(tool_input["summary"], str):
            return json.dumps({
                "status": "error",
                "message": f"'summary' must be a string, got {type(tool_input['summary']).__name__}",
            })

        # Validate each fix entry against the index-keyed contract: the model now owns only
        # its own judgement (suggested_code/explanation/grounded_in), never finding identity.
        fix_errors = []
        for i, fix in enumerate(tool_input["fixes"]):
            if not isinstance(fix, dict):
                fix_errors.append(f"fixes[{i}]: must be an object, got {type(fix).__name__}")
                continue
            if not isinstance(fix.get("index"), int) or isinstance(fix.get("index"), bool):
                fix_errors.append(f"fixes[{i}]: 'index' must be an integer")
            if "suggested_code" in fix and fix["suggested_code"] is not None and not isinstance(
                    fix["suggested_code"], str):
                fix_errors.append(f"fixes[{i}]: 'suggested_code' must be a string or null")
            if not isinstance(fix.get("explanation"), str):
                fix_errors.append(f"fixes[{i}]: 'explanation' must be a string")
            if not isinstance(fix.get("grounded_in"), list):
                fix_errors.append(f"fixes[{i}]: 'grounded_in' must be a list")
            elif not all(isinstance(g, str) for g in fix["grounded_in"]):
                fix_errors.append(f"fixes[{i}]: 'grounded_in' must be a list of strings")

        if fix_errors:
            logger.warning(f"submit_optimization validation failed: {fix_errors}")
            return json.dumps({
                "status": "error",
                "message": "Fix entries failed validation. Correct and resubmit.",
                "errors": fix_errors,
            })

        return json.dumps({
            "status": "success",
            "fixes": tool_input["fixes"],
            "summary": tool_input["summary"],
            "metadata": {
                "total_fixes": len(tool_input["fixes"]),
            },
        }, indent=2)

    except Exception as e:
        logger.error(f"run_optimizer_tool failed unexpectedly for tool '{name}': {e}")
        return json.dumps({
            "status": "error",
            "message": f"run_optimizer_tool failed: {str(e)}",
        })