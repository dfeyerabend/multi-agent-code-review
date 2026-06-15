"""
Local tools for the Evaluator Agent.
submit_evaluation enforces the verdict schema so the orchestrator always receives
consistent per-pair judgments it can map to a status deterministically.
"""

import json

# === ALLOWED VERDICT VALUES ===

_FAITHFULNESS = ["faithful", "partial", "unfaithful"]
_CORRECTNESS  = ["pass", "fail"]
_COMPLETENESS = ["complete", "partial", "incomplete"]

# === TOOL DEFINITION ===

evaluator_local_tools = [
    {
        "name": "submit_evaluation",
        "description": (
            "Submit your verdicts for the fix you were given. "
            "This is how you deliver your result — call it exactly once, "
            "and do not reply with plain text instead."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "reasoning": {                              # first: model argues before it judges
                    "type": "string",
                    "description": (
                        "Reason through the fix here first: how it relates to the code and the "
                        "referenced best practice. Write this before deciding the three verdicts."
                    ),
                },
                "faithfulness": {
                    "type": "string",
                    "enum": _FAITHFULNESS,
                    "description": "Does the fix follow best_practice_refs and cite them honestly in grounded_in?",
                },
                "correctness": {
                    "type": "string",
                    "enum": _CORRECTNESS,
                    "description": "Is suggested_code valid Python that resolves the issue in rationale?",
                },
                "completeness": {
                    "type": "string",
                    "enum": _COMPLETENESS,
                    "description": "Does the fix resolve the whole issue, not just part of it?",
                },
            },
            "required": ["reasoning", "faithfulness", "correctness", "completeness"],
        },
    }
]

# === TOOL EXECUTION ===

def run_evaluator_tool(name: str, tool_input: dict) -> str:
    """
    Executes local evaluator tools by name.

    Args:
        name:       Name of the tool to execute.
        tool_input: Dict of arguments passed by the agents.

    Returns:
        JSON string with status and the validated evaluation verdicts, or an error message.
    """
    if name != "submit_evaluation": # Guard against wrong tool use - there is only one valid tool
        return json.dumps({"status": "error", "message": f"Unknown evaluator tool: {name}"})

    # Catch incorrect calls from LLM early
    try:
        required = ["reasoning", "faithfulness", "correctness", "completeness"]
        missing = [f for f in required if f not in tool_input]
        if missing:
            return json.dumps({
                "status": "error",
                "message": f"submit_evaluation failed — missing required fields: {missing}",
            })

        # Check if LLM used the correct scoring schema
        enums = {
            "faithfulness": _FAITHFULNESS,
            "correctness":  _CORRECTNESS,
            "completeness": _COMPLETENESS,
        }
        for field, allowed in enums.items():               # defensive re-check beyond the API schema
            value = tool_input[field]
            if value not in allowed:
                return json.dumps({
                    "status": "error",
                    "message": f"submit_evaluation failed — {field!r} must be one of {allowed}, got {value!r}",
                })

        return json.dumps({
            "status": "success",
            "evaluation": {
                "reasoning":    tool_input["reasoning"],
                "faithfulness": tool_input["faithfulness"],
                "correctness":  tool_input["correctness"],
                "completeness": tool_input["completeness"],
            },
        }, indent=2)

    except Exception as e:
        return json.dumps({
            "status": "error",
            "message": f"submit_evaluation failed unexpectedly: {str(e)}",
        })