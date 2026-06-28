"""
Local tools for the Evaluator Agent.
submit_evaluation enforces the verdict schema so the orchestrator always receives
consistent per-pair judgments it can map to a status deterministically.
"""

import json

# === ALLOWED VERDICT VALUES ===

_FAITHFULNESS = ["faithful", "unfaithful", "not_applicable"]  # not_applicable: no best_practice_refs to judge against
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
                        "Reason through the fix before deciding the verdicts, in two blocks separated by "
                        "a blank line. First block: objective quality — is the code correct, and does it "
                        "fully resolve the issue? Second block: faithfulness — briefly, is it faithful to "
                        "best_practice_refs; if it deviates, name where and a likely why; if there are no "
                        "refs, say there was no guideline to follow."
                    ),
                },
                "correctness": {                            # judged before faithfulness: schema order drives reasoning order
                    "type": "string",
                    "enum": _CORRECTNESS,
                    "description": "Is suggested_code valid Python that resolves the issue in rationale? Judge this first.",
                },
                "completeness": {
                    "type": "string",
                    "enum": _COMPLETENESS,
                    "description": "Does the fix resolve the whole issue, not just part of it?",
                },
                "faithfulness": {
                    "type": "string",
                    "enum": _FAITHFULNESS,
                    "description": (
                        "Judged ONLY against best_practice_refs: 'faithful' if the fix follows them, "
                        "'unfaithful' if a ref exists but the fix deviates from it, 'not_applicable' if "
                        "best_practice_refs is empty (no guideline to follow). A doc_url citation in "
                        "grounded_in is not itself a violation."
                    ),
                },
            },
            "required": ["reasoning", "correctness", "completeness", "faithfulness"],
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
        required = ["reasoning", "correctness", "completeness", "faithfulness"]
        missing = [f for f in required if f not in tool_input]
        if missing:
            return json.dumps({
                "status": "error",
                "message": f"submit_evaluation failed — missing required fields: {missing}",
            })

        if not isinstance(tool_input["reasoning"], str):
            return json.dumps({
                "status": "error",
                "message": f"submit_evaluation failed — 'reasoning' must be a string, got {type(tool_input['reasoning']).__name__}",
            })

        # Check if LLM used the correct scoring schema
        enums = {
            "correctness":  _CORRECTNESS,
            "completeness": _COMPLETENESS,
            "faithfulness": _FAITHFULNESS,
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
                "correctness":  tool_input["correctness"],
                "completeness": tool_input["completeness"],
                "faithfulness": tool_input["faithfulness"],
            },
        }, indent=2)

    except Exception as e:
        return json.dumps({
            "status": "error",
            "message": f"submit_evaluation failed unexpectedly: {str(e)}",
        })