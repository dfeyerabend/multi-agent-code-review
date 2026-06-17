"""
Local tools for the Enricher Agent.
submit_enrichment enforces the output schema so the Optimizer always receives consistent findings.
"""

import json

# === SCHEMA FRAGMENTS ===

_best_practice_ref_schema = {           # one RAG chunk cited as evidence
    "type": "object",
    "properties": {
        "source":  {"type": "string"},  # "pyguide" or "company"
        "section": {"type": "string"},  # e.g. "3.10.4"
        "text":    {"type": "string"},  # the chunk text used as context
    },
}

_enriched_finding_schema = {                        # one fully enriched finding
    "type": "object",
    "properties": {
        "rule":     {"type": "string"},             # original ruff/bandit rule code, e.g. "B301"
        "line":     {"type": "integer"},            # line number from the Analyzer
        "category": {"type": "string"},             # "Style" | "Logic" | "Maintainability" | "Security"
        "severity": {"type": "string"},             # may override the linter's original severity
        "rationale": {"type": "string"},            # explanation grounded in RAG context
        "best_practice_refs": {
            "type": "array",
            "items": _best_practice_ref_schema,     # empty list when RAG had no good match
        },
        "doc_url": {"type": ["string", "null"]},    # passed through from Analyzer output -> ruff's url field / bandit's more_info field -> can be used when no good match is found in database
        "cwe_id":  {"type": ["integer", "null"]},   # bandit findings only, null otherwise -> bandit's issue_cwe.id field
    },
}

# === TOOL DEFINITION ===

enricher_local_tools = [
    {
        "name": "submit_enrichment",
        "description": (
            "Submit the final reviewed findings after classifying every issue "
            "from the Analyzer's output. Call knowledge_search for each finding "
            "before submitting. You MUST call this tool as your final step. "
            "Do NOT respond with plain text — always submit through this tool."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "findings": {
                    "type": "array",
                    "description": (
                        "One entry per finding from the Analyzer. "
                        "Empty list if the Analyzer reported no issues."
                    ),
                    "items": _enriched_finding_schema,
                },
                "summary": {
                    "type": "string",
                    "description": "1-2 sentence overview of the review findings.",
                },
                "rag_used": {
                    "type": "boolean",
                    "description": "True if knowledge_search was called at least once.",  # Evaluator checks this for Tool Selection score
                },
            },
            "required": ["findings", "summary", "rag_used"],
        },
    }
]

# === TOOL EXECUTION ===

def run_enricher_tool(name: str, tool_input: dict) -> str:
    """
    Executes local reviewer tools by name.

    Args:
        name:       Name of the tool to execute.
        tool_input: Dict of arguments passed by the agents.

    Returns:
        JSON string with status and validated review data, or an error message.
    """
    if name == "submit_enrichment":
        try:
            required_fields = ["findings", "summary", "rag_used"]
            missing = [f for f in required_fields if f not in tool_input]   # checks for all required fields in tool_input
            if missing:
                return json.dumps({
                    "status": "error",
                    "message": f"Missing required fields: {missing}",
                })

            if not isinstance(tool_input["findings"], list):                # guard against the agents passing a dict instead of a list
                return json.dumps({
                    "status": "error",
                    "message": "'findings' must be a list.",
                })

            if not isinstance(tool_input["summary"], str):
                return json.dumps({
                    "status": "error",
                    "message": "'summary' must be a string.",
                })

            if not isinstance(tool_input["rag_used"], bool):
                return json.dumps({
                    "status": "error",
                    "message": "'rag_used' must be a boolean.",
                })

            # Verifies that the output contains the correct formats
            finding_errors = []
            for i, finding in enumerate(tool_input["findings"]):
                if not isinstance(finding, dict):
                    finding_errors.append(f"findings[{i}]: must be an object, got {type(finding).__name__}")
                    continue
                if not isinstance(finding.get("rule"), str):
                    finding_errors.append(f"findings[{i}]: 'rule' must be a string")
                if not isinstance(finding.get("line"), int):
                    finding_errors.append(f"findings[{i}]: 'line' must be an integer")
                if not isinstance(finding.get("category"), str):
                    finding_errors.append(f"findings[{i}]: 'category' must be a string")
                if not isinstance(finding.get("severity"), str):
                    finding_errors.append(f"findings[{i}]: 'severity' must be a string")
                if not isinstance(finding.get("rationale"), str):
                    finding_errors.append(f"findings[{i}]: 'rationale' must be a string")
                if not isinstance(finding.get("best_practice_refs"), list):
                    finding_errors.append(f"findings[{i}]: 'best_practice_refs' must be a list")
                else:
                    for j, ref in enumerate(finding["best_practice_refs"]):
                        if not isinstance(ref, dict):
                            finding_errors.append(
                                f"findings[{i}].best_practice_refs[{j}]: must be an object, got {type(ref).__name__}")
                            continue
                        if not isinstance(ref.get("source"), str):
                            finding_errors.append(f"findings[{i}].best_practice_refs[{j}]: 'source' must be a string")
                        if not isinstance(ref.get("section"), str):
                            finding_errors.append(f"findings[{i}].best_practice_refs[{j}]: 'section' must be a string")
                        if not isinstance(ref.get("text"), str):
                            finding_errors.append(f"findings[{i}].best_practice_refs[{j}]: 'text' must be a string")
                doc_url = finding.get("doc_url")
                if doc_url is not None and not isinstance(doc_url, str):
                    finding_errors.append(f"findings[{i}]: 'doc_url' must be a string or null")
                cwe_id = finding.get("cwe_id")
                if cwe_id is not None and not isinstance(cwe_id, int):
                    finding_errors.append(f"findings[{i}]: 'cwe_id' must be an integer or null")

            if finding_errors:
                return json.dumps({
                    "status": "error",
                    "message": "Finding entries failed validation. Correct and resubmit.",
                    "errors": finding_errors,
                })

            return json.dumps({
                "status": "success",
                "enrichment_results": tool_input,
                "metadata": {
                    "total_reviewed_findings": len(tool_input["findings"]),
                    "rag_used": tool_input["rag_used"],
                },
            }, indent=2)

        except Exception as e:
            return json.dumps({
                "status": "error",
                "message": f"submit_enrichment failed: {str(e)}",
            })

    else:
        return json.dumps({"error": f"Unknown reviewer tool: {name}"})
