"""
Local tools for the Enricher Agent.
submit_enrichment collects ONLY the Enricher's own judgment per finding
(rationale, best_practice_refs, optional severity override), keyed by index.
All pass-through fields (rule, line, lines, occurrences, category, message,
doc_url, cwe_id) are carried forward in Python by agents/enricher_agent.py
from the original findings — never retyped by the LLM.
"""

import json

import logging
logger = logging.getLogger(__name__)

# === SCHEMA FRAGMENTS ===

_best_practice_ref_schema = {           # one RAG chunk cited as evidence
    "type": "object",
    "properties": {
        "source":  {"type": "string"},  # "pyguide" or "company"
        "section": {"type": "string"},  # e.g. "3.10.4"
        "text":    {"type": "string"},  # the chunk text used as context
    },
}

_enrichment_entry_schema = {                            # the model's judgment for ONE finding, referenced by index
    "type": "object",
    "properties": {
        "index": {"type": "integer"},                   # position of the finding within findings_batch
        "rationale": {"type": "string"},                # explanation grounded in RAG context or doc_url
        "best_practice_refs": {
            "type": "array",
            "items": _best_practice_ref_schema,          # empty list when RAG had no good match
        },
        "severity": {"type": "string"},                  # OPTIONAL override; omit to keep the original severity
    },
}

# === TOOL DEFINITION ===

enricher_local_tools = [
    {
        "name": "submit_enrichment",
        "description": (
            "Submit your enrichment judgment for every finding in the batch. "
            "Call knowledge_search for each finding before submitting. Reference "
            "each finding by its 'index' field — do NOT repeat rule, line, "
            "category, message, doc_url, or cwe_id; those are carried forward "
            "automatically. You MUST call this tool as your final step. "
            "Do NOT respond with plain text — always submit through this tool."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "findings": {
                    "type": "array",
                    "description": (
                        "One enrichment entry per finding in findings_batch, "
                        "referenced by index. Empty list if the batch was empty."
                    ),
                    "items": _enrichment_entry_schema,
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
    Executes local enricher tools by name.

    Pipeline: called by the agentic loop in agents/enricher_agent.py whenever
    the model invokes a tool name in LOCAL_TOOL_NAMES. The validated enrichment
    list returned here is later merged with the original findings_batch by
    _merge_enrichment in agents/enricher_agent.py.

    Args:
        name:       Name of the tool to execute.
        tool_input: Dict of arguments passed by the model.

    Returns:
        JSON string with status and the validated enrichment list, or a
        structured error message pinpointing the failing entry/field.
    """
    if not isinstance(name, str):                                    # fail loud on a malformed call site, not a malformed model output
        return json.dumps({
            "status": "error",
            "message": f"run_enricher_tool: 'name' must be a string, got {type(name).__name__}.",
        })
    if not isinstance(tool_input, dict):
        return json.dumps({
            "status": "error",
            "message": f"run_enricher_tool: 'tool_input' must be a dict, got {type(tool_input).__name__}.",
        })

    if name != "submit_enrichment":
        logger.warning("run_enricher_tool: unknown tool requested: %s", name)
        return json.dumps({"status": "error", "message": f"Unknown enricher tool: {name}"})

    try:
        required_fields = ["findings", "summary", "rag_used"]
        missing = [f for f in required_fields if f not in tool_input]
        if missing:
            return json.dumps({
                "status": "error",
                "message": f"submit_enrichment: missing required fields: {missing}",
            })

        if not isinstance(tool_input["findings"], list):
            return json.dumps({
                "status": "error",
                "message": f"submit_enrichment: 'findings' must be a list, got {type(tool_input['findings']).__name__}.",
            })
        if not isinstance(tool_input["summary"], str):
            return json.dumps({
                "status": "error",
                "message": f"submit_enrichment: 'summary' must be a string, got {type(tool_input['summary']).__name__}.",
            })
        if not isinstance(tool_input["rag_used"], bool):
            return json.dumps({
                "status": "error",
                "message": f"submit_enrichment: 'rag_used' must be a boolean, got {type(tool_input['rag_used']).__name__}.",
            })

        # Validate each enrichment entry independently so one malformed item
        # produces a precise, targeted correction instead of a blanket failure.
        entry_errors = []
        for i, entry in enumerate(tool_input["findings"]):
            if not isinstance(entry, dict):
                entry_errors.append(f"findings[{i}]: must be an object, got {type(entry).__name__}")
                continue

            if not isinstance(entry.get("index"), int):
                entry_errors.append(f"findings[{i}]: 'index' must be an integer")
            if not isinstance(entry.get("rationale"), str):
                entry_errors.append(f"findings[{i}]: 'rationale' must be a string")

            refs = entry.get("best_practice_refs")
            if not isinstance(refs, list):
                entry_errors.append(f"findings[{i}]: 'best_practice_refs' must be a list")
            else:
                for j, ref in enumerate(refs):
                    if not isinstance(ref, dict):
                        entry_errors.append(
                            f"findings[{i}].best_practice_refs[{j}]: must be an object, got {type(ref).__name__}")
                        continue
                    if not isinstance(ref.get("source"), str):
                        entry_errors.append(f"findings[{i}].best_practice_refs[{j}]: 'source' must be a string")
                    if not isinstance(ref.get("section"), str):
                        entry_errors.append(f"findings[{i}].best_practice_refs[{j}]: 'section' must be a string")
                    if not isinstance(ref.get("text"), str):
                        entry_errors.append(f"findings[{i}].best_practice_refs[{j}]: 'text' must be a string")

            if "severity" in entry and not isinstance(entry["severity"], str):  # severity is optional, but if present it must be valid
                entry_errors.append(f"findings[{i}]: 'severity' must be a string when present")

        if entry_errors:
            return json.dumps({
                "status": "error",
                "message": "Enrichment entries failed validation. Correct and resubmit.",
                "errors": entry_errors,
            })

        return json.dumps({
            "status": "success",
            "enrichments": tool_input["findings"],
            "summary": tool_input["summary"],
            "rag_used": tool_input["rag_used"],
            "metadata": {
                "total_enriched": len(tool_input["findings"]),
            },
        }, indent=2)

    except Exception as e:
        logger.error("run_enricher_tool: submit_enrichment failed unexpectedly: %s", str(e))
        return json.dumps({
            "status": "error",
            "message": f"submit_enrichment failed unexpectedly: {str(e)}",
        })
