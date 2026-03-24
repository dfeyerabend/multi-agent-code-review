"""
Local tools for the Analyzer Agent.
These are NOT on the MCP server — they run locally in the agent process.
The submit_analysis tool enforces a strict output schema so the
Reviewer agent always gets consistent, structured input.
"""

import json

# === REUSABLE SCHEMA FRAGMENTS ===
# Extracted to keep the main schema readable.
# Each fragment defines the shape of one field in the analysis output.

_syntax_finding_schema = {  # one ruff finding
    "type": "object",
    "properties": {
        "rule": {"type": "string"},  # e.g. "F401", "E302"
        "message": {"type": "string"},  # human-readable description
        "line": {"type": "integer"},  # line number in code
        "severity": {"type": "string"},  # HIGH / MEDIUM / LOW
    }
}

_security_finding_schema = {  # one bandit finding
    "type": "object",
    "properties": {
        "test_id": {"type": "string"},  # e.g. "B608"
        "test_name": {"type": "string"},  # e.g. "hardcoded_sql_expressions"
        "message": {"type": "string"},  # description of the issue
        "line": {"type": "integer"},  # line number
        "severity": {"type": "string"},  # HIGH / MEDIUM / LOW
        "confidence": {"type": "string"},  # HIGH / MEDIUM / LOW
    }
}

_function_schema = {  # one function from AST
    "type": "object",
    "properties": {
        "name": {"type": "string"},
        "line": {"type": "integer"},
        "args": {"type": "array", "items": {"type": "string"}},
        "has_docstring": {"type": "boolean"},
    }
}

_class_schema = {  # one class from AST
    "type": "object",
    "properties": {
        "name": {"type": "string"},
        "line": {"type": "integer"},
        "methods": {"type": "array", "items": {"type": "string"}},
        "base_classes": {"type": "array", "items": {"type": "string"}},
    }
}

_import_schema = {  # one import from AST
    "type": "object",
    "properties": {
        "module": {"type": "string"},
        "alias": {"type": ["string", "null"]},
    }
}

_structure_schema = {  # combined code structure
    "type": "object",
    "description": "Code structure from extract_code_structure",
    "properties": {  # all three go INSIDE properties
        "functions": {
            "type": "array",
            "items": _function_schema,
        },
        "classes": {
            "type": "array",
            "items": _class_schema,
        },
        "imports": {
            "type": "array",
            "items": _import_schema,
        },
    }
}

# === TOOL SCHEMAS ===
analyzer_local_tools = [
    {
        "name": "submit_analysis",
        "description": (
            "Submit the final structured analysis after calling all three MCP tools "
            "(read_code, detect_syntax_errors, extract_code_structure). "
            "You MUST call this tool as your final step. "
            "Do NOT respond with plain text — always submit your findings through this tool."
        ),
        "input_schema": {
            "type": "object",
            "properties": {  # all fields INSIDE properties
                "code": {
                    "type": "string",
                    "description": "The full original code string from read_code",
                },
                "file_path": {
                    "type": ["string", "null"],  # nullable: null when raw code was passed
                    "description": "File path if provided, null if raw code was passed",
                },
                "line_count": {
                    "type": "integer",
                    "description": "The number of lines in the code",
                },
                "syntax_findings": {
                    "type": "array",
                    "description": "Ruff findings from detect_syntax_errors. Empty list if clean.",
                    "items": _syntax_finding_schema,
                },
                "security_findings": {
                    "type": "array",
                    "description": "Bandit findings from detect_syntax_errors. Empty list if clean.",
                    "items": _security_finding_schema,
                },
                "structure": _structure_schema,
                "summary": {  # summary INSIDE properties, not outside
                    "type": "string",
                    "description": "1-2 sentence factual overview of what the analysis found",
                },
            },
            "required": [  # required INSIDE input_schema, not outside
                "code",
                "file_path",
                "line_count",
                "syntax_findings",
                "security_findings",
                "structure",
                "summary",
            ],
        },
    }
]

# === TOOL EXECUTION ===
def run_analyzer_tool(name: str, tool_input: dict) -> str:
    """
    Executes local analyzer tools.
    Currently only submit_analysis — but extensible for future tools.
    """

    if name == "submit_analysis":
        try:
            # Here we validates that required fields are present
            # and return the data as clean JSON.
            required_fields = [
                "code", "file_path", "line_count",
                "syntax_findings", "security_findings",
                "structure", "summary",
            ]

            missing = [f for f in required_fields if f not in tool_input]
            if missing:
                return json.dumps({
                    "status": "error",
                    "message": f"Missing required fields: {missing}",
                })

            # Count total findings for a quick summary
            total_syntax = len(tool_input.get("syntax_findings", []))
            total_security = len(tool_input.get("security_findings", []))

            # Return the validated analysis with metadata
            return json.dumps({
                "status": "success",
                "analysis_results": tool_input,  # pass through the full structured data
                "metadata": {
                    "total_syntax_findings": total_syntax,
                    "total_security_findings": total_security,
                    "total_findings": total_syntax + total_security,
                },
            }, indent=2)

        except Exception as e:
            return json.dumps({
                "status": "error",
                "message": f"submit_analysis failed: {str(e)}",
            })

    else:
        return json.dumps({"error": f"Unknown analyzer tool: {name}"})


