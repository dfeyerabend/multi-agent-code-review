"""
Tests for the local helper functions in agents/analyzer_agent.py.
"""

import json
import pytest
from agents.analyzer_agent import _extract_summary, _assemble_analysis


# === _extract_summary ===

def _tool_result_message(payload: dict) -> dict:
    """Builds a minimal conversation message wrapping one tool_result block."""
    return {
        "role": "user",
        "content": [{"type": "tool_result", "content": json.dumps(payload)}],
    }


def test_extract_summary_finds_valid_result():
    """Returns the summary string from a successful submit_analysis tool_result."""
    messages = [_tool_result_message({"status": "success", "summary": "Clean code, no issues."})]

    assert _extract_summary(messages) == "Clean code, no issues."


def test_extract_summary_ignores_failed_result():
    """Returns None when the only tool_result present is an error, not a success."""
    messages = [_tool_result_message({"status": "error", "message": "bad input"})]

    assert _extract_summary(messages) is None


def test_extract_summary_no_tool_results():
    """Returns None when the conversation has no tool_result blocks at all."""
    messages = [{"role": "assistant", "content": [{"type": "text", "text": "thinking..."}]}]

    assert _extract_summary(messages) is None


def test_extract_summary_picks_latest_match():
    """When multiple successful results exist, returns the most recent one."""
    messages = [
        _tool_result_message({"status": "success", "summary": "First attempt."}),
        _tool_result_message({"status": "success", "summary": "Final attempt."}),
    ]

    assert _extract_summary(messages) == "Final attempt."


# === _assemble_analysis ===

_READ_CODE_OUT = {
    "status": "success",
    "source_type": "raw_string",
    "code": "def get_user(id):\n    return id\n",
    "line_count": 2,
}

_STRUCTURE_OUT = {
    "status": "success",
    "functions": [{"name": "get_user", "line": 1, "args": ["id"], "has_docstring": False}],
    "classes": [],
    "imports": [],
}

_SYNTAX_OUT = {
    "status": "issues_found",
    "results": {
        "ruff": {"findings": [], "error": None},
        "bandit": {
            "findings": [
                {"rule": "B608", "tool": "bandit", "message": "SQL injection risk",
                 "line": 4, "severity": "MEDIUM", "category": "Security", "cwe_id": 89},
            ],
            "error": None,
        },
    },
}

_COMPANY_OUT = {
    "status": "clean",
    "total_findings": 0,
    "rule_errors": {},
    "findings": [],
}


def test_assemble_analysis_builds_correct_shape():
    """Builds analysis_results with real findings and line numbers from MCP outputs."""
    mcp_outputs = {
        "read_code": _READ_CODE_OUT,
        "detect_syntax_errors": _SYNTAX_OUT,
        "extract_code_structure": _STRUCTURE_OUT,
        "check_company_rules": _COMPANY_OUT,
    }
    result = _assemble_analysis(mcp_outputs, "Found one SQL injection risk.")

    assert result["status"] == "success"
    analysis = result["analysis_results"]
    assert analysis["code"] == _READ_CODE_OUT["code"]
    assert analysis["summary"] == "Found one SQL injection risk."
    assert analysis["security_findings"][0]["rule"] == "B608"
    assert analysis["security_findings"][0]["lines"] == [4]        # real line, not 0
    assert result["metadata"]["total_security_findings"] == 1


def test_assemble_analysis_missing_tool_output():
    """Returns a structured error naming the tool the model never called."""
    mcp_outputs = {
        "read_code": _READ_CODE_OUT,
        "extract_code_structure": _STRUCTURE_OUT,
        # detect_syntax_errors missing
    }
    result = _assemble_analysis(mcp_outputs, "summary")

    assert result["status"] == "error"
    assert "detect_syntax_errors" in result["message"]


def test_assemble_analysis_missing_company_tool_output():
    """Returns a structured error naming check_company_rules when the model never called it."""
    mcp_outputs = {
        "read_code": _READ_CODE_OUT,
        "detect_syntax_errors": _SYNTAX_OUT,
        "extract_code_structure": _STRUCTURE_OUT,
        # check_company_rules missing
    }
    result = _assemble_analysis(mcp_outputs, "summary")

    assert result["status"] == "error"
    assert "check_company_rules" in result["message"]


def test_assemble_analysis_read_code_failed():
    """Returns a structured error when read_code itself reported failure."""
    mcp_outputs = {
        "read_code": {"status": "error", "message": "File not found: missing.py"},
        "detect_syntax_errors": _SYNTAX_OUT,
        "extract_code_structure": _STRUCTURE_OUT,
        "check_company_rules": _COMPANY_OUT,
    }
    result = _assemble_analysis(mcp_outputs, "summary")

    assert result["status"] == "error"
    assert "read_code" in result["message"]


def test_assemble_analysis_structure_failed():
    """Returns a structured error when extract_code_structure itself reported failure."""
    mcp_outputs = {
        "read_code": _READ_CODE_OUT,
        "detect_syntax_errors": _SYNTAX_OUT,
        "extract_code_structure": {"status": "error", "message": "Cannot parse code: invalid syntax"},
        "check_company_rules": _COMPANY_OUT,
    }
    result = _assemble_analysis(mcp_outputs, "summary")

    assert result["status"] == "error"
    assert "extract_code_structure" in result["message"]

def test_assemble_analysis_propagates_tool_errors():
    """Marks scan_complete False and carries tool_errors when a scanner failed."""
    syntax_out_partial = {
        **_SYNTAX_OUT,
        "status": "partial",
        "tool_errors": {"bandit": "bandit timed out after 30 seconds"},
    }
    mcp_outputs = {
        "read_code": _READ_CODE_OUT,
        "detect_syntax_errors": syntax_out_partial,
        "extract_code_structure": _STRUCTURE_OUT,
        "check_company_rules": _COMPANY_OUT,
    }
    result = _assemble_analysis(mcp_outputs, "summary")

    assert result["status"] == "success"
    assert result["metadata"]["scan_complete"] is False
    assert result["metadata"]["tool_errors"] == {"bandit": "bandit timed out after 30 seconds"}


def test_assemble_analysis_includes_company_findings():
    """Company findings from check_company_rules land in analysis_results, unmodified."""
    company_out_with_finding = {
        "status": "issues_found",
        "total_findings": 1,
        "rule_errors": {},
        "findings": [
            {"rule": "COMPANY-1.2",
             "message": "Function is missing a '# REASON:' comment (function 'get_user')",
             "severity": "LOW", "category": "Maintainability", "lines": [1], "occurrences": 1},
        ],
    }
    mcp_outputs = {
        "read_code": _READ_CODE_OUT,
        "detect_syntax_errors": _SYNTAX_OUT,
        "extract_code_structure": _STRUCTURE_OUT,
        "check_company_rules": company_out_with_finding,
    }
    result = _assemble_analysis(mcp_outputs, "summary")

    assert result["status"] == "success"
    analysis = result["analysis_results"]
    assert analysis["company_findings"] == company_out_with_finding["findings"]
    assert result["metadata"]["total_company_findings"] == 1
    assert result["metadata"]["total_findings"] == 2  # 1 bandit + 1 company


def test_assemble_analysis_propagates_company_rule_errors():
    """Marks scan_complete False and carries company_rule_errors when a rule failed."""
    company_out_partial = {
        "status": "partial",
        "total_findings": 0,
        "rule_errors": {"COMPANY-1.4": "mechanism 'forbidden_access' crashed: unexpected node type"},
        "findings": [],
    }
    mcp_outputs = {
        "read_code": _READ_CODE_OUT,
        "detect_syntax_errors": _SYNTAX_OUT,
        "extract_code_structure": _STRUCTURE_OUT,
        "check_company_rules": company_out_partial,
    }
    result = _assemble_analysis(mcp_outputs, "summary")

    assert result["status"] == "success"
    assert result["metadata"]["scan_complete"] is False
    assert result["metadata"]["company_rule_errors"] == company_out_partial["rule_errors"]