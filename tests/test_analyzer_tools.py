"""
Tests for the local submit_analysis tool in analyzer_tools.py.
"""

import json
import pytest
from tools.analyzer_tools import run_analyzer_tool

_VALID_INPUT = {
    "code": "def foo(): pass",
    "file_path": None,
    "line_count": 1,
    "syntax_findings": [],
    "security_findings": [],
    "structure": {"functions": [], "classes": [], "imports": []},
    "summary": "No findings.",
}


def test_submit_analysis_valid_input():
    """Returns success status and metadata when all required fields are provided."""
    result = json.loads(run_analyzer_tool("submit_analysis", _VALID_INPUT))

    assert result["status"] == "success"
    assert "analysis_results" in result
    assert result["metadata"]["total_findings"] == 0


def test_submit_analysis_missing_required_field():
    """Returns error status and names the missing field when input is incomplete."""
    incomplete = {k: v for k, v in _VALID_INPUT.items() if k != "summary"}
    result = json.loads(run_analyzer_tool("submit_analysis", incomplete))

    assert result["status"] == "error"
    assert "summary" in result["message"]


def test_submit_analysis_unknown_tool_name():
    """Returns error JSON for an unrecognised tool name."""
    result = json.loads(run_analyzer_tool("nonexistent_tool", {}))

    assert "error" in result


def test_submit_analysis_deduplicates_findings():
    """Multiple findings with the same rule are collapsed into one entry with occurrences and lines."""
    input_with_duplicates = {
        **_VALID_INPUT,
        "syntax_findings": [
            {"rule": "E501", "message": "Line too long", "line": 10, "severity": "LOW", "category": "Style"},
            {"rule": "E501", "message": "Line too long", "line": 22, "severity": "LOW", "category": "Style"},
            {"rule": "E501", "message": "Line too long", "line": 45, "severity": "LOW", "category": "Style"},
        ],
    }
    result = json.loads(run_analyzer_tool("submit_analysis", input_with_duplicates))

    assert result["status"] == "success"
    findings = result["analysis_results"]["syntax_findings"]
    assert len(findings) == 1                        # 3 occurrences collapsed into 1
    assert findings[0]["rule"] == "E501"
    assert findings[0]["occurrences"] == 3
    assert set(findings[0]["lines"]) == {10, 22, 45}
    assert result["metadata"]["total_syntax_findings"] == 1