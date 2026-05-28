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