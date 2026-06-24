"""
Tests for the local submit_analysis tool in analyzer_tools.py.
"""

import json
import pytest
from tools.analyzer_tools import run_analyzer_tool, _deduplicate_findings


# === submit_analysis ===

def test_submit_analysis_valid_input():
    """Returns success status and the summary when the field is provided."""
    result = json.loads(run_analyzer_tool("submit_analysis", {"summary": "No findings."}))

    assert result["status"] == "success"
    assert result["summary"] == "No findings."


def test_submit_analysis_missing_summary():
    """Returns error status naming the missing field when summary is absent."""
    result = json.loads(run_analyzer_tool("submit_analysis", {}))

    assert result["status"] == "error"
    assert "summary" in result["message"]


def test_submit_analysis_summary_wrong_type():
    """Returns error status when summary is not a string."""
    result = json.loads(run_analyzer_tool("submit_analysis", {"summary": 123}))

    assert result["status"] == "error"
    assert "summary" in result["message"]


def test_submit_analysis_unknown_tool_name():
    """Returns error JSON for an unrecognised tool name."""
    result = json.loads(run_analyzer_tool("nonexistent_tool", {}))

    assert "error" in result


# === _deduplicate_findings ===

def test_deduplicate_findings_collapses_same_rule():
    """Multiple findings with the same rule collapse into one entry, keeping the first line."""
    findings = [
        {"rule": "E501", "message": "Line too long", "line": 10, "severity": "LOW"},
        {"rule": "E501", "message": "Line too long", "line": 22, "severity": "LOW"},
        {"rule": "E501", "message": "Line too long", "line": 45, "severity": "LOW"},
    ]
    deduped = _deduplicate_findings(findings)

    assert len(deduped) == 1                          # 3 occurrences collapsed into 1
    assert deduped[0]["rule"] == "E501"
    assert deduped[0]["line"] == 10                   # first occurrence preserved for downstream agents
    assert deduped[0]["lines"] == [10, 22, 45]
    assert deduped[0]["occurrences"] == 3


def test_deduplicate_findings_distinct_rules_stay_separate():
    """Findings with different rule codes are not merged."""
    findings = [
        {"rule": "B608", "message": "SQL injection", "line": 4, "severity": "MEDIUM"},
        {"rule": "F401", "message": "Unused import", "line": 1, "severity": "LOW"},
    ]
    deduped = _deduplicate_findings(findings)

    assert len(deduped) == 2
    rules = {f["rule"] for f in deduped}
    assert rules == {"B608", "F401"}


def test_deduplicate_findings_empty_list():
    """Returns an empty list when there are no findings."""
    assert _deduplicate_findings([]) == []