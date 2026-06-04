"""
Tests for the local submit_review tool in reviewer_tools.py.
"""

import json
import pytest
from tools.reviewer_tools import run_reviewer_tool

_VALID_INPUT = {
    "findings": [
        {
            "rule": "B301",
            "line": 8,
            "category": "Security",
            "severity": "HIGH",
            "rationale": "Pickle deserialization allows arbitrary code execution.",
            "best_practice_refs": [
                {"source": "pyguide", "section": "3.10", "text": "Avoid pickle for untrusted data."}
            ],
            "doc_url": "https://bandit.readthedocs.io/en/latest/blacklists/blacklist_calls.html",
            "cwe_id": 502,
        }
    ],
    "summary": "One high-severity security finding identified.",
    "rag_used": True,
}

def test_submit_review_valid_input():
    """Returns success status and correct metadata when all required fields are provided."""
    result = json.loads(run_reviewer_tool("submit_review", _VALID_INPUT))

    assert result["status"] == "success"
    assert "review_results" in result
    assert result["metadata"]["total_reviewed_findings"] == 1
    assert result["metadata"]["rag_used"] is True


def test_submit_review_missing_required_field():
    """Returns error status and names the missing field when input is incomplete."""
    incomplete = {k: v for k, v in _VALID_INPUT.items() if k != "rag_used"}
    result = json.loads(run_reviewer_tool("submit_review", incomplete))

    assert result["status"] == "error"
    assert "rag_used" in result["message"]


def test_submit_review_findings_not_a_list():
    """Returns error when findings is a dict instead of a list."""
    bad_input = {**_VALID_INPUT, "findings": {"rule": "B301"}}  # dict instead of list
    result = json.loads(run_reviewer_tool("submit_review", bad_input))

    assert result["status"] == "error"
    assert "list" in result["message"]


def test_submit_review_empty_findings_is_valid():
    """Empty findings list is valid — clean code produces no review findings."""
    empty_input = {**_VALID_INPUT, "findings": []}
    result = json.loads(run_reviewer_tool("submit_review", empty_input))

    assert result["status"] == "success"
    assert result["metadata"]["total_reviewed_findings"] == 0


def test_submit_review_unknown_tool_name():
    """Returns error JSON for an unrecognised tool name."""
    result = json.loads(run_reviewer_tool("nonexistent_tool", {}))

    assert "error" in result