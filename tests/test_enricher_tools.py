"""
Tests for the local submit_enrichment tool in enricher_tools.py.
"""

import json
import pytest
from tools.enricher_tools import run_enricher_tool

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

def test_submit_enrichment_valid_input():
    """Returns success status and correct metadata when all required fields are provided."""
    result = json.loads(run_enricher_tool("submit_enrichment", _VALID_INPUT))

    assert result["status"] == "success"
    assert "enrichment_results" in result
    assert result["metadata"]["total_reviewed_findings"] == 1
    assert result["metadata"]["rag_used"] is True


def test_submit_enrichment_missing_required_field():
    """Returns error status and names the missing field when input is incomplete."""
    incomplete = {k: v for k, v in _VALID_INPUT.items() if k != "rag_used"}
    result = json.loads(run_enricher_tool("submit_enrichment", incomplete))

    assert result["status"] == "error"
    assert "rag_used" in result["message"]


def test_submit_enrichment_findings_not_a_list():
    """Returns error when findings is a dict instead of a list."""
    bad_input = {**_VALID_INPUT, "findings": {"rule": "B301"}}
    result = json.loads(run_enricher_tool("submit_enrichment", bad_input))

    assert result["status"] == "error"
    assert "list" in result["message"]


def test_submit_enrichment_empty_findings_is_valid():
    """Empty findings list is valid — clean code produces no enriched findings."""
    empty_input = {**_VALID_INPUT, "findings": []}
    result = json.loads(run_enricher_tool("submit_enrichment", empty_input))

    assert result["status"] == "success"
    assert result["metadata"]["total_reviewed_findings"] == 0


def test_submit_enrichment_unknown_tool_name():
    """Returns error JSON for an unrecognised tool name."""
    result = json.loads(run_enricher_tool("nonexistent_tool", {}))

    assert "error" in result
