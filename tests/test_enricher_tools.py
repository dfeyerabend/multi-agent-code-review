"""
Tests for the local submit_enrichment tool in enricher_tools.py.
"""

import json
import pytest
from tools.enricher_tools import run_enricher_tool

_VALID_INPUT = {
    "findings": [
        {
            "index": 0,
            "rationale": "Pickle deserialization allows arbitrary code execution.",
            "best_practice_refs": [
                {"source": "pyguide", "section": "3.10", "text": "Avoid pickle for untrusted data."}
            ],
            "severity": "HIGH",
        }
    ],
    "summary": "One high-severity security finding identified.",
    "rag_used": True,
}

def test_submit_enrichment_valid_input():
    """Returns success status and correct metadata when all required fields are provided."""
    result = json.loads(run_enricher_tool("submit_enrichment", _VALID_INPUT))

    assert result["status"] == "success"
    assert "enrichments" in result
    assert result["metadata"]["total_enriched"] == 1
    assert result["rag_used"] is True


def test_submit_enrichment_empty_findings_is_valid():
    """Empty findings list is valid — clean code produces no enriched findings."""
    empty_input = {**_VALID_INPUT, "findings": []}
    result = json.loads(run_enricher_tool("submit_enrichment", empty_input))

    assert result["status"] == "success"
    assert result["metadata"]["total_enriched"] == 0


def test_submit_enrichment_findings_not_a_list():
    """Returns error when findings is a dict instead of a list."""
    bad_input = {**_VALID_INPUT, "findings": {"rule": "B301"}}
    result = json.loads(run_enricher_tool("submit_enrichment", bad_input))

    assert result["status"] == "error"
    assert "list" in result["message"]


def test_submit_enrichment_missing_index():
    """Returns error naming the entry and field when 'index' is missing or not an integer."""
    bad_entry = {k: v for k, v in _VALID_INPUT["findings"][0].items() if k != "index"}
    bad_input = {**_VALID_INPUT, "findings": [bad_entry]}
    result = json.loads(run_enricher_tool("submit_enrichment", bad_input))

    assert result["status"] == "error"
    assert any("findings[0]" in e and "index" in e for e in result["errors"])


def test_submit_enrichment_unknown_tool_name():
    """Returns error JSON for an unrecognised tool name."""
    result = json.loads(run_enricher_tool("nonexistent_tool", {}))

    assert result["status"] == "error"
    assert "nonexistent_tool" in result["message"]


def test_submit_enrichment_severity_is_optional():
    """An enrichment entry without a severity override is still valid."""
    entry_without_severity = {k: v for k, v in _VALID_INPUT["findings"][0].items() if k != "severity"}
    valid_input = {**_VALID_INPUT, "findings": [entry_without_severity]}
    result = json.loads(run_enricher_tool("submit_enrichment", valid_input))

    assert result["status"] == "success"
    assert "severity" not in result["enrichments"][0]
