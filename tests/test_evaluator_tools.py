"""
Tests for the local submit_evaluation tool in evaluator_tools.py.
"""

import json
from tools.evaluator_tools import run_evaluator_tool


_VALID_INPUT = {
    "reasoning":    "The fix replaces string concatenation with a parameterized query, directly addressing the SQL injection risk cited in company_rules §1.3.",
    "faithfulness": "faithful",
    "correctness":  "pass",
    "completeness": "complete",
}


def test_submit_evaluation_valid_input():
    """Returns success status and full evaluation dict for a well-formed submission."""
    result = json.loads(run_evaluator_tool("submit_evaluation", _VALID_INPUT))

    assert result["status"] == "success"
    assert "evaluation" in result
    assert result["evaluation"]["faithfulness"] == "faithful"
    assert result["evaluation"]["correctness"]  == "pass"
    assert result["evaluation"]["completeness"] == "complete"


def test_submit_evaluation_missing_reasoning():
    """Returns error and names the missing field when reasoning is absent."""
    incomplete = {k: v for k, v in _VALID_INPUT.items() if k != "reasoning"}
    result = json.loads(run_evaluator_tool("submit_evaluation", incomplete))

    assert result["status"] == "error"
    assert "reasoning" in result["message"]


def test_submit_evaluation_missing_faithfulness():
    """Returns error and names the missing field when faithfulness is absent."""
    incomplete = {k: v for k, v in _VALID_INPUT.items() if k != "faithfulness"}
    result = json.loads(run_evaluator_tool("submit_evaluation", incomplete))

    assert result["status"] == "error"
    assert "faithfulness" in result["message"]


def test_submit_evaluation_missing_correctness():
    """Returns error and names the missing field when correctness is absent."""
    incomplete = {k: v for k, v in _VALID_INPUT.items() if k != "correctness"}
    result = json.loads(run_evaluator_tool("submit_evaluation", incomplete))

    assert result["status"] == "error"
    assert "correctness" in result["message"]


def test_submit_evaluation_missing_completeness():
    """Returns error and names the missing field when completeness is absent."""
    incomplete = {k: v for k, v in _VALID_INPUT.items() if k != "completeness"}
    result = json.loads(run_evaluator_tool("submit_evaluation", incomplete))

    assert result["status"] == "error"
    assert "completeness" in result["message"]


def test_submit_evaluation_invalid_faithfulness():
    """Returns error naming the field and the bad value when faithfulness is out of enum."""
    bad_input = {**_VALID_INPUT, "faithfulness": "yes"}
    result = json.loads(run_evaluator_tool("submit_evaluation", bad_input))

    assert result["status"] == "error"
    assert "faithfulness" in result["message"]


def test_submit_evaluation_invalid_correctness():
    """Returns error naming the field and the bad value when correctness is out of enum."""
    bad_input = {**_VALID_INPUT, "correctness": "maybe"}
    result = json.loads(run_evaluator_tool("submit_evaluation", bad_input))

    assert result["status"] == "error"
    assert "correctness" in result["message"]


def test_submit_evaluation_invalid_completeness():
    """Returns error naming the field and the bad value when completeness is out of enum."""
    bad_input = {**_VALID_INPUT, "completeness": "full"}
    result = json.loads(run_evaluator_tool("submit_evaluation", bad_input))

    assert result["status"] == "error"
    assert "completeness" in result["message"]


def test_submit_evaluation_unknown_tool_name():
    """Returns error JSON for an unrecognised tool name."""
    result = json.loads(run_evaluator_tool("nonexistent_tool", {}))

    assert result["status"] == "error"