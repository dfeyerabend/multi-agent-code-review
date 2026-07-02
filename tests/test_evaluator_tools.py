"""
Tests for the Evaluator: the local submit_evaluation tool (evaluator_tools.py) and the pure
helper functions in evaluator_agent.py.

The agent-helper tests cover the drive-off-fixes path: recover each fix's issue context
(_issue_for_fix), derive a status (_derive_status), and fan one verdict out to one report
entry per covered finding (_entries_for_fix / _evaluated_entry).
"""

import json
from tools.evaluator_tools import run_evaluator_tool
from agents.evaluator_agent import (
    _evaluated_entry,
    _entries_for_fix,
    _issue_for_fix,
    _derive_status,
)


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


def test_submit_evaluation_rejects_dropped_partial_faithfulness():
    """'partial' was removed from the faithfulness enum (replaced by not_applicable) — must be rejected."""
    bad_input = {**_VALID_INPUT, "faithfulness": "partial"}
    result = json.loads(run_evaluator_tool("submit_evaluation", bad_input))

    assert result["status"] == "error"
    assert "faithfulness" in result["message"]


def test_submit_evaluation_not_applicable_faithfulness_is_valid():
    """not_applicable is a valid faithfulness value — no best_practice_refs to judge against."""
    ok_input = {**_VALID_INPUT, "faithfulness": "not_applicable"}
    result = json.loads(run_evaluator_tool("submit_evaluation", ok_input))

    assert result["status"] == "success"
    assert result["evaluation"]["faithfulness"] == "not_applicable"


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


# === _evaluated_entry (evaluator_agent.py) ===

_ENTRY_KEYS = {
    "rule", "lines", "category", "status", "suggested_code",
    "grounded_in", "reasoning", "faithfulness", "correctness", "completeness",
}


def test_evaluated_entry_has_full_schema():
    """Every entry carries the complete key set the render_report report renderer depends on."""
    entry = _evaluated_entry("B608", [4], "Security", "APPROVED", "code", ["ref"], "ok", "faithful", "pass", "complete")

    assert set(entry.keys()) == _ENTRY_KEYS
    assert entry["rule"] == "B608"
    assert entry["lines"] == [4]
    assert entry["status"] == "APPROVED"


def test_evaluated_entry_coerces_non_string_rule():
    """A non-string rule is coerced, not dropped — losing the id is worse than a dirty string."""
    entry = _evaluated_entry(None, [1], "Style", "NOT_EVALUATED", None, [], "r", None, None, None)

    assert isinstance(entry["rule"], str)
    assert entry["rule"] == "None"


def test_evaluated_entry_grounded_in_defaults_to_list():
    """A non-list grounded_in is normalised to an empty list."""
    entry = _evaluated_entry("X", [1], "Style", "APPROVED", "c", "notalist", "r", "faithful", "pass", "complete")

    assert entry["grounded_in"] == []


# === _entries_for_fix (evaluator_agent.py) ===

_VERDICTS = {"faithfulness": "faithful", "correctness": "pass", "completeness": "complete"}


def test_entries_for_fix_fans_out_per_finding_key():
    """One judged fix produces one entry per covered finding, all sharing the verdict."""
    fix = {
        "finding_keys": [
            {"rule": "E401", "lines": [1], "category": "Style"},
            {"rule": "F401", "lines": [1], "category": "Logic"},
        ],
        "suggested_code": "import json",
        "grounded_in": ["pyguide §2.1"],
    }
    entries = _entries_for_fix(fix, "APPROVED", "looks good", _VERDICTS)

    assert len(entries) == 2
    assert {e["rule"] for e in entries} == {"E401", "F401"}
    assert all(e["status"] == "APPROVED" for e in entries)
    assert all(e["reasoning"] == "looks good" for e in entries)
    assert all(e["faithfulness"] == "faithful" for e in entries)
    assert all(e["suggested_code"] == "import json" for e in entries)


def test_entries_for_fix_null_verdicts_when_no_llm():
    """With no verdicts dict (no LLM call), the three verdict fields are None."""
    fix = {"finding_keys": [{"rule": "W291", "lines": [1], "category": "Style"}],
           "suggested_code": None, "grounded_in": []}
    entries = _entries_for_fix(fix, "NO_FIX", "no fix produced")

    assert len(entries) == 1
    assert entries[0]["status"] == "NO_FIX"
    assert entries[0]["faithfulness"] is None
    assert entries[0]["correctness"] is None
    assert entries[0]["completeness"] is None


def test_entries_for_fix_missing_finding_keys_still_surfaces():
    """A fix with no finding_keys still yields one degraded NOT_EVALUATED entry, never vanishes."""
    fix = {"suggested_code": "x", "grounded_in": []}
    entries = _entries_for_fix(fix, "APPROVED", "r", _VERDICTS)

    assert len(entries) == 1
    assert entries[0]["status"] == "NOT_EVALUATED"


def test_entries_for_fix_non_dict_fix_surfaces_once():
    """A non-dict fix is surfaced as a single degraded entry rather than raising."""
    entries = _entries_for_fix("nope", "APPROVED", "r")

    assert len(entries) == 1
    assert entries[0]["status"] == "NOT_EVALUATED"


def test_entries_for_fix_skips_non_dict_key():
    """A malformed finding_key is skipped without dropping the valid ones."""
    fix = {"finding_keys": ["bad", {"rule": "B608", "lines": [4], "category": "Security"}],
           "suggested_code": "c", "grounded_in": []}
    entries = _entries_for_fix(fix, "APPROVED", "r", _VERDICTS)

    assert len(entries) == 1
    assert entries[0]["rule"] == "B608"


# === _issue_for_fix (evaluator_agent.py) ===

def test_issue_for_fix_combines_rationales_of_covered_findings():
    """A multi-finding fix gathers the rationale and refs of every finding it covers."""
    fix = {"finding_keys": [{"rule": "E401", "lines": [1]}, {"rule": "F401", "lines": [1]}]}
    findings = [
        {"rule": "E401", "lines": [1], "rationale": "split imports", "best_practice_refs": [{"s": "a"}]},
        {"rule": "F401", "lines": [1], "rationale": "unused os", "best_practice_refs": [{"s": "b"}]},
    ]
    issue = _issue_for_fix(fix, findings)

    assert "split imports" in issue["rationale"]
    assert "unused os" in issue["rationale"]
    assert {"s": "a"} in issue["best_practice_refs"]
    assert {"s": "b"} in issue["best_practice_refs"]


def test_issue_for_fix_matches_overlapping_lines_for_split_unit():
    """A collapsed multi-line finding still matches the single-line unit it was split into."""
    fix = {"finding_keys": [{"rule": "W291", "lines": [4]}]}
    findings = [{"rule": "W291", "lines": [2, 4], "rationale": "trailing ws", "best_practice_refs": []}]
    issue = _issue_for_fix(fix, findings)

    assert issue["rationale"] == "trailing ws"


def test_issue_for_fix_dedupes_repeated_rationale():
    """Two split units of the same finding contribute its shared rationale only once."""
    fix = {"finding_keys": [{"rule": "W291", "lines": [2]}, {"rule": "W291", "lines": [4]}]}
    findings = [{"rule": "W291", "lines": [2, 4], "rationale": "trailing ws", "best_practice_refs": []}]
    issue = _issue_for_fix(fix, findings)

    assert issue["rationale"] == "trailing ws"


def test_issue_for_fix_non_dict_fix_returns_empty():
    """Invalid fix input returns the empty issue shape, never raises."""
    assert _issue_for_fix("nope", []) == {"rationale": "", "best_practice_refs": []}


# === _derive_status (evaluator_agent.py) ===

def test_derive_status_approved_when_faithful():
    """Correct, complete, and faithful to a retrieved guideline → APPROVED."""
    assert _derive_status({"faithfulness": "faithful", "correctness": "pass", "completeness": "complete"}) == "APPROVED"


def test_derive_status_approved_when_no_guideline():
    """Correct, complete, and not_applicable (no guideline existed) → APPROVED — never blocked by faithfulness."""
    assert _derive_status({"faithfulness": "not_applicable", "correctness": "pass", "completeness": "complete"}) == "APPROVED"


def test_derive_status_correctness_fail_is_incorrect():
    """Broken/wrong code outranks everything, including faithfulness → INCORRECT."""
    assert _derive_status({"faithfulness": "faithful", "correctness": "fail", "completeness": "complete"}) == "INCORRECT"


def test_derive_status_incomplete_outranks_faithfulness():
    """Correct but only partially resolves the issue → INCOMPLETE, even if it would otherwise be faithful."""
    assert _derive_status({"faithfulness": "faithful", "correctness": "pass", "completeness": "partial"}) == "INCOMPLETE"


def test_derive_status_unfaithful_is_noncompliant():
    """Correct and complete, but deviates from a retrieved guideline → NONCOMPLIANT."""
    assert _derive_status({"faithfulness": "unfaithful", "correctness": "pass", "completeness": "complete"}) == "NONCOMPLIANT"


def test_derive_status_unknown_verdict_value_is_not_evaluated():
    """A verdict value outside the known enums is never silently approved → NOT_EVALUATED."""
    assert _derive_status({"faithfulness": "??", "correctness": "pass", "completeness": "complete"}) == "NOT_EVALUATED"


def test_derive_status_non_dict_is_not_evaluated():
    """A malformed verdicts input is treated as NOT_EVALUATED, never raises."""
    assert _derive_status("nope") == "NOT_EVALUATED"