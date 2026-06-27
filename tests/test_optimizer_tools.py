"""
Tests for the Optimizer: the local submit_optimization tool (optimizer_tools.py) and the
pure helper functions in optimizer_agent.py.

Covers the index-keyed contract after the multi-finding change: a single fix now references
the findings it resolves through `indexes` (a non-empty list of ints), so one coherent fix
can cover several findings that share a line. The agent-helper tests cover the routing path
that produces those grouped fixes: explode repeats → group by shared line → route → merge.
"""

import json
from tools.optimizer_tools import run_optimizer_tool
from agents.optimizer_agent import (
    _explode_repeats,
    _group_overlapping,
    _route_findings,
    _merge_fixes,
    _failure_fix,
)


_VALID_INPUT = {
    "fixes": [
        {
            "indexes":        [0],
            "suggested_code": "cursor.execute('SELECT * FROM users WHERE id = %s', (user_id,))",
            "explanation":    "Parameterized query eliminates SQL injection risk.",
            "grounded_in":    ["pyguide §3.10", "company_rules §1.3"],
        }
    ],
    "summary": "One SQL injection fix generated.",
}


def test_submit_optimization_valid_input():
    """Returns success status and correct metadata for a well-formed single-finding submission."""
    result = json.loads(run_optimizer_tool("submit_optimization", _VALID_INPUT))

    assert result["status"] == "success"
    assert "fixes" in result
    assert "summary" in result
    assert result["metadata"]["total_fixes"] == 1


def test_submit_optimization_multi_finding_fix_is_valid():
    """A single fix listing several indexes is accepted — the core conflict-group case."""
    multi = {**_VALID_INPUT, "fixes": [{**_VALID_INPUT["fixes"][0], "indexes": [0, 1, 2]}]}
    result = json.loads(run_optimizer_tool("submit_optimization", multi))

    assert result["status"] == "success"
    assert result["metadata"]["total_fixes"] == 1   # one fix entry, even though it covers three findings


def test_submit_optimization_missing_fixes():
    """Returns error and names the missing field when fixes is absent."""
    incomplete = {k: v for k, v in _VALID_INPUT.items() if k != "fixes"}
    result = json.loads(run_optimizer_tool("submit_optimization", incomplete))

    assert result["status"] == "error"
    assert "fixes" in result["message"]


def test_submit_optimization_missing_summary():
    """Returns error and names the missing field when summary is absent."""
    incomplete = {k: v for k, v in _VALID_INPUT.items() if k != "summary"}
    result = json.loads(run_optimizer_tool("submit_optimization", incomplete))

    assert result["status"] == "error"
    assert "summary" in result["message"]


def test_submit_optimization_fixes_not_a_list():
    """Returns error when fixes is a dict instead of a list."""
    bad_input = {**_VALID_INPUT, "fixes": {"indexes": [0]}}
    result = json.loads(run_optimizer_tool("submit_optimization", bad_input))

    assert result["status"] == "error"
    assert "list" in result["message"]


def test_submit_optimization_summary_not_a_string():
    """Returns error when summary is not a string."""
    bad_input = {**_VALID_INPUT, "summary": 123}
    result = json.loads(run_optimizer_tool("submit_optimization", bad_input))

    assert result["status"] == "error"
    assert "summary" in result["message"]


def test_submit_optimization_empty_fixes_is_valid():
    """Empty fixes list is valid — clean code produces no fixes."""
    empty_input = {**_VALID_INPUT, "fixes": []}
    result = json.loads(run_optimizer_tool("submit_optimization", empty_input))

    assert result["status"] == "success"
    assert result["metadata"]["total_fixes"] == 0


def test_submit_optimization_multiple_fixes():
    """total_fixes counts fix entries, not findings covered."""
    multi_input = {
        **_VALID_INPUT,
        "fixes": [
            _VALID_INPUT["fixes"][0],
            {**_VALID_INPUT["fixes"][0], "indexes": [1]},
        ],
    }
    result = json.loads(run_optimizer_tool("submit_optimization", multi_input))

    assert result["status"] == "success"
    assert result["metadata"]["total_fixes"] == 2


def test_submit_optimization_missing_indexes():
    """Returns error naming the entry when indexes is missing."""
    bad_input = {**_VALID_INPUT, "fixes": [{k: v for k, v in _VALID_INPUT["fixes"][0].items() if k != "indexes"}]}
    result = json.loads(run_optimizer_tool("submit_optimization", bad_input))

    assert result["status"] == "error"
    assert "fixes[0]" in result["errors"][0]
    assert "indexes" in result["errors"][0]


def test_submit_optimization_indexes_not_a_list():
    """Returns error when indexes is a scalar int instead of a list."""
    bad_input = {**_VALID_INPUT, "fixes": [{**_VALID_INPUT["fixes"][0], "indexes": 0}]}
    result = json.loads(run_optimizer_tool("submit_optimization", bad_input))

    assert result["status"] == "error"
    assert "fixes[0]" in result["errors"][0]
    assert "indexes" in result["errors"][0]


def test_submit_optimization_indexes_empty_list():
    """An empty indexes list is rejected — a fix attached to no finding loses the model's work."""
    bad_input = {**_VALID_INPUT, "fixes": [{**_VALID_INPUT["fixes"][0], "indexes": []}]}
    result = json.loads(run_optimizer_tool("submit_optimization", bad_input))

    assert result["status"] == "error"
    assert "fixes[0]" in result["errors"][0]
    assert "indexes" in result["errors"][0]


def test_submit_optimization_indexes_contains_non_int():
    """Returns error when indexes holds a non-integer entry."""
    bad_input = {**_VALID_INPUT, "fixes": [{**_VALID_INPUT["fixes"][0], "indexes": [0, "1"]}]}
    result = json.loads(run_optimizer_tool("submit_optimization", bad_input))

    assert result["status"] == "error"
    assert "fixes[0]" in result["errors"][0]
    assert "indexes" in result["errors"][0]


def test_submit_optimization_indexes_bool_rejected():
    """Booleans are ints in Python — must be explicitly rejected inside indexes."""
    bad_input = {**_VALID_INPUT, "fixes": [{**_VALID_INPUT["fixes"][0], "indexes": [True]}]}
    result = json.loads(run_optimizer_tool("submit_optimization", bad_input))

    assert result["status"] == "error"
    assert "indexes" in result["errors"][0]


def test_submit_optimization_suggested_code_null_is_valid():
    """suggested_code may be null when the model genuinely cannot produce a fix."""
    null_input = {**_VALID_INPUT, "fixes": [{**_VALID_INPUT["fixes"][0], "suggested_code": None}]}
    result = json.loads(run_optimizer_tool("submit_optimization", null_input))

    assert result["status"] == "success"


def test_submit_optimization_grounded_in_not_list_of_strings():
    """Returns error when grounded_in contains a non-string entry."""
    bad_input = {**_VALID_INPUT, "fixes": [{**_VALID_INPUT["fixes"][0], "grounded_in": ["ok", 5]}]}
    result = json.loads(run_optimizer_tool("submit_optimization", bad_input))

    assert result["status"] == "error"
    assert "fixes[0]" in result["errors"][0]
    assert "grounded_in" in result["errors"][0]


def test_submit_optimization_no_longer_accepts_finding_rule_or_line():
    """finding_rule/finding_line are not validated or required — identity is attached in Python."""
    legacy_input = {
        **_VALID_INPUT,
        "fixes": [{"finding_rule": "B608", "finding_line": 12, "indexes": [0],
                   "suggested_code": "x", "explanation": "y", "grounded_in": []}],
    }
    result = json.loads(run_optimizer_tool("submit_optimization", legacy_input))

    assert result["status"] == "success"     # extra legacy fields are simply ignored, not rejected


def test_submit_optimization_unknown_tool_name():
    """Returns error JSON for an unrecognised tool name."""
    result = json.loads(run_optimizer_tool("nonexistent_tool", {}))

    assert result["status"] == "error"
    assert "nonexistent_tool" in result["message"]


# === _explode_repeats (optimizer_agent.py) ===

def test_explode_repeats_splits_multi_occurrence():
    """A finding with occurrences > 1 splits into one single-line unit per line in `lines`."""
    findings = [{"rule": "W291", "lines": [2, 4], "occurrences": 2, "category": "Style"}]
    units = _explode_repeats(findings)

    assert len(units) == 2
    assert {u["lines"][0] for u in units} == {2, 4}
    assert all(u["occurrences"] == 1 and len(u["lines"]) == 1 for u in units)
    assert all("line" not in u for u in units)        # no scalar line reintroduced by the split


def test_explode_repeats_passes_through_single_occurrence():
    """A finding that fired once is returned unchanged."""
    findings = [{"rule": "B608", "lines": [4], "occurrences": 1, "category": "Security"}]
    assert _explode_repeats(findings) == findings


def test_explode_repeats_finding_without_lines_passes_through():
    """A finding with no usable `lines` is kept as-is rather than dropped."""
    findings = [{"rule": "X", "occurrences": 1}]
    assert _explode_repeats(findings) == findings


def test_explode_repeats_skips_non_dict_finding():
    """A non-dict finding is skipped without aborting the rest of the batch."""
    units = _explode_repeats(["bad", {"rule": "B608", "lines": [4], "occurrences": 1}])

    assert len(units) == 1
    assert units[0]["rule"] == "B608"


def test_explode_repeats_non_list_returns_empty():
    """Invalid input type returns an empty list, never raises."""
    assert _explode_repeats("nope") == []


# === _group_overlapping (optimizer_agent.py) ===

def test_group_overlapping_groups_shared_line():
    """Three findings on the same line collapse into one group — the real conflict case."""
    units = [
        {"rule": "E401", "lines": [1]},
        {"rule": "F401", "lines": [1]},
        {"rule": "F401", "lines": [1]},
    ]
    groups = _group_overlapping(units)

    assert len(groups) == 1
    assert len(groups[0]) == 3


def test_group_overlapping_separates_non_overlapping():
    """Findings on different lines are not grouped together."""
    units = [{"rule": "A", "lines": [1]}, {"rule": "B", "lines": [20]}]
    groups = _group_overlapping(units)

    assert len(groups) == 2
    assert all(len(g) == 1 for g in groups)


def test_group_overlapping_is_transitive():
    """A overlaps B and B overlaps C → all three end up in one group."""
    units = [
        {"rule": "A", "lines": [1, 2]},
        {"rule": "B", "lines": [2, 3]},
        {"rule": "C", "lines": [3]},
    ]
    groups = _group_overlapping(units)

    assert len(groups) == 1
    assert len(groups[0]) == 3


def test_group_overlapping_lineless_unit_is_singleton():
    """A unit with no usable line never merges — it isolates into its own group."""
    units = [{"rule": "A", "lines": [1]}, {"rule": "B"}]
    groups = _group_overlapping(units)

    assert len(groups) == 2


def test_group_overlapping_non_list_returns_empty():
    """Invalid input type returns an empty list, never raises."""
    assert _group_overlapping("nope") == []


# === _route_findings (optimizer_agent.py) ===

def test_route_findings_multi_unit_group_is_conflict():
    """A group of 2+ units routes to conflict_groups regardless of mixed categories."""
    groups = [[
        {"rule": "E401", "lines": [1], "category": "Style"},
        {"rule": "F401", "lines": [1], "category": "Logic"},
    ]]
    conflict, individual, style = _route_findings(groups)

    assert len(conflict) == 1
    assert individual == []
    assert style == {}


def test_route_findings_single_style_goes_to_style_groups():
    """A lone Style unit is rule-batched, exactly as before the conflict change."""
    groups = [[{"rule": "W291", "lines": [2], "category": "Style"}]]
    conflict, individual, style = _route_findings(groups)

    assert conflict == []
    assert individual == []
    assert style.get("W291") and len(style["W291"]) == 1


def test_route_findings_single_non_style_goes_individual():
    """A lone Security/Logic/Maintainability unit gets its own individual call."""
    groups = [[{"rule": "B608", "lines": [4], "category": "Security"}]]
    conflict, individual, style = _route_findings(groups)

    assert conflict == []
    assert len(individual) == 1
    assert style == {}


def test_route_findings_non_list_returns_empty_triple():
    """Invalid input type returns the empty three-bucket tuple, never raises."""
    assert _route_findings("nope") == ([], [], {})


# === _merge_fixes (optimizer_agent.py) ===

def test_merge_fixes_fans_multi_index_to_finding_keys():
    """One fix referencing several indexes yields one entry whose finding_keys cover them all."""
    units = [
        {"rule": "E401", "lines": [1], "category": "Style", "index": 0},
        {"rule": "F401", "lines": [1], "category": "Logic", "index": 1},
    ]
    fixes = [{"indexes": [0, 1], "suggested_code": "import json", "explanation": "x", "grounded_in": []}]
    merged = _merge_fixes(units, fixes)

    assert len(merged) == 1
    assert merged[0]["suggested_code"] == "import json"
    assert {k["rule"] for k in merged[0]["finding_keys"]} == {"E401", "F401"}


def test_merge_fixes_uncovered_unit_gets_null_entry():
    """A unit no fix referenced still surfaces, as a null-suggested_code entry."""
    units = [
        {"rule": "A", "lines": [1], "category": "Style", "index": 0},
        {"rule": "B", "lines": [2], "category": "Logic", "index": 1},
    ]
    fixes = [{"indexes": [0], "suggested_code": "fix", "explanation": "", "grounded_in": []}]
    merged = _merge_fixes(units, fixes)

    null_entries = [m for m in merged if m["suggested_code"] is None]
    assert len(merged) == 2
    assert len(null_entries) == 1
    assert null_entries[0]["finding_keys"][0]["rule"] == "B"


def test_merge_fixes_non_list_units_returns_empty():
    """Invalid units input returns an empty list, never raises."""
    assert _merge_fixes("nope", []) == []


# === _failure_fix (optimizer_agent.py) ===

def test_failure_fix_builds_finding_keys_with_null_code():
    """A failed call still surfaces every covered unit as a null-suggested_code entry."""
    units = [{"rule": "B608", "lines": [4], "category": "Security"}]
    entry = _failure_fix(units, "boom")

    assert entry["suggested_code"] is None
    assert entry["explanation"] == "boom"
    assert entry["finding_keys"] == [{"rule": "B608", "lines": [4], "category": "Security"}]


# === explode + group integration (the required scenarios) ===

def test_canonical_line1_conflict_groups_together():
    """The real failure case: E401 + two F401 on line 1 end up in one group after explode + group."""
    findings = [
        {"rule": "E401", "lines": [1], "occurrences": 1, "category": "Style"},
        {"rule": "F401", "lines": [1], "occurrences": 1, "category": "Logic"},
        {"rule": "F401", "lines": [1], "occurrences": 1, "category": "Logic"},
        {"rule": "B608", "lines": [4], "occurrences": 1, "category": "Security"},
        {"rule": "W291", "lines": [2, 4], "occurrences": 2, "category": "Style"},
    ]
    groups = _group_overlapping(_explode_repeats(findings))

    # line 1 → {E401, F401, F401}; line 4 → {B608, W291@4}; line 2 → {W291@2}
    assert sorted(len(g) for g in groups) == [1, 2, 3]
    big = next(g for g in groups if len(g) == 3)
    assert {u["rule"] for u in big} == {"E401", "F401"}


def test_non_overlapping_findings_not_grouped():
    """Findings far apart (line 1 vs line 20) are never grouped together."""
    findings = [
        {"rule": "A", "lines": [1], "occurrences": 1, "category": "Security"},
        {"rule": "B", "lines": [20], "occurrences": 1, "category": "Security"},
    ]
    groups = _group_overlapping(_explode_repeats(findings))

    assert len(groups) == 2
