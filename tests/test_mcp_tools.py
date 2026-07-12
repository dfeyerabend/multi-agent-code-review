"""
Tests for MCP server tool functions.
Tests import and call the tool functions directly — no MCP server or agents needed.
"""

import ast
import json
import os
import pytest

from mcp_server import (
    read_code, detect_syntax_errors, extract_code_structure,
    knowledge_search, generate_fix_suggestion,
)
from mcp_helpers.scanners import _ruff_severity, _ruff_category
from mcp_helpers import company_rules
from mcp_helpers.company_rules import (
    _company_dotted_name,
    _company_check_naming,
    _company_check_comment,
    _company_check_raise,
    _company_check_access,
    _company_load_rules,
    _company_run_checks,
)

# Absolute path to mcp_server.py — used as a real file input in read_code tests.
# os.path.dirname(__file__) is the tests/ folder, so we go one level up.
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_MCP_SERVER_FILE = os.path.join(_PROJECT_ROOT, "mcp_server.py")

# ── read_code ─────────────────────────────────────────────────────────────────

def test_read_code_with_raw_string():
    """Returns success with source_type raw_string when given a code string."""
    result = json.loads(read_code("print('hello')"))

    assert result["status"] == "success"
    assert result["source_type"] == "raw_string"
    assert result["code"] == "print('hello')"
    assert result["line_count"] == 1

def test_read_code_with_file():
    """Returns success with source_type file when given a valid file path."""
    result = json.loads(read_code(_MCP_SERVER_FILE))    # used as always valid file path

    assert result["status"] == "success"
    assert result["source_type"] == "file"
    assert len(result["code"]) > 0
    assert result["line_count"] > 0

def test_read_code_with_nonexistent_path():
    """Returns an error when a .py path is provided but the file does not exist."""
    result = json.loads(read_code("nonexistent_file.py"))

    assert result["status"] == "error"
    assert "nonexistent_file.py" in result["message"]

# ── detect_syntax_errors ──────────────────────────────────────────────────────

def test_detect_syntax_errors_clean_code():
    """Returns clean status and zero findings for valid, issue-free code."""
    result = json.loads(detect_syntax_errors("def add(a, b):\n    return a + b\n"))

    assert result["status"] == "clean"
    assert result["total_findings"] == 0

def test_detect_syntax_errors_finds_unused_import():
    """Ruff reports F401 for an import that is never referenced."""
    code = "import os\n\ndef foo():\n    return 1\n"
    result = json.loads(detect_syntax_errors(code))

    assert result["status"] == "issues_found"
    rules = [f["rule"] for f in result["results"]["ruff"]["findings"]]
    assert "F401" in rules

def test_detect_syntax_errors_finds_security_issue():
    """Bandit reports a finding when eval() is called with user input."""
    code = "eval(input('Enter command: '))\n"
    result = json.loads(detect_syntax_errors(code))

    assert result["status"] == "issues_found"
    assert len(result["results"]["bandit"]["findings"]) > 0

# ── extract_code_structure ────────────────────────────────────────────────────

def test_extract_code_structure_happy_path():
    """Correctly counts functions, classes, and imports via AST parsing."""
    code = (
        "import os\n"
        "from pathlib import Path\n"
        "\n"
        "class FileProcessor:\n"
        "    def __init__(self, path):\n"
        "        self.path = path\n"
        "    def process(self):\n"
        "        pass\n"
        "\n"
        "def helper(x, y):\n"
        '    """A helper function."""\n'
        "    return x + y\n"
    )
    result = json.loads(extract_code_structure(code))

    assert result["status"] == "success"
    assert result["summary"]["function_count"] == 3   # __init__, process, helper
    assert result["summary"]["class_count"] == 1
    assert result["summary"]["import_count"] == 2

def test_extract_code_structure_syntax_error():
    """Returns error status without raising an exception for unparseable code."""
    result = json.loads(extract_code_structure("def foo(:\n    pass"))

    assert result["status"] == "error"
    assert "message" in result


def test_extract_code_structure_empty_code():
    """Returns success with zero counts for an empty code string."""
    result = json.loads(extract_code_structure(""))

    assert result["status"] == "success"
    assert result["summary"]["function_count"] == 0
    assert result["summary"]["class_count"] == 0
    assert result["summary"]["import_count"] == 0

# ── _ruff_severity ────────────────────────────────────────────────────────────

@pytest.mark.parametrize("rule_code, expected", [
    ("S101", "HIGH"),    # security rules
    ("E901", "HIGH"),    # syntax errors
    ("F401", "HIGH"),    # pyflakes logic errors
    ("C901", "MEDIUM"),  # complexity
    ("W291", "MEDIUM"),  # warnings
    ("E501", "LOW"),     # style
    ("XYZ",  "LOW"),     # unknown rule → safe default
])
def test_ruff_severity(rule_code, expected):
    """Maps ruff rule prefixes to the correct severity level."""
    assert _ruff_severity(rule_code) == expected


# ── _ruff_category ────────────────────────────────────────────────────────────

@pytest.mark.parametrize("rule_code, expected", [
    ("S101", "Security"),        # S-prefix
    ("E901", "Logic"),           # E9-prefix → syntax errors
    ("F401", "Logic"),           # F-prefix → pyflakes
    ("B006", "Logic"),           # B-prefix → bugbear
    ("C901", "Maintainability"), # C-prefix → complexity
    ("E501", "Style"),           # E-prefix (non-E9) → style
    ("W291", "Style"),           # W-prefix → style
    ("XYZ",  "Style"),           # unknown → safe default
])
def test_ruff_category(rule_code, expected):
    """Maps ruff rule prefixes to the correct category label."""
    assert _ruff_category(rule_code) == expected


# ── knowledge_search ──────────────────────────────────────────────────────────

def test_knowledge_search_returns_results_with_category(chroma_collection):
    """Returns at least one result when querying with a valid category filter."""
    result = json.loads(knowledge_search("unused import", category="Logic"))

    assert result["status"] == "success"
    assert len(result["results"]) > 0

def test_knowledge_search_returns_results_without_category(chroma_collection):
    """Returns results when no category filter is provided."""
    result = json.loads(knowledge_search("security exception handling"))

    assert result["status"] == "success"
    assert len(result["results"]) > 0

def test_knowledge_search_result_shape(chroma_collection):
    """Every returned chunk contains the required fields."""
    result = json.loads(knowledge_search("naming convention", category="Style"))

    for chunk in result["results"]:
        assert "text" in chunk
        assert "source" in chunk
        assert "section" in chunk
        assert "category" in chunk
        assert "distance" in chunk

def test_knowledge_search_nonsense_query_does_not_crash(chroma_collection):
    """ChromaDB returns nearest matches even for a meaningless query — no exception."""
    result = json.loads(knowledge_search("xqzjwplm12345"))

    assert result["status"] == "success"                            # closest match returned, not an error
    for chunk in result["results"]:
        assert chunk["distance"] > 1.0                              # no real match — all results are semantically distant


# ── generate_fix_suggestion ───────────────────────────────────────────────────

# Two-function module used across multiple tests.
_TWO_FUNC_CODE = (
    "def outer(x):\n"           # line 1
    "    y = x + 1\n"           # line 2
    "    def inner(z):\n"       # line 3
    "        return z * 2\n"    # line 4
    "    return inner(y)\n"     # line 5
    "\n"                        # line 6
    "def standalone(a, b):\n"   # line 7
    "    return a - b\n"        # line 8
)


def test_generate_fix_suggestion_happy_path_innermost():
    """Returns the innermost function when finding_line sits inside a nested function."""
    result = json.loads(generate_fix_suggestion(_TWO_FUNC_CODE, finding_line=4))

    assert result["status"] == "success"
    assert result["function_name"] == "inner"
    assert result["context_type"] == "function"
    assert result["start_line"] == 3
    assert result["end_line"] == 4


def test_generate_fix_suggestion_happy_path_outer():
    """Returns the outer function when finding_line is in outer but not inner."""
    result = json.loads(generate_fix_suggestion(_TWO_FUNC_CODE, finding_line=2))

    assert result["status"] == "success"
    assert result["function_name"] == "outer"
    assert result["context_type"] == "function"


def test_generate_fix_suggestion_happy_path_standalone():
    """Returns the correct function when finding_line is in a top-level function."""
    result = json.loads(generate_fix_suggestion(_TWO_FUNC_CODE, finding_line=8))

    assert result["status"] == "success"
    assert result["function_name"] == "standalone"
    assert result["start_line"] == 7
    assert result["end_line"] == 8


def test_generate_fix_suggestion_function_source_content():
    """function_source contains the actual lines of the matched function."""
    result = json.loads(generate_fix_suggestion(_TWO_FUNC_CODE, finding_line=8))

    assert "def standalone" in result["function_source"]
    assert "return a - b" in result["function_source"]


def test_generate_fix_suggestion_module_level_fallback():
    """Falls back to surrounding lines when finding_line is outside any function."""
    code = (
        "import os\n"
        "X = os.environ['KEY']\n"   # line 2 — module-level, no enclosing function
    )
    result = json.loads(generate_fix_suggestion(code, finding_line=2))

    assert result["status"] == "fallback"
    assert result["context_type"] == "surrounding_lines"
    assert result["function_name"] is None
    assert "fallback_reason" in result


def test_generate_fix_suggestion_syntax_error_fallback():
    """Falls back to surrounding lines when the code cannot be parsed."""
    bad_code = "def broken(:\n    pass\nmore_code = 1\n"
    result = json.loads(generate_fix_suggestion(bad_code, finding_line=3))

    assert result["status"] == "fallback"
    assert result["context_type"] == "surrounding_lines"
    assert result["function_name"] is None
    assert "SyntaxError" in result["fallback_reason"]


def test_generate_fix_suggestion_fallback_source_is_nonempty():
    """Fallback always returns a non-empty function_source so the Optimizer has something to work with."""
    code = "X = 1\nY = 2\nZ = 3\n"
    result = json.loads(generate_fix_suggestion(code, finding_line=2))

    assert result["status"] == "fallback"
    assert len(result["function_source"]) > 0


def test_generate_fix_suggestion_line_out_of_range():
    """Returns error when finding_line exceeds the total line count."""
    result = json.loads(generate_fix_suggestion("x = 1\n", finding_line=99))

    assert result["status"] == "error"
    assert "no usable line" in result["message"]


def test_generate_fix_suggestion_line_zero():
    """Returns error for finding_line=0 — lines are 1-based."""
    result = json.loads(generate_fix_suggestion("x = 1\n", finding_line=0))

    assert result["status"] == "error"


def test_generate_fix_suggestion_empty_code():
    """Returns error when code is an empty string."""
    result = json.loads(generate_fix_suggestion("", finding_line=1))

    assert result["status"] == "error"
    assert "empty" in result["message"]


# ── company rules: _company_dotted_name ───────────────────────────────────────

def test_company_dotted_name_simple_name():
    """A bare Name resolves to its identifier."""
    node = ast.parse("db", mode="eval").body
    assert _company_dotted_name(node) == "db"

def test_company_dotted_name_attribute_chain():
    """An Attribute chain resolves to a dotted string."""
    node = ast.parse("db.query", mode="eval").body
    assert _company_dotted_name(node) == "db.query"

def test_company_dotted_name_non_name_returns_none():
    """An expression that is not a plain Name/Attribute chain resolves to None."""
    node = ast.parse("a[0]", mode="eval").body   # Subscript has no static dotted name
    assert _company_dotted_name(node) is None


# ── company rules: _company_check_naming (rule 1.1) ───────────────────────────

_NAMING_RULE = {
    "id": "COMPANY-1.1",
    "severity": "MEDIUM",
    "category": "Maintainability",
    "message": "DB function must be prefixed",
    "params": {
        "trigger_calls": ["db.query", "db.execute", "db.delete"],
        "required_prefixes": ["db_fetch_", "db_save_", "db_delete_"],
    },
}

def test_company_check_naming_flags_unprefixed_db_function():
    """A DB-accessing function without the required prefix produces one finding."""
    code = "def get_user(uid):\n    return db.query('SELECT 1')\n"
    result = _company_check_naming(ast.parse(code), code.splitlines(), _NAMING_RULE)

    assert result["status"] == "success"
    assert len(result["findings"]) == 1
    assert result["findings"][0]["rule"] == "COMPANY-1.1"
    assert result["findings"][0]["lines"] == [1]

def test_company_check_naming_accepts_prefixed_db_function():
    """A DB function with a valid prefix produces no finding."""
    code = "def db_fetch_user(uid):\n    return db.query('SELECT 1')\n"
    result = _company_check_naming(ast.parse(code), code.splitlines(), _NAMING_RULE)

    assert result["status"] == "success"
    assert result["findings"] == []

def test_company_check_naming_ignores_non_db_function():
    """A function that never touches the DB is not judged by rule 1.1."""
    code = "def add(a, b):\n    return a + b\n"
    result = _company_check_naming(ast.parse(code), code.splitlines(), _NAMING_RULE)

    assert result["status"] == "success"
    assert result["findings"] == []

def test_company_check_naming_rejects_non_ast_tree():
    """A non-AST tree returns a structured error, not an exception."""
    result = _company_check_naming("not an ast", [], _NAMING_RULE)

    assert result["status"] == "error"
    assert "ast.AST" in result["message"]
    assert result["findings"] == []

def test_company_check_naming_rejects_missing_params():
    """A rule without trigger_calls returns a structured error naming the field."""
    rule = {"id": "COMPANY-1.1", "message": "x", "params": {"required_prefixes": ["db_"]}}
    result = _company_check_naming(ast.parse("x = 1\n"), ["x = 1"], rule)

    assert result["status"] == "error"
    assert "trigger_calls" in result["message"]


# ── company rules: _company_check_comment (rule 1.2) ──────────────────────────

_COMMENT_RULE = {
    "id": "COMPANY-1.2",
    "severity": "LOW",
    "category": "Maintainability",
    "message": "missing REASON comment",
    "params": {"marker": "# REASON:", "max_lines_below": 1},
}

def test_company_check_comment_flags_missing_marker():
    """A function without the marker on/below its def line produces a finding."""
    code = "def f():\n    return 1\n"
    result = _company_check_comment(ast.parse(code), code.splitlines(), _COMMENT_RULE)

    assert result["status"] == "success"
    assert len(result["findings"]) == 1
    assert result["findings"][0]["rule"] == "COMPANY-1.2"

def test_company_check_comment_accepts_marker_on_def_line():
    """The marker inline on the def line satisfies the rule."""
    code = "def f():  # REASON: needed\n    return 1\n"
    result = _company_check_comment(ast.parse(code), code.splitlines(), _COMMENT_RULE)

    assert result["status"] == "success"
    assert result["findings"] == []

def test_company_check_comment_accepts_marker_directly_below():
    """The marker on the line directly below the def satisfies the rule."""
    code = "def f():\n    # REASON: needed\n    return 1\n"
    result = _company_check_comment(ast.parse(code), code.splitlines(), _COMMENT_RULE)

    assert result["status"] == "success"
    assert result["findings"] == []

def test_company_check_comment_rejects_non_list_source_lines():
    """source_lines of the wrong type returns a structured error."""
    result = _company_check_comment(ast.parse("def f():\n    pass\n"), "notalist", _COMMENT_RULE)

    assert result["status"] == "error"
    assert "source_lines" in result["message"]

def test_company_check_comment_rejects_missing_marker_param():
    """A rule without a marker returns a structured error naming the field."""
    rule = {"id": "COMPANY-1.2", "message": "x", "params": {"max_lines_below": 1}}
    result = _company_check_comment(ast.parse("def f():\n    pass\n"), ["def f():", "    pass"], rule)

    assert result["status"] == "error"
    assert "marker" in result["message"]


# ── company rules: _company_check_raise (rule 1.3) ────────────────────────────

_RAISE_RULE = {
    "id": "COMPANY-1.3",
    "severity": "MEDIUM",
    "category": "Logic",
    "message": "no built-in exceptions",
    "params": {"forbidden": ["Exception", "ValueError", "FileNotFoundError"]},
}

def test_company_check_raise_flags_builtin_exception():
    """Raising a forbidden built-in exception produces a finding at the raise line."""
    code = "def f():\n    raise ValueError('x')\n"
    result = _company_check_raise(ast.parse(code), code.splitlines(), _RAISE_RULE)

    assert result["status"] == "success"
    assert len(result["findings"]) == 1
    assert result["findings"][0]["rule"] == "COMPANY-1.3"
    assert result["findings"][0]["lines"] == [2]

def test_company_check_raise_accepts_custom_exception():
    """Raising a class outside the forbidden list produces no finding."""
    code = "def f():\n    raise AppError('x')\n"
    result = _company_check_raise(ast.parse(code), code.splitlines(), _RAISE_RULE)

    assert result["status"] == "success"
    assert result["findings"] == []

def test_company_check_raise_ignores_bare_reraise():
    """A bare 'raise' (re-raise) has no class to name and is not flagged."""
    code = "try:\n    pass\nexcept Exception:\n    raise\n"
    result = _company_check_raise(ast.parse(code), code.splitlines(), _RAISE_RULE)

    assert result["status"] == "success"
    assert result["findings"] == []

def test_company_check_raise_rejects_missing_forbidden_param():
    """A rule without a forbidden list returns a structured error naming the field."""
    rule = {"id": "COMPANY-1.3", "message": "x", "params": {}}
    result = _company_check_raise(ast.parse("x = 1\n"), ["x = 1"], rule)

    assert result["status"] == "error"
    assert "forbidden" in result["message"]


# ── company rules: _company_check_access (rule 1.4) ───────────────────────────

_ACCESS_RULE = {
    "id": "COMPANY-1.4",
    "severity": "HIGH",
    "category": "Security",
    "message": "no direct env access",
    "params": {"targets": ["os.getenv", "os.environ"]},
}

def test_company_check_access_flags_os_getenv_call():
    """A call to os.getenv produces a finding."""
    code = "def f():\n    return os.getenv('KEY')\n"
    result = _company_check_access(ast.parse(code), code.splitlines(), _ACCESS_RULE)

    assert result["status"] == "success"
    assert len(result["findings"]) == 1
    assert result["findings"][0]["rule"] == "COMPANY-1.4"

def test_company_check_access_flags_os_environ_subscript():
    """A subscript on os.environ produces a finding."""
    code = "def f():\n    return os.environ['KEY']\n"
    result = _company_check_access(ast.parse(code), code.splitlines(), _ACCESS_RULE)

    assert result["status"] == "success"
    assert len(result["findings"]) == 1

def test_company_check_access_ignores_config_wrapper():
    """Access via a non-targeted call is not flagged."""
    code = "def f():\n    return Config.get('KEY')\n"
    result = _company_check_access(ast.parse(code), code.splitlines(), _ACCESS_RULE)

    assert result["status"] == "success"
    assert result["findings"] == []

def test_company_check_access_rejects_missing_targets_param():
    """A rule without a targets list returns a structured error naming the field."""
    rule = {"id": "COMPANY-1.4", "message": "x", "params": {}}
    result = _company_check_access(ast.parse("x = 1\n"), ["x = 1"], rule)

    assert result["status"] == "error"
    assert "targets" in result["message"]


# ── company rules: _company_load_rules ────────────────────────────────────────

def _write_rules_file(tmp_path, content):
    """Writes content to a temp company_rules.json and returns its path (test helper)."""
    p = tmp_path / "company_rules.json"
    p.write_text(content, encoding="utf-8")
    return str(p)

def test_company_load_rules_happy_path(tmp_path, monkeypatch):
    """Loads and returns the valid rules from a well-formed JSON file."""
    content = '{"version": 1, "rules": [{"id": "COMPANY-1.1", "mechanism": "naming_convention"}]}'
    monkeypatch.setattr(company_rules, "COMPANY_RULES_PATH", _write_rules_file(tmp_path, content))

    result = _company_load_rules()

    assert result["status"] == "success"
    assert len(result["rules"]) == 1
    assert result["rules"][0]["id"] == "COMPANY-1.1"

def test_company_load_rules_missing_file(tmp_path, monkeypatch):
    """Returns an error when the rule set file does not exist."""
    monkeypatch.setattr(company_rules, "COMPANY_RULES_PATH", str(tmp_path / "nope.json"))

    result = _company_load_rules()

    assert result["status"] == "error"
    assert "not found" in result["message"]

def test_company_load_rules_invalid_json(tmp_path, monkeypatch):
    """Returns an error when the file contains malformed JSON."""
    monkeypatch.setattr(company_rules, "COMPANY_RULES_PATH", _write_rules_file(tmp_path, "{not valid json"))

    result = _company_load_rules()

    assert result["status"] == "error"
    assert "invalid JSON" in result["message"]

def test_company_load_rules_bad_structure(tmp_path, monkeypatch):
    """Returns an error when the top-level object has no 'rules' list."""
    monkeypatch.setattr(company_rules, "COMPANY_RULES_PATH", _write_rules_file(tmp_path, '{"version": 1}'))

    result = _company_load_rules()

    assert result["status"] == "error"
    assert "must be an object" in result["message"]

def test_company_load_rules_skips_malformed_rule(tmp_path, monkeypatch):
    """Keeps valid rules and skips entries missing a string id/mechanism."""
    content = (
        '{"version": 1, "rules": ['
        '{"id": "COMPANY-1.1", "mechanism": "naming_convention"},'
        '{"mechanism": "no_id_here"},'
        '{"id": "COMPANY-1.3", "mechanism": "forbidden_raise"}'
        ']}'
    )
    monkeypatch.setattr(company_rules, "COMPANY_RULES_PATH", _write_rules_file(tmp_path, content))

    result = _company_load_rules()

    assert result["status"] == "success"
    assert [r["id"] for r in result["rules"]] == ["COMPANY-1.1", "COMPANY-1.3"]

def test_company_load_rules_empty_after_filtering(tmp_path, monkeypatch):
    """Returns an error when no usable rule remains after filtering."""
    content = '{"version": 1, "rules": [{"mechanism": "no_id"}]}'
    monkeypatch.setattr(company_rules, "COMPANY_RULES_PATH", _write_rules_file(tmp_path, content))

    result = _company_load_rules()

    assert result["status"] == "error"
    assert "no usable rule" in result["message"]


# ── company rules: _company_run_checks ────────────────────────────────────────

_RUN_SAMPLE = "def get_user(uid):\n    return db.query('SELECT 1')\n"

def test_company_run_checks_dispatches_to_mechanism():
    """A wired rule produces its mechanism's findings in the merged result."""
    rules = [{"id": "COMPANY-1.1", "mechanism": "naming_convention", "message": "prefix",
              "params": {"trigger_calls": ["db.query"], "required_prefixes": ["db_fetch_"]}}]
    result = _company_run_checks(ast.parse(_RUN_SAMPLE), _RUN_SAMPLE.splitlines(), rules)

    assert result["rule_errors"] == {}
    assert len(result["findings"]) == 1
    assert result["findings"][0]["rule"] == "COMPANY-1.1"

def test_company_run_checks_unknown_mechanism_records_error():
    """A rule referencing an unregistered mechanism becomes a rule_error, not a crash."""
    rules = [{"id": "COMPANY-9.9", "mechanism": "does_not_exist", "params": {}}]
    result = _company_run_checks(ast.parse("x = 1\n"), ["x = 1"], rules)

    assert result["findings"] == []
    assert "no registered mechanism" in result["rule_errors"]["COMPANY-9.9"]

def test_company_run_checks_mechanism_error_recorded():
    """A mechanism returning status error (bad params) is recorded per rule."""
    rules = [{"id": "COMPANY-1.1", "mechanism": "naming_convention", "message": "prefix", "params": {}}]
    result = _company_run_checks(ast.parse("x = 1\n"), ["x = 1"], rules)

    assert result["findings"] == []
    assert "COMPANY-1.1" in result["rule_errors"]

def test_company_run_checks_rules_not_a_list():
    """A non-list rule set is reported under the synthetic _runner key."""
    result = _company_run_checks(ast.parse("x = 1\n"), ["x = 1"], "not a list")

    assert result["findings"] == []
    assert "_runner" in result["rule_errors"]

def test_company_run_checks_isolates_crashing_mechanism(monkeypatch):
    """A mechanism that raises is caught per rule; siblings are unaffected."""
    def _boom(tree, source_lines, rule):
        raise RuntimeError("boom")

    monkeypatch.setitem(company_rules._COMPANY_CHECKS, "explode", _boom)
    rules = [{"id": "COMPANY-X", "mechanism": "explode", "params": {}}]
    result = _company_run_checks(ast.parse("x = 1\n"), ["x = 1"], rules)

    assert result["findings"] == []
    assert "crashed" in result["rule_errors"]["COMPANY-X"]

def test_company_run_checks_non_dict_result_recorded(monkeypatch):
    """A mechanism returning a non-dict is recorded as a rule_error."""
    monkeypatch.setitem(company_rules._COMPANY_CHECKS, "weird", lambda t, s, r: "not a dict")
    rules = [{"id": "COMPANY-Y", "mechanism": "weird", "params": {}}]
    result = _company_run_checks(ast.parse("x = 1\n"), ["x = 1"], rules)

    assert result["findings"] == []
    assert "non-dict" in result["rule_errors"]["COMPANY-Y"]

