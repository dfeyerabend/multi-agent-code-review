"""
Tests for MCP server tool functions.
Tests import and call the tool functions directly — no MCP server or agent needed.
"""

import json
import os
import pytest

from mcp_server import (
    read_code, detect_syntax_errors, extract_code_structure,
    _ruff_severity, _ruff_category, knowledge_search,
    generate_fix_suggestion,
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
    assert "out of range" in result["message"]


def test_generate_fix_suggestion_line_zero():
    """Returns error for finding_line=0 — lines are 1-based."""
    result = json.loads(generate_fix_suggestion("x = 1\n", finding_line=0))

    assert result["status"] == "error"


def test_generate_fix_suggestion_empty_code():
    """Returns error when code is an empty string."""
    result = json.loads(generate_fix_suggestion("", finding_line=1))

    assert result["status"] == "error"
    assert "empty" in result["message"]

