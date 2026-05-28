"""
Tests for MCP server tool functions.
Tests import and call the tool functions directly — no MCP server or agent needed.
"""

import json
import os
import pytest

from mcp_server import read_code, detect_syntax_errors, extract_code_structure, _ruff_severity, _ruff_category

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