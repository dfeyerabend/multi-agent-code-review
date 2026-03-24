"""
MCP Server for the Code Review Agent pipeline.
Exposes code analysis tools that agents discover dynamically.
Tools: read_code, detect_syntax_errors, extract_code_structure
"""

import os
import ast
import sys
import json
import subprocess
import tempfile
from mcp.server.fastmcp import FastMCP

# Prevents Unicode from crashing the code
sys.stdout.reconfigure(encoding='utf-8')

# Only for debugging
from pprint import pprint

mcp = FastMCP(
    "code-review-mcp",
    instructions=(
        "MCP Server for code review. Provides tools to read, "
        "analyze, and extract structure from Python code."
    )
)

# --- TOOL 1: Read Code ---
@mcp.tool()
def read_code(source: str) -> str:
    """
    Reads code from a file path or accepts a raw code string.

    Args:
        source: Either a file path to a Python file, or a raw code string.
    """
    print(f"\n[read_code] Received input: {source[:80]}...")
    if os.path.isfile(source):
        try:
            print(f"[read_code] -> Detected FILE path: {source}")
            with open(source, "r", encoding = f"utf-8") as f:
                code = f.read()
            print(f"[read_code] ✓ File read successfully ({len(code.splitlines())} lines)")
            return json.dumps({
                "status": "success",
                "source_type": "file",
                "file_path": source,
                "code": code,
                "line_count": len(code.splitlines()), # useful for later agents
            }, indent = 2)
        except Exception as e:
            print(f"[read_code] -> Failed to read file: {str(e)}")
            return json.dumps({
                "status": "error",
                "message": f"Failed to read file: {str(e)}",
            }, indent = 2)

    # If not a file path -> treat as raw string of code
    print(f"[read_code] -> Detected RAW CODE string ({len(source.splitlines())} lines)")
    return json.dumps({
        "status": "success",
        "source_type": "raw_string",
        "code": source,
        "line_count": len(source.splitlines()),
    }, indent = 2)

# --- Helper: write code to temp file for CLI tools ---
def _write_temp_file(code: str) -> str:
    """
    Writes code to a temporary .py file, returns the file path.
    """

    tmp = tempfile.NamedTemporaryFile(      # create temp file that persists after close
        mode = "w",
        suffix = ".py",
        delete = False,
        encoding = "utf-8",
    )

    tmp.write(code)
    if not code.endswith("\n"):
        tmp.write("\n")                     # prevent false W292 from tempfile method
    tmp.close()                             # close so ruff/bandit can read it

    return tmp.name

# --- Helper: run a CLI tool and capture output ---
def _run_cli_tool(command: list[str]) -> dict:
    """
    Runs a CLI command, returns parsed JSON or error info.
    """

    tool_name = command[0]
    print(f"  [{tool_name}] Executing: {' '.join(command[:4])}...")

    try:
        result = subprocess.run(
            command,
            capture_output = True,                  # capture stdout and stderr
            text = True,                            # decode output as string
            timeout = 30,                           # fails if tool call does not work
        )

        print(f"  [{tool_name}] Exit code: {result.returncode}")

        # both ruff and bandit return JSON to stdout
        if result.stdout.strip():
            try:
                return {"status": "success", "data": json.loads(result.stdout)}
            except json.JSONDecodeError:
                return {"status": "success", "raw_output": result.stdout.strip()}

        # no stdout — might be clean (no issues) or an error
        if result.returncode == 0:
            return {"status": "success", "data": []}  # clean run, no issues found

    except FileNotFoundError:  # tool not installed or not in PATH
        tool_name = command[0]
        return {
            "status": "error",
            "message": f"'{tool_name}' not found. Install with: pip install {tool_name}"
        }
    except subprocess.TimeoutExpired:
        return {"status": "error", "message": "Tool timed out after 30 seconds"}



def _ruff_severity(rule_code: str) -> str:
    """
    Maps ruff rule codes to severity levels.
    This is a rough heuristic — ruff doesn't have built-in severity.
    """

    # S-rules are security related (bandit-equivalent rules in ruff)
    if rule_code.startswith("S"):                                       # expected output: "code": "S101" -> security concerns - similar to bandit
        return "HIGH"
    # E9xx are syntax errors, F-rules are pyflakes (logic errors)
    if rule_code.startswith("E9") or rule_code.startswith("F"):         # expected output: "code": "E902" -> syntax errors, or "F401" -> Pyflakes, logic errors
        return "HIGH"
    # C/W are complexity and warnings
    if rule_code.startswith("C") or rule_code.startswith("W"):          # expected output: "code": "C901" -> complexity, or "W291" -> warnings
        return "MEDIUM"
    # everything else (style, formatting)
    return "LOW"

# --- TOOL 2: Detect Syntax Errors ---
@mcp.tool()
def detect_syntax_errors(code: str) -> str:
    """Runs static analysis on Python code using ruff (code quality)
    and bandit (security). Returns structured findings with severity.

    Args:
        code: Python source code as a string.
    """
    print(f"\n[detect_syntax_errors] Received code ({len(code.splitlines())} lines)")
    print(f"[detect_syntax_errors] Writing to temp file...")
    tmp_path = _write_temp_file(code)                   # Always gets passed a code string, because read_code translates already into string
    print(f"[detect_syntax_errors] Temp file: {tmp_path}")

    try:
        print(f"[detect_syntax_errors] Running ruff (code quality)...")
        results = {
            "ruff": {"findings": [], "error": None},
            "bandit": {"findings": [], "error": None}
        }

        # --- Run ruff ---
        ruff_result = _run_cli_tool([
            "ruff", "check",
            "--output-format", "json",                  # structured JSON output
            "--select", "E,F,W,C90,B",                  # Defined selection to ensure that only real errors are reported:
            tmp_path                                    # E=errors, F=pyflakes, W=warnings, C90=complexity, B=bugbear
        ])                                              # No S (Security) because this is handled by bandit


        if ruff_result["status"] == "success" and "data" in ruff_result:
            for issue in ruff_result["data"]:  # each issue is a dict with code, message, location
                results["ruff"]["findings"].append({
                    "rule": issue.get("code", "unknown"),
                    "message": issue.get("message", ""),
                    "line": issue.get("location", {}).get("row"),
                    "column": issue.get("location", {}).get("column"),
                    "severity": _ruff_severity(issue.get("code", ""))
                })

            # only print details when there are actual findings
            if results["ruff"]["findings"]:
                print("\n" + "=" * 20 + "RUFF RESULTS" + "=" * 20)
                pprint(results["ruff"]["findings"], width=120)

            print(f"[detect_syntax_errors] -> Ruff: {len(results['ruff']['findings'])} findings")

        elif ruff_result["status"] == "error":
            print(f"[detect_syntax_errors] Ruff code analysis failed: {ruff_result['message']}")
            results["ruff"]["error"] = ruff_result["message"]

        # --- Run bandit ---
        print(f"[detect_syntax_errors] Running bandit (security)...")
        bandit_result = _run_cli_tool([
            "bandit",
            "-f", "json",  # structured JSON output
            "-q",  # quiet — suppress progress info
            tmp_path
        ])

        if bandit_result["status"] == "success" and "data" in bandit_result:
            bandit_data = bandit_result["data"]
            for issue in bandit_data.get("results", []):  # bandit nests findings under "results" key
                results["bandit"]["findings"].append({
                    "test_id": issue.get("test_id", ""),
                    "test_name": issue.get("test_name", ""),
                    "message": issue.get("issue_text", ""),
                    "line": issue.get("line_number"),
                    "severity": issue.get("issue_severity", "UNKNOWN"),
                    "confidence": issue.get("issue_confidence", "UNKNOWN")
                })

            # only print details when there are actual findings
            if results["bandit"]["findings"]:
                print("\n" + "=" * 20 + "BANDIT RESULTS" + "=" * 20)
                pprint(results["bandit"]["findings"], width=120)

            print(f"[detect_syntax_errors] -> Bandit: {len(results['bandit']['findings'])} findings")

        elif bandit_result["status"] == "error":
            print(f"[detect_syntax_errors] Bandit code analysis failed: {bandit_result['message']}")
            results["bandit"]["error"] = bandit_result["message"]

        # --- Summary ---
        total_findings = (
                len(results["ruff"]["findings"])
                + len(results["bandit"]["findings"])
        )
        print(f"[detect_syntax_errors] ✓ Analysis complete: {total_findings} total findings")

        return json.dumps({
            "status": "clean" if total_findings == 0 else "issues_found",
            "total_findings": total_findings,
            "ruff_findings": len(results["ruff"]["findings"]),
            "bandit_findings": len(results["bandit"]["findings"]),
            "results": results
        }, indent=2)

    finally:
        os.unlink(tmp_path)                             # always clean up temp file


# --- TOOL 3: Extract Code Structure ---
@mcp.tool()
def extract_code_structure(code: str) -> str:
    """
    Extracts functions, classes, and imports from Python code.
    Uses ast (Abstract Syntax Tree)

    Args:
        code: Python source code as a string.
    """

    print(f"\n[extract_code_structure] Received code ({len(code.splitlines())} lines)")

    # Fail fast
    try:
        tree = ast.parse(code)                                       # parse code into AST format (syntax tree)
    except SyntaxError as e:                                         # if this fails, then the code does not follow a valid python structure
        return json.dumps({
            "status": "error",
            "message": f"Cannot parse code: {e.msg} at line {e.lineno}"
        }, indent=2)

    # Walk the AST and collect structural elements
    functions = []
    classes = []
    imports = []

    print(f"[extract_code_structure] Parsing AST...")

    for node in ast.walk(tree):                                     # visit every node in the tree
        if isinstance(node, ast.FunctionDef):
            functions.append({
                "name": node.name,
                "line": node.lineno,                                # catch line number of problem
                "args": [arg.arg for arg in node.args.args],
                "has_docstring": (
                    isinstance(node.body[0], ast.Expr)                  # Three isinstance checks that all need to be TRUE for a clean docstring
                    and isinstance(node.body[0].value, ast.Constant)    # Is first element an expression?; Is second element a parameter?; is the konstant a string?
                    and isinstance(node.body[0].value.value, str)       # If all three are TRUE; THEN has_docstring ist True
                ) if node.body else False                               # Savety logic if node.body does not exist, then  FALSE (would crash otherwise)
            })

        elif isinstance(node, ast.ClassDef):                        # Logic for method nodes
            methods = [
                n.name for n in node.body
                if isinstance(n, ast.FunctionDef)
            ]
            classes.append({
                "name": node.name,
                "line": node.lineno,
                "methods": methods,
                "base_classes": [
                    getattr(base, "id", str(base))
                    for base in node.bases
                ]
            })

        elif isinstance(node, ast.Import):                           # Logic for imports
            for alias in node.names:
                imports.append({
                    "module": alias.name,
                    "alias": alias.asname
                })

        elif isinstance(node, ast.ImportFrom):
            for alias in node.names:
                imports.append({
                    "module": f"{node.module}.{alias.name}",
                    "alias": alias.asname
                })

    ## DEBUG PRINT
    print("\n" + "=" * 20 + "EXTRACT CODE RESULTS" + "=" * 20)
    print("Functions:")
    pprint(functions, width=120)
    print("Classes:")
    pprint(classes, width=120)
    print("Imports:")
    pprint(imports, width=120)
    print("=" * 60 + "\n")

    return json.dumps({
        "status": "success",
        "functions": functions,
        "classes": classes,
        "imports": imports,
        "summary": {
            "function_count": len(functions),
            "class_count": len(classes),
            "import_count": len(imports)
        }
    }, indent=2)


# --- Start the server ---
if __name__ == "__main__":
    mcp.run(transport="stdio")













