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
import chromadb
from config import CHROMA_DB_PATH
from mcp.server.fastmcp import FastMCP

# Setup Logging
import logging
logger = logging.getLogger(__name__)

# Prevents Unicode from crashing the code
sys.stdout.reconfigure(encoding='utf-8')

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

    Returns:
        JSON string with fields:
        - status: "success" or "error"
        - source_type: "file" or "raw_string"
        - code: the full source code as a string
        - line_count: number of lines in the code
        - file_path: the resolved file path (only present for file input)
    """
    logger.debug("read_code input (first 80 chars): %s", source[:80])

    looks_like_path = "\n" not in source and source.strip().endswith(".py")

    if looks_like_path:
        if os.path.isfile(source):
            try:
                logger.debug("Detected file path: %s", source)
                with open(source, "r", encoding = "utf-8") as f:
                    code = f.read()
                logger.info("File read successfully (%d lines)", len(code.splitlines()))
                return json.dumps({
                    "status": "success",
                    "source_type": "file",
                    "file_path": source,
                    "code": code,
                    "line_count": len(code.splitlines()), # useful for later agents
                }, indent = 2)
            except Exception as e:
                logger.error("Failed to read file: %s", str(e))
                return json.dumps({
                    "status": "error",
                    "message": f"Failed to read file: {str(e)}",
                }, indent = 2)
        else:
            logger.warning("File path provided but not found: %s", source)
            return json.dumps({
                "status": "error",
                "message": f"File not found: {source}",
            }, indent=2)

    # no .py extension or contains newlines → treat as raw code
    logger.debug("Detected raw code string (%d lines)", len(source.splitlines()))
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
    logger.debug("Executing: %s", " ".join(command[:4]))

    try:
        result = subprocess.run(
            command,
            capture_output = True,                  # capture stdout and stderr
            text = True,                            # decode output as string
            timeout = 30,                           # fails if tool call does not work
        )

        logger.debug("%s exit code: %d", tool_name, result.returncode)

        # both ruff and bandit return JSON to stdout
        if result.stdout.strip():
            try:
                return {"status": "success", "data": json.loads(result.stdout)}
            except json.JSONDecodeError:
                return {"status": "success", "raw_output": result.stdout.strip()}

        # no stdout — might be clean (no issues) or an error
        if result.returncode == 0:
            return {"status": "success", "data": []}  # clean run, no issues found

        # fallback return for non-zero exit
        return {
            "status": "error",
            "message": f"{tool_name} exited with code {result.returncode}: {result.stderr.strip() or 'no output'}"
        }

    except FileNotFoundError:  # tool not installed or not in PATH
        tool_name = command[0]
        return {
            "status": "error",
            "message": f"'{tool_name}' not found. Install with: pip install {tool_name}"
        }
    except subprocess.TimeoutExpired:
        return {"status": "error", "message": "Tool timed out after 30 seconds"}

# Help function to map ruff categories
def _ruff_category(rule_code: str) -> str:
    """Maps ruff rule prefixes to high-level categories."""
    if rule_code.startswith("S"): # Currently not used because bandit handels security
        return "Security"
    if rule_code.startswith("E9"):
        return "Logic"               # syntax errors
    if rule_code.startswith(("F", "B")):
        return "Logic"               # pyflakes + bugbear (likely bugs)
    if rule_code.startswith("C"):
        return "Maintainability"     # complexity
    if rule_code.startswith(("E", "W")):
        return "Style"
    return "Style"                   # safe default

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
    logger.info("detect_syntax_errors called (%d lines)", len(code.splitlines()))
    tmp_path = _write_temp_file(code)                   # Always gets passed a code string, because read_code translates already into string
    logger.debug("Wrote temp file: %s", tmp_path)

    try:
        logger.debug("Running ruff...")
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
                rule_code  = issue.get("code", "unknown")
                results["ruff"]["findings"].append({
                    "rule": rule_code ,
                    "tool": "ruff",
                    "message": issue.get("message", ""),
                    "line": issue.get("location", {}).get("row"),
                    "column": issue.get("location", {}).get("column"),
                    "severity": _ruff_severity(rule_code),
                    "category": _ruff_category(rule_code),
                    "doc_url": issue.get("url"),
                    "fix_suggestion": issue.get("fix"),
                })

            # only report details when there are actual findings
            if results["ruff"]["findings"]:
                logger.debug("Ruff findings: %s", results["ruff"]["findings"])

            logger.info("Ruff: %d findings", len(results["ruff"]["findings"]))

        elif ruff_result["status"] == "error":
            logger.warning("Ruff failed: %s", ruff_result["message"])
            results["ruff"]["error"] = ruff_result["message"]

        # --- Run bandit ---
        logger.debug("Running bandit...")
        bandit_result = _run_cli_tool([
            "bandit",
            "-f", "json",  # structured JSON output
            "-q",  # quiet — suppress progress info
            tmp_path
        ])

        if bandit_result["status"] == "success" and "data" in bandit_result:
            bandit_data = bandit_result["data"]
            for issue in bandit_data.get("results", []):
                cwe = issue.get("issue_cwe") or {}
                results["bandit"]["findings"].append({
                    "rule": issue.get("test_id", ""),  # umbenannt von test_id für Konsistenz
                    "tool": "bandit",
                    "test_name": issue.get("test_name", ""),
                    "message": issue.get("issue_text", ""),
                    "line": issue.get("line_number"),
                    "severity": issue.get("issue_severity", "UNKNOWN"),
                    "confidence": issue.get("issue_confidence", "UNKNOWN"),
                    "category": "Security", # bandit is always "Security"
                    "doc_url": issue.get("more_info"),
                    "cwe_id": cwe.get("id"),
                    "cwe_url": cwe.get("link"),
                })

            # only report details when there are actual findings
            if results["bandit"]["findings"]:
                logger.debug("Bandit findings: %s", results["bandit"]["findings"])

            logger.info("Bandit: %d findings", len(results["bandit"]["findings"]))

        elif bandit_result["status"] == "error":
            logger.warning("Bandit failed: %s", bandit_result["message"])
            results["bandit"]["error"] = bandit_result["message"]

        # --- Summary ---
        total_findings = (
                len(results["ruff"]["findings"])
                + len(results["bandit"]["findings"])
        )
        logger.info("Analysis complete: %d total findings", total_findings)

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

    logger.info("extract_code_structure called (%d lines)", len(code.splitlines()))

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

    logger.debug("Parsing AST...")

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

    logger.debug("Functions: %s", functions)
    logger.debug("Classes: %s", classes)
    logger.debug("Imports: %s", imports)

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

# --- TOOL 4: Knowledge Search (RAG) ---
@mcp.tool()
def knowledge_search(query: str, category: str = "", n_results: int = 3) -> str:
    """
    Searches the ChromaDB knowledge base for best-practice context.

    Pipeline: called by the Enricher Agent once per finding via MCP STDIO.
    chromadb is imported at module level so the server loads it on startup,
    not on the first tool call (lazy import causes an 8-minute stall under
    FastMCP's worker thread).

    Args:
        query:     Natural language search string, typically rule code + message.
        category:  Optional metadata filter — "Style", "Logic", "Maintainability", or "Security".
        n_results: Number of chunks to return (default 3).

    Returns:
        JSON string with a list of matching chunks, each containing text,
        source, section, category, and relevance distance.
    """
    logger.info("knowledge_search called | query: %s | category: %s", query, category)

    try:
        client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
        collection = client.get_collection("code_best_practices")

        query_kwargs = {
            "query_texts": [query],
            "n_results": n_results,
            "include": ["documents", "metadatas", "distances"],
        }
        if category:
            query_kwargs["where"] = {"category": category}

        results = collection.query(**query_kwargs)

        chunks = []
        documents = results.get("documents", [[]])[0]   # ChromaDB nests one layer per query — [0] unwraps it
        metadatas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]

        for doc, meta, dist in zip(documents, metadatas, distances):
            chunks.append({
                "text": doc,
                "source": meta.get("source"),
                "section": meta.get("section"),
                "category": meta.get("category"),
                "distance": round(dist, 4),
            })

        logger.info("knowledge_search: %d chunks returned", len(chunks))
        logger.debug("knowledge_search results: %s", chunks)

        return json.dumps({
            "status": "success",
            "query": query,
            "category_filter": category or None,
            "results": chunks,
        }, indent=2)

    except Exception as e:
        logger.error("knowledge_search failed: %s", str(e))
        return json.dumps({
            "status": "error",
            "message": str(e),
        }, indent=2)

@mcp.tool()
def generate_fix_suggestion(code: str, finding_line: int) -> str:
    """
        Extracts the function source that contains a given finding line.

        Pipeline: called by the Optimizer Agent once per finding, before generating
        a fix. Returns the narrowest possible context so the Optimizer works on
        real code, not just the flagged line.

        The tool never crashes the pipeline. When full context cannot be extracted
        (syntax error, line outside any function) it falls back to surrounding lines
        and sets status="fallback" so the Evaluator can flag limited-context fixes.

        Args:
            code:           Python source code — either a full file or a raw snippet.
            finding_line:   1-based line number from the finding.

        Returns:
            JSON string with fields:
            - status:           "success" | "fallback" | "error"
            - function_name:    Name of the enclosing function, or null on fallback.
            - function_source:  Extracted source lines as a single string.
            - start_line:       1-based index of the first returned line.
            - end_line:         1-based index of the last returned line.
            - context_type:     "function" | "surrounding_lines"
            - fallback_reason:  Present only when status="fallback".
    """

    logger.info("generate_fix_suggestion called | finding_line=%d", finding_line)

    # --- Guard: empty code ---
    if not code or not code.strip():
        logger.warning("generate_fix_suggestion: empty code received")
        return json.dumps({
            "status": "error",
            "message": "code must not be empty.",
        }, indent=2)

    lines = code.splitlines()
    total_lines = len(lines)

    # --- Guard: line out of range ---
    if finding_line < 1 or finding_line > total_lines:
        logger.warning(
            "generate_fix_suggestion: finding_line=%d out of range (1–%d)",
            finding_line, total_lines,
        )
        return json.dumps({
            "status": "error",
            "message": (
                f"finding_line {finding_line} is out of range "
                f"(file has {total_lines} lines)."
            ),
        }, indent=2)

    # --- Helper: surrounding-lines fallback ---
    # Defined here so it closes over `lines`, `total_lines`, and `finding_line`
    def _surrounding_lines(reason: str) -> str:
        n = 5
        start = max(1, finding_line - n)
        end = min(total_lines, finding_line + n)
        snippet = "\n".join(lines[start - 1: end])
        logger.info(
            "generate_fix_suggestion fallback: %s | lines %d–%d", reason, start, end
        )
        return json.dumps({
            "status": "fallback",
            "fallback_reason": reason,
            "function_name": None,
            "function_source": snippet,
            "start_line": start,
            "end_line": end,
            "context_type": "surrounding_lines",
        }, indent=2)

    # --- Happy path: AST parse ---
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        # Code has a syntax error — surrounding lines is the best we can do
        return _surrounding_lines(f"SyntaxError at line {e.lineno}: {e.msg}")

    # Walk all FunctionDef nodes and keep those whose line range spans finding_line
    candidates = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if node.lineno <= finding_line <= node.end_lineno:
                candidates.append(node)

    if not candidates:
        return _surrounding_lines("finding_line is not inside any function")

    # Innermost function = smallest line range (nested functions have smaller spans)
    innermost = min(candidates, key=lambda n: n.end_lineno - n.lineno)

    start = innermost.lineno
    end = innermost.end_lineno
    snippet = "\n".join(lines[start - 1: end])

    logger.info(
        "generate_fix_suggestion: function '%s' lines %d–%d",
        innermost.name, start, end,
    )
    return json.dumps({
        "status": "success",
        "function_name": innermost.name,
        "function_source": snippet,
        "start_line": start,
        "end_line": end,
        "context_type": "function",
    }, indent=2)

# --- TOOL 5: Create Review Report ---
@mcp.tool()
def create_review_report(fixes_evaluated: list, summary: str) -> str:
    """
    Generates a human-readable markdown review report from evaluated fixes.

    Pipeline: called once by the Evaluator Agent after scoring all fixes,
    before submit_evaluation.

    Args:
        fixes_evaluated: List of evaluated fix dicts, each with finding_rule,
                         finding_line, status, score, reasoning, suggested_code,
                         and grounded_in.
        summary:         Overall summary string produced by the Evaluator.

    Returns:
        JSON string with status and a markdown-formatted report string.
    """
    logger.info("create_review_report called | %d fix(es)", len(fixes_evaluated))

    if not isinstance(fixes_evaluated, list):
        return json.dumps({
            "status": "error",
            "message": f"create_review_report failed — fixes_evaluated must be a list, got {type(fixes_evaluated).__name__}",
        }, indent=2)

    if not isinstance(summary, str):
        return json.dumps({
            "status": "error",
            "message": f"create_review_report failed — summary must be a string, got {type(summary).__name__}",
        }, indent=2)

    try:
        status_emoji = {"APPROVED": "✅", "NEEDS_REVISION": "⚠️", "UNRESOLVABLE": "❌"}

        approved       = [f for f in fixes_evaluated if f.get("status") == "APPROVED"]
        needs_revision = [f for f in fixes_evaluated if f.get("status") == "NEEDS_REVISION"]
        unresolvable   = [f for f in fixes_evaluated if f.get("status") == "UNRESOLVABLE"]

        lines = [
            "# Code Review Report",
            "",
            f"**{summary}**",
            "",
            "| Status | Count |",
            "|--------|-------|",
            f"| ✅ Approved       | {len(approved)} |",
            f"| ⚠️ Needs Revision | {len(needs_revision)} |",
            f"| ❌ Unresolvable   | {len(unresolvable)} |",
            "",
            "---",
            "",
        ]

        for i, fix in enumerate(fixes_evaluated):
            if not isinstance(fix, dict):
                return json.dumps({
                    "status": "error",
                    "message": f"create_review_report failed — fixes_evaluated[{i}] must be a dict, got {type(fix).__name__}",
                }, indent=2)

            rule           = fix.get("finding_rule", "?")
            line           = fix.get("finding_line", "?")
            status         = fix.get("status", "?")
            score          = fix.get("score", "?")
            reasoning      = fix.get("reasoning", "")
            suggested_code = fix.get("suggested_code", "")
            grounded_in    = fix.get("grounded_in", [])
            emoji          = status_emoji.get(status, "•")

            if grounded_in and not isinstance(grounded_in, list):
                return json.dumps({
                    "status": "error",
                    "message": f"create_review_report failed — grounded_in for rule {rule!r} line {line} must be a list, got {type(grounded_in).__name__}",
                }, indent=2)

            lines += [
                f"## {emoji} Rule `{rule}` — Line {line}",
                f"**Status:** {status} | **Score:** {score}/5",
                "",
                f"**Reasoning:** {reasoning}",
                "",
            ]

            if grounded_in:
                lines += [
                    f"**Grounded in:** {', '.join(grounded_in)}",
                    "",
                ]

            if suggested_code:
                lines += [
                    "**Suggested fix:**",
                    "```python",
                    suggested_code,
                    "```",
                    "",
                ]

            lines.append("---")
            lines.append("")

        report = "\n".join(lines)
        logger.info("create_review_report: report generated (%d chars)", len(report))

        return json.dumps({"status": "success", "report": report}, indent=2)

    except Exception as e:
        logger.error("create_review_report failed unexpectedly: %s", str(e))
        return json.dumps({
            "status": "error",
            "message": f"create_review_report failed unexpectedly: {str(e)}",
        }, indent=2)





# --- Start the server ---
if __name__ == "__main__":
    from config import setup_logging
    setup_logging()                     # configure root logger once before server starts

    mcp.run(transport="stdio")













