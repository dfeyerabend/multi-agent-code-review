"""
MCP Server for the Code Review Agent pipeline.
Exposes code analysis tools that agents discover dynamically.
Tools: read_code, detect_syntax_errors, extract_code_structure
"""

import os
import ast
import sys
import json
import chromadb
from config import CHROMA_DB_PATH
from mcp.server.fastmcp import FastMCP
from mcp_helpers.scanners import _write_temp_file, _run_cli_tool, _ruff_category, _ruff_severity
from mcp_helpers.snippets import _enclosing_snippet
from mcp_helpers.company_rules import _company_load_rules, _company_run_checks

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

    Pipeline: first tool called by the Analyzer Agent. Its output is assembled
    into analysis_results by _assemble_analysis (agents/analyzer_agent.py).

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
    # source is model-supplied and drives os.path + string ops below; a non-string
    # would raise uncaught before any logic runs, so reject it loudly first.
    if not isinstance(source, str):
        logger.error("read_code: source must be a str, got %s", type(source).__name__)
        return json.dumps({
            "status": "error",
            "message": f"read_code failed — source must be a string, got {type(source).__name__}",
        }, indent=2)

    try:
        logger.debug("read_code input (first 80 chars): %s", source[:80])

        looks_like_path = "\n" not in source and source.strip().endswith(".py")

        if looks_like_path:
            if os.path.isfile(source):
                try:
                    logger.debug("Detected file path: %s", source)
                    with open(source, "r", encoding="utf-8") as f:
                        code = f.read()
                    logger.info("File read successfully (%d lines)", len(code.splitlines()))
                    return json.dumps({
                        "status": "success",
                        "source_type": "file",
                        "file_path": source,
                        "code": code,
                        "line_count": len(code.splitlines()),  # useful for later agents
                    }, indent=2)
                except Exception as e:
                    logger.error("Failed to read file %s: %s", source, str(e))
                    return json.dumps({
                        "status": "error",
                        "message": f"Failed to read file {source}: {str(e)}",
                    }, indent=2)
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
        }, indent=2)

    except Exception as e:
        logger.error("read_code failed unexpectedly: %s", str(e))
        return json.dumps({
            "status": "error",
            "message": f"read_code failed unexpectedly: {str(e)}",
        }, indent=2)


# --- TOOL 2: Detect Syntax Errors ---
@mcp.tool()
def detect_syntax_errors(code: str) -> str:
    """Runs static analysis on Python code using ruff (code quality)
    and bandit (security). Returns structured findings with severity.

    Pipeline: MCP tool called by the Analyzer agent. Its output is assembled
    into analysis_results by _assemble_analysis (agents/analyzer_agent.py).

    Args:
        code: Python source code as a string.

    Returns:
        JSON string. status is "clean" (no findings, both tools ran),
        "issues_found" (findings present, both tools ran), or "partial"
        (at least one tool failed — scan is incomplete and must not be
        trusted as clean). tool_errors carries the per-tool failure messages.
    """
    # code is model-supplied and feeds splitlines() + the scanners; reject a
    # non-string before touching it, and before a temp file is ever created.
    if not isinstance(code, str):
        logger.error("detect_syntax_errors: code must be a str, got %s", type(code).__name__)
        return json.dumps({
            "status": "error",
            "message": f"detect_syntax_errors failed — code must be a string, got {type(code).__name__}",
        }, indent=2)

    logger.info("detect_syntax_errors called (%d lines)", len(code.splitlines()))

    tmp_path = _write_temp_file(code)
    if tmp_path is None:  # write failed and already logged — nothing to clean up
        return json.dumps({
            "status": "error",
            "message": "detect_syntax_errors failed — could not write temp file for ruff/bandit (see logs).",
        }, indent=2)
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
                rule_code = issue.get("code", "unknown")
                results["ruff"]["findings"].append({
                    "rule": rule_code,
                    "tool": "ruff",
                    "message": issue.get("message", ""),
                    "line": issue.get("location", {}).get("row"),
                    "column": issue.get("location", {}).get("column"),
                    "severity": _ruff_severity(rule_code),
                    "category": _ruff_category(rule_code),
                    "doc_url": issue.get("url"),
                    "fix_suggestion": issue.get("fix"),
                })

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
                    "rule": issue.get("test_id", ""),  # normalized from test_id to match ruff's 'rule' key
                    "tool": "bandit",
                    "test_name": issue.get("test_name", ""),
                    "message": issue.get("issue_text", ""),
                    "line": issue.get("line_number"),
                    "severity": issue.get("issue_severity", "UNKNOWN"),
                    "confidence": issue.get("issue_confidence", "UNKNOWN"),
                    "category": "Security",  # bandit is always "Security"
                    "doc_url": issue.get("more_info"),
                    "cwe_id": cwe.get("id"),
                    "cwe_url": cwe.get("link"),
                })

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

        # A failed tool makes the scan incomplete: a timed-out scanner with zero findings
        # must NOT be indistinguishable from a genuinely clean scan, or a real issue
        # (e.g. a missed security finding) silently disappears downstream.
        tool_errors = {
            tool: results[tool]["error"]
            for tool in ("ruff", "bandit")
            if results[tool]["error"] is not None
        }

        if tool_errors:
            status = "partial"
        elif total_findings == 0:
            status = "clean"
        else:
            status = "issues_found"

        logger.info("Analysis complete: %d findings, status=%s", total_findings, status)
        if tool_errors:
            logger.warning("detect_syntax_errors: incomplete scan — tool_errors: %s", tool_errors)

        return json.dumps({
            "status": status,
            "total_findings": total_findings,
            "ruff_findings": len(results["ruff"]["findings"]),
            "bandit_findings": len(results["bandit"]["findings"]),
            "tool_errors": tool_errors,             # empty dict when both tools ran cleanly
            "results": results
        }, indent=2)

    except Exception as e:
        logger.error("detect_syntax_errors failed unexpectedly: %s", str(e))
        return json.dumps({
            "status": "error",
            "message": f"detect_syntax_errors failed unexpectedly: {str(e)}",
        }, indent=2)

    finally:
        # Cleanup must never crash the call — the file may already be gone.
        try:
            os.unlink(tmp_path)
        except OSError as e:
            logger.warning("detect_syntax_errors: could not remove temp file %s: %s", tmp_path, str(e))


# --- TOOL 3: Extract Code Structure ---
@mcp.tool()
def extract_code_structure(code: str) -> str:
    """
    Extracts functions, classes, and imports from Python code using ast.

    Pipeline: MCP tool called by the Analyzer agent. Its output feeds the
    'structure' field assembled by _assemble_analysis (agents/analyzer_agent.py).

    Args:
        code: Python source code as a string.

    Returns:
        JSON string with status, functions, classes, imports, and a summary
        of counts, or a structured error if the code cannot be parsed.
    """
    if not isinstance(code, str):
        logger.error("extract_code_structure: code must be a str, got %s", type(code).__name__)
        return json.dumps({
            "status": "error",
            "message": f"extract_code_structure failed — code must be a string, got {type(code).__name__}",
        }, indent=2)

    logger.info("extract_code_structure called (%d lines)", len(code.splitlines()))

    try:
        # Fail fast: invalid Python can't be walked — report the parse location.
        try:
            tree = ast.parse(code)
        except SyntaxError as e:
            return json.dumps({
                "status": "error",
                "message": f"Cannot parse code: {e.msg} at line {e.lineno}"
            }, indent=2)

        functions = []
        classes = []
        imports = []

        logger.debug("Parsing AST...")

        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                # has_docstring requires three conditions to all hold; node.body can be
                # empty, which would raise on body[0] — guard that with the trailing else.
                functions.append({
                    "name": node.name,
                    "line": node.lineno,
                    "args": [arg.arg for arg in node.args.args],
                    "has_docstring": (
                        isinstance(node.body[0], ast.Expr)
                        and isinstance(node.body[0].value, ast.Constant)
                        and isinstance(node.body[0].value.value, str)
                    ) if node.body else False,
                })

            elif isinstance(node, ast.ClassDef):
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

            elif isinstance(node, ast.Import):
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

    except Exception as e:
        logger.error("extract_code_structure failed unexpectedly: %s", str(e))
        return json.dumps({
            "status": "error",
            "message": f"extract_code_structure failed unexpectedly: {str(e)}",
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


# --- TOOL 5: Check Company Rules ---
@mcp.tool()
def check_company_rules(code: str) -> str:
    """
    Runs the company-specific coding rules against Python code (AST-based).

    Pipeline: MCP tool called by the Analyzer agent, alongside detect_syntax_errors —
    the company-rule counterpart to the ruff/bandit scan. Loads the rule set from
    company_rules.json, runs each rule's mechanism against the code, and returns the
    findings in the same schema. Its output is assembled into analysis_results by
    _assemble_analysis (agents/analyzer_agent.py).

    Args:
        code: Python source code as a string.

    Returns:
        JSON string. status is "clean" (no violations, all rules ran), "issues_found"
        (violations present), "partial" (at least one rule failed — result incomplete
        and must not be trusted as clean), or "error" (code unparseable or rule set
        unavailable). rule_errors carries the per-rule failure messages.
    """
    # model-supplied input feeds ast.parse/splitlines — reject a non-string up front
    if not isinstance(code, str):
        logger.error("check_company_rules: code must be a str, got %s", type(code).__name__)
        return json.dumps({
            "status": "error",
            "message": f"check_company_rules failed — code must be a string, got {type(code).__name__}",
        }, indent=2)

    logger.info("check_company_rules called (%d lines)", len(code.splitlines()))

    try:
        # AST-based check: unparseable code fails here; the syntax error itself is reported by detect_syntax_errors
        try:
            tree = ast.parse(code)
        except SyntaxError as e:
            return json.dumps({
                "status": "error",
                "message": f"check_company_rules — cannot parse code: {e.msg} at line {e.lineno}",
            }, indent=2)

        rules_result = _company_load_rules()        # mcp_helpers.company_rules: reads + validates company_rules.json
        if rules_result["status"] != "success":
            return json.dumps({
                "status": "error",
                "message": f"check_company_rules — rule set unavailable: {rules_result['message']}",
            }, indent=2)

        # mcp_helpers.company_rules: dispatches each rule to its mechanism and isolates per-rule failures
        run = _company_run_checks(tree, code.splitlines(), rules_result["rules"])
        findings = run["findings"]
        rule_errors = run["rule_errors"]

        if rule_errors:
            status = "partial"                      # a failed rule leaves the run incomplete, not clean
        elif findings:
            status = "issues_found"
        else:
            status = "clean"

        logger.info("check_company_rules complete: %d finding(s), status=%s", len(findings), status)
        if rule_errors:
            logger.warning("check_company_rules: incomplete run — rule_errors: %s", rule_errors)

        return json.dumps({
            "status": status,
            "total_findings": len(findings),
            "rule_errors": rule_errors,             # empty dict when every rule ran cleanly
            "findings": findings,
        }, indent=2)

    except Exception as e:
        logger.error("check_company_rules failed unexpectedly: %s", str(e))
        return json.dumps({
            "status": "error",
            "message": f"check_company_rules failed unexpectedly: {str(e)}",
        }, indent=2)


# --- TOOL 6: Generate Fix Suggestion ---
@mcp.tool()
def generate_fix_suggestion(code: str, finding_line: int) -> str:
    """
    Extracts the function source that contains a given finding line.

    Pipeline: called by the Optimizer Agent once per finding, before generating a
    fix, to work on real surrounding code rather than the flagged line alone. Thin
    wrapper over _enclosing_snippet (this file), which does the AST scoping and
    line numbering; this layer only validates the tool's single-line contract and
    serialises the result.

    The tool never crashes the pipeline. When full context cannot be extracted
    (syntax error, line outside any function) _enclosing_snippet falls back to
    surrounding lines and sets status="fallback" so the Optimizer can flag
    limited-context fixes.

    Args:
        code:         Python source — either a full file or a raw snippet.
        finding_line: 1-based line number from the finding.

    Returns:
        JSON string with the fields from _enclosing_snippet: status, function_name,
        function_source, numbered_source, start_line, end_line, context_type, and
        (on fallback) fallback_reason — or status="error" with a message.
    """
    # finding_line is model-supplied and is the tool's whole contract; validate it
    # here so the caller gets a single-line-specific message, then delegate the rest.
    if not isinstance(finding_line, int) or isinstance(finding_line, bool):
        logger.error("generate_fix_suggestion: finding_line must be an int, got %s", type(finding_line).__name__)
        return json.dumps({
            "status": "error",
            "message": f"generate_fix_suggestion failed — finding_line must be an integer, got {type(finding_line).__name__}",
        }, indent=2)

    logger.info("generate_fix_suggestion called | finding_line=%d", finding_line)

    # code-type, empty, out-of-range and AST handling all live in the shared helper —
    # a single line is just the one-element union case.
    result = _enclosing_snippet(code, [finding_line])
    return json.dumps(result, indent=2)


# --- Start the server ---
if __name__ == "__main__":
    from config import setup_logging
    setup_logging()                     # configure root logger once before server starts

    mcp.run(transport="stdio")
