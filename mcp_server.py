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

# --- Helper: write code to temp file for CLI tools ---
def _write_temp_file(code: str) -> str | None:
    """
    Writes code to a temporary .py file, returns the file path.

    Pipeline: helper used by detect_syntax_errors before invoking ruff and bandit.

    Args:
        code: Python source code to write.

    Returns:
        The temp file path on success, or None on failure — the caller treats
        None as a hard failure and reports it (this helper never raises).
    """
    if not isinstance(code, str):
        logger.error("_write_temp_file: code must be a str, got %s", type(code).__name__)
        return None

    try:
        tmp = tempfile.NamedTemporaryFile(      # persists after close so ruff/bandit can read it
            mode="w",
            suffix=".py",
            delete=False,
            encoding="utf-8",
        )
        tmp.write(code)
        if not code.endswith("\n"):
            tmp.write("\n")                     # prevent false W292 from the tempfile method
        tmp.close()                             # close so ruff/bandit can read it
        return tmp.name

    except Exception as e:
        logger.error("_write_temp_file failed unexpectedly: %s", str(e))
        return None

# --- Helper: run a CLI tool and capture output ---
def _run_cli_tool(command: list[str]) -> dict:
    """
    Runs a CLI command in a detached subprocess, returns parsed JSON or error info.

    Pipeline: helper used by detect_syntax_errors to invoke ruff and bandit.
    Runs inside the MCP server process, which itself is a STDIO child of the agent.

    Args:
        command: Full argument vector, e.g. ["bandit", "-f", "json", "-q", path].

    Returns:
        dict with status "success" plus "data"/"raw_output", or status "error"
        with a message naming the tool and the likely cause.
    """
    if not isinstance(command, list) or not command:
        logger.error("_run_cli_tool: command must be a non-empty list, got %r", command)
        return {
            "status": "error",
            "message": f"_run_cli_tool failed — command must be a non-empty list, got {type(command).__name__}",
        }

    tool_name = command[0]
    logger.debug("Executing: %s", " ".join(command[:4]))

    try:
        result = subprocess.run(
            command,
            capture_output=True,                    # capture stdout and stderr
            text=True,                              # decode output as string
            timeout=30,                             # fails if tool call does not work
            stdin=subprocess.DEVNULL,               # detach the inherited JSON-RPC pipe: bandit's Python startup blocks on it otherwise
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
        return {
            "status": "error",
            "message": f"'{tool_name}' not found. Install with: pip install {tool_name}"
        }
    except subprocess.TimeoutExpired:
        return {
            "status": "error",
            "message": f"{tool_name} timed out after 30 seconds",  # name the tool so a degraded scan is traceable
        }
    except Exception as e:
        logger.error("_run_cli_tool: %s failed unexpectedly: %s", tool_name, str(e))
        return {
            "status": "error",
            "message": f"{tool_name} failed unexpectedly: {str(e)}",
        }

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

        rules_result = _company_load_rules()        # this file: reads + validates company_rules.json
        if rules_result["status"] != "success":
            return json.dumps({
                "status": "error",
                "message": f"check_company_rules — rule set unavailable: {rules_result['message']}",
            }, indent=2)

        # this file: dispatches each rule to its mechanism and isolates per-rule failures
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


def _company_load_rules() -> dict:
    """
    Loads and validates the company rule set from knowledge_base/company_rules.json.

    Pipeline: helper called by the check_company_rules tool (this file) at the start of
    every run, before any code is checked. The path is resolved relative to this file so
    it works on any clone, mirroring create_database.py's __file__-based lookup.

    Individual malformed rules are skipped with a warning so one bad entry cannot disable
    the whole set; only a missing file, invalid JSON, or a broken top-level structure
    fails the load outright.

    Returns:
        dict with "status": "success" and "rules" (list of validated rule dicts), or
        "status": "error" and "message" naming what failed and the likely cause.
    """
    rules_path = os.path.join(os.path.dirname(__file__), "knowledge_base", "company_rules.json")

    try:
        if not os.path.isfile(rules_path):
            logger.error("_company_load_rules: rule set file not found at %s", rules_path)
            return {"status": "error",
                    "message": f"_company_load_rules failed — rule set file not found at {rules_path}"}

        with open(rules_path, "r", encoding="utf-8") as f:
            raw = f.read()

        # invalid JSON is a config error, not a code problem — report it with the parse detail
        try:
            data = json.loads(raw)
        except json.JSONDecodeError as e:
            logger.error("_company_load_rules: invalid JSON in %s: %s", rules_path, e)
            return {"status": "error",
                    "message": f"_company_load_rules failed — invalid JSON in {rules_path}: {e}"}

        if not isinstance(data, dict) or not isinstance(data.get("rules"), list):
            logger.error("_company_load_rules: bad structure in %s (expected object with a 'rules' list)", rules_path)
            return {"status": "error",
                    "message": f"_company_load_rules failed — {rules_path} must be an object with a 'rules' list"}

        # keep only rules the runner can dispatch: string id (for reporting) + string mechanism (for lookup)
        valid_rules = []
        for i, rule in enumerate(data["rules"]):
            if not isinstance(rule, dict) or not isinstance(rule.get("id"), str) or not isinstance(rule.get("mechanism"), str):
                logger.warning("_company_load_rules: skipping malformed rule at index %d (needs string 'id' and 'mechanism'): %r", i, rule)
                continue
            valid_rules.append(rule)

        if not valid_rules:
            logger.error("_company_load_rules: no usable rule in %s", rules_path)
            return {"status": "error",
                    "message": f"_company_load_rules failed — no usable rule found in {rules_path}"}

        logger.info("_company_load_rules: loaded %d rule(s) from %s", len(valid_rules), rules_path)
        return {"status": "success", "rules": valid_rules}

    except Exception as e:
        logger.error("_company_load_rules failed unexpectedly for %s: %s", rules_path, str(e))
        return {"status": "error",
                "message": f"_company_load_rules failed unexpectedly reading {rules_path}: {str(e)}"}


def _company_run_checks(tree: ast.AST, source_lines: list, rules: list) -> dict:
    """
    Runs every company rule against the parsed code and collects the results.

    Pipeline: helper called by the check_company_rules tool (this file) after the code is
    parsed and the rule set is loaded. Dispatches each rule to its mechanism via the
    _COMPANY_CHECKS registry (this file), isolating per-rule failures so one broken rule
    cannot stop the others.

    Args:
        tree:         Parsed AST of the reviewed code.
        source_lines: Code split into lines, forwarded to every mechanism.
        rules:        Validated rule dicts from _company_load_rules.

    Returns:
        dict with "findings" (list of standard-schema finding dicts, all rules merged)
        and "rule_errors" (dict of rule id → error message for rules that failed or had
        no registered mechanism).
    """
    findings = []
    rule_errors = {}

    try:
        if not isinstance(rules, list):
            logger.error("_company_run_checks: 'rules' must be a list, got %s", type(rules).__name__)
            rule_errors["_runner"] = f"rule set must be a list, got {type(rules).__name__}"
            return {"findings": findings, "rule_errors": rule_errors}

        for rule in rules:
            rule_id = rule.get("id", "<unknown>") if isinstance(rule, dict) else "<unknown>"
            mechanism_name = rule.get("mechanism") if isinstance(rule, dict) else None

            mechanism = _COMPANY_CHECKS.get(mechanism_name)
            if mechanism is None:
                logger.warning("_company_run_checks: rule %s has no registered mechanism '%s'", rule_id, mechanism_name)
                rule_errors[rule_id] = f"no registered mechanism '{mechanism_name}'"
                continue

            # wrap the call so an unforeseen crash in one mechanism can't abort the remaining rules
            try:
                result = mechanism(tree, source_lines, rule)
            except Exception as e:
                logger.error("_company_run_checks: mechanism '%s' crashed on rule %s: %s", mechanism_name, rule_id, e)
                rule_errors[rule_id] = f"mechanism '{mechanism_name}' crashed: {e}"
                continue

            if not isinstance(result, dict) or result.get("status") != "success":
                message = result.get("message", "mechanism returned no message") if isinstance(result, dict) else "mechanism returned a non-dict"
                logger.warning("_company_run_checks: rule %s failed: %s", rule_id, message)
                rule_errors[rule_id] = message
                continue

            findings.extend(result.get("findings", []))

        logger.debug("_company_run_checks: %d finding(s), %d rule error(s)", len(findings), len(rule_errors))
        return {"findings": findings, "rule_errors": rule_errors}

    except Exception as e:
        logger.error("_company_run_checks failed unexpectedly: %s", str(e))
        rule_errors["_runner"] = f"runner failed unexpectedly: {str(e)}"
        return {"findings": findings, "rule_errors": rule_errors}


def _company_check_naming(tree: ast.AST, source_lines: list[str], rule: dict) -> dict:
    """
    Flags functions that access the database but lack the required name prefix (rule 1.1).

    Pipeline: company-rule mechanism in mcp_server.py, dispatched by _company_run_checks
    (same module) via the _COMPANY_CHECKS registry for any rule whose "mechanism" is
    "naming_convention". Reached through the check_company_rules MCP tool the Analyzer calls.

    A function counts as a database function when its body calls any name in
    params["trigger_calls"] (e.g. db.query); if so, its name must start with one of
    params["required_prefixes"]. Each violating function becomes one finding.

    Args:
        tree:         Parsed AST of the reviewed code.
        source_lines: Code split into lines. Unused here, but part of the shared mechanism
                      signature so the runner can dispatch every mechanism the same way.
        rule:         One rule object from company_rules.json.

    Returns:
        dict with "status" ("success" | "error"), "findings" (list), "message" on error.
    """
    try:
        if not isinstance(tree, ast.AST):
            return {"status": "error",
                    "message": f"_company_check_naming: 'tree' must be an ast.AST, got {type(tree).__name__}",
                    "findings": []}
        if not isinstance(rule, dict):
            return {"status": "error",
                    "message": f"_company_check_naming: 'rule' must be a dict, got {type(rule).__name__}",
                    "findings": []}

        params = rule.get("params", {})
        trigger_calls = params.get("trigger_calls")
        required_prefixes = params.get("required_prefixes")
        if not isinstance(trigger_calls, list) or not trigger_calls:
            return {"status": "error",
                    "message": f"_company_check_naming: rule '{rule.get('id')}' needs a non-empty 'trigger_calls' list in params",
                    "findings": []}
        if not isinstance(required_prefixes, list) or not required_prefixes:
            return {"status": "error",
                    "message": f"_company_check_naming: rule '{rule.get('id')}' needs a non-empty 'required_prefixes' list in params",
                    "findings": []}

        trigger_set = set(trigger_calls)
        prefixes = tuple(required_prefixes)          # str.startswith accepts a tuple of options
        findings = []

        # a function is judged only if its body actually calls one of the DB entry points
        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue

            accesses_db = any(
                isinstance(call, ast.Call) and _company_dotted_name(call.func) in trigger_set
                for call in ast.walk(node)
            )
            if not accesses_db:
                continue                             # rule 1.1 applies only to DB functions

            if not node.name.startswith(prefixes):
                # function name in the message keeps distinct violations apart under later dedup
                findings.append({
                    "rule": rule["id"],
                    "message": f"{rule['message']} (function '{node.name}')",
                    "severity": rule.get("severity", "MEDIUM"),
                    "category": rule.get("category", "Maintainability"),
                    "lines": [node.lineno],
                    "occurrences": 1,
                })

        logger.debug("_company_check_naming: rule %s produced %d finding(s)", rule.get("id"), len(findings))
        return {"status": "success", "findings": findings}

    except Exception as e:
        rule_id = rule.get("id") if isinstance(rule, dict) else "<unknown>"
        logger.error("_company_check_naming crashed on rule %s: %s", rule_id, e)
        return {"status": "error",
                "message": f"_company_check_naming failed unexpectedly — likely a malformed AST node: {e}",
                "findings": []}


def _company_dotted_name(node: ast.AST) -> str | None:
    """
    Reconstructs the dotted name of an expression, e.g. a Name/Attribute chain → 'db.query'.

    Pipeline: shared private helper in mcp_server.py for the company-rule mechanisms
    (_company_check_naming, _company_check_raise, _company_check_access) to match calls/attributes
    against the string patterns in company_rules.json.

    Args:
        node: The func/value expression of a Call or Subscript (ast.Name or ast.Attribute).

    Returns:
        The dotted name (e.g. 'os.getenv'), or None if the expression is not a plain
        Name/Attribute chain (e.g. a call on a subscript or list literal).
    """
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        base = _company_dotted_name(node.value)              # recurse into the left side: Name('db') + '.query'
        return f"{base}.{node.attr}" if base else None
    return None                                      # Call/Subscript/etc. have no static dotted name

def _company_check_comment(tree: ast.AST, source_lines: list[str], rule: dict) -> dict:
    """
    Flags functions missing the required marker comment on/below the def line (rule 1.2).

    Pipeline: company-rule mechanism in mcp_server.py, dispatched by _company_run_checks
    (same module) via the _COMPANY_CHECKS registry for any rule whose "mechanism" is
    "required_comment". Reached through the check_company_rules MCP tool the Analyzer calls.

    Comments are not part of the AST, so this reads source_lines directly: the marker must
    appear on the def line or within params["max_lines_below"] lines beneath it.

    Args:
        tree:         Parsed AST of the reviewed code.
        source_lines: Code split into lines — the marker is matched against these.
        rule:         One rule object from company_rules.json.

    Returns:
        dict with "status" ("success" | "error"), "findings" (list), "message" on error.
    """
    try:
        if not isinstance(tree, ast.AST):
            return {"status": "error",
                    "message": f"_company_check_comment: 'tree' must be an ast.AST, got {type(tree).__name__}",
                    "findings": []}
        if not isinstance(source_lines, list):
            return {"status": "error",
                    "message": f"_company_check_comment: 'source_lines' must be a list, got {type(source_lines).__name__}",
                    "findings": []}
        if not isinstance(rule, dict):
            return {"status": "error",
                    "message": f"_company_check_comment: 'rule' must be a dict, got {type(rule).__name__}",
                    "findings": []}

        params = rule.get("params", {})
        marker = params.get("marker")
        max_lines_below = params.get("max_lines_below", 1)
        if not isinstance(marker, str) or not marker:
            return {"status": "error",
                    "message": f"_company_check_comment: rule '{rule.get('id')}' needs a non-empty string 'marker' in params",
                    "findings": []}
        if not isinstance(max_lines_below, int) or isinstance(max_lines_below, bool) or max_lines_below < 0:
            return {"status": "error",
                    "message": f"_company_check_comment: rule '{rule.get('id')}' needs a non-negative int 'max_lines_below' in params",
                    "findings": []}

        findings = []
        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue

            window = source_lines[node.lineno - 1: node.lineno + max_lines_below]   # def line + lines directly below, where the marker may sit
            if any(marker in line for line in window):
                continue

            findings.append({
                "rule": rule["id"],
                "message": f"{rule['message']} (function '{node.name}')",
                "severity": rule.get("severity", "LOW"),
                "category": rule.get("category", "Maintainability"),
                "lines": [node.lineno],
                "occurrences": 1,
            })

        logger.debug("_company_check_comment: rule %s produced %d finding(s)", rule.get("id"), len(findings))
        return {"status": "success", "findings": findings}

    except Exception as e:
        rule_id = rule.get("id") if isinstance(rule, dict) else "<unknown>"
        logger.error("_company_check_comment crashed on rule %s: %s", rule_id, e)
        return {"status": "error",
                "message": f"_company_check_comment failed unexpectedly — likely malformed source lines: {e}",
                "findings": []}


def _company_check_raise(tree: ast.AST, source_lines: list[str], rule: dict) -> dict:
    """
    Flags raise statements that raise a forbidden built-in exception (rule 1.3).

    Pipeline: company-rule mechanism in mcp_server.py, dispatched by _company_run_checks
    (same module) via the _COMPANY_CHECKS registry for any rule whose "mechanism" is
    "forbidden_raise". Reached through the check_company_rules MCP tool the Analyzer calls.

    Args:
        tree:         Parsed AST of the reviewed code.
        source_lines: Code split into lines. Unused here, but part of the shared mechanism
                      signature so the runner can dispatch every mechanism the same way.
        rule:         One rule object from company_rules.json.

    Returns:
        dict with "status" ("success" | "error"), "findings" (list), "message" on error.
    """
    try:
        if not isinstance(tree, ast.AST):
            return {"status": "error",
                    "message": f"_company_check_raise: 'tree' must be an ast.AST, got {type(tree).__name__}",
                    "findings": []}
        if not isinstance(rule, dict):
            return {"status": "error",
                    "message": f"_company_check_raise: 'rule' must be a dict, got {type(rule).__name__}",
                    "findings": []}

        forbidden = rule.get("params", {}).get("forbidden")
        if not isinstance(forbidden, list) or not forbidden:
            return {"status": "error",
                    "message": f"_company_check_raise: rule '{rule.get('id')}' needs a non-empty 'forbidden' list in params",
                    "findings": []}

        forbidden_set = set(forbidden)
        findings = []

        for node in ast.walk(tree):
            if not isinstance(node, ast.Raise) or node.exc is None:
                continue                             # bare 'raise' re-raises the active exception — nothing to name

            target = node.exc.func if isinstance(node.exc, ast.Call) else node.exc   # 'raise Foo(...)' holds the class in .func; 'raise Foo' holds it directly
            name = _company_dotted_name(target)
            if name in forbidden_set:
                findings.append({
                    "rule": rule["id"],
                    "message": f"{rule['message']} (raises '{name}')",
                    "severity": rule.get("severity", "MEDIUM"),
                    "category": rule.get("category", "Logic"),
                    "lines": [node.lineno],
                    "occurrences": 1,
                })

        logger.debug("_company_check_raise: rule %s produced %d finding(s)", rule.get("id"), len(findings))
        return {"status": "success", "findings": findings}

    except Exception as e:
        rule_id = rule.get("id") if isinstance(rule, dict) else "<unknown>"
        logger.error("_company_check_raise crashed on rule %s: %s", rule_id, e)
        return {"status": "error",
                "message": f"_company_check_raise failed unexpectedly — likely malformed AST node: {e}",
                "findings": []}

def _company_check_access(tree: ast.AST, source_lines: list[str], rule: dict) -> dict:
    """
    Flags forbidden direct accesses such as os.getenv(...) or os.environ[...] (rule 1.4).

    Pipeline: company-rule mechanism in mcp_server.py, dispatched by _company_run_checks
    (same module) via the _COMPANY_CHECKS registry for any rule whose "mechanism" is
    "forbidden_access". Reached through the check_company_rules MCP tool the Analyzer calls.

    Covers two AST shapes with one target list: a Call (os.getenv("K")) and a Subscript
    (os.environ["K"]).

    Args:
        tree:         Parsed AST of the reviewed code.
        source_lines: Code split into lines. Unused here, but part of the shared mechanism
                      signature so the runner can dispatch every mechanism the same way.
        rule:         One rule object from company_rules.json.

    Returns:
        dict with "status" ("success" | "error"), "findings" (list), "message" on error.
    """
    try:
        if not isinstance(tree, ast.AST):
            return {"status": "error",
                    "message": f"_company_check_access: 'tree' must be an ast.AST, got {type(tree).__name__}",
                    "findings": []}
        if not isinstance(rule, dict):
            return {"status": "error",
                    "message": f"_company_check_access: 'rule' must be a dict, got {type(rule).__name__}",
                    "findings": []}

        targets = rule.get("params", {}).get("targets")
        if not isinstance(targets, list) or not targets:
            return {"status": "error",
                    "message": f"_company_check_access: rule '{rule.get('id')}' needs a non-empty 'targets' list in params",
                    "findings": []}

        targets_set = set(targets)
        findings = []

        for node in ast.walk(tree):
            # a Call names the target in .func (os.getenv); a Subscript names it in .value (os.environ)
            if isinstance(node, ast.Call):
                name = _company_dotted_name(node.func)
            elif isinstance(node, ast.Subscript):
                name = _company_dotted_name(node.value)
            else:
                continue

            if name in targets_set:
                findings.append({
                    "rule": rule["id"],
                    "message": f"{rule['message']} (uses '{name}')",
                    "severity": rule.get("severity", "HIGH"),
                    "category": rule.get("category", "Security"),
                    "lines": [node.lineno],
                    "occurrences": 1,
                })

        logger.debug("_company_check_access: rule %s produced %d finding(s)", rule.get("id"), len(findings))
        return {"status": "success", "findings": findings}

    except Exception as e:
        rule_id = rule.get("id") if isinstance(rule, dict) else "<unknown>"
        logger.error("_company_check_access crashed on rule %s: %s", rule_id, e)
        return {"status": "error",
                "message": f"_company_check_access failed unexpectedly — likely malformed AST node: {e}",
                "findings": []}


# only wired mechanisms live here; the runner turns a missing mechanism into a rule_error, not a crash
_COMPANY_CHECKS = {
    "naming_convention": _company_check_naming,
    "required_comment": _company_check_comment,
    "forbidden_raise": _company_check_raise,
    "forbidden_access": _company_check_access,
}

# === SNIPPET HELPERS ===

@mcp.tool()
def _enclosing_snippet(code: str, lines: list) -> dict:
    """
    Extracts the smallest function source enclosing a set of finding lines.

    Pipeline: server-side core of the generate_fix_suggestion tool (this file).
    Called there with a single finding line during the Optimizer agent loop, and
    reused for the union of a fix's lines so the Evaluator can later judge against
    a scoped, line-numbered snippet instead of the whole file.

    Falls back to a fixed window around the anchor lines when the code does not
    parse or no single function spans every anchor line — partial context beats
    none.

    Args:
        code:  Python source — full file or raw snippet.
        lines: 1-based line numbers to enclose. Non-int and out-of-range entries
               are dropped; the call fails only if none remain.

    Returns:
        dict with status "success" | "fallback" | "error".
        On success/fallback: function_name (str|None), function_source (str),
        numbered_source (str, same lines prefixed with their 1-based number),
        start_line (int), end_line (int), context_type ("function" |
        "surrounding_lines"), and fallback_reason (str, only when "fallback").
        On error: message (str).
    """
    # code and lines feed splitlines()/ast.parse — a wrong type would raise deep
    # inside, so reject it loudly before any work begins.
    if not isinstance(code, str):
        logger.error("_enclosing_snippet: code must be a str, got %s", type(code).__name__)
        return {"status": "error", "message": f"_enclosing_snippet failed — code must be a string, got {type(code).__name__}"}

    if not isinstance(lines, list):
        logger.error("_enclosing_snippet: lines must be a list, got %s", type(lines).__name__)
        return {"status": "error", "message": f"_enclosing_snippet failed — lines must be a list, got {type(lines).__name__}"}

    try:
        if not code.strip():
            logger.warning("_enclosing_snippet: empty code received")
            return {"status": "error", "message": "_enclosing_snippet failed — code must not be empty."}

        source_lines = code.splitlines()
        total_lines = len(source_lines)

        # bool is an int subclass — exclude it so True/False cannot pose as a line number.
        usable = sorted({
            ln for ln in lines
            if isinstance(ln, int) and not isinstance(ln, bool) and 1 <= ln <= total_lines
        })
        if not usable:
            logger.warning(
                "_enclosing_snippet: no usable in-range line among %r (file has %d lines)",
                lines, total_lines,
            )
            return {"status": "error", "message": f"_enclosing_snippet failed — no usable line in range 1–{total_lines} among {lines}."}

        anchor_lo, anchor_hi = usable[0], usable[-1]

        # Number every returned line with its real 1-based file position: the explicit
        # anchor field is only trustworthy if the model can SEE the matching number in
        # the snippet — LLMs miscount unannotated source lines.
        def _render(start: int, end: int) -> str:
            width = len(str(end))
            return "\n".join(
                f"{n:>{width}} | {source_lines[n - 1]}"
                for n in range(start, end + 1)
            )

        # Defined inline so it closes over source_lines/anchors: callers must always get a
        # snippet, even when AST scoping is impossible.
        def _surrounding(reason: str) -> dict:
            start = max(1, anchor_lo - 5)
            end = min(total_lines, anchor_hi + 5)
            logger.info("_enclosing_snippet fallback: %s | lines %d–%d", reason, start, end)
            return {
                "status": "fallback",
                "fallback_reason": reason,
                "function_name": None,
                "function_source": "\n".join(source_lines[start - 1:end]),
                "numbered_source": _render(start, end),
                "start_line": start,
                "end_line": end,
                "context_type": "surrounding_lines",
            }

        try:
            tree = ast.parse(code)
        except SyntaxError as e:
            return _surrounding(f"SyntaxError at line {e.lineno}: {e.msg}")

        # Keep only functions spanning EVERY anchor line: a single fix's findings must
        # share one common scope, or the snippet would frame the wrong code.
        candidates = [
            node for node in ast.walk(tree)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.lineno <= anchor_lo and anchor_hi <= node.end_lineno
        ]
        if not candidates:
            return _surrounding("no single function encloses all anchor lines")

        # Innermost = smallest line span, so a nested function wins over its outer scope.
        innermost = min(candidates, key=lambda n: n.end_lineno - n.lineno)
        start, end = innermost.lineno, innermost.end_lineno

        logger.info("_enclosing_snippet: function '%s' lines %d–%d", innermost.name, start, end)
        return {
            "status": "success",
            "function_name": innermost.name,
            "function_source": "\n".join(source_lines[start - 1:end]),
            "numbered_source": _render(start, end),
            "start_line": start,
            "end_line": end,
            "context_type": "function",
        }

    except Exception as e:
        logger.error("_enclosing_snippet failed unexpectedly for lines %r: %s", lines, str(e))
        return {"status": "error", "message": f"_enclosing_snippet failed unexpectedly: {str(e)}"}

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













