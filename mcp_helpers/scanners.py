"""
Scanner helpers for the MCP server (ruff/bandit plumbing).
Extracted verbatim from mcp_server.py; imported back by detect_syntax_errors.
"""

import json
import subprocess
import tempfile
import logging

logger = logging.getLogger(__name__)


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
