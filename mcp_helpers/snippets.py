"""
Snippet helper for the MCP server.
Extracted verbatim from mcp_server.py (minus the stray @mcp.tool() registration);
imported back by the generate_fix_suggestion tool.
"""

import ast
import logging

logger = logging.getLogger(__name__)


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
