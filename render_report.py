"""
Rendering layer for the Code Review pipeline — the single source of truth for
all user-facing output.

Every function turns a pipeline result dict into a Markdown string. The console
entrypoint (orchestrator __main__) and a future Gradio adapter both call these same
functions, so a change here changes every surface at once. Functions are pure
(str/dict in -> str out) except write_report_file, which is the only one that touches
disk.
"""

import os
import uuid
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


# === CATEGORY LABELS ===

# Analyzer findings carry a machine `category`; the user should see a plain label, not a
# linter's internal bucket name. This map is the ONLY place those labels are defined.
_CATEGORY_LABELS = {
    "Security": "Security",
    "Logic": "Logic",
    "Style": "Style / Layout",
    "Maintainability": "Maintainability",
}

# Display order for the analyzer table: most safety-relevant first.
_CATEGORY_ORDER = ["Security", "Logic", "Style", "Maintainability"]


def friendly_category(category) -> str:
    """
    Maps a machine finding category to a human-readable label.

    Pipeline: used by render_analyzer_summary (this module) when building the per-category
    finding table. Kept as the single definition point so relabeling is a one-line change.

    Args:
        category: The finding's `category` field, expected to be a str.

    Returns:
        The mapped label, or the input coerced to str when the category is unknown —
        an unmapped category must still be shown, never dropped.
    """
    try:
        if not isinstance(category, str):
            return str(category)
        return _CATEGORY_LABELS.get(category, category)
    except Exception as e:
        logger.error("friendly_category failed for %r: %s", category, str(e))
        return "Unknown"


# === STATUS METADATA ===

# (status, emoji, label) in report priority order. Canonical copy for the whole
# presentation layer — the counts table and the per-fix headers both read from here.
# Deliberately only three symbols across six statuses: ✅ good, ❌ code is wrong,
# ⚠️ needs a look — more distinct emoji read as visual clutter.
_STATUS_META = [
    ("APPROVED",      "✅", "Approved"),
    ("INCORRECT",     "❌", "Incorrect"),
    ("INCOMPLETE",    "❌", "Incomplete"),
    ("NONCOMPLIANT",  "⚠️", "Noncompliant"),
    ("NO_FIX",        "⚠️", "No fix"),
    ("NOT_EVALUATED", "⚠️", "Not evaluated"),
]
_STATUS_EMOJI = {s: e for s, e, _ in _STATUS_META}

# Human phrasing for WHERE a non-approved finding stalled, so the summary points at the
# stage responsible instead of only naming the status.
_STALL_LOCATION = {
    "INCORRECT":     "Evaluator — fix judged incorrect",
    "INCOMPLETE":    "Evaluator — fix judged incomplete",
    "NONCOMPLIANT":  "Evaluator — deviates from a retrieved guideline",
    "NO_FIX":        "Optimizer — no fix produced",
    "NOT_EVALUATED": "Evaluator — evaluation failed",
}


# === SHARED HELPERS ===

def format_duration(seconds) -> str:
    """
    Formats a duration in seconds as "Xm Ys" or "Ys".

    Pipeline: used by the orchestrator for the per-stage "done in" line and by
    render_run_overview (this module) for the total runtime.

    Args:
        seconds: Duration in seconds (int/float), or None.

    Returns:
        A short human string; "?" when the value is unusable.
    """
    try:
        if seconds is None or not isinstance(seconds, (int, float)):
            return "?"
        seconds = int(round(seconds))
        if seconds < 60:
            return f"{seconds}s"
        return f"{seconds // 60}m {seconds % 60:02d}s"
    except Exception as e:
        logger.error("format_duration failed for %r: %s", seconds, str(e))
        return "?"


def _md_table(headers: list, rows: list) -> str:
    """
    Builds a GitHub-flavoured Markdown table from headers and row cells.

    Pipeline: internal helper for the render_* functions in this module.

    Args:
        headers: Column titles.
        rows:    List of rows, each a list of cells (stringified here).

    Returns:
        The table as a single Markdown string. Returns an empty string on bad input
        rather than raising, so a malformed table never breaks the surrounding output.
    """
    try:
        if not isinstance(headers, list) or not isinstance(rows, list):
            logger.error("_md_table: headers and rows must be lists")
            return ""

        str_headers = [str(h) for h in headers]
        str_rows = [[str(c) for c in row] for row in rows]

        # Pad every column to its widest cell so the raw Markdown is column-aligned when read
        # in a plain terminal (which never renders tables); a Markdown renderer ignores the
        # extra spaces, so Gradio/GitHub/the .md file are unaffected. Width is by code point,
        # so an emoji column stays slightly off — good enough, and only the status column has one.
        cols = len(str_headers)
        widths = list(map(len, str_headers))
        for row in str_rows:
            for i in range(min(cols, len(row))):
                widths[i] = max(widths[i], len(row[i]))

        def _fmt(cells):
            padded = [(cells[i] if i < len(cells) else "").ljust(widths[i]) for i in range(cols)]
            return "| " + " | ".join(padded) + " |"

        head = _fmt(str_headers)
        sep = "| " + " | ".join("-" * w for w in widths) + " |"
        body = [_fmt(r) for r in str_rows]
        return "\n".join([head, sep, *body])
    except Exception as e:
        logger.error("_md_table failed unexpectedly: %s", str(e))
        return ""


def _first_line_number(anchor_lines, finding_keys) -> int:
    """
    Determines the first physical line a fix concerns, for sorting report blocks.

    Pipeline: internal helper for render_full_report (this module).

    Args:
        anchor_lines: The fix's comma-separated anchor-line string (may be empty/None).
        finding_keys: The fix's finding_keys list (fallback source of line numbers).

    Returns:
        The smallest line number found, or a large sentinel so line-less fixes sort last.
    """
    try:
        nums = []
        if isinstance(anchor_lines, str) and anchor_lines.strip():
            for part in anchor_lines.split(","):
                part = part.strip()
                if part.isdigit():
                    nums.append(int(part))
        if not nums and isinstance(finding_keys, list):
            for key in finding_keys:
                if isinstance(key, dict):
                    for ln in key.get("lines") or []:
                        if isinstance(ln, int):
                            nums.append(ln)
        return min(nums) if nums else 10**9   # line-less fixes sort to the end, never crash
    except Exception as e:
        logger.error("_first_line_number failed unexpectedly: %s", str(e))
        return 10**9


def _findings_for_fix(fix: dict, enriched_findings: list) -> list:
    """
    Collects the human descriptions of every finding a fix covers.

    Pipeline: internal helper for render_full_report (this module). Looks each of the
    fix's finding_keys up in enriched_findings by rule + overlapping lines — the same
    match idea used by the Evaluator's _issue_for_fix — so two same-line findings (e.g.
    unused `os` and unused `sys` on line 1) can be listed distinctly by their message.

    Args:
        fix:               An Optimizer fix dict with finding_keys.
        enriched_findings: The Enricher's findings, the source of the human `message`.

    Returns:
        List of {"rule", "message"} dicts, de-duplicated and order-preserving. Empty on
        bad input — a fix with no resolvable findings still renders (just without a list).
    """
    try:
        if not isinstance(fix, dict) or not isinstance(enriched_findings, list):
            return []

        finding_keys = fix.get("finding_keys")
        if not isinstance(finding_keys, list):
            return []

        seen = set()
        out = []
        for key in finding_keys:
            if not isinstance(key, dict):
                continue
            key_rule = key.get("rule")
            key_lineset = set(key.get("lines") or [])

            for finding in enriched_findings:
                if not isinstance(finding, dict) or finding.get("rule") != key_rule:
                    continue
                f_lines = finding.get("lines") if isinstance(finding.get("lines"), list) else []
                # Overlap (not equality) so a collapsed multi-line finding still matches the
                # single-line key it was split into; an empty key line-set falls back to rule-only.
                if key_lineset and not (key_lineset & set(f_lines)):
                    continue
                message = finding.get("message", "")
                dedup_key = (key_rule, message)
                if dedup_key in seen:
                    continue
                seen.add(dedup_key)
                out.append({"rule": key_rule, "message": message})
        return out
    except Exception as e:
        logger.error("_findings_for_fix failed unexpectedly: %s", str(e))
        return []


def _verdict_for_fix(fix: dict, fixes_evaluated: list) -> dict:
    """
    Finds the Evaluator verdict belonging to one Optimizer fix.

    Pipeline: internal helper for render_full_report (this module). The Evaluator fans one
    verdict out to one entry per covered finding, all sharing the same status/reasoning, so
    matching any entry whose (rule, lines) equals one of the fix's finding_keys recovers the
    fix's single verdict — this is what regroups the fanned entries back to one block per fix
    without the Evaluator having to tag them.

    Args:
        fix:             An Optimizer fix dict with finding_keys.
        fixes_evaluated: The Evaluator's fanned-out entry list.

    Returns:
        The first matching evaluated entry dict, or {} when none matches (rendered as
        "not evaluated" rather than dropped).
    """
    try:
        if not isinstance(fix, dict) or not isinstance(fixes_evaluated, list):
            return {}

        identities = set()
        for key in fix.get("finding_keys") or []:
            if isinstance(key, dict):
                identities.add((key.get("rule"), tuple(key.get("lines") or [])))

        for entry in fixes_evaluated:
            if not isinstance(entry, dict):
                continue
            ident = (entry.get("rule"), tuple(entry.get("lines") or []))
            if ident in identities:
                return entry
        return {}
    except Exception as e:
        logger.error("_verdict_for_fix failed unexpectedly: %s", str(e))
        return {}


def _anchor_label(fix: dict) -> str:
    """
    Produces the line label for a fix's report heading.

    Pipeline: internal helper for _render_fix_block (this module). Prefers the Optimizer's
    explicit anchor_lines; falls back to the fix's finding_keys so NO_FIX fixes (which carry
    no anchor_lines) still show their line instead of "?".

    Args:
        fix: An Optimizer fix dict.

    Returns:
        A comma-separated line string, or "?" when no line is known.
    """
    try:
        anchor = fix.get("anchor_lines")
        if isinstance(anchor, str) and anchor.strip():
            return anchor
        nums = sorted({
            ln for key in (fix.get("finding_keys") or []) if isinstance(key, dict)
            for ln in (key.get("lines") or []) if isinstance(ln, int) and not isinstance(ln, bool)
        })
        return ", ".join(str(n) for n in nums) if nums else "?"
    except Exception as e:
        logger.error("_anchor_label failed unexpectedly: %s", str(e))
        return "?"


def _original_snippet(fix: dict, code: str) -> str:
    """
    Returns the original code a fix concerns, line-numbered, for the before/after view.

    Pipeline: internal helper for _render_fix_block (this module). Uses the enclosing-function
    snippet the Optimizer attached (bounded to one function, so the user sees where the block
    starts and ends). NO_FIX fixes carry no snippet, so it falls back to the offending line(s)
    pulled straight from the source — even an unfixed finding still shows its original code.

    Args:
        fix:  An Optimizer fix dict (may carry code_context + finding_keys).
        code: The full source code, used only for the NO_FIX fallback.

    Returns:
        A line-numbered snippet string, or an empty string when nothing can be shown.
    """
    try:
        ctx = fix.get("code_context")
        if isinstance(ctx, str) and ctx.strip():
            return ctx.rstrip("\n")

        if not isinstance(code, str) or not code:
            return ""
        src_lines = code.splitlines()
        targets = sorted({
            ln for key in (fix.get("finding_keys") or []) if isinstance(key, dict)
            for ln in (key.get("lines") or []) if isinstance(ln, int) and not isinstance(ln, bool)
        })
        numbered = [f"{ln} | {src_lines[ln - 1]}" for ln in targets if 1 <= ln <= len(src_lines)]
        return "\n".join(numbered)
    except Exception as e:
        logger.error("_original_snippet failed unexpectedly: %s", str(e))
        return ""


# === INPUT SUMMARY (pipeline start) ===

def render_input_summary(code_input: str) -> str:
    """
    Renders the pipeline title plus a one-line summary of what was submitted.

    Pipeline: emitted by the orchestrator at the very start of a run, before the Analyzer.

    Why a local heuristic: this is a display preview shown before read_code runs, so it
    reuses read_code's own path test ("no newline and ends with .py") to guess file vs raw
    code — the Analyzer still produces the authoritative line_count later.

    Args:
        code_input: The raw code string or file path handed to the pipeline.

    Returns:
        A Markdown header + input line. Falls back to a minimal header on bad input.
    """
    try:
        if not isinstance(code_input, str):
            return "# \U0001f50d Code Review Pipeline\n\n**Input:** (invalid input)"

        is_path = "\n" not in code_input and code_input.strip().endswith(".py")
        if is_path:
            descriptor = f"file · `{os.path.basename(code_input.strip())}`"
        else:
            line_count = len(code_input.splitlines()) if code_input.strip() else 0
            descriptor = f"raw code · {line_count} lines"

        return f"# \U0001f50d Code Review Pipeline\n\n**Input:** {descriptor}"
    except Exception as e:
        logger.error("render_input_summary failed unexpectedly: %s", str(e))
        return "# \U0001f50d Code Review Pipeline"


# === STAGE OUTPUT (header before, table after) ===

def render_stage_header(step: int, total: int, name: str) -> str:
    """
    Renders the headline emitted before an agent runs.

    Pipeline: emitted by the orchestrator immediately before each stage, so the user sees
    which step is active while it works. The matching "done in Xs" line + summary table are
    emitted after the stage returns.

    Args:
        step:  1-based position of this stage.
        total: Total number of stages.
        name:  Agent name.

    Returns:
        A Markdown H2 line.
    """
    try:
        return f"## ▶ Step {step}/{total} · {name}"
    except Exception as e:
        logger.error("render_stage_header failed unexpectedly: %s", str(e))
        return f"## {name}"


def render_analyzer_summary(analyzer_result: dict) -> str:
    """
    Renders the Analyzer feedback table: findings per human category.

    Pipeline: emitted by the orchestrator after the Analyzer returns success, before the
    findings move on to the Enricher. Translates raw rule codes into the categories a human
    reads (Security / Logic / Style-Layout / Maintainability).

    Args:
        analyzer_result: The Analyzer's result dict.

    Returns:
        A Markdown table, plus a "scan incomplete" warning line when a scanner failed.
    """
    try:
        if not isinstance(analyzer_result, dict):
            return "_Analyzer summary unavailable (malformed result)._"

        analysis = analyzer_result.get("analysis_results", {})
        findings = (analysis.get("syntax_findings") or []) + (analysis.get("security_findings") or [])

        counts = {cat: 0 for cat in _CATEGORY_ORDER}
        extra = {}   # any category outside the known order still gets counted, never dropped
        for f in findings:
            cat = f.get("category") if isinstance(f, dict) else None
            if cat in counts:
                counts[cat] += 1
            else:
                extra[cat] = extra.get(cat, 0) + 1

        rows = [[friendly_category(cat), counts[cat]] for cat in _CATEGORY_ORDER if counts[cat]]
        rows += [[friendly_category(cat), n] for cat, n in extra.items()]
        rows.append(["**Total**", f"**{len(findings)}**"])

        parts = [_md_table(["Category", "Findings"], rows)]

        # A failed scanner with zero findings must not look like clean code — surface it.
        meta = analyzer_result.get("metadata", {})
        if isinstance(meta, dict) and meta.get("scan_complete") is False:
            tool_errors = meta.get("tool_errors") or {}
            detail = ", ".join(tool_errors.keys()) if isinstance(tool_errors, dict) and tool_errors else "unknown"
            parts += ["", f"⚠️ scan incomplete — {detail} did not run cleanly; findings may be partial."]

        return "\n".join(parts)
    except Exception as e:
        logger.error("render_analyzer_summary failed unexpectedly: %s", str(e))
        return "_Analyzer summary unavailable (render error)._"


def render_enricher_summary(enricher_result: dict, findings_in: int = None) -> str:
    """
    Renders the Enricher feedback table: findings carried through + whether RAG was used.

    Pipeline: emitted by the orchestrator after the Enricher returns success. Shows in/out
    counts on separate rows so a dropped finding would be immediately visible.

    Args:
        enricher_result: The Enricher's result dict.
        findings_in:     How many findings the orchestrator handed in (for conservation).

    Returns:
        A Markdown table.
    """
    try:
        if not isinstance(enricher_result, dict):
            return "_Enricher summary unavailable (malformed result)._"

        results = enricher_result.get("enrichment_results", {})
        out_findings = results.get("findings") if isinstance(results, dict) else []
        out_count = len(out_findings) if isinstance(out_findings, list) else 0
        rag_used = bool(results.get("rag_used")) if isinstance(results, dict) else False

        in_display = str(findings_in) if isinstance(findings_in, int) else "—"
        # A tick only when nothing was lost between in and out; a mismatch is left un-ticked
        # so the user's eye catches it.
        out_display = f"{out_count} ✓" if findings_in == out_count else str(out_count)

        rows = [
            ["Findings in", in_display],
            ["Findings out", out_display],
            ["Knowledge base used", "yes" if rag_used else "no"],
        ]
        return _md_table(["Metric", "Result"], rows)
    except Exception as e:
        logger.error("render_enricher_summary failed unexpectedly: %s", str(e))
        return "_Enricher summary unavailable (render error)._"


def render_optimizer_summary(optimizer_result: dict, findings_in: int = None) -> str:
    """
    Renders the Optimizer feedback table: fixes generated, findings covered, unresolved.

    Pipeline: emitted by the orchestrator after the Optimizer returns success. "Findings
    covered" vs "Fixes generated" makes the many-findings-to-one-fix grouping transparent;
    "Unresolved" surfaces findings the Optimizer could not fix (failed_count).

    Args:
        optimizer_result: The Optimizer's result dict.
        findings_in:      How many findings the orchestrator handed in (for conservation).

    Returns:
        A Markdown table.
    """
    try:
        if not isinstance(optimizer_result, dict):
            return "_Optimizer summary unavailable (malformed result)._"

        results = optimizer_result.get("optimization_results", {})
        fixes = results.get("fixes") if isinstance(results, dict) else []
        fixes = fixes if isinstance(fixes, list) else []

        meta = optimizer_result.get("metadata", {})
        total_fixes = meta.get("total_fixes", len(fixes)) if isinstance(meta, dict) else len(fixes)
        failed = meta.get("failed_count", 0) if isinstance(meta, dict) else 0

        # Findings covered = every finding_key across all fixes; this is the count that must
        # reconcile against what the Enricher handed in.
        covered = 0
        for fix in fixes:
            if isinstance(fix, dict) and isinstance(fix.get("finding_keys"), list):
                covered += len(fix["finding_keys"])

        in_display = str(findings_in) if isinstance(findings_in, int) else "—"
        rows = [
            ["Findings in", in_display],
            ["Fixes generated", total_fixes],
            ["Findings covered", str(covered)],
            ["Unresolved", failed],
        ]
        return _md_table(["Metric", "Result"], rows)
    except Exception as e:
        logger.error("render_optimizer_summary failed unexpectedly: %s", str(e))
        return "_Optimizer summary unavailable (render error)._"


# === PIPELINE SUMMARY (end of run) ===

def render_run_overview(stats: dict) -> str:
    """
    Renders the end-of-run overview: status, duration, and finding conservation.

    Pipeline: emitted by the orchestrator once the pipeline finishes (success or failure).
    The status line answers "did it work, and if not, where did it break", and the
    conservation line proves no finding silently vanished between stages.

    Args:
        stats: Orchestrator-built dict with keys: status, failed_stage, failed_message,
               duration_total_s, stages_completed, findings_trace (list of (stage, count)),
               lost_findings (list of dicts).

    Returns:
        A Markdown block.
    """
    try:
        if not isinstance(stats, dict):
            return "# Pipeline Summary\n\n_Overview unavailable (malformed stats)._"

        # Neutral stats symbol in the heading — the success/failure signal lives on the
        # Status line below, so the heading should read as "summary", not as a verdict.
        heading = "# \U0001f4ca Pipeline Summary"
        if stats.get("status") == "success":
            status_line = "**Status:** SUCCESS"
        else:
            stage = stats.get("failed_stage") or "unknown stage"
            msg = stats.get("failed_message") or "no message"
            status_line = f"**Status:** FAILED at {stage} — {msg}"

        duration = format_duration(stats.get("duration_total_s"))
        stages = stats.get("stages_completed", "?")
        meta_line = f"**Duration:** {duration} · {stages} stage(s) completed"

        parts = [heading, "", status_line, meta_line]

        # Conservation line: Analyzer N -> Enricher N -> ... so a drop is visible at a glance.
        trace = stats.get("findings_trace")
        if isinstance(trace, list) and trace:
            chain = " → ".join(f"{stage} {count}" for stage, count in trace)
            lost = stats.get("lost_findings") or []
            tail = "  ·  none lost ✓" if not lost else f"  ·  ⚠️ {len(lost)} lost"
            parts += ["", f"**Findings tracked:** {chain}{tail}"]

            # A lost finding is never swallowed — it is listed here with where it disappeared.
            for item in lost:
                if not isinstance(item, dict):
                    continue
                rule = item.get("rule", "?")
                lines = item.get("lines")
                loc = ", ".join(str(n) for n in lines) if isinstance(lines, list) and lines else "?"
                message = item.get("message", "")
                after = item.get("lost_after", "unknown stage")
                parts.append(f"   - Line {loc} · `{rule}` · {message} (lost after {after}, kept as unresolved)")

        return "\n".join(parts)
    except Exception as e:
        logger.error("render_run_overview failed unexpectedly: %s", str(e))
        return "# Pipeline Summary\n\n_Overview unavailable (render error)._"


def render_results_overview(evaluator_result: dict) -> str:
    """
    Renders the compact results table plus a "where it stalled" list for open findings.

    Pipeline: emitted by the orchestrator after a successful run. Counts are per finding
    (matching the Evaluator's own metadata); the detailed report groups these into per-fix
    blocks.

    Args:
        evaluator_result: The Evaluator's result dict.

    Returns:
        A Markdown counts table, plus a follow-up table naming where each non-approved
        finding stalled.
    """
    try:
        if not isinstance(evaluator_result, dict):
            return "## Results\n\n_Results unavailable (malformed result)._"

        results = evaluator_result.get("evaluation_results", {})
        entries = results.get("fixes_evaluated") if isinstance(results, dict) else []
        entries = entries if isinstance(entries, list) else []

        counts = {s: 0 for s, _, _ in _STATUS_META}
        for e in entries:
            st = e.get("status") if isinstance(e, dict) else None
            if st in counts:
                counts[st] += 1

        rows = [[f"{emoji} {label}", counts[st]] for st, emoji, label in _STATUS_META]
        parts = ["## Results  (per finding)", "", _md_table(["Status", "Count"], rows)]

        # Everything not APPROVED gets a "where did it stall" line so the user knows which
        # stage to look at, not just that something is open.
        open_entries = [e for e in entries if isinstance(e, dict) and e.get("status") != "APPROVED"]
        if open_entries:
            stall_rows = []
            for e in open_entries:
                rule = e.get("rule", "?")
                lines = e.get("lines")
                loc = ", ".join(str(n) for n in lines) if isinstance(lines, list) and lines else "?"
                status = e.get("status", "?")
                emoji = _STATUS_EMOJI.get(status, "•")
                where = _STALL_LOCATION.get(status, "—")
                stall_rows.append([loc, f"`{rule}`", f"{emoji} {status.title()}", where])
            parts += ["", "### Needs attention — where it stalled", "",
                      _md_table(["Line", "Rule", "Status", "Where"], stall_rows)]

        return "\n".join(parts)
    except Exception as e:
        logger.error("render_results_overview failed unexpectedly: %s", str(e))
        return "## Results\n\n_Results unavailable (render error)._"


# === FULL REPORT (written to file / shown on demand) ===

def render_full_report(evaluator_result: dict, optimizer_fixes: list, enriched_findings: list,
                       code: str = "") -> str:
    """
    Renders the detailed Markdown report — one block per fix, titled by its line(s).

    Pipeline: built by the orchestrator after a successful run and written to a file (or
    shown on demand in Gradio). Drives off the Optimizer fixes so each fix appears once (no
    per-finding triplication), pulls the shared verdict from the Evaluator entries, and
    lists every covered finding with its human message.

    Args:
        evaluator_result:  The Evaluator's result dict (source of verdicts + counts).
        optimizer_fixes:   The Optimizer's fixes list (the grouping unit + suggested code).
        enriched_findings: The Enricher's findings (source of the human "Found" text).
        code:              The full source code, for the original-code fallback on NO_FIX fixes.

    Returns:
        A complete Markdown document. Returns a minimal error document on unexpected
        failure so a broken report never crashes the run.
    """
    try:
        results = evaluator_result.get("evaluation_results", {}) if isinstance(evaluator_result, dict) else {}
        entries = results.get("fixes_evaluated") if isinstance(results, dict) else []
        entries = entries if isinstance(entries, list) else []
        fixes = optimizer_fixes if isinstance(optimizer_fixes, list) else []
        enriched = enriched_findings if isinstance(enriched_findings, list) else []

        # Header + per-finding counts (same tally the results overview shows).
        counts = {s: 0 for s, _, _ in _STATUS_META}
        for e in entries:
            st = e.get("status") if isinstance(e, dict) else None
            if st in counts:
                counts[st] += 1
        total = sum(counts.values())
        needs = total - counts["APPROVED"]

        lines = [
            "# Code Review Report",
            "",
            f"**{total} finding(s) across {len(fixes)} fix(es) — "
            f"{counts['APPROVED']} approved, {needs} need attention.**",
            "",
            _md_table(["Status", "Count"], [[f"{e} {l}", counts[s]] for s, e, l in _STATUS_META]),
            "",
            "---",
            "",
        ]

        # Order blocks by first affected line so the report reads top-to-bottom like the file.
        ordered = sorted(
            [f for f in fixes if isinstance(f, dict)],
            key=lambda f: _first_line_number(f.get("anchor_lines"), f.get("finding_keys")),
        )

        for fix in ordered:
            lines.append(_render_fix_block(fix, entries, enriched, code))
            lines.append("")

        return "\n".join(lines)
    except Exception as e:
        logger.error("render_full_report failed unexpectedly: %s", str(e))
        return "# Code Review Report\n\n_Report unavailable (render error): " + str(e) + "_"


def _render_fix_block(fix: dict, fixes_evaluated: list, enriched_findings: list, code: str = "") -> str:
    """
    Renders one fix as a report block: findings, before/after code, verdicts, reasoning.

    Pipeline: internal helper for render_full_report (this module), called once per Optimizer
    fix. Shows the original enclosing code next to the suggested fix so the user can apply the
    change with full context; the reasoning is tucked into a collapsible <details> so the long
    block no longer floods the report while staying one click away.

    Args:
        fix:               One Optimizer fix dict.
        fixes_evaluated:   The Evaluator's entries (source of this fix's verdict).
        enriched_findings: The Enricher's findings (source of the "Found" text).
        code:              The full source, for the original-code fallback on NO_FIX fixes.

    Returns:
        A Markdown block string.
    """
    try:
        verdict = _verdict_for_fix(fix, fixes_evaluated)
        status = verdict.get("status", "NOT_EVALUATED")
        emoji = _STATUS_EMOJI.get(status, "•")
        anchor = _anchor_label(fix)
        block = [f"### Line {anchor} — {emoji} {status}"]

        # What is wrong here — every finding this one fix covers, by its human message.
        found = _findings_for_fix(fix, enriched_findings)
        if found:
            block += ["", "**Found:**"]
            block += [f"- `{f['rule']}` — {f['message']}" for f in found]

        # Before/after: the original enclosing snippet (line-numbered, function-bounded) next
        # to the fix, so the user sees the exact code, its extent, and what to change it to.
        original = _original_snippet(fix, code)
        if original:
            block += ["", "**Original code:**", "", "```text", original, "```"]

        suggested = fix.get("suggested_code")
        if suggested is None:
            block += ["", "**Suggested fix:** _no fix produced — see original code above._"]
        elif isinstance(suggested, str) and suggested.strip() == "":
            block += ["", f"**Suggested fix — remove line(s) {anchor}.**"]
        else:
            block += ["", "**Suggested fix:**", "", "```python", str(suggested).rstrip("\n"), "```"]

        # Verdicts only exist when the fix was actually judged (NO_FIX/failed leave them None).
        correctness = verdict.get("correctness")
        completeness = verdict.get("completeness")
        faithfulness = verdict.get("faithfulness")
        if correctness or completeness or faithfulness:
            faith_display = (faithfulness or "n/a").replace("_", " ")
            block += ["", f"**Verdicts:** correctness `{correctness}` · "
                          f"completeness `{completeness}` · faithfulness `{faith_display}`"]

        grounded = verdict.get("grounded_in")
        if isinstance(grounded, list) and grounded:
            block += ["", f"**Grounded in:** {', '.join(str(g) for g in grounded)}"]

        reasoning = verdict.get("reasoning")
        if isinstance(reasoning, str) and reasoning.strip():
            block += ["", "<details><summary>Evaluator reasoning</summary>", "",
                      reasoning.strip(), "", "</details>"]

        block += ["", "---"]
        return "\n".join(block)
    except Exception as e:
        logger.error("_render_fix_block failed unexpectedly: %s", str(e))
        return "### (fix block unavailable — render error)\n\n---"


# === FILE OUTPUT (the only impure function) ===

def write_report_file(markdown: str, out_dir: str) -> str:
    """
    Writes a rendered report to a uniquely named Markdown file.

    Pipeline: called by the CLI entrypoint (orchestrator __main__) after render_full_report.
    Kept separate from the pure renderers so a Gradio adapter can instead offer the same
    string as an inline panel or download without ever touching disk.

    Why a unique name: a hosted deployment may run several reviews; timestamp + short uuid
    avoids one run overwriting another's report.

    Args:
        markdown: The report content.
        out_dir:  Directory to write into; created if missing.

    Returns:
        The absolute path written, or an empty string on failure (a failed write must not
        abort the run — the report is optional).
    """
    try:
        if not isinstance(markdown, str) or not isinstance(out_dir, str):
            logger.error("write_report_file: markdown and out_dir must be str")
            return ""

        os.makedirs(out_dir, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        filename = f"review_{stamp}_{uuid.uuid4().hex[:6]}.md"
        path = os.path.join(out_dir, filename)
        with open(path, "w", encoding="utf-8") as fh:   # utf-8: the report carries status emoji
            fh.write(markdown)
        return os.path.abspath(path)
    except Exception as e:
        logger.error("write_report_file failed unexpectedly: %s", str(e))
        return ""
