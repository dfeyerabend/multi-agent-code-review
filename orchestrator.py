"""
Orchestrator — linear pipeline driver for the Code Review Agent.
Owns full pipeline state. Builds each agent's input contract and passes forward only what
that agent needs, and owns all user-facing output: it tracks the status of every finding
across stages and emits the progress, overviews, and final report through the render_report
layer.
"""

import os
import time
import asyncio
import logging

logger = logging.getLogger(__name__)

from agents.analyzer_agent import run_analyzer
from agents.enricher_agent import run_enricher
from agents.optimizer_agent import run_optimizer
from agents.evaluator_agent import run_evaluator

import render_report

# Fields the Enricher needs from each finding — everything else stays in the orchestrator.
_ENRICHER_FIELDS = {"rule", "lines", "occurrences", "category", "severity", "message", "doc_url", "cwe_id"}


# === FINDING TRACKING ===

def _covers(entry_or_key: dict, rule, lineset: set) -> bool:
    """
    Tests whether an evaluated entry / fix key refers to a given finding by rule + lines.

    Pipeline: internal helper for _reconcile_findings (this module). Matches on overlapping
    lines (not equality) so a collapsed multi-line finding still matches the single-line unit
    it was exploded into; an empty finding line-set falls back to a rule-only match.

    Args:
        entry_or_key: An evaluated entry or a fix finding_key, with `rule` and `lines`.
        rule:         The finding's rule code.
        lineset:      The finding's line numbers as a set.

    Returns:
        True when the entry refers to the finding.
    """
    if not isinstance(entry_or_key, dict) or entry_or_key.get("rule") != rule:
        return False
    other = set(entry_or_key.get("lines") or [])
    return (not lineset) or bool(lineset & other)


def _reconcile_findings(enriched_findings: list, fixes: list, fixes_evaluated: list) -> list:
    """
    Checks every enriched finding still has a home in the final output, and reports any drop.

    Pipeline: called by run_pipeline (this module) after the Evaluator returns. This is the
    safety net behind "nothing is silently dropped": each stage is designed to conserve
    findings, and this verifies it — a finding with no downstream representation is surfaced
    (with where it disappeared) instead of vanishing.

    Args:
        enriched_findings: The Enricher's findings — the ledger of what must be resolved.
        fixes:             The Optimizer's fixes (checked to attribute a drop to Optimizer).
        fixes_evaluated:   The Evaluator's entries (the final home each finding should reach).

    Returns:
        List of lost-finding dicts {rule, lines, message, lost_after}. Empty when every
        finding is accounted for (the expected case).
    """
    try:
        if not isinstance(enriched_findings, list):
            return []
        fixes = fixes if isinstance(fixes, list) else []
        fixes_evaluated = fixes_evaluated if isinstance(fixes_evaluated, list) else []

        # Flatten fix finding_keys once so each finding can be checked against them cheaply.
        fix_keys = [k for f in fixes if isinstance(f, dict) for k in (f.get("finding_keys") or [])]

        lost = []
        for finding in enriched_findings:
            if not isinstance(finding, dict):
                continue
            rule = finding.get("rule")
            lineset = set(finding.get("lines") or [])

            in_eval = any(_covers(e, rule, lineset) for e in fixes_evaluated)
            if in_eval:
                continue   # reached the final report — nothing lost

            # Attribute the drop: if the finding never became a fix it was lost before the
            # Optimizer produced fixes; if it had a fix but no verdict it was lost at the Evaluator.
            in_fixes = any(_covers(k, rule, lineset) for k in fix_keys)
            lost_after = "Optimizer" if in_fixes else "Enricher"
            lost.append({
                "rule": rule,
                "lines": finding.get("lines"),
                "message": finding.get("message", ""),
                "lost_after": lost_after,
            })
        return lost
    except Exception as e:
        logger.error("_reconcile_findings failed unexpectedly: %s", str(e))
        return []


# === PIPELINE ===

async def run_pipeline(code_input: str, emit=None) -> dict:
    """
    Runs the full code review pipeline from Analyzer through Evaluator.

    Pipeline: top-level entry point. Called by __main__ (console) or app.py (Gradio). Each
    stage receives only the fields it needs; identity, metadata, timing, and all output are
    managed here.

    Why the emit callback: the orchestrator owns output but must not print — printing is the
    entrypoint's job. `emit` (a callable(str) or None) receives each rendered Markdown block
    as the run progresses, so the console passes `print` and Gradio can stream to a widget,
    both reusing the same render_report functions.

    Args:
        code_input: File path or raw code string.
        emit:       Optional callable(str) invoked with each rendered Markdown block. When
                    None the pipeline runs silently (returns the same result).

    Returns:
        dict with the final pipeline output, augmented with `pipeline_stats` and (on success)
        `review_report_markdown`. On a stage failure the failing stage's dict is returned,
        also augmented with `pipeline_stats`. Always returns — never raises.
    """
    def _emit(text) -> None:
        # Output must never break the pipeline: a broken emit callback is logged and ignored.
        if emit is None or not isinstance(text, str) or not text.strip():
            return
        try:
            emit(text)
        except Exception as e:
            logger.error("emit callback failed (output suppressed): %s", str(e))

    t_start = time.perf_counter()
    trace = []   # (stage_name, finding_count) pairs for the conservation line

    def _fail(stage_name: str, stage_result: dict) -> dict:
        # Build the failure overview once, from wherever the pipeline short-circuited.
        stats = {
            "status": "error",
            "failed_stage": stage_name,
            "failed_message": stage_result.get("message") if isinstance(stage_result, dict) else "unknown",
            "duration_total_s": time.perf_counter() - t_start,
            "stages_completed": len(trace),
            "findings_trace": trace,
            "lost_findings": [],
        }
        _emit(render_report.render_run_overview(stats))
        if isinstance(stage_result, dict):
            stage_result = dict(stage_result)
            stage_result["pipeline_stats"] = stats
        return stage_result

    try:
        _emit(render_report.render_input_summary(code_input))

        # --- Step 1: Analyzer ---
        _emit(render_report.render_stage_header(1, 4, "Analyzer"))
        logger.info("Pipeline — step 1: Analyzer")
        t0 = time.perf_counter()
        analyzer_result = await run_analyzer(code_input)
        dur = time.perf_counter() - t0

        if analyzer_result.get("status") != "success":
            logger.error("Analyzer failed: %s", analyzer_result.get("message"))
            return _fail("Analyzer", analyzer_result)

        analysis = analyzer_result["analysis_results"]
        findings = [
            {k: v for k, v in f.items() if k in _ENRICHER_FIELDS}
            for f in (
                    analysis.get("syntax_findings", [])
                    + analysis.get("security_findings", [])
                    + analysis.get("company_findings", [])
            )
        ]
        trace.append(("Analyzer", len(findings)))
        _emit(f"Done in {render_report.format_duration(dur)}\n\n"
              + render_report.render_analyzer_summary(analyzer_result))

        # --- Step 2: Enricher ---
        _emit(render_report.render_stage_header(2, 4, "Enricher"))
        logger.info("Pipeline — step 2: Enricher — %d finding(s)", len(findings))
        t0 = time.perf_counter()
        enricher_result = await run_enricher(findings)
        dur = time.perf_counter() - t0

        if enricher_result.get("status") != "success":
            logger.error("Enricher failed: %s", enricher_result.get("message"))
            return _fail("Enricher", enricher_result)

        enriched_findings = enricher_result["enrichment_results"]["findings"]
        code = analysis["code"]
        trace.append(("Enricher", len(enriched_findings)))
        _emit(f"Done in {render_report.format_duration(dur)}\n\n"
              + render_report.render_enricher_summary(enricher_result, findings_in=len(findings)))

        # --- Step 3: Optimizer ---
        _emit(render_report.render_stage_header(3, 4, "Optimizer"))
        logger.info("Pipeline — step 3: Optimizer — %d finding(s)", len(enriched_findings))
        t0 = time.perf_counter()
        optimizer_result = await run_optimizer(code, enriched_findings)
        dur = time.perf_counter() - t0

        if optimizer_result.get("status") != "success":
            logger.error("Optimizer failed: %s", optimizer_result.get("message"))
            return _fail("Optimizer", optimizer_result)

        fixes = optimizer_result["optimization_results"]["fixes"]
        # Covered = every finding_key across all fixes; this is what must reconcile with the input.
        covered = sum(len(f.get("finding_keys") or []) for f in fixes if isinstance(f, dict))
        trace.append(("Optimizer", covered))
        _emit(f"Done in {render_report.format_duration(dur)}\n\n"
              + render_report.render_optimizer_summary(optimizer_result, findings_in=len(enriched_findings)))

        # --- Step 4: Evaluator ---
        _emit(render_report.render_stage_header(4, 4, "Evaluator"))
        logger.info("Pipeline — step 4: Evaluator — %d fix(es)", len(fixes))
        t0 = time.perf_counter()
        evaluator_result = await run_evaluator(code, enriched_findings, fixes)
        dur = time.perf_counter() - t0

        if evaluator_result.get("status") != "success":
            logger.error("Evaluator failed: %s", evaluator_result.get("message"))
            return _fail("Evaluator", evaluator_result)

        entries = evaluator_result.get("evaluation_results", {}).get("fixes_evaluated", [])
        trace.append(("Evaluator", len(entries)))
        _emit(f"## ▶ Step 4/4 · Evaluator — judged {len(fixes)} fix(es) in "
              f"{render_report.format_duration(dur)}")

        # Verify nothing was silently dropped between stages, and report any drop by name.
        lost = _reconcile_findings(enriched_findings, fixes, entries)

        stats = {
            "status": "success",
            "failed_stage": None,
            "failed_message": None,
            "duration_total_s": time.perf_counter() - t_start,
            "stages_completed": 4,
            "findings_trace": trace,
            "lost_findings": lost,
        }

        _emit(render_report.render_run_overview(stats))
        _emit(render_report.render_results_overview(evaluator_result))

        # Report is built here (the orchestrator holds fixes + enriched findings) and returned
        # so the entrypoint can write it to a file / Gradio can offer it inline.
        report_md = render_report.render_full_report(evaluator_result, fixes, enriched_findings, code)

        result = dict(evaluator_result)
        result["pipeline_stats"] = stats
        result["review_report_markdown"] = report_md
        return result

    except Exception as e:
        # guards against unexpected key errors in inter-agent data handoffs
        logger.error("run_pipeline failed unexpectedly: %s", str(e))
        stats = {
            "status": "error",
            "failed_stage": "orchestrator",
            "failed_message": str(e),
            "duration_total_s": time.perf_counter() - t_start,
            "stages_completed": len(trace),
            "findings_trace": trace,
            "lost_findings": [],
        }
        _emit(render_report.render_run_overview(stats))
        return {
            "status": "error",
            "message": f"run_pipeline failed unexpectedly: {str(e)}",
            "pipeline_stats": stats,
        }


# === ENTRY POINT ===

if __name__ == "__main__":
    import sys
    from config import setup_logging, PROJECT_ROOT
    setup_logging()

    # The report + overviews carry status emoji; the default Windows console is cp1252 and
    # would crash on them. Force utf-8 so the rendered Markdown prints on any platform.
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception as e:
        logger.warning("could not reconfigure stdout to utf-8 (%s) — non-ascii output may fail", str(e))

    if len(sys.argv) > 1:
        test_input = sys.argv[1]
    else:
        test_input = (
            "import os, sys\n"  # E401 (Style) + F401 x2 os/sys (Logic) — unused, multiple imports on one line
            "import json\n"  # F401 json (Logic) — unused import
            "\n"
            "def get_user(id):  # REASON: fetch user record by id\n"  # REASON present -> COMPANY-1.2 does NOT fire on this function
            "    query = 'SELECT * FROM users WHERE id = ' + id\n"  # B608 (Security, bandit) — SQL injection via string concatenation
            "    return query\n"
            "\n"
            "def calculate_total(a, b):\n"  # PURE case: no REASON comment, no other findings on this function at all
            "    return a + b\n"  # -> expect exactly one finding: COMPANY-1.2 (Maintainability), isolated
            "\n"
            "def process_input(data):  # REASON: parse and validate raw input\n"  # REASON present -> COMPANY-1.2 does NOT fire on this function
            "    try:\n"
            "        return int(data)\n"
            "    except ValueError:\n"
            "        raise ValueError('invalid data')\n"
        # COMBINED case, same line: ruff B904 raise-without-from-inside-except (Logic)
            # + COMPANY-1.3 forbidden raise of built-in ValueError (Logic)
        )

    # emit prints each rendered block as it arrives — print lives in the entrypoint, never in
    # module logic, so the same run_pipeline stays reusable by Gradio.
    result = asyncio.run(run_pipeline(test_input, emit=lambda md: print("\n" + md)))

    # The detailed report is written to a file so a long review never floods the console.
    report_md = result.get("review_report_markdown")
    if report_md:
        path = render_report.write_report_file(report_md, os.path.join(PROJECT_ROOT, "reports"))
        if path:
            print(f"\n\U0001f4c4 Full report written to: {path}")
        else:
            print("\n(report file could not be written — see logs)")
