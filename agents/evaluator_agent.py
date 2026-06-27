"""
Evaluator Agent — Step 4 in the Code Review Pipeline.
Receives enriched findings and optimizer fixes from the orchestrator, judges each (finding, fix) pair independently, and returns a structured evaluation with a markdown report.
"""

import asyncio
import json
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

import logging
logger = logging.getLogger(__name__)

from config import (
    client,
    MODEL,
    MAX_TOKENS,
    MAX_ITERATIONS,
    MCP_SERVER_PATH,
    EVALUATOR_PROMPT,
)

from tools.evaluator_tools import (
    evaluator_local_tools,
    run_evaluator_tool,
)

# === HELPER FUNCTIONS ===

def _derive_status(verdicts: dict) -> str:
    """
    Derives a deterministic status string from the three LLM verdicts.

    Pipeline: called by run_evaluator once per pair after _run_evaluator_pair
    returns successfully and all validation layers pass.

    Args:
        verdicts: Dict with keys faithfulness, correctness, completeness.

    Returns:
        One of: 'APPROVED', 'NEEDS_REVISION', 'UNRESOLVABLE'.
        Returns 'UNRESOLVABLE' on bad input — a missing verdict is treated as
        an unresolvable state rather than a silent guess.
    """
    if not isinstance(verdicts, dict):
        logger.error(
            "_derive_status: verdicts must be a dict, got %s",
            type(verdicts).__name__,
        )
        return "UNRESOLVABLE"   # a missing or malformed verdict cannot be assumed to be passing

    try:
        if verdicts.get("correctness") == "fail":   # broken code overrides everything — no other verdict can salvage it
            return "UNRESOLVABLE"

        if (
            verdicts.get("faithfulness") == "faithful"
            and verdicts.get("correctness") == "pass"
            and verdicts.get("completeness") == "complete"
        ):
            return "APPROVED"

        return "NEEDS_REVISION"

    except Exception as e:
        logger.error("_derive_status failed unexpectedly: %s", str(e))
        return "UNRESOLVABLE"

async def _run_evaluator_pair(code: str, issue: dict, fix: dict) -> dict:
    """
    Runs one Evaluator LLM call for a single (issue, fix) pair.

    Pipeline: called by run_evaluator once per pair where suggested_code
    is not None. Each call is fully independent — no shared state between pairs.

    Args:
        code:  Full source code string from the Analyzer.
        issue: Minimal issue dict with rationale and best_practice_refs.
        fix:   Minimal fix dict with suggested_code, explanation, grounded_in.

    Returns:
        dict with evaluation verdicts on success, or a structured error dict
        with status 'max_iterations_reached' or 'error'.
    """
    try:
        if not isinstance(code, str):
            logger.error("_run_evaluator_pair: code must be a str, got %s", type(code).__name__)
            return {"status": "error", "message": f"Invalid input: code must be str, got {type(code).__name__}"}

        if not isinstance(issue, dict):
            logger.error("_run_evaluator_pair: issue must be a dict, got %s", type(issue).__name__)
            return {"status": "error", "message": f"Invalid input: issue must be dict, got {type(issue).__name__}"}

        if not isinstance(fix, dict):
            logger.error("_run_evaluator_pair: fix must be a dict, got %s", type(fix).__name__)
            return {"status": "error", "message": f"Invalid input: fix must be dict, got {type(fix).__name__}"}

        batch_input = {
            "code": code,
            "issue": issue,     # only rationale + best_practice_refs — identity fields stripped by caller
            "fix": fix,         # only suggested_code + explanation + grounded_in
        }
        messages = [{"role": "user", "content": json.dumps(batch_input, indent=2)}]

        for iteration in range(MAX_ITERATIONS):
            logger.debug("Iteration %d/%d", iteration + 1, MAX_ITERATIONS)

            response = client.messages.create(
                model=MODEL,
                max_tokens=MAX_TOKENS,
                system=EVALUATOR_PROMPT,
                tools=evaluator_local_tools,    # local only — no MCP tools needed for judging
                messages=messages,
            )

            logger.debug("Stop reason: %s", response.stop_reason)

            if response.stop_reason == "end_turn":
                final_output = _extract_final_output(messages, response)
                logger.info("Pair evaluated after %d iteration(s)", iteration + 1)
                return final_output

            elif response.stop_reason == "tool_use":
                messages.append({"role": "assistant", "content": response.content})

                tool_results = []
                for block in response.content:
                    if hasattr(block, "text") and block.text:
                        logger.debug("Claude says: %s", block.text[:200])

                    if block.type == "tool_use":
                        logger.debug("Tool call: %s | args: %s", block.name, str(block.input)[:200])
                        tool_output = run_evaluator_tool(block.name, block.input)
                        logger.debug("Tool result: %s", tool_output[:300])

                        tool_results.append({
                            "type": "tool_result",
                            "tool_use_id": block.id,
                            "content": tool_output,
                        })

                messages.append({"role": "user", "content": tool_results})

        # loop exhausted without a valid submit_evaluation call — expected budget failure, not a bug
        logger.warning("Reached max iterations (%d) without valid output", MAX_ITERATIONS)
        return {
            "status": "max_iterations_reached",
            "message": "Max iterations reached without valid output",
        }

    except Exception as e:
        # unexpected: Anthropic API error, network failure, malformed response object, etc.
        logger.error("Unexpected error in _run_evaluator_pair: %s", str(e))
        return {
            "status": "error",
            "message": f"Unexpected error — likely API or network failure: {str(e)}",
        }


def _extract_final_output(messages: list, final_response) -> dict:
    """
    Pulls the submit_evaluation result out of the conversation history.

    Pipeline: called by _run_evaluator_pair once the model stops with
    stop_reason='end_turn' (i.e. after it called submit_evaluation).

    Args:
        messages:       Full conversation message list for this pair.
        final_response: Last Anthropic API response object (fallback source only).

    Returns:
        dict with the validated evaluation output, or error info.
    """
    if not isinstance(messages, list):
        logger.error("_extract_final_output: messages must be a list, got %s", type(messages).__name__)
        return {"status": "error", "message": f"Invalid input: messages must be list, got {type(messages).__name__}"}

    try:
        for msg in reversed(messages):
            if msg.get("role") == "user" and isinstance(msg.get("content"), list):
                for block in msg["content"]:
                    if block.get("type") == "tool_result":
                        try:
                            result = json.loads(block["content"])
                            if result.get("status") == "success" and "evaluation" in result:
                                return result
                        except (json.JSONDecodeError, TypeError):
                            continue

        # no valid tool_result found — attempt to parse the raw text response as a fallback
        final_text = ""
        for block in final_response.content:
            if hasattr(block, "text"):
                final_text += block.text

        try:
            return json.loads(final_text)
        except json.JSONDecodeError:
            logger.error("Failed to parse final output — raw: %s", final_text[:200])
            return {"status": "error", "raw_output": final_text}

    except Exception as e:
        logger.error("_extract_final_output failed unexpectedly: %s", str(e))
        return {"status": "error", "message": f"Unexpected error extracting output: {str(e)}"}


def _evaluated_entry(
    rule,
    lines,
    category: str,
    status: str,
    suggested_code,
    grounded_in,
    reasoning: str,
    faithfulness,
    correctness,
    completeness,
) -> dict:
    """
    Builds one fixes_evaluated entry with the full report schema.

    Pipeline: called by _entries_for_fix (this module) for every covered finding key, so
    every entry create_review_report renders carries an identical set of keys — a missing
    field in any one of them would break the report.

    Args:
        rule:           Rule code of the covered finding.
        lines:          Line number(s) the finding affects.
        category:       Finding category, for the report matrix.
        status:         Derived status — APPROVED / NEEDS_REVISION / UNRESOLVABLE.
        suggested_code: The optimizer's fix, or None.
        grounded_in:    Grounding sources from the optimizer.
        reasoning:      Human-readable verdict reasoning.
        faithfulness:   Faithfulness verdict, or None when no LLM call was made.
        correctness:    Correctness verdict, or None.
        completeness:   Completeness verdict, or None.

    Returns:
        dict conforming to the fixes_evaluated entry schema.
    """
    try:
        return {
            "rule":           rule if isinstance(rule, str) else str(rule),  # losing the rule id is worse than a coerced string
            "lines":          lines,
            "category":       category if isinstance(category, str) else str(category),
            "status":         status,
            "suggested_code": suggested_code,
            "grounded_in":    grounded_in if isinstance(grounded_in, list) else [],
            "reasoning":      reasoning if isinstance(reasoning, str) else str(reasoning),
            "faithfulness":   faithfulness,
            "correctness":    correctness,
            "completeness":   completeness,
        }
    except Exception as e:
        logger.error("_evaluated_entry failed unexpectedly: %s", str(e))
        return {
            "rule": str(rule), "lines": lines, "category": "Unknown", "status": "UNRESOLVABLE",
            "suggested_code": None, "grounded_in": [], "reasoning": f"Entry construction failed: {str(e)}",
            "faithfulness": None, "correctness": None, "completeness": None,
        }


def _entries_for_fix(fix: dict, status: str, reasoning: str, verdicts: dict = None) -> list:
    """
    Fans one judged fix out to one evaluated entry per finding it covers.

    Pipeline: called by run_evaluator (this module) after a fix is judged, or determined
    UNRESOLVABLE without an LLM call. The single verdict is shared across every finding in
    the fix's finding_keys, so a fix that resolved three conflicting findings produces three
    report entries that all carry the same status and reasoning.

    Args:
        fix:       Optimizer fix dict with finding_keys, suggested_code, grounded_in.
        status:    Derived status applied to every covered finding.
        reasoning: Verdict reasoning applied to every covered finding.
        verdicts:  Verdict dict (faithfulness/correctness/completeness), or None when no LLM
                   call was made — then all three verdicts are None.

    Returns:
        List of fixes_evaluated entries, one per finding_key. A fix with no usable
        finding_keys still yields one degraded UNRESOLVABLE entry so it never vanishes.
    """
    if not isinstance(fix, dict):
        logger.error("_entries_for_fix: fix must be a dict, got %s", type(fix).__name__)
        return [_evaluated_entry(None, None, "Unknown", "UNRESOLVABLE", None, [],
                                 "Fix was not a dict — cannot evaluate.", None, None, None)]

    try:
        verdicts     = verdicts if isinstance(verdicts, dict) else {}
        faithfulness = verdicts.get("faithfulness")
        correctness  = verdicts.get("correctness")
        completeness = verdicts.get("completeness")

        suggested_code = fix.get("suggested_code")
        grounded_in    = fix.get("grounded_in", [])
        finding_keys   = fix.get("finding_keys")

        # A fix with no identity still must surface once, attributed to nothing, rather than vanish.
        if not isinstance(finding_keys, list) or not finding_keys:
            logger.error("_entries_for_fix: fix has no usable finding_keys — emitting one degraded entry")
            return [_evaluated_entry(None, None, "Unknown", "UNRESOLVABLE", suggested_code, grounded_in,
                                     "Fix carried no finding identity — cannot attribute to a finding.",
                                     None, None, None)]

        entries = []
        for key in finding_keys:
            if not isinstance(key, dict):  # one malformed key must not drop the rest
                logger.warning("_entries_for_fix: skipping non-dict finding_key: %s", type(key).__name__)
                continue
            entries.append(_evaluated_entry(
                key.get("rule"),
                key.get("lines"),
                key.get("category", "Unknown"),
                status,
                suggested_code,
                grounded_in,
                reasoning,
                faithfulness,
                correctness,
                completeness,
            ))

        if not entries:  # every key was malformed — still surface the fix once
            return [_evaluated_entry(None, None, "Unknown", "UNRESOLVABLE", suggested_code, grounded_in,
                                     "Fix finding_keys were all malformed — cannot attribute to a finding.",
                                     None, None, None)]

        return entries

    except Exception as e:
        logger.error("_entries_for_fix failed unexpectedly: %s", str(e))
        return [_evaluated_entry(None, None, "Unknown", "UNRESOLVABLE", None, [],
                                 f"Entry fan-out failed: {str(e)}", None, None, None)]

# === AGENT LOOP ===

def _issue_for_fix(fix: dict, enriched_findings: list) -> dict:
    """
    Assembles the issue context (rationale + best-practice refs) a fix is judged against.

    Pipeline: called by run_evaluator (this module) for each fix with a real suggested_code,
    before the LLM call. A fix may resolve several findings; this gathers the rationale and
    refs of every finding it covers so the model judges the fix against all of them at once.

    Why a lookup: the optimizer's fix carries finding identity (rule + lines) but not the
    original rationale, so the issue text is recovered from enriched_findings by matching
    rule and overlapping lines. Occurrences collapsed under one rule share a rationale, so an
    exact per-occurrence match is unnecessary — overlapping lines is enough.

    Args:
        fix:               Optimizer fix dict with finding_keys.
        enriched_findings: The Enricher's findings, source of rationale/best_practice_refs.

    Returns:
        dict with 'rationale' (combined, de-duplicated) and 'best_practice_refs' (combined,
        de-duplicated). Returns empty strings/lists on invalid input or unexpected failure.
    """
    empty = {"rationale": "", "best_practice_refs": []}

    if not isinstance(fix, dict):
        logger.error("_issue_for_fix: fix must be a dict, got %s", type(fix).__name__)
        return empty

    if not isinstance(enriched_findings, list):
        logger.error("_issue_for_fix: enriched_findings must be a list, got %s", type(enriched_findings).__name__)
        return empty

    try:
        finding_keys = fix.get("finding_keys")
        if not isinstance(finding_keys, list):
            finding_keys = []

        rationales = []  # ordered, de-duplicated: collapsed occurrences share one rationale
        refs = []

        for key in finding_keys:
            if not isinstance(key, dict):
                continue
            key_rule    = key.get("rule")
            key_lines   = key.get("lines") if isinstance(key.get("lines"), list) else []
            key_lineset = set(key_lines)

            for finding in enriched_findings:
                if not isinstance(finding, dict) or finding.get("rule") != key_rule:
                    continue
                f_lines = finding.get("lines") if isinstance(finding.get("lines"), list) else []
                # Match on overlapping lines so a collapsed multi-line finding still matches the
                # single-line unit it was split into; an empty key line-set falls back to rule-only.
                if key_lineset and not (key_lineset & set(f_lines)):
                    continue

                rationale = finding.get("rationale")
                if isinstance(rationale, str) and rationale and rationale not in rationales:
                    rationales.append(rationale)
                for ref in finding.get("best_practice_refs", []) or []:
                    if ref not in refs:
                        refs.append(ref)

        return {"rationale": "\n".join(rationales), "best_practice_refs": refs}

    except Exception as e:
        logger.error("_issue_for_fix failed unexpectedly: %s", str(e))
        return empty


async def run_evaluator(code: str, enriched_findings: list, fixes: list) -> dict:
    """
    Orchestrates all evaluator pair calls, derives statuses, and builds the final report.

    Pipeline: Step 4 in the pipeline. Called by the orchestrator with the full source
    code, all enriched findings from the Enricher, and all fixes from the Optimizer.

    Args:
        code:              Full source code string from the Analyzer.
        enriched_findings: List of enriched finding dicts from the Enricher.
        fixes:             List of fix dicts from the Optimizer.

    Returns:
        dict with evaluation results, open findings, markdown report, and metadata.
        Always returns a structured dict — never raises.
    """
    if not isinstance(code, str):
        logger.error("run_evaluator: code must be a str, got %s", type(code).__name__)
        return {"status": "error", "message": f"Invalid input: code must be str, got {type(code).__name__}"}

    if not isinstance(enriched_findings, list):
        logger.error("run_evaluator: enriched_findings must be a list, got %s", type(enriched_findings).__name__)
        return {"status": "error", "message": f"Invalid input: enriched_findings must be list, got {type(enriched_findings).__name__}"}

    if not isinstance(fixes, list):
        logger.error("run_evaluator: fixes must be a list, got %s", type(fixes).__name__)
        return {"status": "error", "message": f"Invalid input: fixes must be list, got {type(fixes).__name__}"}

    if not enriched_findings:           # legitimate path: analyzer found no issues in clean code (Test Case 2)
        return {
            "status": "success",        # success — the pipeline ran correctly, there was just nothing to evaluate
            "evaluation_results": {
                "fixes_evaluated": [],
                "open_findings": [],
                "report": "",
                "summary": "No findings to evaluate.",
            },
            "metadata": {"total": 0, "approved": 0, "needs_revision": 0, "unresolvable": 0},
        }

    try:
        fixes_evaluated = []

        # Drive off fixes, not findings: the optimizer already emits one fix-result per unit
        # (real, null, or failure), so judging each fix once and fanning its verdict out to
        # every finding it covers reconstructs the full per-finding report without re-matching.
        for i, fix in enumerate(fixes):
            if not isinstance(fix, dict):  # a non-dict fix cannot be judged — surface it, do not crash the loop
                logger.warning("run_evaluator: fixes[%d] is not a dict (%s) — marking UNRESOLVABLE", i, type(fix).__name__)
                fixes_evaluated.extend(_entries_for_fix(fix, "UNRESOLVABLE", "Optimizer fix entry was malformed."))
                continue

            suggested_code = fix.get("suggested_code")

            # No suggested code → optimizer produced no actionable fix; nothing to judge.
            if suggested_code is None:
                reason = fix.get("explanation") or "No fix was produced for this finding."
                logger.warning("run_evaluator: fixes[%d] has null suggested_code — UNRESOLVABLE without LLM call", i)
                fixes_evaluated.extend(_entries_for_fix(fix, "UNRESOLVABLE", reason))
                continue

            # Minimal LLM input: issue context recovered from the findings plus the fix's own
            # output. Identity is deliberately excluded — matching is already done in Python.
            issue     = _issue_for_fix(fix, enriched_findings)
            fix_input = {
                "suggested_code": suggested_code,
                "explanation":    fix.get("explanation", ""),
                "grounded_in":    fix.get("grounded_in", []),
            }

            logger.info(
                "Evaluating fix %d/%d covering %d finding(s)",
                i + 1, len(fixes), len(fix.get("finding_keys") or []),
            )
            result = await _run_evaluator_pair(code, issue, fix_input)

            # Layer 1a: agent ran out of iterations — predictable budget failure, not a bug.
            if result.get("status") == "max_iterations_reached":
                logger.warning("run_evaluator: fixes[%d] hit max iterations — marking UNRESOLVABLE", i)
                fixes_evaluated.extend(_entries_for_fix(
                    fix, "UNRESOLVABLE",
                    "Evaluator reached max iterations without producing a valid verdict.",
                ))
                continue

            # Layer 1b: unexpected error (API failure, network issue, etc.).
            if result.get("status") == "error":
                logger.error("run_evaluator: fixes[%d] errored: %s", i, result.get("message", "unknown error"))
                fixes_evaluated.extend(_entries_for_fix(
                    fix, "UNRESOLVABLE",
                    f"Unexpected error during evaluation: {result.get('message', 'unknown error')}",
                ))
                continue

            # Validate model output before use: a partial tool call can still reach here, and
            # guessing a missing verdict is worse than failing loudly.
            evaluation = result.get("evaluation")
            if not isinstance(evaluation, dict):
                logger.error("run_evaluator: fixes[%d] — no 'evaluation' dict (got %s)", i, type(evaluation).__name__)
                fixes_evaluated.extend(_entries_for_fix(
                    fix, "UNRESOLVABLE", "Evaluator returned malformed output — no evaluation dict present.",
                ))
                continue

            required = ["reasoning", "faithfulness", "correctness", "completeness"]
            missing  = [f for f in required if f not in evaluation]
            if missing:
                logger.error("run_evaluator: fixes[%d] — evaluation missing fields %s", i, missing)
                fixes_evaluated.extend(_entries_for_fix(
                    fix, "UNRESOLVABLE", f"Evaluator returned incomplete verdicts — missing: {missing}",
                ))
                continue

            status = _derive_status(evaluation)
            logger.info(
                "run_evaluator: fixes[%d] → %s | faith=%s correct=%s complete=%s",
                i, status, evaluation["faithfulness"], evaluation["correctness"], evaluation["completeness"],
            )
            # One verdict, fanned out to every finding this fix resolved.
            fixes_evaluated.extend(_entries_for_fix(fix, status, evaluation["reasoning"], evaluation))

        open_findings  = [e for e in fixes_evaluated if e["status"] == "UNRESOLVABLE"]
        approved       = sum(1 for e in fixes_evaluated if e["status"] == "APPROVED")
        needs_revision = sum(1 for e in fixes_evaluated if e["status"] == "NEEDS_REVISION")
        unresolvable   = len(open_findings)
        total          = len(fixes_evaluated)

        summary = (
            f"{total} finding(s) evaluated: {approved} approved, "
            f"{needs_revision} need revision, {unresolvable} unresolvable."
        )

        # Call create_review_report via MCP — once, after all pairs are processed
        report = ""
        server_params = StdioServerParameters(command="python", args=[MCP_SERVER_PATH])
        try:
            async with stdio_client(server_params) as (read, write):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    logger.info("Calling create_review_report via MCP")
                    mcp_result = await session.call_tool(
                        "create_review_report",
                        arguments={"fixes_evaluated": fixes_evaluated, "summary": summary},
                    )
                    raw    = mcp_result.content[0].text if mcp_result.content else ""
                    parsed = json.loads(raw)
                    if parsed.get("status") == "success" and "report" in parsed:
                        report = parsed["report"]
                    else:
                        logger.error(
                            "create_review_report returned error: %s",
                            parsed.get("message", "unknown"),
                        )
        except Exception as e:
            # report generation failing must not suppress the evaluation results
            logger.error("create_review_report MCP call failed — report will be empty: %s", str(e))

        return {
            "status": "success",
            "evaluation_results": {
                "fixes_evaluated": fixes_evaluated,
                "open_findings":   open_findings,
                "report":          report,
                "summary":         summary,
            },
            "metadata": {
                "total":          total,
                "approved":       approved,
                "needs_revision": needs_revision,
                "unresolvable":   unresolvable,
            },
        }

    except Exception as e:
        logger.error("run_evaluator failed unexpectedly: %s", str(e))
        return {
            "status": "error",
            "message": f"run_evaluator failed unexpectedly: {str(e)}",
        }

# === ENTRY POINT ===

if __name__ == "__main__":
    from config import setup_logging
    setup_logging()

    test_code = (
        "import os, sys\n"
        "def get_user(id):\n"
        "    query = 'SELECT * FROM users WHERE id = ' + id\n"
        "    return query\n"
    )

    # Findings carry `lines` only (no `line`). _issue_for_fix recovers each fix's rationale
    # from here by matching rule + overlapping lines.
    test_enriched_findings = [
        {
            "rule": "E401",
            "lines": [1],
            "category": "Style",
            "severity": "LOW",
            "rationale": "Multiple imports on one line should be split onto separate lines.",
            "best_practice_refs": [],
            "doc_url": "https://docs.astral.sh/ruff/rules/multiple-imports-on-one-line",
        },
        {
            "rule": "F401",
            "lines": [1],
            "category": "Logic",
            "severity": "LOW",
            "rationale": "'os' is imported but never used and should be removed.",
            "best_practice_refs": [],
            "doc_url": "https://docs.astral.sh/ruff/rules/unused-import",
        },
        {
            "rule": "B608",
            "lines": [3],
            "category": "Security",
            "severity": "HIGH",
            "rationale": "String-based SQL query construction allows injection attacks.",
            "best_practice_refs": [
                {
                    "source": "company_rules",
                    "section": "1.3",
                    "text": "Never construct SQL queries via string concatenation.",
                }
            ],
            "doc_url": "https://bandit.readthedocs.io/en/latest/plugins/b608_hardcoded_sql_expressions.html",
        },
        {
            "rule": "W291",
            "lines": [1],
            "category": "Style",
            "severity": "LOW",
            "rationale": "Trailing whitespace should be removed.",
            "best_practice_refs": [],
            "doc_url": "https://docs.astral.sh/ruff/rules/trailing-whitespace",
        },
    ]

    # Fixes mirror real Optimizer output: each carries finding_keys plus the model's own
    # output. Three shapes are exercised: a multi-finding fix (fans out to two report
    # entries), a single-finding fix, and a null fix (UNRESOLVABLE without an LLM call).
    test_fixes = [
        {
            "finding_keys": [
                {"rule": "E401", "lines": [1], "category": "Style"},
                {"rule": "F401", "lines": [1], "category": "Logic"},
            ],
            "suggested_code": "import sys\n",
            "explanation": "Removed the unused 'os' import and split the combined import line.",
            "grounded_in": ["pyguide §2.1"],
        },
        {
            "finding_keys": [
                {"rule": "B608", "lines": [3], "category": "Security"},
            ],
            "suggested_code": (
                "def get_user(id):\n"
                "    query = 'SELECT * FROM users WHERE id = ?'\n"
                "    return query, (id,)\n"
            ),
            "explanation": "Replaced string concatenation with a parameterized query.",
            "grounded_in": ["company_rules §1.3"],
        },
        {
            "finding_keys": [
                {"rule": "W291", "lines": [1], "category": "Style"},
            ],
            "suggested_code": None,
            "explanation": "Optimizer did not return a fix for this finding.",
            "grounded_in": [],
        },
    ]

    print("=" * 60)
    print("EVALUATOR AGENT — TEST RUN")
    print("=" * 60)

    result = asyncio.run(run_evaluator(test_code, test_enriched_findings, test_fixes))

    print("\n" + "=" * 60)
    print("FINAL OUTPUT:")
    print("=" * 60)
    print(json.dumps(result, indent=2))