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

def _match_pairs(enriched_findings: list, fixes: list) -> list:
    """
    Pairs each finding with its corresponding fix by rule + line.

    Pipeline: called once by run_evaluator before any LLM calls, after input
    validation passes.

    Args:
        enriched_findings: List of enriched finding dicts from the Enricher.
        fixes:             List of fix dicts from the Optimizer.

    Returns:
        List of dicts with keys 'finding' and 'fix' (fix is None when the
        Optimizer produced no entry for that finding), or an error dict on
        unexpected failure.
    """
    if not isinstance(enriched_findings, list):
        logger.error(
            "_match_pairs: enriched_findings must be a list, got %s",
            type(enriched_findings).__name__,
        )
        return []

    if not isinstance(fixes, list):
        logger.error(
            "_match_pairs: fixes must be a list, got %s",
            type(fixes).__name__,
        )
        return []

    try:
        fix_index = {
            (f.get("finding_rule"), f.get("finding_line")): f
            for f in fixes
            if isinstance(f, dict)      # non-dict entries are silently skipped — a bad fix must not block all matches
        }

        pairs = []
        for finding in enriched_findings:
            if not isinstance(finding, dict):  # a malformed finding cannot be matched or evaluated
                logger.warning("_match_pairs: skipping non-dict finding: %s", type(finding).__name__)
                continue
            key = (finding.get("rule"), finding.get("line"))
            pairs.append({
                "finding": finding,
                "fix": fix_index.get(key),  # None if optimizer produced no fix for this finding
            })

        logger.info(
            "_match_pairs: %d finding(s) → %d matched, %d unmatched",
            len(enriched_findings),
            sum(1 for p in pairs if p["fix"] is not None),
            sum(1 for p in pairs if p["fix"] is None),
        )
        return pairs

    except Exception as e:
        logger.error("_match_pairs failed unexpectedly: %s", str(e))
        return []

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


def _unresolvable_entry(
    finding_rule: str,
    finding_line,
    category: str,
    fix,
    reason: str,
) -> dict:
    """
    Builds a standardised UNRESOLVABLE entry without an LLM call.

    Pipeline: called by run_evaluator for null fixes and all validation
    failures. Centralised here so every UNRESOLVABLE entry has an identical
    schema — a missing field in any one of them would crash create_review_report.

    Args:
        finding_rule: Rule code from the original finding.
        finding_line: Line number from the original finding.
        category:     Category from the original finding.
        fix:          The fix dict, or None if no fix was produced.
        reason:       Human-readable explanation of why this is unresolvable.

    Returns:
        dict conforming to the fixes_evaluated entry schema with status UNRESOLVABLE.
    """
    if not isinstance(finding_rule, str):
        logger.error("_unresolvable_entry: finding_rule must be str, got %s", type(finding_rule).__name__)
        finding_rule = str(finding_rule)    # coerce rather than drop — losing the rule id is worse than a dirty string

    if not isinstance(category, str):
        logger.error("_unresolvable_entry: category must be str, got %s", type(category).__name__)
        category = str(category)

    if not isinstance(reason, str):
        logger.error("_unresolvable_entry: reason must be str, got %s", type(reason).__name__)
        reason = str(reason)

    if fix is not None and not isinstance(fix, dict):
        logger.error(
            "_unresolvable_entry: fix must be a dict or None, got %s — treating as None",
            type(fix).__name__,
        )
        fix = None  # a non-dict fix cannot be safely read; drop it to prevent AttributeError on .get()

    try:
        return {
            "finding_rule": finding_rule,
            "finding_line": finding_line,
            "category":     category,
            "status":       "UNRESOLVABLE",
            "suggested_code": fix.get("suggested_code") if fix else None,
            "grounded_in":    fix.get("grounded_in", []) if fix else [],
            "reasoning":    reason,
            "faithfulness": None,   # no LLM call was made — verdicts are undefined, not zero
            "correctness":  None,
            "completeness": None,
        }

    except Exception as e:
        logger.error("_unresolvable_entry failed unexpectedly: %s", str(e))
        return {
            "finding_rule": finding_rule,
            "finding_line": finding_line,
            "category":     category,
            "status":       "UNRESOLVABLE",
            "suggested_code": None,
            "grounded_in":    [],
            "reasoning":    f"Entry construction failed: {str(e)}",
            "faithfulness": None,
            "correctness":  None,
            "completeness": None,
        }

# === AGENT LOOP ===

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
        pairs = _match_pairs(enriched_findings, fixes)  # links each finding to its fix by rule + line, once before the loop
        fixes_evaluated = []

        for pair in pairs:
            finding = pair["finding"]
            fix     = pair["fix"]                               # None if optimizer produced no fix for this finding

            finding_rule = finding.get("rule", "unknown")       # carried by orchestrator — LLM never sees these
            finding_line = finding.get("line")                  # identity fields only needed for the output entry
            category     = finding.get("category", "Unknown")   # needed by create_review_report for the category matrix

            # No fix produced by optimizer → skip LLM call entirely
            if fix is None or fix.get("suggested_code") is None:
                # two cases: fix is None = optimizer produced no entry for this finding at all
                #            suggested_code is None = optimizer ran but explicitly failed (recorded as unresolved)
                # in both cases: no point calling the LLM — there is nothing to evaluate
                logger.warning(
                    "No fix for rule %s line %s — marking UNRESOLVABLE without LLM call",
                    finding_rule, finding_line,
                )
                fixes_evaluated.append(_unresolvable_entry(
                    finding_rule, finding_line, category,
                    fix,
                    reason="No fix was produced for this finding.",
                ))
                continue    # move to next pair — do not abort the whole loop

        # Build minimal LLM input — strip identity/metadata fields
            issue = {
                "rationale":          finding.get("rationale", ""),
                "best_practice_refs": finding.get("best_practice_refs", []),
                # rule, line, severity, category deliberately excluded — the LLM does not need them to judge the fix, and extra fields invite hallucination
            }

            fix_input = {
                "suggested_code": fix.get("suggested_code"),
                "explanation":    fix.get("explanation", ""),
                "grounded_in":    fix.get("grounded_in", []),
                # finding_rule and finding_line excluded — matching is already done, LLM must not re-interpret identity
            }

            logger.info("Evaluating rule %s line %s", finding_rule, finding_line)
            result = await _run_evaluator_pair(code, issue, fix_input)

            # Layer 1a: agent ran out of iterations — expected budget failure, not a bug
            if result.get("status") == "max_iterations_reached":
                logger.warning(
                    "Max iterations reached for rule %s line %s — marking UNRESOLVABLE",
                    finding_rule, finding_line,
                )
                fixes_evaluated.append(_unresolvable_entry(
                    finding_rule, finding_line, category, fix,
                    reason="Evaluator reached max iterations without producing a valid verdict.",
                ))
                continue

            # Layer 1b: unexpected error (API failure, network issue, etc.)
            if result.get("status") == "error":
                logger.error(
                    "Unexpected error for rule %s line %s: %s",
                    finding_rule, finding_line, result.get("message", "unknown error"),
                )
                fixes_evaluated.append(_unresolvable_entry(
                    finding_rule, finding_line, category, fix,
                    reason=f"Unexpected error during evaluation: {result.get('message', 'unknown error')}",
                ))
                continue

            # Validate model output before use: a partial tool call can still reach here,
            # and guessing a missing field is worse than failing loudly.
            evaluation = result.get("evaluation")
            if not isinstance(evaluation, dict):
                logger.error(
                    "Rule %s line %s: response has no 'evaluation' dict — type was %s",
                    finding_rule, finding_line, type(evaluation).__name__,
                )
                fixes_evaluated.append(_unresolvable_entry(
                    finding_rule, finding_line, category, fix,
                    reason="Evaluator returned malformed output — no evaluation dict present.",
                ))
                continue

            required = ["reasoning", "faithfulness", "correctness", "completeness"]
            missing  = [f for f in required if f not in evaluation]
            if missing:
                logger.error(
                    "Rule %s line %s: evaluation dict missing fields %s",
                    finding_rule, finding_line, missing,
                )
                fixes_evaluated.append(_unresolvable_entry(
                    finding_rule, finding_line, category, fix,
                    reason=f"Evaluator returned incomplete verdicts — missing: {missing}",
                ))
                continue

            status = _derive_status(evaluation)
            logger.info(
                "Rule %s line %s → %s | faith=%s correct=%s complete=%s",
                finding_rule, finding_line, status,
                evaluation["faithfulness"], evaluation["correctness"], evaluation["completeness"],
            )
            fixes_evaluated.append({
                "finding_rule": finding_rule,                   # orchestrator supplies identity — LLM never carried this
                "finding_line": finding_line,
                "category":     category,                       # orchestrator supplies category for report matrix
                "status":       status,                         # derived deterministically in Python, not by LLM
                "suggested_code": fix.get("suggested_code"),    # passed through unchanged from optimizer
                "grounded_in":  fix.get("grounded_in", []),     # passed through unchanged from optimizer
                "reasoning":    evaluation["reasoning"],        # from LLM
                "faithfulness": evaluation["faithfulness"],     # from LLM
                "correctness":  evaluation["correctness"],      # from LLM
                "completeness": evaluation["completeness"],     # from LLM
            })

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
        "import os\n"
        "def get_user(id):\n"
        "    query = 'SELECT * FROM users WHERE id = ' + id\n"
        "    return query\n"
    )

    test_enriched_findings = [
        {
            "rule": "B608",
            "line": 3,
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
    ]

    test_fixes = [
        {
            "finding_rule": "B608",
            "finding_line": 3,
            "suggested_code": (
                "def get_user(id):\n"
                "    query = 'SELECT * FROM users WHERE id = ?'\n"
                "    return query, (id,)\n"
            ),
            "explanation": "Replaced string concatenation with a parameterized query.",
            "grounded_in": ["company_rules §1.3"],
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