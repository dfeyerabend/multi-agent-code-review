"""
Optimizer Agent — Step 3 in the Code Review Pipeline.
Receives enriched findings and source code from the orchestrator, generates concrete
fix suggestions per finding in batches, and returns merged structured output.
"""

import asyncio
import json
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from agents.agent_utils import convert_mcp_tools_to_anthropic, chunk_list

import logging
logger = logging.getLogger(__name__)

from config import (
    client,
    MODEL,
    MAX_TOKENS,
    MAX_ITERATIONS,
    MCP_SERVER_PATH,
    OPTIMIZER_PROMPT,
    OPTIMIZER_TOOLS,
    OPTIMIZER_STYLE_BATCH_SIZE,
    OPTIMIZER_FORCE_GROUPED,
    OPTIMIZER_FORCE_INDIVIDUAL,
)

from tools.optimizer_tools import (
    optimizer_local_tools,
    run_optimizer_tool,
)

LOCAL_TOOL_NAMES = {t["name"] for t in optimizer_local_tools}

# === HELPER FUNCTIONS ===

def _route_findings(findings: list) -> tuple[list, dict[str, list]]:
    """
    Splits findings into individual and grouped buckets for routing.

    Pipeline: called once by run_optimizer after MCP session opens, before
    any LLM calls. Determines which findings get their own call and which
    are batched together by rule code.

    Style findings are grouped by rule code so all instances of the same
    rule (e.g. 20 × W291) are resolved in one LLM call. Security, Logic,
    and Maintainability findings get individual calls by default — mixed
    context degrades fix quality for complex issues.

    Config overrides (OPTIMIZER_FORCE_GROUPED / OPTIMIZER_FORCE_INDIVIDUAL)
    let specific rule IDs opt out of the default without changing this logic.

    Args:
        findings: List of enriched finding dicts from the Enricher.

    Returns:
        Tuple of:
        - individual_findings: flat list, one LLM call each.
        - style_groups: dict mapping rule_code → list of findings,
          one LLM call per group (chunked later at OPTIMIZER_STYLE_BATCH_SIZE).
        Returns ([], {}) on invalid input or unexpected failure.
    """
    if not isinstance(findings, list):
        logger.error("_route_findings: findings must be a list, got %s", type(findings).__name__)
        return [], {}

    try:
        individual_findings = []
        style_groups: dict[str, list] = {}

        for finding in findings:
            if not isinstance(finding, dict):  # a non-dict finding cannot be routed — skip rather than crash the whole batch
                logger.warning("_route_findings: skipping non-dict finding: %s", type(finding).__name__)
                continue

            category  = finding.get("category", "")
            rule_code = finding.get("rule", "unknown")

            if category == "Style" and rule_code not in OPTIMIZER_FORCE_INDIVIDUAL:
                style_groups.setdefault(rule_code, []).append(finding)

            elif category != "Style" and rule_code in OPTIMIZER_FORCE_GROUPED:
                # Config override: treat this Security/Logic/Maintainability rule as grouped
                style_groups.setdefault(rule_code, []).append(finding)

            else:
                individual_findings.append(finding)

        logger.info(
            "_route_findings: %d individual | %d style group(s) covering %d finding(s)",
            len(individual_findings),
            len(style_groups),
            sum(len(v) for v in style_groups.values()),
        )
        return individual_findings, style_groups

    except Exception as e:
        logger.error("_route_findings failed unexpectedly: %s", str(e))
        return [], {}


def _extract_final_output(messages: list, final_response) -> dict:
    """
    Extracts the submit_optimization result from the conversation history.

    Pipeline: called by _run_optimizer_batch once the model stops with
    stop_reason='end_turn' (i.e. after it called submit_optimization).

    Args:
        messages:       Full conversation message list.
        final_response: The last response object from the Anthropic API (fallback source only).

    Returns:
        dict containing the validated optimization output, or error info.
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
                            if result.get("status") == "success" and "fixes" in result:  # tool now returns fixes/summary flat, not nested
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


def _merge_fixes(findings_batch: list, fixes: list) -> list:
    """
    Reattaches finding identity (finding_rule, finding_line, lines, category) from the
    original findings onto the model's index-keyed fix output.

    Pipeline: called by _run_optimizer_batch after a successful submit_optimization
    result, before the batch's fixes are returned to run_optimizer.

    Identity fields must never come from the model — see the de-laundering rule in
    CLAUDE.md. findings_batch entries carry an 'index' field (attached by
    _run_optimizer_batch before the LLM call) that fixes reference to say which
    finding they address.

    Args:
        findings_batch: Findings sent to the model for this batch, each with an
                         'index' field already attached.
        fixes:          Validated fix list from submit_optimization, each item
                         carrying 'index', 'suggested_code', 'explanation', 'grounded_in'.

    Returns:
        List of merged fix dicts, one per finding in findings_batch, in the same
        finding_rule/finding_line/suggested_code/explanation/grounded_in shape the
        Evaluator already expects, plus carried-through lines/category. A finding
        with no matching fix gets a null-suggested_code entry instead of being dropped.
    """
    if not isinstance(findings_batch, list):
        logger.error("_merge_fixes: findings_batch must be a list, got %s", type(findings_batch).__name__)
        return []

    if not isinstance(fixes, list):
        logger.error("_merge_fixes: fixes must be a list, got %s", type(fixes).__name__)
        fixes = []  # treat as no fixes — every finding still needs an entry, see fallback below

    try:
        # Index the model's fixes once so each finding can look itself up by position,
        # not by retyped identity — that lookup IS the de-laundering boundary.
        fixes_by_index = {}
        for i, fix in enumerate(fixes):
            if not isinstance(fix, dict):
                logger.warning("_merge_fixes: skipping fixes[%d] — not a dict (%s)", i, type(fix).__name__)
                continue
            idx = fix.get("index")
            if not isinstance(idx, int) or isinstance(idx, bool):
                logger.warning("_merge_fixes: skipping fixes[%d] — invalid index %r", i, idx)
                continue
            fixes_by_index[idx] = fix

        merged = []
        for finding in findings_batch:
            try:
                idx = finding.get("index")
                fix = fixes_by_index.get(idx)

                if fix is None:  # model never addressed this finding — must not silently vanish from the report
                    logger.warning(
                        "_merge_fixes: no fix for finding index %s (rule %s line %s) — emitting null fix",
                        idx, finding.get("rule"), finding.get("line"),
                    )
                    merged.append({
                        "finding_rule": finding.get("rule"),
                        "finding_line": finding.get("line"),
                        "lines": finding.get("lines"),
                        "category": finding.get("category"),
                        "suggested_code": None,
                        "explanation": "Optimizer did not return a fix for this finding.",
                        "grounded_in": [],
                    })
                    continue

                merged.append({
                    "finding_rule": finding.get("rule"),
                    "finding_line": finding.get("line"),
                    "lines": finding.get("lines"),
                    "category": finding.get("category"),
                    "suggested_code": fix.get("suggested_code"),
                    "explanation": fix.get("explanation", ""),
                    "grounded_in": fix.get("grounded_in", []),
                })

            except Exception as e:
                # one malformed finding must not drop the rest of the batch from the report
                logger.error("_merge_fixes: failed to merge finding %r: %s", finding, str(e))
                merged.append({
                    "finding_rule": finding.get("rule") if isinstance(finding, dict) else None,
                    "finding_line": finding.get("line") if isinstance(finding, dict) else None,
                    "lines": finding.get("lines") if isinstance(finding, dict) else None,
                    "category": finding.get("category") if isinstance(finding, dict) else None,
                    "suggested_code": None,
                    "explanation": f"Optimizer merge failed for this finding: {str(e)}",
                    "grounded_in": [],
                })

        return merged

    except Exception as e:
        logger.error("_merge_fixes failed unexpectedly: %s", str(e))
        # last-resort fallback: still emit one null-fix entry per finding rather than dropping the batch
        return [
            {
                "finding_rule": f.get("rule") if isinstance(f, dict) else None,
                "finding_line": f.get("line") if isinstance(f, dict) else None,
                "lines": f.get("lines") if isinstance(f, dict) else None,
                "category": f.get("category") if isinstance(f, dict) else None,
                "suggested_code": None,
                "explanation": f"Optimizer merge failed unexpectedly: {str(e)}",
                "grounded_in": [],
            }
            for f in findings_batch if isinstance(f, dict)
        ]

# === AGENT LOOP ===

async def _run_optimizer_batch(
    code: str,
    findings_batch: list,
    session: ClientSession,
    all_tools: list,
) -> dict:
    """
    Runs the Optimizer agent loop on a single batch of findings.

    Pipeline: called by run_optimizer_single (one finding) and
    run_optimizer_group (one rule-code chunk). Each call is fully
    independent — no shared state between batches.

    Each finding gets a batch-local 'index' attached before being sent to the
    model, so the model can reference findings without retyping their identity.
    After a successful submit_optimization call, _merge_fixes reattaches
    finding_rule/finding_line/lines/category from the original findings —
    those fields are never trusted from the model's output.

    Args:
        code:           Full source code string, passed once per batch.
        findings_batch: Subset of enriched findings for this iteration.
        session:        Active MCP client session, shared across all batches.
        all_tools:      Combined MCP + local tool list, built once by the wrapper.

    Returns:
        dict with fix suggestions for this batch, or error info.
        Status is 'max_iterations_reached' when the loop exhausts its budget
        (predictable end-state), or 'error' for unexpected failures.
    """
    if not isinstance(code, str):
        logger.error("_run_optimizer_batch: code must be a str, got %s", type(code).__name__)
        return {"status": "error", "message": f"Invalid input: code must be str, got {type(code).__name__}"}

    if not isinstance(findings_batch, list):
        logger.error("_run_optimizer_batch: findings_batch must be a list, got %s", type(findings_batch).__name__)
        return {"status": "error", "message": f"Invalid input: findings_batch must be list, got {type(findings_batch).__name__}"}

    if not findings_batch:
        logger.warning("_run_optimizer_batch: received empty findings_batch — nothing to process")
        return {"status": "error", "message": "Empty findings_batch — nothing to process"}

    if not isinstance(all_tools, list):
        logger.error("_run_optimizer_batch: all_tools must be a list, got %s", type(all_tools).__name__)
        return {"status": "error", "message": f"Invalid input: all_tools must be list, got {type(all_tools).__name__}"}

    try:
        # Tag each finding with its batch-local position so the model can reference
        # it by index instead of retyping finding_rule/finding_line into its output.
        indexed_batch = [
            {**finding, "index": i} if isinstance(finding, dict) else finding
            for i, finding in enumerate(findings_batch)
        ]

        batch_input = {
            "code": code,
            "findings": indexed_batch,
        }
        messages = [{"role": "user", "content": json.dumps(batch_input, indent=2)}]

        for iteration in range(MAX_ITERATIONS):
            logger.debug("Iteration %d/%d", iteration + 1, MAX_ITERATIONS)

            response = client.messages.create(
                model=MODEL,
                max_tokens=MAX_TOKENS,
                system=OPTIMIZER_PROMPT,
                tools=all_tools,
                messages=messages,
            )

            logger.debug("Stop reason: %s", response.stop_reason)

            if response.stop_reason == "end_turn":
                final_output = _extract_final_output(messages, response)
                logger.info("Batch completed after %d iteration(s)", iteration + 1)

                if final_output.get("status") == "success" and isinstance(final_output.get("fixes"), list):
                    final_output["fixes"] = _merge_fixes(indexed_batch, final_output["fixes"])

                return final_output

            elif response.stop_reason == "tool_use":
                messages.append({"role": "assistant", "content": response.content})

                tool_results = []
                for block in response.content:
                    if hasattr(block, "text") and block.text:
                        logger.debug("Claude says: %s", block.text[:200])

                    if block.type == "tool_use":
                        logger.debug("Tool call: %s | args: %s", block.name, str(block.input)[:200])
                        tool_name = block.name
                        tool_args = block.input

                        if tool_name in LOCAL_TOOL_NAMES:
                            logger.debug("Calling tool: %s (local)", tool_name)
                            tool_output = run_optimizer_tool(tool_name, tool_args)
                        else:
                            logger.debug("Calling tool: %s (MCP)", tool_name)
                            try:
                                result = await session.call_tool(tool_name, arguments=tool_args)
                                tool_output = result.content[0].text if result.content else ""
                            except Exception as e:
                                tool_output = json.dumps({"status": "error", "message": str(e)})
                                logger.warning("Tool error on %s: %s", tool_name, e)

                        logger.debug("Tool result for %s: %s", tool_name, tool_output[:300])
                        tool_results.append({
                            "type": "tool_result",
                            "tool_use_id": block.id,
                            "content": tool_output,
                        })

                messages.append({"role": "user", "content": tool_results})

        # loop exhausted without a valid submit_optimization call — expected budget failure, not a bug
        logger.warning("Reached max iterations (%d) without valid output", MAX_ITERATIONS)
        return {
            "status": "max_iterations_reached",
            "message": "Max iterations reached without valid output",
        }

    except Exception as e:
        # unexpected: Anthropic API error, network failure, malformed response object, etc.
        logger.error("Unexpected error in _run_optimizer_batch: %s", str(e))
        return {
            "status": "error",
            "message": f"Unexpected error — likely API or network failure: {str(e)}",
        }


async def run_optimizer_single(
    finding: dict,
    code: str,
    session: ClientSession,
    all_tools: list,
) -> dict:
    """
    Runs one Optimizer LLM call for a single finding.

    Pipeline: called by run_optimizer for every Security, Logic, and
    Maintainability finding, and for any Style rule overridden via
    OPTIMIZER_FORCE_INDIVIDUAL.

    Args:
        finding:    Single enriched finding dict from the Enricher.
        code:       Full source code string from the Analyzer.
        session:    Active MCP client session, shared across all calls.
        all_tools:  Combined MCP + local tool list.

    Returns:
        dict with fix output for this finding, or error info.
    """
    if not isinstance(finding, dict):
        logger.error("run_optimizer_single: finding must be a dict, got %s", type(finding).__name__)
        return {"status": "error", "message": f"Invalid input: finding must be dict, got {type(finding).__name__}"}

    if not isinstance(code, str):
        logger.error("run_optimizer_single: code must be a str, got %s", type(code).__name__)
        return {"status": "error", "message": f"Invalid input: code must be str, got {type(code).__name__}"}

    if not isinstance(all_tools, list):
        logger.error("run_optimizer_single: all_tools must be a list, got %s", type(all_tools).__name__)
        return {"status": "error", "message": f"Invalid input: all_tools must be list, got {type(all_tools).__name__}"}

    try:
        return await _run_optimizer_batch(code, [finding], session, all_tools)
    except Exception as e:
        logger.error("run_optimizer_single failed unexpectedly: %s", str(e))
        return {"status": "error", "message": f"Unexpected error in run_optimizer_single: {str(e)}"}


async def run_optimizer_group(
    findings: list,
    code: str,
    session: ClientSession,
    all_tools: list,
) -> dict:
    """
    Runs one Optimizer LLM call for a group of same-rule findings.

    Pipeline: called by run_optimizer for each Style rule-code group
    (chunked at OPTIMIZER_STYLE_BATCH_SIZE before this is called).

    Args:
        findings:   List of findings sharing the same rule code.
        code:       Full source code string from the Analyzer.
        session:    Active MCP client session, shared across all calls.
        all_tools:  Combined MCP + local tool list.

    Returns:
        dict with fix output for this group, or error info.
    """
    if not isinstance(findings, list):
        logger.error("run_optimizer_group: findings must be a list, got %s", type(findings).__name__)
        return {"status": "error", "message": f"Invalid input: findings must be list, got {type(findings).__name__}"}

    if not findings:
        logger.warning("run_optimizer_group: received empty findings list — nothing to process")
        return {"status": "error", "message": "Empty findings list — nothing to process"}

    if not isinstance(code, str):
        logger.error("run_optimizer_group: code must be a str, got %s", type(code).__name__)
        return {"status": "error", "message": f"Invalid input: code must be str, got {type(code).__name__}"}

    if not isinstance(all_tools, list):
        logger.error("run_optimizer_group: all_tools must be a list, got %s", type(all_tools).__name__)
        return {"status": "error", "message": f"Invalid input: all_tools must be list, got {type(all_tools).__name__}"}

    try:
        return await _run_optimizer_batch(code, findings, session, all_tools)
    except Exception as e:
        logger.error("run_optimizer_group failed unexpectedly: %s", str(e))
        return {"status": "error", "message": f"Unexpected error in run_optimizer_group: {str(e)}"}


async def run_optimizer(code: str, enriched_findings: list) -> dict:
    """
    Connects to MCP, routes findings, generates fixes, returns merged output.

    Pipeline: Step 3 in the pipeline. Called by the orchestrator with the
    full source code and all enriched findings from the Enricher.

    Security, Logic, and Maintainability findings are processed individually
    — one LLM call each — to keep fix reasoning focused. Style findings are
    grouped by rule code and processed in batches of OPTIMIZER_STYLE_BATCH_SIZE
    so repeated instances of the same rule (e.g. trailing whitespace) are
    resolved in a single call.

    Args:
        code:              Full source code string from the Analyzer.
        enriched_findings: List of enriched finding dicts from the Enricher.

    Returns:
        dict with merged fix suggestions across all findings, or error info.
        Always returns a structured dict — never raises.
    """
    if not isinstance(code, str):
        logger.error("run_optimizer: code must be a str, got %s", type(code).__name__)
        return {"status": "error", "message": f"Invalid input: code must be str, got {type(code).__name__}"}

    if not isinstance(enriched_findings, list):
        logger.error("run_optimizer: enriched_findings must be a list, got %s", type(enriched_findings).__name__)
        return {"status": "error", "message": f"Invalid input: enriched_findings must be list, got {type(enriched_findings).__name__}"}

    if not enriched_findings:           # legitimate path: analyzer found no issues in clean code (Test Case 2)
        return {
            "status": "success",
            "optimization_results": {"fixes": [], "summary": "No findings to optimize."},
            "metadata": {"total_fixes": 0},
        }

    server_params = StdioServerParameters(
        command="python",
        args=[MCP_SERVER_PATH],
    )

    try:
        async with stdio_client(server_params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()

                tool_list = await session.list_tools()
                mcp_tools = convert_mcp_tools_to_anthropic(tool_list.tools)
                mcp_tools = [t for t in mcp_tools if t["name"] in OPTIMIZER_TOOLS]
                all_tools = mcp_tools + optimizer_local_tools

                tool_summary = [
                    f"{t['name']} ({'local' if t['name'] in LOCAL_TOOL_NAMES else 'MCP'})"
                    for t in all_tools
                ]
                logger.info("Connected to MCP. Tools: %s", ", ".join(tool_summary))

                individual_findings, style_groups = _route_findings(enriched_findings)

                total_calls = len(individual_findings) + sum(
                    -(-len(v) // OPTIMIZER_STYLE_BATCH_SIZE)
                    for v in style_groups.values()
                )
                logger.info(
                    "Routing: %d individual call(s) + %d style group(s) → %d total call(s)",
                    len(individual_findings), len(style_groups), total_calls,
                )

                all_fixes = []
                failed_count = 0

                # --- Individual findings ---
                for i, finding in enumerate(individual_findings):
                    logger.info(
                        "Individual call %d/%d — rule %s line %d",
                        i + 1, len(individual_findings),
                        finding.get("rule"), finding.get("line"),
                    )
                    result = await run_optimizer_single(finding, code, session, all_tools)

                    if result.get("status") != "success":
                        logger.error(
                            "Individual call failed for rule %s line %s — skipping. Reason: %s",
                            finding.get("rule"), finding.get("line"), result.get("message"),
                        )
                        all_fixes.append({
                            "finding_rule": finding.get("rule"),
                            "finding_line": finding.get("line"),
                            "lines": finding.get("lines"),
                            "category": finding.get("category"),
                            "suggested_code": None,
                            "explanation": f"Optimizer failed: {result.get('message', 'unknown error')}",
                            "grounded_in": [],
                        })
                        failed_count += 1
                        continue

                    fixes = result.get("fixes")
                    if not isinstance(fixes, list):     # internal contract violation — silently extending with a non-list would corrupt all_fixes
                        logger.error(
                            "Individual call for rule %s returned malformed fixes — expected list, got %s",
                            finding.get("rule"), type(fixes).__name__,
                        )
                        all_fixes.append({
                            "finding_rule": finding.get("rule"),
                            "finding_line": finding.get("line"),
                            "lines": finding.get("lines"),
                            "category": finding.get("category"),
                            "suggested_code": None,
                            "explanation": "Optimizer returned malformed output — no fixes list present.",
                            "grounded_in": [],
                        })
                        failed_count += 1
                        continue

                    all_fixes.extend(fixes)

                # --- Style groups ---
                for rule_code, group_findings in style_groups.items():
                    chunks = chunk_list(group_findings, OPTIMIZER_STYLE_BATCH_SIZE)
                    logger.info(
                        "Style group '%s': %d finding(s) → %d chunk(s)",
                        rule_code, len(group_findings), len(chunks),
                    )
                    for j, chunk in enumerate(chunks):
                        logger.info(
                            "Style group '%s' chunk %d/%d — %d finding(s)",
                            rule_code, j + 1, len(chunks), len(chunk),
                        )
                        result = await run_optimizer_group(chunk, code, session, all_tools)

                        if result.get("status") != "success":
                            logger.error(
                                "Style group '%s' chunk %d failed — skipping. Reason: %s",
                                rule_code, j + 1, result.get("message"),
                            )
                            for finding in chunk:
                                all_fixes.append({
                                    "finding_rule": finding.get("rule"),
                                    "finding_line": finding.get("line"),
                                    "lines": finding.get("lines"),
                                    "category": finding.get("category"),
                                    "suggested_code": None,
                                    "explanation": f"Optimizer failed: {result.get('message', 'unknown error')}",
                                    "grounded_in": [],
                                })
                            failed_count += len(chunk)
                            continue

                        fixes = result.get("fixes")
                        if not isinstance(fixes, list):     # internal contract violation — silently extending with a non-list would corrupt all_fixes
                            logger.error(
                                "Style group '%s' chunk %d returned malformed fixes — expected list, got %s",
                                rule_code, j + 1, type(fixes).__name__,
                            )
                            for finding in chunk:
                                all_fixes.append({
                                    "finding_rule": finding.get("rule"),
                                    "finding_line": finding.get("line"),
                                    "lines": finding.get("lines"),
                                    "category": finding.get("category"),
                                    "suggested_code": None,
                                    "explanation": "Optimizer returned malformed output — no fixes list present.",
                                    "grounded_in": [],
                                })
                            failed_count += len(chunk)
                            continue

                        all_fixes.extend(fixes)

                total = len(all_fixes)
                summary = (
                    f"{total - failed_count} fix(es) generated, "
                    f"{failed_count} finding(s) unresolved."
                    if failed_count
                    else f"{total} fix(es) generated for {len(enriched_findings)} finding(s)."
                )

                return {
                    "status": "success",
                    "optimization_results": {
                        "fixes": all_fixes,
                        "summary": summary,
                    },
                    "metadata": {
                        "total_fixes": total,
                        "failed_count": failed_count,
                    },
                }

    except Exception as e:
        logger.error("run_optimizer failed — MCP connection or session error: %s", str(e))
        return {
            "status": "error",
            "message": f"MCP connection failed — likely mcp_server.py is missing or has a syntax error: {str(e)}",
        }


# === ENTRY POINT ===

if __name__ == "__main__":
    from config import setup_logging
    setup_logging()

    # Trailing whitespace deliberately placed on lines 2 and 4 so the W291 finding
    # below carries a real multi-entry `lines` list — exercises the duplicate-line path.
    test_code = (
        "import os, sys\n"
        "import json   \n"                                          # line 2: trailing whitespace
        "def get_user(id):\n"
        "    query = 'SELECT * FROM users WHERE id = ' + id   \n"    # line 4: SQL injection + trailing whitespace
        "    return query\n"
    )

    # Findings mirror real Enricher output: `lines` and `occurrences` are ALWAYS present,
    # and `line` always equals `lines[0]`. The W291 finding spans two lines to prove the
    # optimizer addresses every entry in `lines`, not just the first.
    test_findings = [
        {
            "rule": "B608",
            "line": 4,
            "lines": [4],
            "occurrences": 1,
            "category": "Security",
            "severity": "HIGH",
            "rationale": "String-based SQL query construction allows injection attacks.",
            "best_practice_refs": [],
            "doc_url": "https://bandit.readthedocs.io/en/latest/plugins/b608_hardcoded_sql_expressions.html",
            "cwe_id": 89,
        },
        {
            "rule": "W291",
            "line": 2,
            "lines": [2, 4],
            "occurrences": 2,
            "category": "Style",
            "severity": "LOW",
            "rationale": "Trailing whitespace should be removed.",
            "best_practice_refs": [],
            "doc_url": "https://docs.astral.sh/ruff/rules/trailing-whitespace",
            "cwe_id": None,
        },
    ]

    print("=" * 60)
    print("OPTIMIZER AGENT — TEST RUN")
    print("=" * 60)

    result = asyncio.run(run_optimizer(test_code, test_findings))

    print("\n" + "=" * 60)
    print("FINAL OUTPUT:")
    print("=" * 60)
    print(json.dumps(result, indent=2))