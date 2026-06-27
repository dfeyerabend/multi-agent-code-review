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

def _explode_repeats(findings: list) -> list:
    """
    Splits repetitive findings into one single-line unit per affected line.

    Pipeline: called by run_optimizer (this module) as the first step on the enriched
    findings, before _group_overlapping and _route_findings.

    A finding with occurrences > 1 is the same problem firing on several lines (e.g.
    W291 trailing whitespace on lines [2, 4]) that _deduplicate_findings (in
    tools/analyzer_tools.py) collapsed into one entry. During line-overlap grouping a
    multi-line entry would act as "glue", dragging unrelated lines into one giant batch.
    Splitting it into single-line units removes that glue and is lossless — the
    occurrences shared one identical message, so every unit inherits the same rationale.
    The gate is occurrences > 1, not len(lines) > 1: occurrences is what actually means
    "repetition", so a genuine single multi-line span (if ever introduced upstream) is
    left intact instead of being shattered into meaningless per-line fragments.

    Args:
        findings: List of enriched finding dicts from the Enricher.

    Returns:
        Flat list of finding dicts where no entry spans more than one line. Repetitive
        findings are expanded; all others pass through unchanged. Returns [] on invalid
        input, or the original findings unchanged on unexpected failure so the pipeline
        degrades without losing findings.
    """
    if not isinstance(findings, list):
        logger.error("_explode_repeats: findings must be a list, got %s", type(findings).__name__)
        return []

    try:
        units = []
        for finding in findings:
            try:
                if not isinstance(finding, dict):  # a non-dict finding cannot be exploded — skip, don't crash the batch
                    logger.warning("_explode_repeats: skipping non-dict finding: %s", type(finding).__name__)
                    continue

                occurrences = finding.get("occurrences")
                lines       = finding.get("lines")

                # A finding with no usable `lines` cannot be split or grouped on a line; pass
                # it through untouched so it is never dropped — grouping later isolates it as
                # its own singleton rather than crashing.
                if not isinstance(lines, list) or not lines:
                    logger.warning(
                        "_explode_repeats: finding has no usable lines (rule %s) — passing through unchanged",
                        finding.get("rule"),
                    )
                    units.append(finding)
                    continue

                # Repetition is the only multi-line case in this pipeline; gate on occurrences,
                # falling back to the line count only when occurrences is missing/malformed.
                occ = occurrences if isinstance(occurrences, int) and not isinstance(occurrences, bool) else len(lines)

                if occ > 1:
                    for ln in lines:
                        units.append({**finding, "lines": [ln], "occurrences": 1})
                else:
                    units.append(finding)

            except Exception as e:
                # one malformed finding must not drop its siblings — pass it through unexploded
                logger.error("_explode_repeats: failed to process finding %r: %s", finding, str(e))
                units.append(finding)

        logger.info("_explode_repeats: %d finding(s) → %d single-line unit(s)", len(findings), len(units))
        return units

    except Exception as e:
        logger.error("_explode_repeats failed unexpectedly: %s", str(e))
        return findings  # degrade gracefully: no explosion, but no findings lost


def _group_overlapping(units: list) -> list:
    """
    Groups units whose line-sets intersect into connected components.

    Pipeline: called by run_optimizer (this module) on the single-line units produced by
    _explode_repeats, before each group is routed to an LLM call.

    Two units belong together when they touch a common physical line, and the grouping is
    transitive — A with B and B with C puts all three in one group — so every problem on a
    line is resolved in one coherent call. Because _explode_repeats already removed
    multi-line repeats, no unit can bridge unrelated lines, which keeps groups local
    instead of collapsing the whole file into one batch. A unit with no usable line owns no
    lines and therefore lands in its own singleton group rather than crashing.

    Args:
        units: List of single-line finding units from _explode_repeats.

    Returns:
        List of groups, each a non-empty list of units. A non-overlapping unit yields a
        group of one. Returns [] on invalid input; on unexpected failure, falls back to one
        group per unit so nothing merges wrongly and no unit is lost.
    """
    if not isinstance(units, list):
        logger.error("_group_overlapping: units must be a list, got %s", type(units).__name__)
        return []

    try:
        # Build each unit's set of physical lines once. A non-dict or line-less unit gets an
        # empty set, which never intersects anything and so stays its own singleton.
        line_sets = []
        for unit in units:
            if not isinstance(unit, dict):
                logger.warning("_group_overlapping: non-dict unit isolated as singleton: %s", type(unit).__name__)
                line_sets.append(set())
                continue
            lines = unit.get("lines")
            if not isinstance(lines, list):
                lines = []
            usable = {ln for ln in lines if isinstance(ln, int) and not isinstance(ln, bool)}
            if not usable:
                logger.warning("_group_overlapping: unit has no usable line (rule %s) — isolating as singleton", unit.get("rule"))
            line_sets.append(usable)

        # Connected components by shared line: each component tracks its accumulated line-set
        # and its units. A new unit folds every component it touches into the first match,
        # which makes the grouping transitive across chains of overlaps.
        components = []
        for unit, lset in zip(units, line_sets):
            hits = [c for c in components if c["lines"] & lset]
            if not hits:
                components.append({"lines": set(lset), "units": [unit]})
                continue
            target = hits[0]
            target["units"].append(unit)
            target["lines"] |= lset
            for other in hits[1:]:               # absorb any further components this unit bridges
                target["units"].extend(other["units"])
                target["lines"] |= other["lines"]
                components.remove(other)

        groups = [c["units"] for c in components]
        logger.info("_group_overlapping: %d unit(s) → %d group(s)", len(units), len(groups))
        return groups

    except Exception as e:
        logger.error("_group_overlapping failed unexpectedly: %s", str(e))
        return [[u] for u in units]   # degrade: isolate every unit, lose nothing


def _route_findings(groups: list) -> tuple[list, list, dict[str, list]]:
    """
    Classifies overlap-groups into conflict batches, individual calls, and style rule-batches.

    Pipeline: called once by run_optimizer after _explode_repeats and _group_overlapping,
    before any LLM calls. Decides how many calls happen and which units share each one.

    A group with more than one unit is a genuine line conflict (e.g. E401 + two F401 on the
    same line): it always becomes ONE batch so a single coherent fix covers all of them,
    regardless of category — this is why a conflict group takes precedence over the Style
    rule-code batching below. A group of one is routed exactly as before this change: Style
    units batch by rule code (so 25 × W291 share one call), Security/Logic/Maintainability
    units get their own call. The OPTIMIZER_FORCE_GROUPED / OPTIMIZER_FORCE_INDIVIDUAL
    overrides let specific rule IDs opt out of their default.

    Args:
        groups: List of unit-groups from _group_overlapping; each group is a list of 1+ units.

    Returns:
        Tuple of:
        - conflict_groups: list of groups (each 2+ units), one batch LLM call each.
        - individual_units: flat list of units, one LLM call each.
        - style_groups: dict mapping rule_code → list of units, rule-batched later
          (chunked at OPTIMIZER_STYLE_BATCH_SIZE).
        Returns ([], [], {}) on invalid input or unexpected failure.
    """
    if not isinstance(groups, list):
        logger.error("_route_findings: groups must be a list, got %s", type(groups).__name__)
        return [], [], {}

    try:
        conflict_groups: list = []
        individual_units: list = []
        style_groups: dict[str, list] = {}

        for group in groups:
            if not isinstance(group, list) or not group:  # a malformed/empty group cannot be routed — skip rather than crash
                logger.warning("_route_findings: skipping malformed group: %r", group)
                continue

            # More than one unit means these lines genuinely overlap: one merged fix, no
            # category split — the conflict batch wins over style rule-code grouping.
            if len(group) > 1:
                conflict_groups.append(group)
                continue

            unit = group[0]
            if not isinstance(unit, dict):  # a non-dict singleton cannot be routed — skip rather than crash
                logger.warning("_route_findings: skipping non-dict unit: %s", type(unit).__name__)
                continue

            category  = unit.get("category", "")
            rule_code = unit.get("rule", "unknown")

            if category == "Style" and rule_code not in OPTIMIZER_FORCE_INDIVIDUAL:
                style_groups.setdefault(rule_code, []).append(unit)

            elif category != "Style" and rule_code in OPTIMIZER_FORCE_GROUPED:
                # Config override: treat this Security/Logic/Maintainability rule as grouped
                style_groups.setdefault(rule_code, []).append(unit)

            else:
                individual_units.append(unit)

        logger.info(
            "_route_findings: %d conflict group(s) | %d individual | %d style group(s) covering %d unit(s)",
            len(conflict_groups),
            len(individual_units),
            len(style_groups),
            sum(len(v) for v in style_groups.values()),
        )
        return conflict_groups, individual_units, style_groups

    except Exception as e:
        logger.error("_route_findings failed unexpectedly: %s", str(e))
        return [], [], {}


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


def _merge_fixes(units_batch: list, fixes: list) -> list:
    """
    Fans each fix out to the identity of every unit its `indexes` cover.

    Pipeline: called by _run_optimizer_batch (this module) after a successful
    submit_optimization result, before the batch's fixes are returned to run_optimizer.

    Identity (rule, lines, category) must never come from the model — see the de-laundering
    rule in CLAUDE.md. units_batch entries carry an 'index' field (attached by
    _run_optimizer_batch before the LLM call) that fixes reference via their 'indexes' list
    to say which units they resolve. One fix may cover several units (a line conflict
    resolved in one rewrite); its identity is the list `finding_keys`, one
    {rule, lines, category} per covered unit. category is carried per unit because a
    conflict group can mix categories (e.g. Style E401 + Logic F401 on one line).

    Args:
        units_batch: Units sent to the model for this batch, each with an 'index' field.
        fixes:       Validated fix list from submit_optimization, each carrying 'indexes',
                     'suggested_code', 'explanation', 'grounded_in'.

    Returns:
        List of fix-result dicts, one per fix, each with finding_keys plus the model's
        suggested_code/explanation/grounded_in. Every unit with no covering fix still gets
        its own null-suggested_code entry so it is never dropped from the report.
    """
    if not isinstance(units_batch, list):
        logger.error("_merge_fixes: units_batch must be a list, got %s", type(units_batch).__name__)
        return []

    if not isinstance(fixes, list):
        logger.error("_merge_fixes: fixes must be a list, got %s", type(fixes).__name__)
        fixes = []  # treat as no fixes — every unit still needs an entry, see fallback below

    try:
        # Index the batch units once so a fix can resolve each index it lists back to real
        # identity — that lookup IS the de-laundering boundary (Python carries identity, not the LLM).
        units_by_index = {}
        for unit in units_batch:
            if isinstance(unit, dict) and isinstance(unit.get("index"), int) and not isinstance(unit.get("index"), bool):
                units_by_index[unit["index"]] = unit

        merged = []
        covered_indexes = set()

        for i, fix in enumerate(fixes):
            try:
                if not isinstance(fix, dict):
                    logger.warning("_merge_fixes: skipping fixes[%d] — not a dict (%s)", i, type(fix).__name__)
                    continue

                indexes = fix.get("indexes")
                if not isinstance(indexes, list) or not indexes:  # validated upstream, but never trust LLM output unchecked
                    logger.warning("_merge_fixes: skipping fixes[%d] — invalid indexes %r", i, indexes)
                    continue

                # Build identity for every covered unit; an index with no matching unit is
                # dropped from this fix's keys and logged, never guessed.
                finding_keys = []
                for idx in indexes:
                    unit = units_by_index.get(idx)
                    if unit is None:
                        logger.warning("_merge_fixes: fixes[%d] references unknown index %r — ignoring", i, idx)
                        continue
                    finding_keys.append({
                        "rule":     unit.get("rule"),
                        "lines":    unit.get("lines"),
                        "category": unit.get("category"),
                    })
                    covered_indexes.add(idx)

                if not finding_keys:  # the fix resolved no real unit — do not emit an identity-less entry
                    logger.warning("_merge_fixes: fixes[%d] covered no known unit — dropped", i)
                    continue

                merged.append({
                    "finding_keys":   finding_keys,
                    "suggested_code": fix.get("suggested_code"),
                    "explanation":    fix.get("explanation", ""),
                    "grounded_in":    fix.get("grounded_in", []),
                })

            except Exception as e:
                # one malformed fix must not drop the rest of the batch
                logger.error("_merge_fixes: failed to merge fixes[%d]: %s", i, str(e))
                continue

        # Any unit the model never addressed must still surface — one null fix per uncovered unit.
        for unit in units_batch:
            if not isinstance(unit, dict):
                continue
            idx = unit.get("index")
            if idx in covered_indexes:
                continue
            logger.warning(
                "_merge_fixes: no fix for unit index %s (rule %s lines %s) — emitting null fix",
                idx, unit.get("rule"), unit.get("lines"),
            )
            merged.append({
                "finding_keys": [{
                    "rule":     unit.get("rule"),
                    "lines":    unit.get("lines"),
                    "category": unit.get("category"),
                }],
                "suggested_code": None,
                "explanation": "Optimizer did not return a fix for this finding.",
                "grounded_in": [],
            })

        return merged

    except Exception as e:
        logger.error("_merge_fixes failed unexpectedly: %s", str(e))
        # last-resort fallback: one null-fix entry per unit rather than dropping the batch
        return [
            {
                "finding_keys": [{
                    "rule":     u.get("rule"),
                    "lines":    u.get("lines"),
                    "category": u.get("category"),
                }],
                "suggested_code": None,
                "explanation": f"Optimizer merge failed unexpectedly: {str(e)}",
                "grounded_in": [],
            }
            for u in units_batch if isinstance(u, dict)
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
    After a successful submit_optimization call, _merge_fixes reattaches each
    fix's finding_keys (rule/lines/category per covered finding) from the original
    findings — that identity is never trusted from the model's output.

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
        # it by index instead of retyping its identity (rule/lines) into its output.
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


def _failure_fix(units: list, explanation: str) -> dict:
    """
    Builds one null-suggested fix-result entry covering the given units.

    Pipeline: called by run_optimizer (this module) on every call-failure path so a failed
    group still surfaces in the report instead of vanishing. Mirrors the finding_keys shape
    _merge_fixes produces for successful fixes, so the Evaluator sees one consistent contract.

    Args:
        units:       Units the failed call was meant to fix (1+).
        explanation: Human-readable reason the fix is missing.

    Returns:
        Fix-result dict with finding_keys for every dict unit and suggested_code=None.
    """
    try:
        keys = [
            {"rule": u.get("rule"), "lines": u.get("lines"), "category": u.get("category")}
            for u in units if isinstance(u, dict)
        ]
        return {
            "finding_keys": keys,
            "suggested_code": None,
            "explanation": explanation,
            "grounded_in": [],
        }
    except Exception as e:
        logger.error("_failure_fix failed unexpectedly: %s", str(e))
        return {"finding_keys": [], "suggested_code": None, "explanation": explanation, "grounded_in": []}


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

                # Explode repeats into single-line units, group overlapping units, then route:
                # this is the conflict-resolution pass — overlapping findings share one call.
                units            = _explode_repeats(enriched_findings)
                groups           = _group_overlapping(units)
                conflict_groups, individual_units, style_groups = _route_findings(groups)

                total_calls = (
                    len(conflict_groups)
                    + len(individual_units)
                    + sum(-(-len(v) // OPTIMIZER_STYLE_BATCH_SIZE) for v in style_groups.values())
                )
                logger.info(
                    "Routing: %d conflict group(s) + %d individual + %d style group(s) → %d total call(s)",
                    len(conflict_groups), len(individual_units), len(style_groups), total_calls,
                )

                all_fixes = []
                failed_count = 0

                # --- Conflict groups: overlapping units resolved by one merged fix per group ---
                for i, group in enumerate(conflict_groups):
                    logger.info("Conflict group %d/%d — %d unit(s)", i + 1, len(conflict_groups), len(group))
                    result = await run_optimizer_group(group, code, session, all_tools)
                    fixes  = result.get("fixes")

                    if result.get("status") != "success" or not isinstance(fixes, list):
                        reason = result.get("message") or "optimizer returned no usable fixes list"
                        logger.error("Conflict group %d failed — skipping. Reason: %s", i + 1, reason)
                        all_fixes.append(_failure_fix(group, f"Optimizer failed: {reason}"))
                        failed_count += len(group)
                        continue

                    all_fixes.extend(fixes)

                # --- Individual units: one call each (Security / Logic / Maintainability) ---
                for i, unit in enumerate(individual_units):
                    logger.info(
                        "Individual call %d/%d — rule %s lines %s",
                        i + 1, len(individual_units), unit.get("rule"), unit.get("lines"),
                    )
                    result = await run_optimizer_single(unit, code, session, all_tools)
                    fixes  = result.get("fixes")

                    if result.get("status") != "success" or not isinstance(fixes, list):
                        reason = result.get("message") or "optimizer returned no usable fixes list"
                        logger.error(
                            "Individual call for rule %s lines %s failed — skipping. Reason: %s",
                            unit.get("rule"), unit.get("lines"), reason,
                        )
                        all_fixes.append(_failure_fix([unit], f"Optimizer failed: {reason}"))
                        failed_count += 1
                        continue

                    all_fixes.extend(fixes)

                # --- Style groups: same-rule units rule-batched (chunked) ---
                for rule_code, group_units in style_groups.items():
                    chunks = chunk_list(group_units, OPTIMIZER_STYLE_BATCH_SIZE)
                    logger.info(
                        "Style group '%s': %d unit(s) → %d chunk(s)",
                        rule_code, len(group_units), len(chunks),
                    )
                    for j, chunk in enumerate(chunks):
                        logger.info(
                            "Style group '%s' chunk %d/%d — %d unit(s)",
                            rule_code, j + 1, len(chunks), len(chunk),
                        )
                        result = await run_optimizer_group(chunk, code, session, all_tools)
                        fixes  = result.get("fixes")

                        if result.get("status") != "success" or not isinstance(fixes, list):
                            reason = result.get("message") or "optimizer returned no usable fixes list"
                            logger.error("Style group '%s' chunk %d failed — skipping. Reason: %s", rule_code, j + 1, reason)
                            # independent units — one failure entry each so each surfaces separately
                            for u in chunk:
                                all_fixes.append(_failure_fix([u], f"Optimizer failed: {reason}"))
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

    # Findings mirror real Enricher output: every finding carries `lines` (a list) and
    # `occurrences` — there is no `line` scalar. This input exercises the conflict path:
    # line 1 holds three overlapping findings (E401 + two F401) that must collapse into one
    # fix; line 4 holds two (B608 + W291); and W291 spans lines 2 and 4 with occurrences=2,
    # so it is exploded into single-line units before grouping.
    test_code = (
        "import os, sys\n"                                           # line 1: E401 + F401(os) + F401(sys)
        "import json   \n"                                           # line 2: trailing whitespace
        "def get_user(id):\n"
        "    query = 'SELECT * FROM users WHERE id = ' + id   \n"    # line 4: SQL injection + trailing whitespace
        "    return query\n"
    )

    test_findings = [
        {
            "rule": "E401",
            "lines": [1],
            "occurrences": 1,
            "category": "Style",
            "severity": "LOW",
            "rationale": "Multiple imports on one line should be split onto separate lines.",
            "best_practice_refs": [],
            "doc_url": "https://docs.astral.sh/ruff/rules/multiple-imports-on-one-line",
            "cwe_id": None,
        },
        {
            "rule": "F401",
            "lines": [1],
            "occurrences": 1,
            "category": "Logic",
            "severity": "LOW",
            "rationale": "'os' is imported but never used and should be removed.",
            "best_practice_refs": [],
            "doc_url": "https://docs.astral.sh/ruff/rules/unused-import",
            "cwe_id": None,
        },
        {
            "rule": "F401",
            "lines": [1],
            "occurrences": 1,
            "category": "Logic",
            "severity": "LOW",
            "rationale": "'sys' is imported but never used and should be removed.",
            "best_practice_refs": [],
            "doc_url": "https://docs.astral.sh/ruff/rules/unused-import",
            "cwe_id": None,
        },
        {
            "rule": "B608",
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