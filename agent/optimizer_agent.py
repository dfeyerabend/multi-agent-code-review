"""
Optimizer Agent — Step 3 in the Code Review Pipeline.
Receives enriched findings and source code from the orchestrator, generates concrete
fix suggestions per finding in batches, and returns merged structured output.
"""

import asyncio
import json
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from agent.agent_utils import convert_mcp_tools_to_anthropic, chunk_list

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
    return await _run_optimizer_batch(code, [finding], session, all_tools)


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
    return await _run_optimizer_batch(code, findings, session, all_tools)


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
    """
    if not enriched_findings:
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
                            "Individual call failed for rule %s line %s after max iterations — skipping. Reason: %s",
                            finding.get("rule"), finding.get("line"), result.get("message"),
                        )
                        all_fixes.append({
                            "finding_rule": finding.get("rule"),
                            "finding_line": finding.get("line"),
                            "suggested_code": None,
                            "explanation": f"Optimizer failed after max iterations: {result.get('message', 'unknown error')}",
                            "grounded_in": [],
                        })
                        failed_count += 1
                        continue

                    all_fixes.extend(result["optimization_results"]["fixes"])

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
                                "Style group '%s' chunk %d failed after max iterations — skipping. Reason: %s",
                                rule_code, j + 1, result.get("message"),
                            )
                            for finding in chunk:
                                all_fixes.append({
                                    "finding_rule": finding.get("rule"),
                                    "finding_line": finding.get("line"),
                                    "suggested_code": None,
                                    "explanation": f"Optimizer failed after max iterations: {result.get('message', 'unknown error')}",
                                    "grounded_in": [],
                                })
                            failed_count += len(chunk)
                            continue

                        all_fixes.extend(result["optimization_results"]["fixes"])

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
        logger.error("Optimizer failed to connect to MCP server: %s", str(e))
        return {
            "status": "error",
            "message": f"MCP connection failed — likely mcp_server.py is missing or has a syntax error: {str(e)}",
        }


async def _run_optimizer_batch(code: str, findings_batch: list, session: ClientSession, all_tools: list) -> dict:
    """
    Runs the Optimizer agent loop on a single batch of findings.

    Args:
        code:           Full source code string, passed once per batch.
        findings_batch: Subset of enriched findings for this iteration.
        session:        Active MCP client session, shared across all batches.
        all_tools:      Combined MCP + local tool list, built once by the wrapper.

    Returns:
        dict with fix suggestions for this batch, or error info.
    """
    batch_input = {
        "code": code,
        "findings": findings_batch,
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

    logger.warning("Reached max iterations (%d) without valid output", MAX_ITERATIONS)
    return {"status": "error", "message": "Max iterations reached without valid output"}


def _extract_final_output(messages: list, final_response) -> dict:
    """
    Extracts the submit_optimization result from the conversation history.

    Args:
        messages:       Full conversation message list.
        final_response: The last response object from the Anthropic API.

    Returns:
        dict containing the validated optimization output, or error info.
    """
    for msg in reversed(messages):          # start from last message
        if msg.get("role") == "user" and isinstance(msg.get("content"), list):
            for block in msg["content"]:
                if block.get("type") == "tool_result":
                    try:
                        result = json.loads(block["content"])
                        if result.get("status") == "success" and "optimization_results" in result:
                            return result
                    except (json.JSONDecodeError, TypeError):
                        continue

    final_text = ""
    for block in final_response.content:
        if hasattr(block, "text"):
            final_text += block.text

    try:
        return json.loads(final_text)
    except json.JSONDecodeError:
        logger.error("Failed to parse final output — raw: %s", final_text[:200])
        return {"status": "error", "raw_output": final_text}


if __name__ == "__main__":
    from config import setup_logging
    setup_logging()

    test_code = (
        "import os, sys\n"
        "import json\n"
        "def get_user(id):\n"
        "    query = 'SELECT * FROM users WHERE id = ' + id\n"
        "    return query\n"
    )

    test_findings = [
        {
            "rule": "B608",
            "line": 4,
            "category": "Security",
            "severity": "HIGH",
            "rationale": "String-based SQL query construction allows injection attacks.",
            "best_practice_refs": [],
            "doc_url": "https://bandit.readthedocs.io/en/latest/plugins/b608_hardcoded_sql_expressions.html",
            "cwe_id": 89,
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

def _route_findings(
    findings: list,
) -> tuple[list, dict[str, list]]:
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
    """
    individual_findings = []
    style_groups: dict[str, list] = {}

    for finding in findings:
        category  = finding.get("category", "")
        rule_code = finding.get("rule", "unknown")

        if category == "Style" and rule_code not in OPTIMIZER_FORCE_INDIVIDUAL:
            # Group by rule code — all instances of the same Style rule in one call
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