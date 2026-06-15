"""
Evaluator Agent — Step 4 in the Code Review Pipeline.
Receives enriched findings and optimizer fixes from the orchestrator, judges each (finding, fix) pair independently, and returns a structured evaluation with a markdown report.
"""

import asyncio
import json
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from agents.agent_utils import convert_mcp_tools_to_anthropic

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

LOCAL_TOOL_NAMES = {t["name"] for t in evaluator_local_tools}

# === HELPER FUNCTIONS ===

def _match_pairs(enriched_findings: list, fixes: list) -> list:
    """
    Pairs each finding with its corresponding fix by rule + line.

    Pipeline: called once by run_evaluator before any LLM calls.

    Args:
        enriched_findings: List of enriched finding dicts from the Enricher.
        fixes:             List of fix dicts from the Optimizer.

    Returns:
        List of dicts, each with keys 'finding' and 'fix' (fix may be None
        if no match was found for that finding).
    """
    fix_index = {
        (f.get("finding_rule"), f.get("finding_line")): f
        for f in fixes
    }

    pairs = []
    for finding in enriched_findings:
        key = (finding.get("rule"), finding.get("line"))
        pairs.append({
            "finding": finding,
            "fix": fix_index.get(key),      # None if optimizer produced no fix for this finding
        })

    logger.info(
        "_match_pairs: %d finding(s) → %d matched, %d unmatched",
        len(enriched_findings),
        sum(1 for p in pairs if p["fix"] is not None),
        sum(1 for p in pairs if p["fix"] is None),
    )
    return pairs

def _derive_status(verdicts: dict) -> str:
    """
    Derives a deterministic status string from the three LLM verdicts.

    Pipeline: called by run_evaluator once per pair after run_evaluator_single returns.

    Args:
        verdicts: Dict with keys faithfulness, correctness, completeness.

    Returns:
        One of: 'APPROVED', 'NEEDS_REVISION', 'UNRESOLVABLE'.
    """
    if verdicts.get("correctness") == "fail":       # broken code overrides everything
        return "UNRESOLVABLE"

    if (
        verdicts.get("faithfulness") == "faithful"
        and verdicts.get("correctness") == "pass"
        and verdicts.get("completeness") == "complete"
    ):
        return "APPROVED"

    return "NEEDS_REVISION"

# Function to call LLM with only the required fields
async def _run_evaluator_pair (code: str, issue: dict, fix: dict) -> dict:
    """
    Runs one Evaluator LLM call for a single (issue, fix) pair.

    Pipeline: called by run_evaluator once per matched pair where suggested_code is not None.

    Args:
        code:  Full source code string from the Analyzer.
        issue: Minimal issue dict with rationale and best_practice_refs.
        fix:   Minimal fix dict with suggested_code, explanation, grounded_in.

    Returns:
        dict with evaluation verdicts on success, or error info.
    """
    batch_input = {
        "code": code,
        "issue": issue,
        "fix": fix,
    }
    messages = [{"role": "user", "content": json.dumps(batch_input, indent=2)}]

    for iteration in range(MAX_ITERATIONS):
        logger.debug("Iteration %d/%d", iteration + 1, MAX_ITERATIONS)

        response = client.messages.create(
            model=MODEL,
            max_tokens=MAX_TOKENS,
            system=EVALUATOR_PROMPT,
            tools=evaluator_local_tools,        # local only — no MCP tools
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

    logger.warning("Reached max iterations (%d) without valid output", MAX_ITERATIONS)
    return {"status": "error", "message": "Max iterations reached without valid output"}


def _extract_final_output(messages: list, final_response) -> dict:
    """
    Extracts the submit_evaluation result from the conversation history.

    Pipeline: called by run_evaluator_single on end_turn.

    Args:
        messages:       Full conversation message list.
        final_response: The last response object from the Anthropic API.

    Returns:
        dict with validated evaluation verdicts, or error info.
    """
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

    final_text = ""
    for block in final_response.content:
        if hasattr(block, "text"):
            final_text += block.text

    try:
        return json.loads(final_text)
    except json.JSONDecodeError:
        logger.error("Failed to parse final output — raw: %s", final_text[:200])
        return {"status": "error", "raw_output": final_text}