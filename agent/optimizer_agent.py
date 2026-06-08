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
    OPTIMIZER_BATCH_SIZE,
)

from tools.optimizer_tools import (
    optimizer_local_tools,
    run_optimizer_tool,
)

LOCAL_TOOL_NAMES = {t["name"] for t in optimizer_local_tools}


async def run_optimizer(code: str, enriched_findings: list) -> dict:
    """
    Connects to MCP, generates fixes for enriched findings in batches, returns merged output.

    Args:
        code:              Full source code string from the Analyzer.
        enriched_findings: List of enriched finding dicts from the Enricher.

    Returns:
        dict with merged fix suggestions or error info.
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

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            tools_result = await session.list_tools()
            mcp_tools = convert_mcp_tools_to_anthropic(tools_result.tools)
            mcp_tools = [t for t in mcp_tools if t["name"] in OPTIMIZER_TOOLS]
            all_tools = mcp_tools + optimizer_local_tools

            tool_summary = [
                f"{t['name']} ({'local' if t['name'] in LOCAL_TOOL_NAMES else 'MCP'})"
                for t in all_tools
            ]
            logger.info("Connected to MCP. Tools: %s", ", ".join(tool_summary))

            batches = chunk_list(enriched_findings, OPTIMIZER_BATCH_SIZE)
            logger.info("Processing %d finding(s) in %d batch(es)", len(enriched_findings), len(batches))

            all_fixes = []

            for i, batch in enumerate(batches):
                logger.info("Batch %d/%d — %d finding(s)", i + 1, len(batches), len(batch))
                batch_result = await _run_optimizer_batch(code, batch, session, all_tools)

                if batch_result.get("status") != "success":
                    logger.error("Batch %d failed: %s", i + 1, batch_result.get("message"))
                    return batch_result

                all_fixes.extend(batch_result["optimization_results"]["fixes"])

            total = len(all_fixes)
            summary = f"{total} fix(es) generated for {len(enriched_findings)} finding(s)."

            return {
                "status": "success",
                "optimization_results": {
                    "fixes": all_fixes,
                    "summary": summary,
                },
                "metadata": {"total_fixes": total},
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

    logger.warning("Reached max iterations (%d)", MAX_ITERATIONS)
    return {"status": "error", "message": "Max iterations reached"}


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