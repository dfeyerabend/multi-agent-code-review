"""
Enricher Agent — Step 2 in the Code Review Pipeline.
Receives a flat list of findings from the orchestrator, enriches each with RAG context
via knowledge_search in batches, and returns merged structured output.
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
    ENRICHER_PROMPT,
    ENRICHER_TOOLS,
    ENRICHER_BATCH_SIZE,
)

from tools.enricher_tools import (
    enricher_local_tools,
    run_enricher_tool,
)

LOCAL_TOOL_NAMES = {t["name"] for t in enricher_local_tools}

async def run_enricher(findings: list) -> dict:
    """
    Connects to MCP, enriches findings in batches, returns merged enriched output.

    Args:
        findings: Flat list of finding dicts from the orchestrator.

    Returns:
        dict with merged enriched findings or error info.
    """
    if not findings:
        return {
            "status": "success",
            "enrichment_results": {"findings": [], "summary": "No findings to enrich.", "rag_used": False},
            "metadata": {"total_reviewed_findings": 0, "rag_used": False},
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
            mcp_tools = [t for t in mcp_tools if t["name"] in ENRICHER_TOOLS]
            all_tools = mcp_tools + enricher_local_tools

            tool_summary = [
                f"{t['name']} ({'local' if t['name'] in LOCAL_TOOL_NAMES else 'MCP'})"
                for t in all_tools
            ]
            logger.info("Connected to MCP. Tools: %s", ", ".join(tool_summary))

            batches = chunk_list(findings, ENRICHER_BATCH_SIZE)
            logger.info("Processing %d finding(s) in %d batch(es)", len(findings), len(batches))

            all_enriched_findings = []
            rag_used = False

            for i, batch in enumerate(batches):
                logger.info("Batch %d/%d — %d finding(s)", i + 1, len(batches), len(batch))
                batch_result = await _run_enricher_batch(batch, session, all_tools)

                if batch_result.get("status") != "success":
                    logger.error("Batch %d failed: %s", i + 1, batch_result.get("message"))
                    return batch_result

                enrichment = batch_result["enrichment_results"]
                all_enriched_findings.extend(enrichment["findings"])
                rag_used = rag_used or enrichment["rag_used"]

            by_category = {}
            for f in all_enriched_findings:
                cat = f.get("category", "Unknown")
                by_category[cat] = by_category.get(cat, 0) + 1
            category_summary = " / ".join(f"{v} {k}" for k, v in sorted(by_category.items()))
            total = len(all_enriched_findings)
            summary = f"{total} finding(s) enriched: {category_summary}. RAG {'used' if rag_used else 'not used'}."

            return {
                "status": "success",
                "enrichment_results": {
                    "findings": all_enriched_findings,
                    "summary": summary,
                    "rag_used": rag_used,
                },
                "metadata": {
                    "total_reviewed_findings": total,
                    "rag_used": rag_used,
                },
            }


async def _run_enricher_batch(findings_batch: list, session: ClientSession, all_tools: list) -> dict:
    """
    Runs the Enricher agent loop on a single batch of findings.

    Args:
        findings_batch: Subset of findings/issues for this iteration.
        session:        Active MCP client session, shared across all batches.
        all_tools:      Combined MCP + local tool list, built once by the wrapper.

    Returns:
        dict with enriched findings for this batch, or error info.
    """
    messages = [{"role": "user", "content": json.dumps(findings_batch, indent=2)}]

    for iteration in range(MAX_ITERATIONS):
        logger.debug("Iteration %d/%d", iteration + 1, MAX_ITERATIONS)

        response = client.messages.create(
            model=MODEL,
            max_tokens=MAX_TOKENS,
            system=ENRICHER_PROMPT,
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
                        tool_output = run_enricher_tool(tool_name, tool_args)
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
    Extracts the submit_enrichment result from the conversation history.

    Args:
        messages:       Full conversation message list.
        final_response: The last response object from the Anthropic API.

    Returns:
        dict containing the validated review output, or error info.
    """
    for msg in reversed(messages):
        if msg.get("role") == "user" and isinstance(msg.get("content"), list):
            for block in msg["content"]:
                if block.get("type") == "tool_result":
                    try:
                        result = json.loads(block["content"])
                        if result.get("status") == "success" and "enrichment_results" in result:
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

    # Default test: flat findings list as the orchestrator will produce
    test_findings = [
        {"rule": "F401", "message": "`os` imported but unused", "line": 1, "severity": "HIGH", "category": "Logic",
         "doc_url": "https://docs.astral.sh/ruff/rules/unused-import"},
        {"rule": "B608", "message": "Possible SQL injection via string-based query construction", "line": 3,
         "severity": "HIGH", "category": "Security",
         "doc_url": "https://bandit.readthedocs.io/en/latest/plugins/b608_hardcoded_sql_expressions.html",
         "cwe_id": 89},
    ]

    print("=" * 60)
    print("ENRICHER AGENT — TEST RUN")
    print("=" * 60)

    result = asyncio.run(run_enricher(test_findings))

    print("\n" + "=" * 60)
    print("FINAL OUTPUT:")
    print("=" * 60)
    print(json.dumps(result, indent=2))