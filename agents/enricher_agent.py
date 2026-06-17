"""
Enricher Agent — Step 2 in the Code Review Pipeline.
Receives a flat list of findings from the orchestrator, enriches each with RAG context
via knowledge_search in batches, and returns merged structured output.
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
    ENRICHER_PROMPT,
    ENRICHER_TOOLS,
    ENRICHER_BATCH_SIZE,
)

from tools.enricher_tools import (
    enricher_local_tools,
    run_enricher_tool,
)

LOCAL_TOOL_NAMES = {t["name"] for t in enricher_local_tools}

# === HELPER FUNCTIONS ===

def _extract_final_output(messages: list, final_response) -> dict:
    """
    Extracts the submit_enrichment result from the conversation history.

    Pipeline: called by _run_enricher_batch once the model stops with
    stop_reason='end_turn' (i.e. after it called submit_enrichment).

    Args:
        messages:       Full conversation message list for this batch.
        final_response: Last Anthropic API response object (fallback source only).

    Returns:
        dict containing the validated enrichment output, or error info.
    """
    if not isinstance(messages, list):
        logger.error(
            "_extract_final_output: messages must be a list, got %s",
            type(messages).__name__,
        )
        return {"status": "error", "message": f"Invalid input: messages must be list, got {type(messages).__name__}"}

    try:
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

        # no valid tool_result found — fall back to parsing the raw text response
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


# === AGENT LOOP ===

async def _run_enricher_batch(findings_batch: list, session: ClientSession, all_tools: list) -> dict:
    """
    Runs the Enricher agent loop on a single batch of findings.

    Pipeline: called by run_enricher once per batch, after the MCP session and tool
    list are set up. Each batch call is independent — the session is shared, but
    message history is not.

    Args:
        findings_batch: Subset of findings for this batch.
        session:        Active MCP client session, shared across all batches.
        all_tools:      Combined MCP + local tool list, built once by run_enricher.

    Returns:
        dict with enriched findings for this batch, or error info.
    """
    if not isinstance(findings_batch, list):
        logger.error(
            "_run_enricher_batch: findings_batch must be a list, got %s",
            type(findings_batch).__name__,
        )
        return {"status": "error", "message": f"Invalid input: findings_batch must be list, got {type(findings_batch).__name__}"}

    if not isinstance(all_tools, list):
        logger.error(
            "_run_enricher_batch: all_tools must be a list, got %s",
            type(all_tools).__name__,
        )
        return {"status": "error", "message": f"Invalid input: all_tools must be list, got {type(all_tools).__name__}"}

    if session is None:
        logger.error("_run_enricher_batch: session must not be None")
        return {"status": "error", "message": "Invalid input: session is None"}

    try:
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

        # loop exhausted without a valid submit_enrichment call — expected budget failure, not a bug
        logger.warning("Reached max iterations (%d) without valid output", MAX_ITERATIONS)
        return {
            "status": "max_iterations_reached",
            "message": "Max iterations reached without valid output",
        }

    except Exception as e:
        # unexpected: Anthropic API error, network failure, malformed response object, etc.
        logger.error("Unexpected error in _run_enricher_batch: %s", str(e))
        return {
            "status": "error",
            "message": f"Unexpected error — likely API or network failure: {str(e)}",
        }


async def run_enricher(findings: list) -> dict:
    """
    Connects to MCP, enriches findings in batches, returns merged enriched output.

    Pipeline: Step 2 in the pipeline. Called by the orchestrator with the flat findings
    list produced by the Analyzer. Manages the MCP session for the full enrichment run.

    Args:
        findings: Flat list of finding dicts from the orchestrator.

    Returns:
        dict with merged enriched findings and metadata, or error info.
        Always returns a structured dict — never raises.
    """
    if not isinstance(findings, list):
        logger.error("run_enricher: findings must be a list, got %s", type(findings).__name__)
        return {"status": "error", "message": f"Invalid input: findings must be list, got {type(findings).__name__}"}

    if not findings:    # legitimate path: analyzer found no issues in clean code (Test Case 2)
        return {
            "status": "success",
            "enrichment_results": {"findings": [], "summary": "No findings to enrich.", "rag_used": False},
            "metadata": {"total_reviewed_findings": 0, "rag_used": False},
        }

    try:
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
                        # one failed batch aborts all — partial enrichment would produce an inconsistent output
                        logger.error(
                            "Batch %d failed with status '%s': %s",
                            i + 1, batch_result.get("status"), batch_result.get("message"),
                        )
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

    except Exception as e:
        logger.error("run_enricher failed unexpectedly: %s", str(e))
        return {
            "status": "error",
            "message": f"run_enricher failed unexpectedly: {str(e)}",
        }


# === ENTRY POINT ===

if __name__ == "__main__":
    from config import setup_logging
    setup_logging()

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