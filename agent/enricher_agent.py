"""
Enricher Agent — Step 2 in the Code Review Pipeline.
Receives Analyzer output, enriches each finding with RAG context via knowledge_search, and submits structured enriched findings.
"""

import asyncio
import json
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from agent.agent_utils import convert_mcp_tools_to_anthropic

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
)

from tools.enricher_tools import (
    enricher_local_tools,
    run_enricher_tool,
)

LOCAL_TOOL_NAMES = {t["name"] for t in enricher_local_tools}

async def run_enricher(analyzer_output: dict) -> dict:
    """
    Connects to MCP, runs the Reviewer agent loop, returns enriched findings.

    Args:
        analyzer_output: The full dict returned by run_analyzer().

    Returns:
        dict with reviewed findings or error info.
    """
    server_params = StdioServerParameters(
        command="python",
        args=[MCP_SERVER_PATH],
    )

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            tools_result = await session.list_tools()
            mcp_tools = convert_mcp_tools_to_anthropic(tools_result.tools)
            mcp_tools = [t for t in mcp_tools if t["name"] in ENRICHER_TOOLS] # whitelist filter from config.py
            all_tools = mcp_tools + enricher_local_tools

            tool_summary = [
                f"{t['name']} ({'local' if t['name'] in LOCAL_TOOL_NAMES else 'MCP'})"
                for t in all_tools
            ]
            logger.info("Connected to MCP. Tools: %s", ", ".join(tool_summary))

            # Pass the Analyzer output as the user message
            enricher_input = analyzer_output.copy()
            enricher_input["analysis_results"] = {
                k: v for k, v in analyzer_output["analysis_results"].items()
                if k != "code"     # Remove code section as this is not relevant for the Reviewer agent
            }
            messages = [{"role": "user", "content": json.dumps(enricher_input, indent=2)}]

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
                    logger.info("Completed after %d iteration(s)", iteration + 1)
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
                        if result.get("status") == "success" and "enrichment_results" in result:  # reviewer-specific key
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

    # Default test: pipe in a realistic Analyzer output
    test_analyzer_output = {
        "status": "success",
        "analysis_results": {
            "file_path": None,
            "line_count": 4,
            "syntax_findings": [
                {"rule": "F401", "message": "`os` imported but unused", "line": 1, "severity": "HIGH", "category": "Logic", "doc_url": "https://docs.astral.sh/ruff/rules/unused-import"}
            ],
            "security_findings": [
                {"rule": "B608", "message": "Possible SQL injection via string-based query construction", "line": 3, "severity": "HIGH", "category": "Security", "doc_url": "https://bandit.readthedocs.io/en/latest/plugins/b608_hardcoded_sql_expressions.html", "cwe_id": 89, "cwe_url": "https://cwe.mitre.org/data/definitions/89.html"}
            ],
            "structure": {"functions": [{"name": "get_user", "line": 2, "args": ["id"], "has_docstring": False}], "classes": [], "imports": [{"module": "os", "alias": None}]},
            "summary": "Two findings: one unused import and one SQL injection risk.",
        },
        "metadata": {"total_syntax_findings": 1, "total_security_findings": 1, "total_findings": 2},
    }

    print("=" * 60)
    print("ENRICHER AGENT — TEST RUN")
    print("=" * 60)

    result = asyncio.run(run_enricher(test_analyzer_output))

    print("\n" + "=" * 60)
    print("FINAL OUTPUT:")
    print("=" * 60)
    print(json.dumps(result, indent=2))