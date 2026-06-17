"""
Analyzer Agent — Step 1 in the Code Review Pipeline.
Connects to the MCP server for code analysis tools,
and uses a local submit_analysis tool for structured output.
"""

import asyncio
import json
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

import logging
logger = logging.getLogger(__name__)

from agents.agent_utils import convert_mcp_tools_to_anthropic
from config import (
    client,
    MODEL,
    MAX_TOKENS,
    MAX_ITERATIONS,
    MCP_SERVER_PATH,
    ANALYZER_PROMPT,
    ANALYZER_TOOLS,
)

from tools.analyzer_tools import (
    analyzer_local_tools,
    run_analyzer_tool,
)

LOCAL_TOOL_NAMES = {t["name"] for t in analyzer_local_tools}


# === HELPER FUNCTIONS ===

def _extract_final_output(messages: list, final_response) -> dict:
    """
    Pulls the submit_analysis result out of the conversation history.

    Pipeline: called by run_analyzer once the model stops with
    stop_reason='end_turn' (i.e. after it called submit_analysis).

    Args:
        messages:       Full conversation message list for this agent run.
        final_response: Last Anthropic API response object (fallback source only).

    Returns:
        dict with the validated analysis output, or error info.
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
                            if result.get("status") == "success" and "analysis_results" in result:
                                return result
                        except (json.JSONDecodeError, TypeError):
                            continue

        # submit_analysis result not found — raw text is the only remaining source
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

async def run_analyzer(code_input: str) -> dict:
    """
    Connects to the MCP server, runs the Analyzer agent loop, and returns the structured analysis.

    Pipeline: Step 1 in the pipeline. Called by the orchestrator with either a file path
    or a raw code string. Returns structured findings consumed by the Enricher.

    Args:
        code_input: Either a file path (.py) or a raw code string.

    Returns:
        dict with structured analysis results on success, or error info.
        Always returns a structured dict — never raises.
    """
    if not isinstance(code_input, str):
        logger.error("run_analyzer: code_input must be a str, got %s", type(code_input).__name__)
        return {"status": "error", "message": f"Invalid input: code_input must be str, got {type(code_input).__name__}"}

    if not code_input.strip():  # empty input is Test Case 3 — return a clean empty result rather than an unnecessary API call
        logger.warning("run_analyzer: code_input is empty — returning empty analysis")
        return {
            "status": "success",
            "analysis_results": {
                "code": "",
                "file_path": None,
                "line_count": 0,
                "syntax_findings": [],
                "security_findings": [],
                "structure": {"functions": [], "classes": [], "imports": []},
                "summary": "No code provided.",
            },
            "metadata": {"total_syntax_findings": 0, "total_security_findings": 0, "total_findings": 0},
        }

    try:
        server_params = StdioServerParameters(command="python", args=[MCP_SERVER_PATH])

        async with stdio_client(server_params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()

                tools_result = await session.list_tools()
                mcp_tools = convert_mcp_tools_to_anthropic(tools_result.tools)
                mcp_tools = [t for t in mcp_tools if t["name"] in ANALYZER_TOOLS]  # whitelist from config prevents scope creep

                all_tools = mcp_tools + analyzer_local_tools

                tool_summary = [
                    f"{t['name']} ({'local' if t['name'] in LOCAL_TOOL_NAMES else 'MCP'})"
                    for t in all_tools
                ]
                logger.info("Connected to MCP. Tools: %s", ", ".join(tool_summary))

                messages = [{"role": "user", "content": code_input}]

                for iteration in range(MAX_ITERATIONS):
                    logger.debug("Iteration %d/%d", iteration + 1, MAX_ITERATIONS)

                    response = client.messages.create(
                        model=MODEL,
                        max_tokens=MAX_TOKENS,
                        system=ANALYZER_PROMPT,
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

                                if block.name in LOCAL_TOOL_NAMES:
                                    tool_output = run_analyzer_tool(block.name, block.input)
                                else:
                                    try:
                                        result = await session.call_tool(block.name, arguments=block.input)
                                        tool_output = result.content[0].text if result.content else ""
                                    except Exception as e:
                                        tool_output = json.dumps({"status": "error", "message": str(e)})
                                        logger.warning("MCP tool %s failed: %s", block.name, str(e))

                                logger.debug("Tool result for %s: %s", block.name, tool_output[:300])
                                tool_results.append({
                                    "type": "tool_result",
                                    "tool_use_id": block.id,
                                    "content": tool_output,
                                })

                        messages.append({"role": "user", "content": tool_results})

                # loop exhausted without a valid submit_analysis call — expected budget failure, not a bug
                logger.warning("Reached max iterations (%d) without valid output", MAX_ITERATIONS)
                return {
                    "status": "max_iterations_reached",
                    "message": "Max iterations reached without valid output",
                }

    except Exception as e:
        # unexpected: MCP connection failure, Anthropic API error, network issue, etc.
        logger.error("run_analyzer failed unexpectedly: %s", str(e))
        return {
            "status": "error",
            "message": f"run_analyzer failed unexpectedly — likely MCP or API failure: {str(e)}",
        }


# === ENTRY POINT ===

if __name__ == "__main__":
    import sys
    from config import setup_logging
    setup_logging()

    if len(sys.argv) > 1:
        test_input = sys.argv[1]
    else:
        test_input = (
            "import os, sys\n"
            "import json\n"
            "def get_user(id):\n"
            "    query = 'SELECT * FROM users WHERE id = ' + id\n"
            "    return query\n"
        )

    print("=" * 60)
    print("ANALYZER AGENT — TEST RUN")
    print("=" * 60)

    result = asyncio.run(run_analyzer(test_input))

    print("\n" + "=" * 60)
    print("FINAL OUTPUT:")
    print("=" * 60)
    print(json.dumps(result, indent=2))