"""
Analyzer Agent — Step 1 in the Code Review Pipeline.
Connects to the MCP server for code analysis tools,
and uses a local submit_analysis tool for structured output.
"""

import asyncio
import json
from mcp import ClientSession, StdioServerParameters       # MCP client SDK
from mcp.client.stdio import stdio_client

import logging
logger = logging.getLogger(__name__)

from agents.agent_utils import convert_mcp_tools_to_anthropic
from config import (
    client,             # shared Anthropic client instance
    MODEL,
    MAX_TOKENS,
    MAX_ITERATIONS,
    MCP_SERVER_PATH,
    ANALYZER_PROMPT,
    ANALYZER_TOOLS,
)

from tools.analyzer_tools import (
    analyzer_local_tools,       # local tool schemas (submit_analysis)
    run_analyzer_tool,          # local tool executor
)

# --- Names of tools that run locally (not via MCP) ---
LOCAL_TOOL_NAMES = {t["name"] for t in analyzer_local_tools}



async def run_analyzer(code_input: str) -> dict:
    """
        Main function: connects to MCP, runs the Analyzer agents loop,
        returns the structured analysis result.

        Args:
            code_input: Either a file path or a raw code string.
        Returns:
            dict with the structured analysis (or error info).
    """

    # --- Connect to MCP server via STDIO ---
    server_params = StdioServerParameters(
        command="python",               # command to start the MCP server
        args=[MCP_SERVER_PATH],         # path to our mcp_server.py
    )

    async with stdio_client(server_params) as (read, write):    # open STDIO streams
        async with ClientSession(read, write) as session:       # create MCP session
            await session.initialize()                          # handshake with server

            # --- Get tools from MCP server ---
            tools_result = await session.list_tools()
            mcp_tools = convert_mcp_tools_to_anthropic(tools_result.tools)
            mcp_tools = [t for t in mcp_tools if t["name"] in ANALYZER_TOOLS]  # whitelist filter from config.py

            # Combine MCP tools + local tools into one list for Claude
            all_tools = mcp_tools + analyzer_local_tools

            # Log as a single summary line at INFO, full list at DEBUG
            tool_summary = [f"{t['name']} ({'local' if t['name'] in LOCAL_TOOL_NAMES else 'MCP'})" for t in all_tools]
            logger.info("Connected to MCP. Tools: %s", ", ".join(tool_summary))

            # --- Agent loop ---
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

                # -- Analyzer is done
                if response.stop_reason == "end_turn":
                    final_output = _extract_final_output(messages, response)
                    logger.info("Completed after %d iteration(s)", iteration + 1)
                    return final_output

                # --- Analyzer wants to use tools ---
                elif response.stop_reason == "tool_use":
                    messages.append({"role": "assistant", "content": response.content})

                    tool_results = []
                    for block in response.content:
                        if hasattr(block, "text") and block.text:
                            logger.debug("Claude says: %s", block.text[:200])
                        if block.type == "tool_use":
                            logger.debug("Tool call: %s | args: %s", block.name, str(block.input)[:200])

                        if block.type == "tool_use":
                            tool_name = block.name
                            tool_args = block.input

                            # Route: local tool or MCP tool?
                            if tool_name in LOCAL_TOOL_NAMES:
                                logger.debug("Calling tool: %s (local)", tool_name)
                                tool_output = run_analyzer_tool(tool_name, tool_args)
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

            # Max iterations reached
            logger.warning("Reached max iterations (%d)", MAX_ITERATIONS)
            return {"status": "error", "message": "Max iterations reached"}


def _extract_final_output(messages: list, final_response) -> dict:
    """
    Extracts the structured analysis from the conversation.
    Prefers the submit_analysis tool result over raw text output,
    because the tool enforces the correct schema.
    """

    # Look backwards through messages for the last submit_analysis result
    for msg in reversed(messages):
        if msg.get("role") == "user" and isinstance(msg.get("content"), list):
            for block in msg["content"]:  # iterate tool_result blocks
                if block.get("type") == "tool_result":
                    try:
                        result = json.loads(block["content"])
                        if result.get("status") == "success" and "analysis_results" in result:
                            return result  # found the submit_analysis output
                    except (json.JSONDecodeError, TypeError):
                        continue

    # Fallback: try to parse text from final response
    final_text = ""
    for block in final_response.content:
        if hasattr(block, "text"):
            final_text += block.text

    try:
        return json.loads(final_text)
    except json.JSONDecodeError:
        logger.error("Failed to parse final output — raw: %s", final_text[:200])
        return {"status": "error", "raw_output": final_text}


# --- Entry point for testing ---
if __name__ == "__main__":
    import sys
    from config import setup_logging
    setup_logging()

    if len(sys.argv) > 1:
        test_input = sys.argv[1]  # e.g.: python analyzer_agent.py test_code.py
    else:
        # Default test: SQL injection + missing docstring + unused import
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



