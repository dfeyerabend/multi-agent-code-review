"""
Shared utilities for all agents in the Code Review pipeline.
"""


def convert_mcp_tools_to_anthropic(mcp_tools: list) -> list:
    """
    Converts MCP tool definitions to Anthropic's expected tool format.

    Args:
        mcp_tools: List of tool objects returned by session.list_tools().

    Returns:
        List of tool dicts in Anthropic API format.
    """
    return [
        {
            "name": tool.name,
            "description": tool.description or "",
            "input_schema": tool.inputSchema,
        }
        for tool in mcp_tools
    ]
