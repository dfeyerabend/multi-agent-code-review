"""
Shared utilities for all agents in the Code Review pipeline.
Includes MCP tool format conversion and list chunking for batched agents runs.
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

def chunk_list(items: list, size: int) -> list[list]:
    """
    Splits a list into successive sublists of at most `size` items.

    Args:
        items: The list to split.
        size:  Maximum number of items per chunk.

    Returns:
        List of sublists. If items is empty, returns an empty list.
    """
    return [items[i : i + size] for i in range(0, len(items), size)]
