"""
Analyzer Agent — Step 1 in the Code Review Pipeline.
Connects to the MCP server for code analysis tools, and uses a local submit_analysis tool for the model's own summary.
All deterministic data (findings, structure) is assembled in Python directly from MCP tool outputs — never retyped by the LLM.
"""

import asyncio
import json
from typing import Optional
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
    _deduplicate_findings,
)

LOCAL_TOOL_NAMES = {t["name"] for t in analyzer_local_tools}


# === HELPER FUNCTIONS ===

def _extract_summary(messages: list) -> Optional[str]:
    """
    Pulls the validated summary out of the conversation history.

    Pipeline: called by run_analyzer once the model stops with stop_reason="end_turn"
    (i.e. after it called submit_analysis).

    Args:
        messages: Full conversation message list for this agent run.

    Returns:
        The summary string if a successful submit_analysis result is found,
        None if no valid result is present.
    """
    for msg in reversed(messages):
        if msg.get("role") != "user" or not isinstance(msg.get("content"), list):
            continue
        for block in msg["content"]:
            if block.get("type") != "tool_result":
                continue
            try:
                result = json.loads(block["content"])
            except (json.JSONDecodeError, TypeError):
                continue
            if result.get("status") == "success" and "summary" in result:
                return result["summary"]
    return None


def _assemble_analysis(mcp_outputs: dict, summary: str) -> dict:
    """
    Builds the final analysis_results dict directly from MCP tool outputs.

    Pipeline: called by run_analyzer once the model has produced a valid summary
    via submit_analysis. Replaces having the LLM retype linter/AST data — every
    deterministic field here came verbatim from an MCP tool the agent already
    called earlier in this same loop.

    Args:
        mcp_outputs: Parsed JSON results keyed by MCP tool name
                     ("read_code", "detect_syntax_errors", "extract_code_structure",
                     "check_company_rules").
        summary:     The Analyzer's own factual summary from submit_analysis.

    Returns:
        dict with status "success" and "analysis_results"/"metadata" on success,
        or status "error" naming the missing or failed tool output. metadata
        carries scan_complete (False if a scanner or a company rule failed) and
        tool_errors/company_rule_errors.
    """
    try:
        read_code_out = mcp_outputs.get("read_code")
        syntax_out = mcp_outputs.get("detect_syntax_errors")
        structure_out = mcp_outputs.get("extract_code_structure")
        company_out = mcp_outputs.get("check_company_rules")

        # All four are required inputs to the assembly — a missing one means the
        # model never called that tool, which is a predictable agent failure, not a crash.
        missing = [
            name for name, out in [
                ("read_code", read_code_out),
                ("detect_syntax_errors", syntax_out),
                ("extract_code_structure", structure_out),
                ("check_company_rules", company_out),
            ] if out is None
        ]
        if missing:
            logger.warning("_assemble_analysis: model never called: %s", missing)
            return {
                "status": "error",
                "message": f"Cannot assemble analysis — model never called: {missing}",
            }

        if read_code_out.get("status") != "success":
            return {
                "status": "error",
                "message": f"read_code did not succeed: {read_code_out.get('message')}",
            }

        if structure_out.get("status") != "success":
            return {
                "status": "error",
                "message": f"extract_code_structure did not succeed: {structure_out.get('message')}",
            }

        results = syntax_out.get("results", {})
        ruff_findings = results.get("ruff", {}).get("findings", [])
        bandit_findings = results.get("bandit", {}).get("findings", [])

        # Surface any scanner failure loudly. A failed security scan that produced zero
        # findings must never be presented downstream as a trustworthy clean result —
        # the pipeline still proceeds (ruff results stay useful), but the gap is recorded.
        tool_errors = syntax_out.get("tool_errors") or {}
        if tool_errors:
            logger.warning("_assemble_analysis: incomplete scan — tool_errors: %s", tool_errors)

        # Company findings already arrive in final shape (lines/occurrences, message
        # includes the function name) — no dedup pass needed or safe to apply here.
        company_findings = company_out.get("findings", [])
        company_rule_errors = company_out.get("rule_errors") or {}
        if company_rule_errors:
            logger.warning("_assemble_analysis: incomplete company-rule scan — rule_errors: %s", company_rule_errors)

        analysis_results = {
            "code": read_code_out["code"],
            "file_path": read_code_out.get("file_path"),
            "line_count": read_code_out["line_count"],
            "syntax_findings": _deduplicate_findings(ruff_findings),
            "security_findings": _deduplicate_findings(bandit_findings),
            "company_findings": company_findings,
            "structure": {
                "functions": structure_out.get("functions", []),
                "classes": structure_out.get("classes", []),
                "imports": structure_out.get("imports", []),
            },
            "summary": summary,
        }

        return {
            "status": "success",
            "analysis_results": analysis_results,
            "metadata": {
                "total_syntax_findings": len(analysis_results["syntax_findings"]),
                "total_security_findings": len(analysis_results["security_findings"]),
                "total_company_findings": len(analysis_results["company_findings"]),
                "total_findings": (
                    len(analysis_results["syntax_findings"])
                    + len(analysis_results["security_findings"])
                    + len(analysis_results["company_findings"])
                ),
                "scan_complete": not tool_errors and not company_rule_errors,  # False if any scanner or rule failed
                "tool_errors": tool_errors,                     # empty dict when both scanners ran
                "company_rule_errors": company_rule_errors,     # empty dict when every company rule ran
            },
        }

    except Exception as e:
        logger.error("_assemble_analysis failed unexpectedly: %s", str(e))
        return {
            "status": "error",
            "message": f"_assemble_analysis failed unexpectedly: {str(e)}",
        }


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
                "company_findings": [],
                "structure": {"functions": [], "classes": [], "imports": []},
                "summary": "No code provided.",
            },
            "metadata": {
                "total_syntax_findings": 0,
                "total_security_findings": 0,
                "total_company_findings": 0,
                "total_findings": 0,
                "scan_complete": True,
                "tool_errors": {},
                "company_rule_errors": {},
            },
        }

    # Scanners receive the exact code captured from read_code, never the model's retyped
    # copy — a retyped string can be silently corrupted and make ruff/bandit scan the wrong code.
    _CODE_ARG_TOOLS = {"detect_syntax_errors", "extract_code_structure", "check_company_rules"}

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
                mcp_outputs = {}  # captured verbatim per tool name, used by _assemble_analysis after the loop ends
                captured_code = None  # the authoritative code string from read_code, forced into later scanner calls

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
                        summary = _extract_summary(messages)
                        if summary is None:
                            logger.warning("run_analyzer: ended turn without a valid submit_analysis result")
                            return {
                                "status": "error",
                                "message": "Model ended turn without a valid submit_analysis call.",
                            }

                        final_output = _assemble_analysis(mcp_outputs, summary)
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

                                # Force the real code into scanner calls; the model's retyped copy is discarded.
                                tool_input = block.input
                                if block.name in _CODE_ARG_TOOLS:
                                    if captured_code is not None:
                                        tool_input = {**block.input, "code": captured_code}
                                    else:  # scanner requested before read_code ran — predictable agent ordering failure
                                        logger.warning(
                                            "run_analyzer: %s called before read_code — using model-provided code as fallback",
                                            block.name,
                                        )

                                if block.name in LOCAL_TOOL_NAMES:
                                    tool_output = run_analyzer_tool(block.name, tool_input)
                                else:
                                    try:
                                        result = await session.call_tool(block.name, arguments=tool_input)
                                        tool_output = result.content[0].text if result.content else ""
                                        try:
                                            mcp_outputs[block.name] = json.loads(tool_output)  # captured for Python-side assembly, not for the LLM to retype
                                            # Capture the authoritative code the moment read_code succeeds.
                                            if block.name == "read_code":
                                                code_val = mcp_outputs[block.name].get("code")
                                                if isinstance(code_val, str):
                                                    captured_code = code_val
                                        except json.JSONDecodeError:
                                            logger.warning("Could not parse MCP output for %s as JSON", block.name)
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
            "\n"
            "def get_user(id, cache={}):  # REASON: demo function\n"
            "    query = 'SELECT * FROM users WHERE id = ' + id\n"
            "    return db.execute(query)\n"
        )

    print("=" * 60)
    print("ANALYZER AGENT — TEST RUN")
    print("=" * 60)

    result = asyncio.run(run_analyzer(test_input))

    print("\n" + "=" * 60)
    print("FINAL OUTPUT:")
    print("=" * 60)
    print(json.dumps(result, indent=2))