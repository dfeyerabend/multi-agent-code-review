"""
Orchestrator — linear pipeline driver for the Code Review Agent.
Owns full pipeline state. Builds each agent's input contract and passes forward only what that agent needs.
"""

import asyncio
import json
import logging

logger = logging.getLogger(__name__)

from agent.analyzer_agent import run_analyzer
from agent.enricher_agent import run_enricher
from agent.optimizer_agent import run_optimizer

# Fields the Enricher needs from each finding — everything else stays in the orchestrator.
_ENRICHER_FIELDS = {"rule", "line", "category", "severity", "message", "doc_url", "cwe_id"}


async def run_pipeline(code_input: str) -> dict:
    """
    Runs the code review pipeline from Analyzer through to the last built agent.

    Args:
        code_input: File path or raw code string.

    Returns:
        dict with the final pipeline output, or error info from the failing stage.
    """
    # --- Step 1: Analyzer ---
    logger.info("Pipeline — step 1: Analyzer")
    analyzer_result = await run_analyzer(code_input)

    if analyzer_result.get("status") != "success":
        logger.error("Analyzer failed: %s", analyzer_result.get("message"))
        return analyzer_result

    analysis = analyzer_result["analysis_results"]

    # --- Build Enricher input: project and flatten findings ---
    findings = [
        {k: v for k, v in f.items() if k in _ENRICHER_FIELDS}
        for f in analysis.get("syntax_findings", []) + analysis.get("security_findings", [])
    ]

    # --- Step 2: Enricher ---
    logger.info("Pipeline — step 2: Enricher — %d finding(s)", len(findings))
    enricher_result = await run_enricher(findings)

    if enricher_result.get("status") != "success":
        logger.error("Enricher failed: %s", enricher_result.get("message"))
        return enricher_result

    # --- Step 3: Optimizer ---
    enriched_findings = enricher_result["enrichment_results"]["findings"]
    code = analysis["code"]

    logger.info("Pipeline — step 3: Optimizer — %d finding(s)", len(enriched_findings))
    optimizer_result = await run_optimizer(code, enriched_findings)

    if optimizer_result.get("status") != "success":
        logger.error("Optimizer failed: %s", optimizer_result.get("message"))
        return optimizer_result

    # --- Step 4: Evaluator (stub) ---
    # TODO: wire run_evaluator

    return optimizer_result


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
    print("PIPELINE — TEST RUN")
    print("=" * 60)

    result = asyncio.run(run_pipeline(test_input))

    print("\n" + "=" * 60)
    print("FINAL OUTPUT:")
    print("=" * 60)
    print(json.dumps(result, indent=2))
