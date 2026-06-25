"""
Orchestrator — linear pipeline driver for the Code Review Agent.
Owns full pipeline state. Builds each agent's input contract and passes forward only what that agent needs.
"""

import asyncio
import json
import logging

logger = logging.getLogger(__name__)

from agents.analyzer_agent import run_analyzer
from agents.enricher_agent import run_enricher
from agents.optimizer_agent import run_optimizer
from agents.evaluator_agent import run_evaluator

# Fields the Enricher needs from each finding — everything else stays in the orchestrator.
_ENRICHER_FIELDS = {"rule", "line", "lines", "occurrences", "category", "severity", "message", "doc_url", "cwe_id"}


async def run_pipeline(code_input: str) -> dict:
    """
    Runs the full code review pipeline from Analyzer through Evaluator.

    Pipeline: top-level entry point. Called directly from __main__ or from app.py.
    Each stage receives only the fields it needs; identity and metadata are managed here.

    Args:
        code_input: File path or raw code string.

    Returns:
        dict with the final pipeline output, or a structured error dict from the
        failing stage. Always returns — never raises.
    """
    try:
        # --- Step 1: Analyzer ---
        logger.info("Pipeline — step 1: Analyzer")
        analyzer_result = await run_analyzer(code_input)

        if analyzer_result.get("status") != "success":
            logger.error("Analyzer failed: %s", analyzer_result.get("message"))
            return analyzer_result

        analysis = analyzer_result["analysis_results"]

        # --- Build Enricher input: flatten and project findings ---
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

        enriched_findings = enricher_result["enrichment_results"]["findings"]
        code = analysis["code"]

        # --- Step 3: Optimizer ---
        logger.info("Pipeline — step 3: Optimizer — %d finding(s)", len(enriched_findings))
        optimizer_result = await run_optimizer(code, enriched_findings)

        if optimizer_result.get("status") != "success":
            logger.error("Optimizer failed: %s", optimizer_result.get("message"))
            return optimizer_result

        fixes = optimizer_result["optimization_results"]["fixes"]

        # --- Step 4: Evaluator ---
        logger.info("Pipeline — step 4: Evaluator — %d fix(es)", len(fixes))
        evaluator_result = await run_evaluator(code, enriched_findings, fixes)

        if evaluator_result.get("status") != "success":
            logger.error("Evaluator failed: %s", evaluator_result.get("message"))
            return evaluator_result

        return evaluator_result

    except Exception as e:
        # guards against unexpected key errors in inter-agent data handoffs
        logger.error("run_pipeline failed unexpectedly: %s", str(e))
        return {
            "status": "error",
            "message": f"run_pipeline failed unexpectedly: {str(e)}",
        }


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