"""
Tests for RAG retrieval from the ChromaDB knowledge base.
Each company rule has a targeted test that proves the correct chunk is retrieved — verifying RAG is working, not the model's training data.
"""

import pytest


def _get_retrieved_ids(chroma_collection, query: str, n_results: int = 3) -> list[str]:
    """
    Queries the collection and returns the document IDs of the top results.

    Args:
        chroma_collection: The ChromaDB collection fixture.
        query: Natural language query string.
        n_results: Number of top results to retrieve.

    Returns:
        List of document ID strings.
    """
    results = chroma_collection.query(query_texts=[query], n_results=n_results)
    return results["ids"][0]


# ── Company rule retrieval ────────────────────────────────────────────────────

def test_retrieves_db_naming_rule(chroma_collection):
    """Rule 1.1: db_ prefix requirement is retrieved for database function violations."""
    ids = _get_retrieved_ids(
        chroma_collection,
        query="function querying database without db_ prefix",
    )
    assert any("company_1.1" in doc_id for doc_id in ids)


def test_retrieves_reason_comment_rule(chroma_collection):
    """Rule 1.2: REASON comment requirement is retrieved for undocumented functions."""
    ids = _get_retrieved_ids(
        chroma_collection,
        query="function missing reason comment explaining its purpose",
    )
    assert any("company_1.2" in doc_id for doc_id in ids)


def test_retrieves_exception_hierarchy_rule(chroma_collection):
    """Rule 1.3: Custom exception hierarchy is retrieved for generic Exception usage."""
    ids = _get_retrieved_ids(
        chroma_collection,
        query="raising generic Exception instead of custom error class",
    )
    assert any("company_1.3" in doc_id for doc_id in ids)


def test_retrieves_config_access_rule(chroma_collection):
    """Rule 1.4: Config.get() requirement is retrieved for direct os.environ access."""
    ids = _get_retrieved_ids(
        chroma_collection,
        query="direct os.environ access for reading configuration",
    )
    assert any("company_1.4" in doc_id for doc_id in ids)


# ── Metadata filtering ────────────────────────────────────────────────────────

def test_category_filter_returns_only_matching_chunks(chroma_collection):
    """where filter on category metadata excludes chunks from other categories."""
    results = chroma_collection.query(
        query_texts=["security vulnerability"],
        n_results=5,
        where={"category": "Security"},
    )
    categories = [m["category"] for m in results["metadatas"][0]]
    assert all(c == "Security" for c in categories)


def test_query_on_unrelated_topic_does_not_crash(chroma_collection):
    """Returns results without error for a query unrelated to any stored content."""
    results = chroma_collection.query(
        query_texts=["quantum entanglement in distributed systems"],
        n_results=3,
    )
    assert len(results["ids"][0]) > 0