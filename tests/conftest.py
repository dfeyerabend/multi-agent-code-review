"""
Shared pytest fixtures for the code review pipeline test suite.
Fixtures defined here are available to all test files automatically — no imports needed.
"""

import pytest
import chromadb
from config import CHROMA_DB_PATH

COLLECTION_NAME = "code_best_practices"

@pytest.fixture(scope="session")
def chroma_collection():
    """
    Provides the ChromaDB knowledge base collection for RAG retrieval tests.

    Session-scoped: opened once and shared across all tests that declare this fixture.

    Returns:
        chromadb.Collection: The 'code_best_practices' collection.

    Raises:
        pytest.fail: If the collection does not exist at CHROMA_DB_PATH.
        Run knowledge_base/create_database.py to initialise it.
    """
    client = chromadb.PersistentClient(path=str(CHROMA_DB_PATH))
    try:
        return client.get_collection(name=COLLECTION_NAME)
    except Exception:
        pytest.fail(
            f"Collection '{COLLECTION_NAME}' not found at {CHROMA_DB_PATH}. "
            "Run: python knowledge_base/create_database.py"
        )