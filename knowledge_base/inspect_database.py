"""
inspect_database.py
Displays random sample entries from ChromaDB per source.
Usage: python inspect_database.py
"""
import sys
import random                                                       # for random sampling
import chromadb                                                     # vector database client
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))        # add project root to path so config.py is found
from config import CHROMA_DB_PATH                                   # path to persistent storage

COLLECTION_NAME = "code_best_practices"                             # must match the name used in create_database.py
SAMPLES_PER_SOURCE = 3                                              # how many random entries to show per source
PREVIEW_LENGTH = 300                                                # max characters shown for document content


def inspect_database():
    """Connects to ChromaDB and prints random sample entries per source."""

    client = chromadb.PersistentClient(path=str(CHROMA_DB_PATH))
    collection = client.get_or_create_collection(name=COLLECTION_NAME)

    total = collection.count()                                      # total number of chunks across all sources
    print(f"Collection '{COLLECTION_NAME}': {total} total chunks\n")

    for source in ["pyguide", "company"]:                          # loop over known sources

        # fetch all entries for this source (documents + metadata)
        results = collection.get(
            where={"source": source},                              # filter by source tag in metadata
            include=["documents", "metadatas"]
        )

        count = len(results["ids"])

        if count == 0:
            print(f"[{source}] No entries found.\n")
            continue

        sample_size = min(SAMPLES_PER_SOURCE, count)               # don't sample more than what exists
        sample_indices = random.sample(range(count), sample_size)  # pick random positions

        print(f"[{source}] {count} chunks total — showing {sample_size} random samples:")
        print("=" * 70)

        for idx in sample_indices:
            meta = results["metadatas"][idx]
            doc = results["documents"][idx]
            preview = doc[:PREVIEW_LENGTH].replace("\n", " ")      # flatten newlines for readable one-liner preview

            print(f"  ID:       {results['ids'][idx]}")
            print(f"  Section:  {meta.get('section', '—')}")
            print(f"  Title:    {meta.get('title', '—')}")
            print(f"  Category: {meta.get('category', '—')}")
            print(f"  Preview:  {preview}...")
            print()

        print()


if __name__ == "__main__":
    inspect_database()