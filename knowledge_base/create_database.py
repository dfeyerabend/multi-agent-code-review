"""
create_database.py
Reads knowledge documents and loads them into ChromaDB as vector embeddings.
Run once before starting the agent pipeline: python create_database.py
"""

import re
import sys
import chromadb
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))        # add project root to path so config.py is found
from config import CHROMA_DB_PATH                                   # persistent storage path

# --- Configuration ---


DOCUMENTS_DIR = Path(__file__).parent / "documents" #  folder with knowledge source .md

COLLECTION_NAME = "code_best_practices"                             # single collection for all coding practices

SOURCES = {
    "pyguide.md": "pyguide",
    "company_rules.md": "company",
}

# --- Helpers ---
def _infer_category(title: str) -> str:
    """Infers a review category from the heading title using keyword matching."""
    t = title.lower()
    if any(w in t for w in ["exception", "error", "import", "thread", "type"]):
        return "Logic"
    if any(w in t for w in ["security", "sql", "injection", "password", "environ", "secret", "config"]):
        return "Security"
    if any(w in t for w in ["decorator", "lambda", "default", "global", "mutable", "nested", "complexity", "naming", "convention", "comment"]):
        return "Maintainability"
    return "Style"                                                     # fallback: most pyguide sections are style-related

def _extract_section(heading_title: str) -> str:
    """Extracts the section number from a heading like '3.10.4 Lambda Functions'."""
    match = re.match(r"^([\d.]+)\s+", heading_title)             # look for digits and dots at the start
    return match.group(1) if match else ""                              # return '3.10.4' or empty string if not found

def chunk_by_headings(text: str) -> list[dict]:
    """
    Splits markdown text into chunks at every ### heading.
    Each chunk includes the heading line + all content until the next heading.
    Returns a list of dicts with 'heading' and 'content'.
    """
    chunks = []
    current_heading = ""
    current_lines = []

    for line in text.split("\n"):
        if line.startswith("### "):                                     # start of a new chunk
            if current_lines:                                           # save the previous chunk before starting a new one
                content = "\n".join(current_lines).strip()
                if content:
                    chunks.append({"heading": current_heading, "content": content})
            current_heading = line[4:].strip()                          # strip the '### ' prefix to get the title
            current_lines = [line]                                      # include the heading line in the chunk content
        else:
            current_lines.append(line)

    # save the final lines after the loop ends
    if current_lines:
        content = "\n".join(current_lines).strip()
        if content:
            chunks.append({"heading": current_heading, "content": content})

    return chunks

def _clean_text(text: str) -> str:
    """Removes HTML tags and collapses excessive blank lines from markdown source."""
    text = re.sub(r"<[^>]+>", "", text)          # strip all HTML tags, e.g. <a id="..."></a>
    text = re.sub(r"\n{3,}", "\n\n", text)       # collapse 3+ consecutive newlines into 2
    return text.strip()

# --- Main ---

def create_database():
    """Loads all knowledge documents into ChromaDB. Safe to re-run (uses upsert)."""

    # connect to (or create) the persistent ChromaDB storage folder
    client = chromadb.PersistentClient(path=str(CHROMA_DB_PATH))

    # get_or_create_collection: creates on first run, loads existing on subsequent runs
    collection = client.get_or_create_collection(name=COLLECTION_NAME)

    total_chunks = 0

    for filename, source_tag in SOURCES.items():
        filepath = DOCUMENTS_DIR / filename

        if not filepath.exists():                                   # warn and skip missing files instead of crashing
            print(f"[WARNING] File not found, skipping: {filepath}")
            continue

        text = filepath.read_text(encoding="utf-8")  # read the full markdown file
        chunks = chunk_by_headings(text)
        chunks = [c for c in chunks if c["heading"]]  # skip text without a ### heading (no title) - prevents introduction texts to be included

        if not chunks:
            print(f"[WARNING] No ### headings found in {filename}, skipping.")
            continue

        # prepare the three parallel lists ChromaDB expects
        ids = [
            f"{source_tag}_{_extract_section(chunk['heading']) or i}"  # section number if available, fallback to index if not
            for i, chunk in enumerate(chunks)
        ]


        documents = [_clean_text(chunk["content"]) for chunk in chunks]  # clean HTML noise before embedding
        metadatas = [
            {
                "source": source_tag,                                   # 'pyguide' or 'company'
                "section": _extract_section(chunk["heading"]),          # e.g. '3.10.4'
                "title": chunk["heading"],                              # e.g. '3.10.4 Lambda Functions'
                "category": _infer_category(chunk["heading"]),          # 'Style' | 'Logic' | 'Maintainability' | 'Security'
            }
            for chunk in chunks
        ]

        # upsert = insert new chunks, update existing ones (safe to re-run)
        collection.upsert(documents=documents, ids=ids, metadatas=metadatas)

        print(f"[{source_tag}] {filename}: {len(chunks)} chunks loaded")
        total_chunks += len(chunks)

    print(f"\nDone. {total_chunks} total chunks stored in ChromaDB.")
    print(f"Location: {CHROMA_DB_PATH}")


if __name__ == "__main__":
    create_database()





