"""
Run this script once before starting the app to build the ChromaDB index.

Usage:
    python scripts/build_index.py
"""

from pathlib import Path
import sys

# Allow imports from project root
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.ingestion import ingest_directory, get_chroma_collection

POLICIES_DIR = Path(__file__).parent.parent / "data" / "policies"


def main():
    if not POLICIES_DIR.exists():
        print(f"Error: policies directory not found at {POLICIES_DIR}")
        sys.exit(1)

    print(f"Building index from: {POLICIES_DIR}")
    print("-" * 40)

    total = ingest_directory(POLICIES_DIR)

    print("-" * 40)
    collection = get_chroma_collection()
    print(f"Index built successfully. Total chunks in DB: {collection.count()}")


if __name__ == "__main__":
    main()
