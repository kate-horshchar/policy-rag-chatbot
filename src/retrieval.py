import os
from typing import Optional

import chromadb
from dotenv import load_dotenv

from src.ingestion import get_embedding_model, get_chroma_collection

load_dotenv()

TOP_K = int(os.getenv("TOP_K", 5))
MMR_LAMBDA = float(os.getenv("MMR_LAMBDA", 0.7))

# BGE instruction prefix — improves retrieval quality for this model
BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


def embed_query(query: str) -> list[float]:
    """Embed a user query with the BGE instruction prefix."""
    model = get_embedding_model()
    prefixed = BGE_QUERY_PREFIX + query
    embedding = model.encode(prefixed, normalize_embeddings=True)
    return embedding.tolist()


def retrieve(query: str, k: int = TOP_K) -> list[dict]:
    """
    Retrieve the top-k most relevant chunks from ChromaDB.
    Returns list of dicts with keys: text, source, section_title, distance.
    """
    collection = get_chroma_collection()
    query_embedding = embed_query(query)

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=min(k, collection.count()),
        include=["documents", "metadatas", "distances"],
    )

    chunks = []
    documents = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]
    distances = results.get("distances", [[]])[0]

    for doc, meta, dist in zip(documents, metadatas, distances):
        chunks.append({
            "text": doc,
            "source": meta.get("source", "unknown"),
            "section_title": meta.get("section_title", "General"),
            "distance": round(dist, 4),
        })

    return chunks


def search(query: str) -> list[dict]:
    """
    Main search function used by the pipeline.
    Retrieves top-k most relevant chunks.
    """
    return retrieve(query, k=TOP_K)
