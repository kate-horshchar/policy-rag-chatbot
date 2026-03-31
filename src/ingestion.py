import os
import hashlib
from pathlib import Path
from typing import Optional

import chromadb
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

load_dotenv()

CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")
CHROMA_COLLECTION_NAME = os.getenv("CHROMA_COLLECTION_NAME", "company_policies")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")
CHUNK_SIZE = 512
CHUNK_OVERLAP = 64

_model: Optional[SentenceTransformer] = None


def get_embedding_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer(EMBEDDING_MODEL)
    return _model


def get_chroma_collection() -> chromadb.Collection:
    client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)
    return client.get_or_create_collection(
        name=CHROMA_COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )


# --- Document loaders ---

def load_markdown(file_path: Path) -> str:
    return file_path.read_text(encoding="utf-8")


def load_txt(file_path: Path) -> str:
    return file_path.read_text(encoding="utf-8")


def load_pdf(file_path: Path) -> str:
    from pypdf import PdfReader
    reader = PdfReader(str(file_path))
    return "\n".join(page.extract_text() or "" for page in reader.pages)


def load_html(file_path: Path) -> str:
    from bs4 import BeautifulSoup
    html = file_path.read_text(encoding="utf-8")
    soup = BeautifulSoup(html, "lxml")
    return soup.get_text(separator="\n")


def load_document(file_path: Path) -> str:
    suffix = file_path.suffix.lower()
    if suffix in (".md", ".markdown"):
        return load_markdown(file_path)
    elif suffix == ".txt":
        return load_txt(file_path)
    elif suffix == ".pdf":
        return load_pdf(file_path)
    elif suffix in (".html", ".htm"):
        return load_html(file_path)
    else:
        raise ValueError(f"Unsupported file type: {suffix}")


# --- Metadata extraction ---

def extract_section_title(text: str, char_offset: int) -> str:
    """Find the last markdown heading before char_offset."""
    snippet = text[:char_offset]
    lines = snippet.split("\n")
    for line in reversed(lines):
        stripped = line.strip()
        if stripped.startswith("##"):
            return stripped.lstrip("#").strip()
    return "General"


# --- Chunking ---

def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[dict]:
    """
    Recursively split text by separators, return list of
    {"text": ..., "char_offset": ...}
    """
    separators = ["\n\n", "\n", ". ", " "]
    chunks = []
    _recursive_split(text, separators, chunk_size, overlap, 0, chunks)
    return chunks


def _recursive_split(text: str, separators: list[str], chunk_size: int, overlap: int, base_offset: int, result: list):
    if len(text) <= chunk_size:
        if text.strip():
            result.append({"text": text.strip(), "char_offset": base_offset})
        return

    sep = separators[0] if separators else " "
    remaining_seps = separators[1:] if len(separators) > 1 else []

    parts = text.split(sep)
    current = ""
    current_offset = base_offset

    for part in parts:
        candidate = current + (sep if current else "") + part
        if len(candidate) <= chunk_size:
            current = candidate
        else:
            if current.strip():
                if len(current) > chunk_size and remaining_seps:
                    _recursive_split(current, remaining_seps, chunk_size, overlap, current_offset, result)
                else:
                    result.append({"text": current.strip(), "char_offset": current_offset})
            # overlap: carry last `overlap` chars into next chunk
            overlap_text = current[-overlap:] if len(current) > overlap else current
            current_offset = base_offset + text.find(part)
            current = overlap_text + (sep if overlap_text else "") + part

    if current.strip():
        if len(current) > chunk_size and remaining_seps:
            _recursive_split(current, remaining_seps, chunk_size, overlap, current_offset, result)
        else:
            result.append({"text": current.strip(), "char_offset": current_offset})


# --- Embedding ---

def embed_documents(texts: list[str]) -> list[list[float]]:
    model = get_embedding_model()
    embeddings = model.encode(texts, normalize_embeddings=True)
    return embeddings.tolist()


# --- Main ingestion ---

def ingest_file(file_path: Path, collection: chromadb.Collection) -> int:
    """Ingest a single file into ChromaDB. Returns number of chunks added."""
    text = load_document(file_path)
    chunks = chunk_text(text)
    source_name = file_path.name

    texts, metadatas, ids = [], [], []

    for chunk in chunks:
        chunk_text_val = chunk["text"]
        char_offset = chunk["char_offset"]
        section_title = extract_section_title(text, char_offset)

        # Use content hash as ID for idempotent re-indexing
        chunk_id = hashlib.md5(f"{source_name}:{chunk_text_val}".encode()).hexdigest()

        texts.append(chunk_text_val)
        metadatas.append({
            "source": source_name,
            "section_title": section_title,
            "char_offset": char_offset,
        })
        ids.append(chunk_id)

    if not texts:
        return 0

    embeddings = embed_documents(texts)

    # Upsert — safe to re-run without duplicating
    collection.upsert(
        documents=texts,
        embeddings=embeddings,
        metadatas=metadatas,
        ids=ids,
    )

    return len(texts)


def ingest_directory(directory: Path) -> int:
    """Ingest all supported documents from a directory."""
    collection = get_chroma_collection()
    supported = {".md", ".markdown", ".txt", ".pdf", ".html", ".htm"}
    total = 0

    for file_path in sorted(directory.iterdir()):
        if file_path.suffix.lower() in supported:
            print(f"Ingesting: {file_path.name} ...", end=" ")
            count = ingest_file(file_path, collection)
            print(f"{count} chunks")
            total += count

    print(f"\nTotal chunks indexed: {total}")
    return total
