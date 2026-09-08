import os
import re
import time

from groq import Groq
from dotenv import load_dotenv

from src.retrieval import search
from src.prompts import SYSTEM_PROMPT, build_rag_prompt
from src.guardrails import (
    validate_input,
    validate_output,
    GuardrailError,
    FALLBACK_PHRASE,
)

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")

SNIPPET_LENGTH = 200

_client = None


def build_snippet(text: str, length: int = SNIPPET_LENGTH) -> str:
    """
    Flatten a chunk into a readable one-line preview.

    Chunks are raw markdown and often start mid-table, so the markup is stripped
    rather than rendered: a truncated fragment cannot produce valid markup.
    """
    lines, heading_at = [], None
    for line in text.split("\n"):
        line = line.strip()
        if not line or re.fullmatch(r"[|\s:-]+", line):
            continue
        if line.startswith("#") and heading_at is None:
            heading_at = len(lines)
        if "|" in line:
            cells = [c.strip() for c in line.split("|") if c.strip()]
            line = " · ".join(cells)
        line = line.lstrip("#").strip()
        line = re.sub(r"^([-*+]|\d+\.)\s+", "", line)
        lines.append(line)

    # Chunks are cut at a fixed size, so they routinely open mid-word. Starting at
    # the chunk's own heading gives a preview that reads from a real boundary.
    truncated_start = False
    if heading_at and len(" ".join(lines[heading_at:])) >= length // 2:
        lines, truncated_start = lines[heading_at:], True
    else:
        heading_at = None

    flat = re.sub(r"\s+", " ", " ".join(lines))
    flat = re.sub(r"\*\*(.+?)\*\*|\*(.+?)\*|`(.+?)`", r"\1\2\3", flat)
    flat = re.sub(r"[*`]", "", flat)

    if heading_at is None:
        cleaned = re.sub(r"^[^\w(]+|^\S*?[a-z]\s+(?=[a-z])", "", flat).strip()
        truncated_start = cleaned != flat
        flat = cleaned

    prefix = "..." if truncated_start else ""
    if len(flat) > length:
        return prefix + flat[:length].rstrip() + "..."
    return prefix + flat


def get_groq_client() -> Groq:
    global _client
    if _client is None:
        _client = Groq(api_key=GROQ_API_KEY)
    return _client


def ask(question: str) -> dict:
    """
    Main RAG pipeline function.

    Args:
        question: Raw user question string.

    Returns:
        dict with keys:
            - answer (str): Model response with citations.
            - sources (list): List of source dicts used for the answer.
            - latency_ms (int): Total time from question to answer in milliseconds.
            - error (str | None): Error message if something went wrong.
    """
    start = time.perf_counter()

    # Step 1: Validate input
    try:
        question = validate_input(question)
    except GuardrailError as e:
        return {
            "answer": str(e),
            "sources": [],
            "latency_ms": int((time.perf_counter() - start) * 1000),
            "error": str(e),
        }

    # Step 2: Retrieve relevant chunks
    chunks = search(question)

    # Step 3: Build prompt
    user_message = build_rag_prompt(question, chunks)

    # Step 4: Call Groq API
    try:
        client = get_groq_client()
        response = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_message},
            ],
            max_tokens=1024,
            temperature=0,
            # Reasoning tokens count against max_tokens and, left unbounded, can
            # consume the whole budget and return empty content.
            extra_body={"reasoning_effort": "low"},
        )
        answer = (response.choices[0].message.content or "").strip()
        if not answer:
            raise ValueError("model returned an empty answer")
    except Exception as e:
        return {
            "answer": "Sorry, I was unable to process your request. Please try again later.",
            "sources": [],
            "latency_ms": int((time.perf_counter() - start) * 1000),
            "error": str(e),
        }

    # Step 5: Validate output
    answer = validate_output(answer, chunks)

    # Step 6: Format sources for response. A refusal has no sources — retrieval
    # still returns the nearest chunks, but none of them support an answer.
    if FALLBACK_PHRASE in answer:
        chunks = []

    sources = [
        {
            "source": chunk["source"],
            "section_title": chunk["section_title"],
            "url": f"/policies/{chunk['source']}",
            "snippet": build_snippet(chunk["text"]),
        }
        for chunk in chunks
    ]

    latency_ms = int((time.perf_counter() - start) * 1000)

    return {
        "answer": answer,
        "sources": sources,
        "latency_ms": latency_ms,
        "error": None,
    }
