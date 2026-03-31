import os
import time

from groq import Groq
from dotenv import load_dotenv

from src.retrieval import search
from src.prompts import SYSTEM_PROMPT, build_rag_prompt
from src.guardrails import validate_input, validate_output, GuardrailError

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")

_client = None


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
            temperature=0.1,
        )
        answer = response.choices[0].message.content.strip()
    except Exception as e:
        return {
            "answer": "Sorry, I was unable to process your request. Please try again later.",
            "sources": [],
            "latency_ms": int((time.perf_counter() - start) * 1000),
            "error": str(e),
        }

    # Step 5: Validate output
    answer = validate_output(answer, chunks)

    # Step 6: Format sources for response
    sources = [
        {
            "source": chunk["source"],
            "section_title": chunk["section_title"],
            "snippet": chunk["text"][:200] + "..." if len(chunk["text"]) > 200 else chunk["text"],
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
