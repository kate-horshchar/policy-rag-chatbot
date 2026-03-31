import os
import re

from dotenv import load_dotenv

load_dotenv()

MAX_QUERY_LENGTH = int(os.getenv("MAX_QUERY_LENGTH", 500))
MAX_RESPONSE_LENGTH = 2000

BLOCKED_PATTERNS = [
    r"ignore\s+(previous|all|prior)\s+instructions",
    r"you\s+are\s+now\s+a",
    r"pretend\s+you\s+are",
    r"act\s+as\s+if\s+you",
    r"forget\s+(everything|all|your)\s+(you|instructions)",
    r"disregard\s+(your|all|previous)",
    r"system\s*prompt",
    r"jailbreak",
]

FALLBACK_PHRASE = "I don't have enough information in the current policies"


class GuardrailError(Exception):
    pass


def validate_input(query: str) -> str:
    """
    Validate and clean user input.
    Raises GuardrailError if the input is invalid or suspicious.
    Returns the cleaned query string.
    """
    query = query.strip()

    if not query:
        raise GuardrailError("Query cannot be empty.")

    if len(query) > MAX_QUERY_LENGTH:
        raise GuardrailError(
            f"Query is too long. Please keep it under {MAX_QUERY_LENGTH} characters."
        )

    query_lower = query.lower()
    for pattern in BLOCKED_PATTERNS:
        if re.search(pattern, query_lower):
            raise GuardrailError(
                "Your query contains content that cannot be processed. "
                "Please ask a question about Sunshine Consulting policies."
            )

    return query


def validate_output(answer: str, chunks: list[dict]) -> str:
    """
    Validate the model's response.
    - If chunks were found, ensure the answer contains at least one citation.
    - Truncate if response is too long.
    Returns the validated (possibly truncated) answer.
    """
    if len(answer) > MAX_RESPONSE_LENGTH:
        answer = answer[:MAX_RESPONSE_LENGTH].rstrip() + "..."

    # Only check for citations if we had relevant chunks and it's not a fallback
    if chunks and FALLBACK_PHRASE not in answer:
        if "[Source:" not in answer:
            answer += (
                "\n\n*Note: This answer is based on Sunshine Consulting policy documents. "
                "Please verify with People Operations for official confirmation.*"
            )

    return answer
