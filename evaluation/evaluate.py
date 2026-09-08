"""
Evaluation script for the Policy RAG Chatbot.

Metrics:
- Groundedness %: answer is supported by the retrieved chunks
- Citation Accuracy %: correct source document is cited
- Citation Validity %: every citation names a real document and a real section
- Partial Match %: answer agrees with the short gold answer
- Latency p50/p95: response time percentiles
- Fallback Rate: unanswerable questions correctly refused

Usage:
    python evaluation/evaluate.py
"""

import json
import os
import re
import sys
import time
from pathlib import Path
from datetime import datetime

import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.pipeline import ask  # noqa: E402
from src.retrieval import search  # noqa: E402

QA_PAIRS_PATH = Path(__file__).parent / "qa_pairs.json"
POLICIES_DIR = Path(__file__).parent.parent / "data" / "policies"
CITATION_RE = re.compile(r"\[Source:([^\]]+)\]")
RESULTS_DIR = Path(__file__).parent / "results"
FALLBACK_PHRASE = "I don't have enough information in the current policies"
API_ERROR_PHRASE = "unable to process your request"
SLEEP_BETWEEN_REQUESTS = 1  # seconds — avoids Groq rate limits
RETRIES = 2  # a transient Groq error should not be scored as a wrong answer


def load_qa_pairs() -> list[dict]:
    with open(QA_PAIRS_PATH, encoding="utf-8") as f:
        return json.load(f)


def normalize(text: str) -> str:
    """
    Lowercase and flatten typographic variation so that string comparison is not
    defeated by formatting: en/em dashes vs hyphens, markdown emphasis, and
    punctuation the model may add around an otherwise correct answer.
    """
    text = text.lower()
    text = re.sub(r"[‐-―]", "-", text)  # dash variants -> hyphen
    text = text.replace("*", "").replace("_", "")  # markdown emphasis
    text = re.sub(r"[^\w\s%$.-]", " ", text)
    return re.sub(r"\s+", " ", text)


def key_terms(expected: str) -> list[str]:
    """
    Terms a correct answer is expected to contain: any token carrying a digit
    (amounts, percentages, durations) plus words longer than three characters.
    """
    terms = []
    for token in normalize(expected).replace("-", " ").split():
        token = token.strip(".,")
        if not token:
            continue
        if any(ch.isdigit() for ch in token) or len(token) > 3:
            terms.append(token)
    return terms


def _term_coverage(terms: list[str], text: str) -> float:
    if not terms:
        return 0.0
    haystack = normalize(text).replace("-", " ")
    return sum(1 for t in terms if t in haystack) / len(terms)


def is_grounded(chunks: list[dict], expected: str) -> bool:
    """
    Groundedness: is the expected answer actually supported by the evidence the
    retriever returned? Scored against the full chunk text — not the truncated
    snippet shown in the API response, which would hide evidence past 200 chars.
    """
    evidence = " ".join(c.get("text", "") for c in chunks)
    return _term_coverage(key_terms(expected), evidence) >= 0.5


def is_partial_match(answer: str, expected: str) -> bool:
    """
    Optional Exact/Partial Match metric: does the answer itself carry the
    substance of the short gold answer?
    """
    return _term_coverage(key_terms(expected), answer) >= 0.5


def load_document_headings() -> dict[str, set[str]]:
    """Map each policy filename to the set of section headings it actually contains."""
    headings = {}
    for path in POLICIES_DIR.glob("*.md"):
        found = set()
        for line in path.read_text(encoding="utf-8").split("\n"):
            if line.strip().startswith("##"):
                found.add(re.sub(r"\s+", " ", line.lstrip("#").strip().lower()))
        headings[path.name] = found
    return headings


def check_citations(answer: str, headings: dict[str, set[str]]) -> tuple[int, int]:
    """
    Citation validity: does every citation point somewhere real?

    Unlike citation accuracy, this does not ask whether the *expected* document was
    named — it checks that each document and section the model cites actually
    exists, which is what catches an invented attribution.

    Returns (valid, total).
    """
    valid = total = 0
    for raw in CITATION_RE.findall(answer):
        total += 1
        doc, _, section = raw.partition("Section:")
        doc = doc.strip().rstrip(",").strip()
        section = re.sub(r"\s+", " ", section.strip().lower())

        if doc not in headings:
            continue
        if section and section not in headings[doc]:
            continue
        valid += 1
    return valid, total


def is_citation_accurate(answer: str, expected_source: str) -> bool:
    """
    Check if the expected source document is cited in the answer.
    """
    if not expected_source:
        return True  # unanswerable — no citation expected
    return expected_source.lower() in answer.lower()


def is_fallback(answer: str) -> bool:
    return FALLBACK_PHRASE.lower() in answer.lower()


def run_evaluation():
    qa_pairs = load_qa_pairs()
    RESULTS_DIR.mkdir(exist_ok=True)

    results = []
    latencies = []

    grounded_count = 0
    partial_match_count = 0
    partial_match_total = 0
    citation_correct_count = 0
    citation_total = 0
    fallback_correct = 0
    fallback_total = 0
    api_errors = 0
    citations_valid = 0
    citations_total = 0
    headings = load_document_headings()

    print(f"Running evaluation on {len(qa_pairs)} questions...\n")
    print("-" * 60)

    for i, qa in enumerate(qa_pairs, start=1):
        qid = qa["id"]
        question = qa["question"]
        expected = qa["expected_answer"]
        expected_source = qa.get("source_document")
        category = qa.get("category", "factual")

        print(f"[{i:02d}/{len(qa_pairs)}] {qid}: {question[:60]}...")

        for attempt in range(RETRIES):
            result = ask(question)
            answer = result.get("answer", "")
            if API_ERROR_PHRASE not in answer:
                break
            if attempt < RETRIES - 1:
                print("         transient API error - retrying")
                time.sleep(SLEEP_BETWEEN_REQUESTS * 3)

        sources = result.get("sources", [])
        latency_ms = result.get("latency_ms", 0)
        latencies.append(latency_ms)

        if API_ERROR_PHRASE in answer:
            api_errors += 1

        # Grounding is scored against the full retrieved chunks. Retrieval is
        # deterministic, so this returns the same evidence the answer was built
        # from, without truncating it to the API's 200-char snippets.
        chunks = search(question) if category != "unanswerable" else []

        grounded = False
        if category == "unanswerable":
            grounded = is_fallback(answer)
        else:
            grounded = is_grounded(chunks, expected)

        if grounded:
            grounded_count += 1

        # Partial match and citation accuracy (only for factual/multi-hop)
        partial_ok = None
        citation_ok = None
        if category != "unanswerable":
            partial_ok = is_partial_match(answer, expected)
            partial_match_total += 1
            if partial_ok:
                partial_match_count += 1

            citation_ok = is_citation_accurate(answer, expected_source)
            citation_total += 1
            if citation_ok:
                citation_correct_count += 1

        valid, cited = check_citations(answer, headings)
        citations_valid += valid
        citations_total += cited

        # Fallback rate (only for unanswerable)
        if category == "unanswerable":
            fallback_total += 1
            if is_fallback(answer):
                fallback_correct += 1

        record = {
            "id": qid,
            "category": category,
            "question": question,
            "expected": expected,
            "answer": answer,
            "sources": [s["source"] for s in sources],
            "latency_ms": latency_ms,
            "grounded": grounded,
            "partial_match": partial_ok,
            "citation_accurate": citation_ok,
        }
        results.append(record)

        status = "PASS" if grounded else "FAIL"
        src_list = [s["source"] for s in sources[:2]]
        print(f"         {status} grounded | {latency_ms}ms | sources: {src_list}")

        if i < len(qa_pairs):
            time.sleep(SLEEP_BETWEEN_REQUESTS)

    print("-" * 60)

    # Calculate metrics
    total = len(qa_pairs)
    groundedness_pct = round(grounded_count / total * 100, 1)
    partial_match_pct = (
        round(partial_match_count / partial_match_total * 100, 1)
        if partial_match_total
        else 0
    )
    citation_pct = (
        round(citation_correct_count / citation_total * 100, 1) if citation_total else 0
    )
    citation_validity_pct = (
        round(citations_valid / citations_total * 100, 1) if citations_total else 0
    )
    fallback_rate = (
        round(fallback_correct / fallback_total * 100, 1) if fallback_total else 0
    )
    p50 = int(np.percentile(latencies, 50))
    p95 = int(np.percentile(latencies, 95))

    summary = {
        "timestamp": datetime.now().isoformat(),
        "total_questions": total,
        "model": os.getenv("GROQ_MODEL", ""),
        "embedding_model": os.getenv("EMBEDDING_MODEL", ""),
        "top_k": os.getenv("TOP_K", ""),
        "metrics": {
            "groundedness_pct": groundedness_pct,
            "citation_accuracy_pct": citation_pct,
            "citation_validity_pct": citation_validity_pct,
            "partial_match_pct": partial_match_pct,
            "fallback_rate_pct": fallback_rate,
            "latency_p50_ms": p50,
            "latency_p95_ms": p95,
        },
        "counts": {
            "grounded": grounded_count,
            "partial_match": partial_match_count,
            "partial_match_total": partial_match_total,
            "citation_correct": citation_correct_count,
            "citation_total": citation_total,
            "citations_valid": citations_valid,
            "citations_total": citations_total,
            "fallback_correct": fallback_correct,
            "fallback_total": fallback_total,
            "api_errors": api_errors,
        },
        "results": results,
    }

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = RESULTS_DIR / f"eval_{timestamp}.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    # Print summary
    print("\nEVALUATION SUMMARY")
    print("=" * 60)
    print(f"Groundedness:      {groundedness_pct}%  ({grounded_count}/{total})")
    print(
        f"Citation Accuracy: {citation_pct}%  ({citation_correct_count}/{citation_total})"
    )
    cites = f"{citations_valid}/{citations_total} citations resolve"
    print(f"Citation Validity: {citation_validity_pct}%  ({cites})")
    matched = f"{partial_match_count}/{partial_match_total}"
    print(f"Partial Match:     {partial_match_pct}%  ({matched})")
    refused = f"{fallback_correct}/{fallback_total} unanswerable correctly refused"
    print(f"Fallback Rate:     {fallback_rate}%  ({refused})")
    print(f"Latency p50:       {p50}ms")
    print(f"Latency p95:       {p95}ms")
    print("=" * 60)
    print(f"\nFull results saved to: {output_path}")


if __name__ == "__main__":
    run_evaluation()
