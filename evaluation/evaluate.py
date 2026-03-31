"""
Evaluation script for the Policy RAG Chatbot.

Metrics:
- Groundedness %: answer is supported by retrieved chunks
- Citation Accuracy %: correct source document is cited
- Latency p50/p95: response time percentiles
- Fallback Rate: unanswerable questions correctly refused

Usage:
    python evaluation/evaluate.py
"""

import json
import sys
import time
from pathlib import Path
from datetime import datetime

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.pipeline import ask

QA_PAIRS_PATH = Path(__file__).parent / "qa_pairs.json"
RESULTS_DIR = Path(__file__).parent / "results"
FALLBACK_PHRASE = "I don't have enough information in the current policies"
SLEEP_BETWEEN_REQUESTS = 1  # seconds — avoids Groq rate limits


def load_qa_pairs() -> list[dict]:
    with open(QA_PAIRS_PATH, encoding="utf-8") as f:
        return json.load(f)


def is_grounded(answer: str, sources: list[dict], expected: str) -> bool:
    """
    Check if key terms from the expected answer appear in the retrieved chunks.
    """
    expected_lower = expected.lower()
    # Extract key terms: numbers and words longer than 3 chars
    key_terms = [w for w in expected_lower.split() if len(w) > 3 or w.replace("$", "").replace("%", "").isdigit()]

    if not key_terms:
        return False

    all_chunk_text = " ".join(s.get("snippet", "").lower() for s in sources)
    matched = sum(1 for term in key_terms if term in all_chunk_text)
    return matched / len(key_terms) >= 0.5


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
    citation_correct_count = 0
    citation_total = 0
    fallback_correct = 0
    fallback_total = 0

    print(f"Running evaluation on {len(qa_pairs)} questions...\n")
    print("-" * 60)

    for i, qa in enumerate(qa_pairs, start=1):
        qid = qa["id"]
        question = qa["question"]
        expected = qa["expected_answer"]
        expected_source = qa.get("source_document")
        category = qa.get("category", "factual")

        print(f"[{i:02d}/{len(qa_pairs)}] {qid}: {question[:60]}...")

        result = ask(question)
        answer = result.get("answer", "")
        sources = result.get("sources", [])
        latency_ms = result.get("latency_ms", 0)
        latencies.append(latency_ms)

        # Groundedness
        grounded = False
        if category == "unanswerable":
            grounded = is_fallback(answer)
        else:
            grounded = is_grounded(answer, sources, expected)

        if grounded:
            grounded_count += 1

        # Citation accuracy (only for factual/multi-hop)
        citation_ok = None
        if category != "unanswerable":
            citation_ok = is_citation_accurate(answer, expected_source)
            citation_total += 1
            if citation_ok:
                citation_correct_count += 1

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
            "citation_accurate": citation_ok,
        }
        results.append(record)

        status = "✓" if grounded else "✗"
        print(f"         {status} grounded | {latency_ms}ms | sources: {[s['source'] for s in sources[:2]]}")

        if i < len(qa_pairs):
            time.sleep(SLEEP_BETWEEN_REQUESTS)

    print("-" * 60)

    # Calculate metrics
    total = len(qa_pairs)
    groundedness_pct = round(grounded_count / total * 100, 1)
    citation_pct = round(citation_correct_count / citation_total * 100, 1) if citation_total else 0
    fallback_rate = round(fallback_correct / fallback_total * 100, 1) if fallback_total else 0
    p50 = int(np.percentile(latencies, 50))
    p95 = int(np.percentile(latencies, 95))

    summary = {
        "timestamp": datetime.now().isoformat(),
        "total_questions": total,
        "metrics": {
            "groundedness_pct": groundedness_pct,
            "citation_accuracy_pct": citation_pct,
            "fallback_rate_pct": fallback_rate,
            "latency_p50_ms": p50,
            "latency_p95_ms": p95,
        },
        "counts": {
            "grounded": grounded_count,
            "citation_correct": citation_correct_count,
            "citation_total": citation_total,
            "fallback_correct": fallback_correct,
            "fallback_total": fallback_total,
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
    print(f"Citation Accuracy: {citation_pct}%  ({citation_correct_count}/{citation_total})")
    print(f"Fallback Rate:     {fallback_rate}%  ({fallback_correct}/{fallback_total} unanswerable correctly refused)")
    print(f"Latency p50:       {p50}ms")
    print(f"Latency p95:       {p95}ms")
    print("=" * 60)
    print(f"\nFull results saved to: {output_path}")


if __name__ == "__main__":
    run_evaluation()
