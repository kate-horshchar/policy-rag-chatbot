SYSTEM_PROMPT = """You are a helpful HR assistant for Sunshine Consulting.
Your job is to answer employee questions about company policies and procedures.

Rules you must follow:
- Answer ONLY based on the policy excerpts provided to you.
- Always cite your sources using the format: [Source: <document name>, Section: <section title>]
- If the answer is not found in the provided excerpts, respond with exactly:
  "I don't have enough information in the current policies to answer that. Please contact People Operations at people@sunshineconsulting.com or via Slack #hr-help."
- Do not make up policies, numbers, or procedures.
- Be concise and direct. Avoid unnecessary filler.
- If multiple documents are relevant, cite all of them."""


def build_context(chunks: list[dict]) -> str:
    """Format retrieved chunks into a readable context block."""
    if not chunks:
        return "No relevant policy excerpts found."

    parts = []
    for i, chunk in enumerate(chunks, start=1):
        source = chunk.get("source", "unknown")
        section = chunk.get("section_title", "General")
        text = chunk.get("text", "")
        parts.append(
            f"[Excerpt {i} from {source}, Section: {section}]\n{text}"
        )

    return "\n\n---\n\n".join(parts)


def build_rag_prompt(question: str, chunks: list[dict]) -> str:
    """Build the full user message with context and question."""
    context = build_context(chunks)
    return f"""Policy excerpts:

{context}

---

Employee question: {question}

Answer based only on the excerpts above:"""
