# Policy Assistant — Sunshine Consulting

A retrieval-augmented chat assistant that answers employee questions about company
policies. It answers **only** from the policy corpus in `data/policies/`, cites the
document and section behind every answer, links to the source document, and refuses
to answer when the corpus does not cover the question.

Built with Flask, ChromaDB, `sentence-transformers`, and Groq. The RAG pipeline is
implemented directly, without LangChain.

---

## Setup

**Requirements:** Python 3.12 (see `.python-version`) and a free
[Groq API key](https://console.groq.com).

```bash
# 1. Create and activate a virtual environment
python -m venv venv
venv\Scripts\activate          # Windows
source venv/bin/activate       # macOS / Linux

# 2. Install dependencies
pip install -r requirements.txt

# 3. Configure environment
cp .env.example .env           # then open .env and set GROQ_API_KEY

# 4. Build the vector index  ← required before the first run
python scripts/build_index.py
```

Step 4 is **not optional**. The Chroma index is not committed to the repository, so
a fresh clone has nothing to retrieve from until it is built. It reads every
document in `data/policies/`, chunks and embeds them, and writes to `./chroma_db`
(~1 minute; the embedding model is downloaded on first run). The script is
idempotent — re-run it any time the policy documents change.

## Running

```bash
python app/web.py                      # development server → http://localhost:5000
gunicorn app.web:app --bind 0.0.0.0:5000   # production server
```

## Endpoints

| Method | Path | Description |
|---|---|---|
| `GET` | `/` | Chat interface |
| `POST` | `/chat` | Accepts `{"question": "..."}`, returns the answer, its sources (document, section, snippet, link) and latency |
| `GET` | `/health` | Service status, index size, active model |
| `GET` | `/policies/<filename>` | Serves a source policy document, so citations link to the original |

```bash
curl -X POST http://localhost:5000/chat \
  -H "Content-Type: application/json" \
  -d '{"question": "How many days of PTO do I accrue per month?"}'
```

## Configuration

All settings are read from `.env` (see `.env.example`):

| Variable | Default | Purpose |
|---|---|---|
| `GROQ_API_KEY` | — | Groq API key (required) |
| `GROQ_MODEL` | `openai/gpt-oss-20b` | Generation model |
| `EMBEDDING_MODEL` | `BAAI/bge-small-en-v1.5` | Embedding model |
| `CHROMA_PERSIST_DIR` | `./chroma_db` | Index location |
| `TOP_K` | `5` | Chunks retrieved per question |
| `MAX_QUERY_LENGTH` | `500` | Maximum question length |

## Tests

```bash
pip install -r requirements-dev.txt
pytest tests/ -v
```

21 tests covering chunking, section extraction, query embedding, guardrails, and
all four endpoints. GitHub Actions runs `black`, `flake8`, an import check, and the
test suite on every push and pull request (`.github/workflows/ci.yml`).

## Evaluation

25 questions across all ten policy areas:

```bash
python evaluation/evaluate.py
```

| Metric | Result |
|---|---|
| Groundedness | 100.0% (25/25) |
| Citation accuracy | 100.0% (23/23) |
| Citation validity | 100.0% (28/28 citations resolve) |
| Partial match | 95.7% (22/23) |
| Fallback rate | 100.0% (2/2) |
| Latency p50 / p95 | 3,495 ms / 5,761 ms |

Approach, limitations, and the design rationale behind these choices:
**[design-and-evaluation.md](design-and-evaluation.md)**.

AI tools used during development: **[ai-tooling.md](ai-tooling.md)**.

---

## Project structure

```
app/web.py              Flask app — routes and request handling
src/ingestion.py        Document loaders (md/txt/pdf/html), chunking, embedding, indexing
src/retrieval.py        Query embedding and top-k search
src/prompts.py          System prompt and RAG prompt construction
src/guardrails.py       Input and output validation
src/pipeline.py         Orchestrates the full question → answer flow
scripts/build_index.py  Builds the Chroma index from data/policies/
evaluation/             Evaluation harness and question set
data/policies/          The policy corpus (10 documents)
```
