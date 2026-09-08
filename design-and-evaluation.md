# Design and Evaluation

Design and architecture decisions behind the Policy Assistant, and the evaluation
of the RAG system.

---

# Part 1 — Design and architecture

## Architecture

```
Question → guardrails → embed query → Chroma top-k search
         → prompt (rules + labelled excerpts) → Groq → guardrails → answer + citations
```

Six modules, each with one job: `ingestion` (load, chunk, embed, index),
`retrieval` (query embedding, search), `prompts` (prompt construction),
`guardrails` (input and output validation), `pipeline` (orchestration), and
`app/web.py` (HTTP). The pipeline is implemented directly rather than through
LangChain — at this size the framework would add a dependency and a layer of
indirection without removing any real work, and writing the six steps explicitly
makes the data flow auditable, which matters for a system whose main promise is
that it does not invent policy.

## Technology choices

| Choice | Why |
|---|---|
| **ChromaDB**, local and persistent | 136 chunks does not justify a hosted vector database. Runs in-process, persists to one SQLite file, keeps the project free and reproducible offline. Cosine distance, matching the L2-normalised embeddings. Trade-off: the index does not survive a stateless deploy and must be rebuilt at build time. |
| **BAAI/bge-small-en-v1.5** embeddings | Free and local — no API key, quota, or network round-trip per query. Strong retrieval quality at 384 dimensions and small enough for a CPU-only container. Queries receive BGE's `"Represent this sentence for searching relevant passages: "` prefix and documents do not, which is the asymmetric usage the model was trained for. `render.yaml` substitutes the smaller `all-MiniLM-L6-v2`; the prefix logic is model-aware, so both work unchanged. |
| **Flask** | The brief asks for three endpoints and a chat page. Flask covers that in one file; Django or FastAPI would add structure this size of app cannot use. |
| **Groq**, `openai/gpt-oss-20b` | Free tier, fast, and reliably follows the citation format, with more rate-limit headroom than the 120b variant. |
| **`reasoning_effort: low`** | gpt-oss spends hidden reasoning tokens before answering, and those count against `max_tokens`. Left unbounded, reasoning intermittently consumed the entire budget and the model returned **empty content** — a blank answer with only the guardrail's note attached. Capping reasoning fixed it and cut a typical call from ~1.5 s to ~0.4 s. The pipeline also treats empty content as an error rather than passing a blank answer to the user. |
| **No LangChain** | See above — the pipeline is six explicit steps. |

## Chunking — 512 characters, 64 overlap

Recursive splitting on a separator hierarchy: paragraphs → lines → sentences →
spaces. The splitter only falls back to a cruder separator when a run of text is
genuinely too long, so chunks almost always end on a natural boundary rather than
mid-sentence.

512 characters (~100–130 tokens) keeps a chunk mostly signal — policy prose is
dense and a single clause often answers the question — and lets five chunks fit
comfortably in context. The 64-character overlap (~12%) stops a fact that straddles
a boundary from being lost by both neighbours, at little duplication cost.

The unit is **characters, not tokens**: exact, dependency-free and deterministic,
which matters more here than the modest packing efficiency a tokeniser would buy.

Two supporting decisions:

- **Section titles as metadata.** Each chunk stores the nearest preceding `##`
  heading, which is what lets a citation name *Section: Accrual* rather than just
  a filename.
- **Content-hash IDs.** Each chunk's ID is `md5(source:text)` and writes use
  `upsert`, so re-indexing is idempotent — unchanged chunks overwrite themselves
  instead of duplicating.

## Retrieval — top-k = 5, no re-ranking

Five chunks give roughly 2,500 characters of context: enough to cover a question
spanning two documents (remote work during probation, for instance) without
diluting the prompt with unrelated text. Below k=3 multi-document questions began
missing a source; much above k=5 the extra chunks were mostly noise.

Re-ranking is optional in the brief and was skipped deliberately: a cross-encoder
would add a second model to load and several hundred milliseconds per query, and
with a 136-chunk corpus first-stage retrieval already returns the supporting
evidence for every question in the evaluation set — there is no headroom for a
re-ranker to recover.

## Prompt format

Rules live in the system message; evidence and the question live in the user
message, so instructions stay constant while only evidence varies.

```
[Excerpt 1 from pto_policy.md, Section: Accrual]
<chunk text>
---
[Excerpt 2 from ...]
---
Employee question: <question>
Answer based only on the excerpts above:
```

- **Each excerpt is labelled inline with its filename and section**, so everything
  needed for a correct citation sits next to the text being cited and the model
  never has to invent an attribution. This is the largest single contributor to
  citation accuracy.
- **The citation format and the refusal sentence are fixed strings**, shared as
  literals between the prompt, the guardrails and the evaluation script. Both can
  therefore be checked exactly rather than guessed at.
- **The closing instruction** repeats the grounding constraint immediately before
  generation, where it carries most weight.

## Guardrails

| Requirement | Mechanism |
|---|---|
| Refuse outside the corpus | Fixed refusal sentence in the system prompt, directing the employee to People Operations |
| Limit output length | `max_tokens=1024` at the API plus a 2,000-character truncation in `validate_output()` |
| Always cite sources | Citation format in the prompt; if `[Source:` is missing, a verification notice is appended |

Input validation additionally rejects empty queries, queries over 500 characters,
and eight regex patterns for common prompt-injection phrasings. This is a cheap
first filter, not a complete defence — the substantive protection is that the model
sees only retrieved policy text and has no tools or write access.

## Reproducibility

No stage of the pipeline draws on a random number generator: chunking is a pure
function of the text, chunk IDs are content hashes, embedding is deterministic,
retrieval is an exhaustive search over 136 chunks, and the evaluation set is fixed
and evaluated in order. `temperature=0` is the corresponding control for
generation — the one otherwise stochastic step — which is why there is no
`random.seed()` to set. Dependencies are pinned in `requirements.txt` and the
Python version in `.python-version`.

Reading the model from an environment variable rather than hard-coding it proved
its worth mid-project: Groq decommissioned `llama-3.1-8b-instant`, which the
project originally used, and the switch to `openai/gpt-oss-20b` was a one-line
change.

---

# Part 2 — Evaluation

**Run of 7 September 2026** — `openai/gpt-oss-20b`, `BAAI/bge-small-en-v1.5`,
`top_k=5`, `temperature=0`. Raw output: `evaluation/results/baseline.json`.

```bash
python evaluation/evaluate.py
```

## Approach

25 questions in `evaluation/qa_pairs.json`, covering all ten policy documents.
Each carries a short gold answer and the document that should be cited.

| Category | Count | Purpose |
|---|---|---|
| `factual` | 22 | Single-document lookups — PTO, expenses, remote work, security, holidays, performance, onboarding, travel |
| `multi-hop` | 1 | Requires combining two documents |
| `unanswerable` | 2 | Not covered by the corpus — must be refused, not guessed |

Key terms of a gold answer are its digit-bearing tokens plus words longer than
three characters; a check passes when at least half of them appear.

| Metric | What it asks |
|---|---|
| **Groundedness** | Is the gold answer present in the retrieved chunks? Matched against the full chunk text. For `unanswerable` questions, it means emitting the refusal. |
| **Citation Accuracy** | Does the answer cite the document that contains the answer? Not scored for `unanswerable`. |
| **Citation Validity** | Does every citation point somewhere real? Each `[Source: <document>, Section: <section>]` is checked against the corpus — the file must exist and the section must match a heading in it. Catches invented attributions. |
| **Partial Match** *(optional)* | Does the answer itself carry the gold answer's substance? The same term test, applied to the answer rather than the evidence. |
| **Fallback Rate** | Of the unanswerable questions, how many produced the exact refusal. |
| **Latency** | Wall-clock time inside `ask()`: guardrails, embedding, search, the Groq call, validation. |

## Results

| Metric | Result | |
|---|---|---|
| **Groundedness** | **100.0%** | 25 / 25 |
| **Citation Accuracy** | **100.0%** | 23 / 23 |
| **Citation Validity** | **100.0%** | 28 / 28 citations resolve |
| **Partial Match** *(optional)* | **95.7%** | 22 / 23 |
| **Fallback Rate** | **100.0%** | 2 / 2 unanswerable correctly refused |
| **Latency p50** | **3,495 ms** | |
| **Latency p95** | **5,761 ms** | |

No API errors occurred during the run.

### Latency

The percentiles describe the harness more than the application. The raw
distribution is bimodal, not noisy:

```
348  379  431  437  438  536  621  626  658  661  856   ← requests 1-11
2587 3495 4481 4501 4667 4748 5560 5623 5669 5706 ...   ← thereafter
```

The first eleven questions finish in under a second; everything after that costs
seconds. That boundary is Groq's free tier throttling a burst of sequential
requests. Timing the stages confirms it: embedding and search together take about
100 ms, an unthrottled Groq call 401 ms, a throttled one up to 6.6 s.

So a user asking occasional questions sees roughly **half a second** end to end,
and the p50 of 3.5 s is an artifact of firing 25 questions back to back. The
percentiles are correspondingly unstable — earlier runs the same day gave p95
values of 11,808 ms and 7,384 ms on identical inputs.

## A correction to the measurement

An earlier version of this evaluation reported 88% groundedness. That number was
wrong, and the reason is worth recording.

The original scorer compared gold-answer terms against the `snippet` field of the
API response — which is **truncated to 200 characters**. Evidence past that cutoff
was invisible to the scorer, so correct answers were marked ungrounded:

| Question | Gold answer | Model's answer | Old verdict |
|---|---|---|---|
| `security-003` | Public, Internal, Confidential, Restricted | "Public, Internal, Confidential, Restricted" | ✗ |
| `performance-001` | November-December, Q4 | "November–December (Q4)" | ✗ |
| `remote-003` | Once per quarter | "quarterly in-person visits" | ✗ |

Three of four failures were artifacts of the metric, not model errors; the fourth
was a transient API error. The scorer now compares against the full chunk text,
normalises dash variants, markdown emphasis and punctuation before matching, counts
any token containing a digit as a key term so amounts like `1.5` and `8%` are
checked, and retries once on a transient API error rather than scoring it as a
wrong answer.

## Limitations

These results mean *"no failures were detected by these checks"*, not *"the system
is correct"*.

**The metrics are lexical.** They test term overlap, so a fully paraphrased answer
can score badly and an answer with the right terms inside a wrong claim can score
well. A judge model or human annotation would measure faithfulness properly; term
overlap is a cheap approximation suited to a 25-question set.

The single Partial Match miss shows exactly this ceiling. `multihop-001` expects
*"No, remote work requires completing the 90-day probationary period"*; the model
answered *"No. New employees must work on-site for the first 90 days of
employment."* That is correct and cites both relevant documents, but it says
"on-site" rather than "remote work" and never uses "probationary", so too few gold
terms appear. No normalisation rule fixes paraphrase.

**Groundedness here is closer to retrieval coverage.** It asks whether the
supporting evidence was retrieved, not whether every sentence of the answer follows
from it. An answer that cited correctly but added an unsupported sentence would
still pass. That is the main gap against the definition in the brief.

**Citation checks stop short of attribution.** Validity confirms each cited
document and section exists; accuracy confirms the expected document was named.
Neither confirms the cited section actually supports the claim beside it. Five of
the 23 answers named a second document — correct for the multi-hop question, and
defensible for the expense/travel pairs, whose policies overlap.

**100% reflects a favourable setting.** Ten short documents written for this
project, with questions written from them, mostly single-fact lookups. On a larger,
messier corpus the numbers would fall. The two `unanswerable` questions are the only
adversarial pressure in the set.

**Results move between runs.** `temperature=0` makes decoding greedy but does not
make the hosted API bit-for-bit reproducible: the same question passed Partial Match
in one run and missed in the next on identical inputs.

**Single run, no ablations.** Retrieval `k`, chunk size and prompt variants were
not swept; `k=5` and 512-character chunks were chosen on the reasoning in Part 1
rather than by measurement.
