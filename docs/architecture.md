# Architecture

How the DFSA compliance assistant is put together: the components, one request end to end, the data model, and how each part fails. The [README](../README.md) is the overview; [WRITEUP.md](WRITEUP.md) is the case study.

---

## Components

| Component | Role | Holds |
|---|---|---|
| **SQLite** (`data/app.db`, WAL) | System of record | Every chunk (also the ones never embedded), cross-references, Defined Terms, ingestion runs, conversations, messages, request traces, eval runs, cache entries |
| **Qdrant** `dfsa_rules__text-embedding-3-small-1536` | Search index | One point per embeddable chunk: dense vector, BM25 vector, payload with text and metadata. Point id = `chunks.id` |
| **Qdrant** `dfsa_cache__text-embedding-3-small-1536` | Answer-cache index | One dense point per cache entry (id = `cache_entries.id`). Used only by the semantic layer, which is off |
| **OpenAI** `text-embedding-3-small` (1536) | Embeddings | Documents and queries, through one client (`src/embeddings/client.py`) |
| **fastembed** `Qdrant/bm25` | Sparse vectors | Local; `avg_len` 124, measured on the corpus |
| **Cross-encoder** `BAAI/bge-reranker-v2-m3` | Reranking + confidence | Local, GPU fp16, pinned to a Hugging Face commit, loaded offline |
| **LLM** GPT-4o | Query rewriting, answers, eval judges | Temperature 0; answers and judges in JSON mode |
| **Langfuse v4** (self-hosted) | Tracing, prompt store, eval datasets | Prompt versions and labels; one trace per turn; dataset `dfsa-golden` |
| **FastAPI** | Serving | `src/api/app.py`; `X-API-Key` on everything but `/health` |

Everything except the two model APIs runs locally in Docker (`infra/docker-compose.yml`), bound to `127.0.0.1`.

**Rule of thumb:** SQLite answers "what is true and what happened", Qdrant answers "what is near this query", Langfuse is where a person looks at traces and edits prompts. Nothing in Qdrant or Langfuse is the only copy of anything.

---

## One chat turn

```mermaid
sequenceDiagram
    autonumber
    actor U as Client
    participant API as FastAPI
    participant DB as SQLite
    participant LLM as GPT-4o
    participant C as Answer cache
    participant E as OpenAI embeddings
    participant Q as Qdrant
    participant R as Reranker (GPU)

    U->>API: POST /conversations/{id}/messages
    API->>DB: last 6 messages (never from the client)
    opt a follow-up (history exists)
        API->>LLM: query-rewrite prompt
        LLM-->>API: standalone question
    end
    API->>C: lookup(namespace, normalised question)
    alt exact repeat in this namespace, cited chunks unchanged
        C-->>API: stored answer + citations (no retrieval, no LLM)
    else miss
        API->>E: embed the question (once; reused by retrieval)
        API->>DB: exact citations named in the question ("COB 3.3.2")
        API->>Q: dense + BM25 search, one batched request, filters in Qdrant
        API->>R: rerank exact hits + top 40 candidates
        alt top rerank score < 0.40, or no context
            API-->>API: refusal, no LLM call
        else confident
            API->>DB: context: scope rules, Guidance, sections, definitions
            API->>LLM: answer prompt over numbered blocks (JSON)
            API-->>API: citation validator (fail = refusal)
        end
    end
    API->>DB: user + assistant message + trace row (one transaction)
    opt answered, first turn, production prompts
        API->>C: store the answer (SQLite row + Qdrant point)
    end
    API-->>U: answer or refusal, citations, trace id, cost, latency
```

**Stages, with the code behind each:**

| Stage | Code | Notes |
|---|---|---|
| History | `ChatService.send` (`src/chat/service.py`) | `CHAT_HISTORY_TURNS=6` messages, each cut to 1,500 characters |
| Rewrite | `src/chat/rewrite.py` | Only when there is history; an empty or cut-off rewrite falls back to the original |
| Cache | `src/chat/cache.py` | See [The answer cache](#the-answer-cache) |
| Citation router | `src/retrieval/citations.py` | Resolves "COB 3.3.2", "Article 20 of the Regulatory Law", sections; reports unresolved and too-broad citations |
| Hybrid search | `src/retrieval/hybrid.py` | RRF fusion client-side, so each candidate keeps its dense and BM25 ranks |
| Rerank | `src/retrieval/rerank.py` | Sigmoid scores 0–1; truncation counted with the reranker's own tokenizer |
| Context | `src/retrieval/context.py` | One block = one provision, labelled binding / non-binding; 6,000-token budget, never drops a hit |
| Gate + answer | `src/generation/answer.py` | Below `RETRIEVAL_REFUSAL_THRESHOLD` (0.40) refuses before the LLM |
| Citation check | `src/generation/citations.py` | Every cited block exists and carries that citation; every provision named is a listed block or a reference quoted from one |
| Store | `src/storage/repositories.py` | Messages and trace in one transaction; the cache write afterwards, best effort |

**Latency and cost per turn** (RTX 3050 laptop GPU): about 4.5–8 s and $0.013–0.02 for an answer, almost all of it GPT-4o input; a refusal at the gate costs only the query embedding (plus the rewrite for a follow-up); an exact cache hit takes about 5 ms and costs $0.

---

## The answer cache

`src/chat/cache.py`. **Exact repeats only**, by decision (see below).

- **Key:** the standalone (rewritten) question, normalised (case, spacing, closing punctuation), hashed.
- **Namespace:** sha256 of everything that shapes an answer:
  - the corpus (every embedded chunk's content hash, recomputed when a new ingestion run finishes);
  - the embedding and BM25 fingerprints;
  - the retrieval config (incl. the reranker revision and the refusal threshold);
  - the answer prompt's content hash and the LLM deployment;
  - the request filters.

  A lookup only searches its own namespace, so a re-ingest, a prompt promotion or a config change makes old entries unreachable at once. `scripts/cache.py purge` deletes them.
- **Verified on every hit:** each cited chunk must still have the content hash it was answered from.
- **Stored only:** answered turns that passed the calibrated confidence gate, on production prompts, from first turns (a follow-up's answer can lean on the conversation). A follow-up can still *hit*, through its rewritten question.
- **Audit:** a cached turn's trace row has `cache_status='hit'` and the entry id, and the entry points at the trace of the turn that produced the answer, with its retrieval scores and prompt versions.

**Why not semantic matching.** The semantic layer (nearest cached question in Qdrant, plus guards: same provisions, Defined Terms and numbers) is built and tested but switched off (`CACHE_SIMILARITY_THRESHOLD` unset). `src/eval/cache_eval.py` measured it on 32 paraphrases and 32 hard negatives: real paraphrases score 0.83–0.98 cosine, and "CFD margin on a **major** currency pair" vs "… **minor** currency pair" (different answers) scores 0.978. No threshold serves a paraphrase without serving a wrong answer. A similarity-only cache would have served 17 of 32 wrong answers at 0.80 and 6 at 0.90. Enabling a threshold has to pass that eval's gate (0 false hits).

---

## Ingestion

```
data/manifest.yaml ─► Docling parse (cached by file hash + Docling version + options)
                   ─► element stream (reading order fixed, headers/footers/TOC dropped)
                   ─► chunker (Rulebook / Glossary / Laws) ─► split at legal boundaries (≤ 512 tokens)
                   ─► cross-references + Defined Terms ─► SQLite ─► embed changed chunks ─► Qdrant
```

- **Chunk types:** `rule` (binding), `guidance` (non-binding), `definition` (binding), `article` (binding), `text` (binding in a Law, not in the Rulebook), `section` (a parent: stored, never embedded).
- **Idempotent.** `content_hash = sha256(embedding fingerprint + embedded text)` per chunk; an unchanged re-run embeds nothing and costs $0. A metadata-only change rewrites the payload (`payload_hash`) without re-embedding.
- **Self-healing.** The index plan is diffed against the points actually in Qdrant, not SQLite's `is_embedded` flag, so a crash between the two writes, a lost collection or a `--no-embed` reload repairs itself on the next run.
- **Reconcile.** `ingest --reconcile` proves SQLite and Qdrant agree (exit 1 if not); `--fix` deletes orphan points.
- **No crawling.** PDFs are downloaded by hand and listed in the manifest with their sha256; ingestion never touches a live URL.

---

## Data model

### SQLite (Alembic migrations 0001–0004)

```
documents 1 ──< chunks >── chunk_refs      what each chunk references (target_key indexed)
                   └────── chunk_terms     which Defined Terms it uses (term indexed)
ingestion_runs                             counts, tokens, $ per ingest

conversations 1 ──< messages               user / assistant turns; rewritten query, citations, trace id
traces                                     one per request (= Langfuse trace id): retrieval and answer
                                           traces as JSON, tokens and $ per stage, latency per stage,
                                           cache status / entry / lookup trace
cache_entries                              cached answers: namespace + components, guards, cited chunks
                                           with content hashes, source trace, hits, expiry

eval_runs 1 ──< eval_results               config, prompt hashes, golden-set hash, metrics, gate report
```

Only portable SQLAlchemy types are used, and all access goes through `src/storage/repositories.py`, so moving to Postgres is a configuration change.

### Qdrant: the chunk collection

- Vectors: `dense` (1536, cosine) and `bm25` (sparse, `Modifier.IDF`).
- Collection metadata: embedding fingerprint, model, dimensions, sparse fingerprint, schema version. A mismatch raises `EmbeddingModelMismatch` / `CollectionMismatch` at startup.
- Payload: identity (`key`, `citation`, `chunk_type`, `binding`, `regulator`, `doc_type`, `module`, `doc_id`, `version`), structure (`section`, `parent_key`, `part_of`, `guidance_for`, `rule_number`, `term`, `scope_rule`), links (`defined_terms`, `cross_refs`), content (`header_text`, `text`, pages), provenance (`content_hash`, `embedding_fingerprint`, `payload_hash`).
- 12 payload indexes: 10 keyword (`regulator`, `doc_type`, `module`, `doc_id`, `chunk_type`, `citation`, `key`, `parent_key`, `term`, `defined_terms`) and 2 boolean (`binding`, `scope_rule`).

---

## Prompts and the release gate

```
prompts/*.yaml (git) ─sync_prompts─► Langfuse version, label "staging"
                                        │
                     src.eval.run --label staging  (e2e: answers + judges + gate)
                                        │ passed, and the prompt file is committed
                     promote_prompt ────► label "production"
```

- Five prompts: `answer`, `refusal`, `query-rewrite`, `judge-faithfulness`, `judge-relevance`. Each declares its variables; compiling with a missing or unknown one raises.
- One content hash (type + messages + config) for the git file and the Langfuse version. It drives sync, promotion and drift checks, and is recorded in every trace.
- `promote_prompt` moves `production` only to a version whose content equals the **committed** git file, and only if the newest e2e eval run with exactly that content passed `eval/thresholds.yaml` (floors, ceilings, ≤ 3% regression against the last passing run). `--skip-gate REASON` is a logged override.
- **Rollback** is a git revert, a sync and a promote.
- If Langfuse can't serve a prompt, the registry falls back to the git copy and records `local-fallback` with the reason.
- The cache has its own gate: `src.eval.cache_eval` at the configured threshold must serve 0 hard negatives.

---

## Failure modes

| What fails | What happens | Why |
|---|---|---|
| Retrieval isn't confident (top rerank < 0.40) | Refusal, no LLM call | A wrong answer is worse than none |
| The model cites something it wasn't given, or isn't confident | Refusal (citation validator) | Never a patched answer |
| LLM, OpenAI or Qdrant down | HTTP 503 with the trace id; an error trace is stored | An outage is not "no confident answer" |
| Any other exception | HTTP 500 with the trace id | |
| Langfuse down | Answers still work; prompts fall back to git; tracing is lost | `/health` reports it but stays `ok` |
| Embedding model changed in config | Startup fails (`EmbeddingModelMismatch`), for both collections | Vectors from two models never mix |
| BM25 settings changed | Startup fails (`CollectionMismatch`) | Document weights depend on them |
| Database behind the code | Startup fails: "run `alembic upgrade head`" | |
| Reranker not in the local Hugging Face cache | Startup fails with a hint to download it once | Loaded offline, pinned to a commit |
| A cached answer's sources changed | Not served (namespace and per-chunk content hash) | |
| Writing a cache entry fails | Logged; the turn still succeeds | The cache is an optimisation |

---

## Configuration

Settings come from `.env` and the environment (`src/config/settings.py`), one section per concern:

| Prefix | Section |
|---|---|
| `AZURE_OPENAI_` | Generation: endpoint, key, deployment, prices |
| `EMBEDDING_` (+ `OPENAI_API_KEY`) | The single embedding config: model, dimensions, price. Used by ingestion and queries |
| `QDRANT_` | URL, key, BM25 parameters (`k`, `b`, `avg_len`) |
| `CHUNK_` | Chunking limits |
| `RETRIEVAL_` | Mode, `candidate_k`, rerank on/off and model, `top_n`, `refusal_threshold`, context policy and budget |
| `CHAT_` | History length and message limits |
| `CACHE_` | Answer cache: `enabled`, `similarity_threshold` (unset = exact only), `ttl_days` |
| `LANGFUSE_` | Host, keys, prompt label and cache TTL |
| `API_` | `API_KEY`, auth switch |

`127.0.0.1` is used everywhere instead of `localhost`: on Windows, `localhost` tries IPv6 first and each new connection waited about 2 s.
