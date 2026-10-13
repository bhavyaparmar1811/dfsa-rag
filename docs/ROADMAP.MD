# DFSA RAG: implementation roadmap and tracker

> **Purpose:** one Markdown file for tracking the whole build: what's done (with the decisions and findings behind it), what's in progress, and what's next.
> **After approval:** saved to the repo as `docs/ROADMAP.md` and updated at every checkpoint.

## Context
This is a compliance Q&A system over the DFSA Rulebook and the DIFC Laws. It answers in plain language with **exact rule-level citations**, and **refuses when it isn't confident**. It is portfolio project 1, so it has to show production RAG practice end to end: structure-aware ingestion, hybrid retrieval with reranking, grounded generation, observability, evaluation with a release gate, and cost tracking.

The stack is fixed in CLAUDE.md: Azure OpenAI GPT-4o, OpenAI `text-embedding-3-small`, Qdrant, SQLite, Docling, FastAPI. Langfuse (self-hosted) was added for prompt management, tracing and evals.

**Status key:** ✅ done · 🔄 in progress · ⬜ not started · ⏸ deferred

| Phase | Scope | Status |
|---|---|---|
| A | Foundation: settings, embedding client, smoke tests, Qdrant container | ✅ |
| B | Extraction and chunking of all 11 documents | ✅ |
| C | Storage (SQLite + Qdrant) and embedding | 🔄 C1–C3 done |
| D | Retrieval: hybrid search, citation lookup, rerank, parent expansion | ⬜ |
| E | Langfuse: self-hosted tracing and prompt store | ⬜ |
| F | Generation, grounding and refusal | ⬜ |
| G | Chat history, query rewriting, FastAPI | ⬜ |
| H | Evaluation: golden set, metrics, LLM judges, release gate | ⬜ |
| I | Cost reporting, docs, polish | ⬜ |
| J | VARA as a second regulator | ⏸ |

---

## Phase A: Foundation ✅
| Item | Where | Notes |
|---|---|---|
| ✅ uv project, dependencies, `pyproject.toml`, Ruff, pytest | root | Python 3.12 |
| ✅ Settings with a separate section per provider | `src/config/settings.py` | `AZURE_OPENAI_*` (generation), `OPENAI_API_KEY` (embeddings), `QDRANT_*`, `CHUNK_*`, paths |
| ✅ The single embedding config + fingerprint | `EmbeddingSettings.fingerprint` → `text-embedding-3-small-1536` | It also names the Qdrant collection, so changing the model means a new collection |
| ✅ The shared `EmbeddingClient` (ingestion and queries) | `src/embeddings/client.py` | Batching, dimension check, `assert_compatible()`, token and cost logging, rejects inputs over 8,191 tokens |
| ✅ Smoke tests | `scripts/check_azure_openai.py`, `scripts/check_openai_embeddings.py`, `scripts/check_qdrant.py` | |
| ✅ Qdrant 1.19.0 in Docker, localhost only, API key enforced | `infra/docker-compose.yml` | Key in `.env`; `:?` makes compose refuse to start without it; named volume `qdrant_data` |
| ✅ Corpus manifest | `data/manifest.yaml` | 6 Rulebook modules (GEN, COB, AML, MKT, FPR, GLO) and 5 Laws; sha256 of each PDF |

## Phase B: Extraction and chunking ✅
**Pipeline:** `manifest → Docling parse (cached) → elements (cleaned, ordered) → chunker → split → cross-references → defined terms → JSONL + quality report`
**Run it:** `uv run python -m src.ingestion.chunk_corpus` writes `data/parsed/chunks/*.jsonl` and `data/parsed/reports/*.json`

| Step | What | Key files |
|---|---|---|
| ✅ B1 | Manifest loader: relative paths, lowercase sha256, fails loudly on a mismatch | `src/ingestion/manifest.py` |
| ✅ B2 | Docling parse (OCR off, images off), cache keyed on file hash + Docling version + options | `src/ingestion/parse.py`, `scripts/inspect_docling.py` |
| ✅ B3 | Element stream: reads `orig`, orders by page position, drops headers/footers, TOC and figures, flattens layout tables | `src/ingestion/elements.py` |
| ✅ B4 | Rulebook chunker: one chunk per rule and per Guidance block, section parents, scope rules, sequence guards | `src/ingestion/chunkers/rulebook.py`, `patterns.py`, `models.py`, `header.py`, `ids.py`, `validate.py` |
| ✅ B5 | Splitting at legal boundaries (512 tokens including the header) and cross-reference extraction | `src/ingestion/split.py`, `src/ingestion/refs.py` |
| ✅ B6 | GEN, AML, MKT and FPR brought to zero issues | as above |
| ✅ B7 | Glossary definitions and tagging every chunk with the Defined Terms it uses | `src/ingestion/chunkers/glossary.py`, `src/ingestion/terms.py` |
| ✅ B8 | DIFC Laws: articles, Schedule paragraphs, Law definitions | `src/ingestion/chunkers/law.py` |

**Chunk types:**
- `rule` (binding)
- `guidance` (non-binding)
- `definition` (binding; from GLO and the Laws' Schedules)
- `article` (binding; Law articles and Schedule paragraphs)
- `text` (unnumbered text: binding in a Law, non-binding in the Rulebook)
- `section` (a parent: stored, never embedded)

**Corpus:**
- 4,060 chunks in total, **3,397 embeddable**: 1,495 rule chunks (including split parts), 609 Guidance, 402 articles, 872 definitions, 19 text
- 496 parents
- 587,492 embedding tokens, about **$0.012** for a full embed
- 2,114 cross-references (2,495 `chunk_refs` rows, because each part of a split chunk carries its own; the earlier figure of 1,469 predates GLO and the Laws)
- 97% of chunks tagged with at least one Defined Term

**Findings from building it:**
- **Docling's output needed correcting:**
  - it moves rule numbers into a separate `marker` field;
  - its reading order is wrong on 89 of 244 COB pages;
  - it labelled real FPR rule pages as a TOC, which silently dropped 21 rules;
  - it reads column-aligned rule text as tables.
- **MKT** writes appendix numbers two ways (`A.1.1.1` and `A1.1.1`), and its prospectus tables run on for about 26 pages.
- **AML Figure 4** is drawn with real text, so its labels came through as headings.
- **The Laws:**
  - the Collective Investment Law writes its chapter headings in title case;
  - some Schedule titles are split from their heading;
  - six deleted articles are printed on a single line.
- **Two typos in the DFSA source PDFs**, both found by the reference check: COB cites "Rule 16.6.4", which doesn't exist, and GEN cites "A2.3.2" where it means A2.3.1.

**Tests:** 138 at the end of Phase B (168 after C3), including corpus tests against the real PDFs (`-m corpus`).

---

## Phase C: Storage and embedding 🔄
**Goal:** persist everything with SQLite as the system of record and Qdrant as the search index, through an idempotent `ingest` command that only re-embeds chunks whose content changed.

### ✅ C1: Qdrant connectivity
`scripts/check_qdrant.py` reports the server version, that requests without the key get HTTP 401, that the key works, and lists collections with their point counts and embedding model.

### ✅ C2: SQLite schema (Alembic 0001)
- `src/storage/models.py` defines `documents`, `chunks` (30 columns; `id` is the Qdrant point id; `key` is unique across the corpus), `chunk_refs` (indexed on `target_key` for reverse lookups), `chunk_terms` (indexed on `term`) and `ingestion_runs` (counts, tokens, $).
- `src/storage/db.py` sets WAL, `foreign_keys=ON` and `synchronous=NORMAL`; `session_scope()` commits on success and rolls back on error.
- `alembic.ini` and `src/storage/migrations/`: the database URL comes from settings; batch mode is on; `0001_ingestion_tables.py` has been applied to `data/app.db`.
- `tests/storage/test_schema.py` runs the migration on a temporary database and checks cascading deletes, the unique key, the database settings, and that the models and migrations agree.

### ✅ C3: Load SQLite, no embedding yet
**Run it:** `uv run python -m src.ingestion.ingest --dry-run` (writes nothing), then `--no-embed`. Without either flag the command refuses until C4 is built.

| What | Where |
|---|---|
| `content_hash(chunk, fingerprint)` = sha256(fingerprint + `embedding_text()`), defined once next to `embedding_text()` | `src/ingestion/models.py` |
| `diff_chunks()`, a pure function: new / changed / unchanged / removed, plus `to_embed`, `stale_point_ids` and estimated tokens | `src/ingestion/diff.py` |
| `DocumentRepository`, `ChunkRepository` (`existing`, `upsert`, `delete`, `mark_embedded`, `by_key`, `children_of`, `parts_of`, `referenced_by`, `with_term`, `refs_of`, `terms_of`), `IngestionRunRepository` + `RunStats` | `src/storage/repositories.py` |
| `assert_migrated(engine)`: `ingest` fails at once with "run `alembic upgrade head`" if the database is behind | `src/storage/db.py` |
| `chunk_document()` returns `ChunkedDocument(chunks, report)` and still writes the JSONL and the report | `src/ingestion/chunk_corpus.py` |
| `ingest()` (takes a chunker callable, so tests use fakes) and its CLI: GLO first, one transaction per document, an `ingestion_runs` row per run (failed runs too, with the error) | `src/ingestion/ingest.py` |
| Checkpoint queries | `scripts/sql/c3_checkpoint.sql` |
| 25 new tests: diff classification, repositories, `ingest` end to end, migration check | `tests/ingestion/test_diff.py`, `tests/ingestion/test_ingest.py`, `tests/storage/test_repositories.py` |

**Decisions made while building it:**
- `existing(doc_id)` returns `{key: StoredChunk(id, content_hash, is_embedded)}`, not just the hash. A row loaded with `--no-embed` is "unchanged" but still needs a vector, and C4 must know which old points to delete.
- **Changed** means a different `content_hash` **or a different id**. A new document version changes every id (`uuid5(doc_id|version|key)`), so the whole document is re-embedded under new ids and the old row with the same key is replaced.
- `stale_point_ids` lists the points no upsert will overwrite: removed chunks, chunks whose id changed, and chunks that stopped being embeddable (e.g. a rule that grew past 512 tokens and is now a whole with parts). C4 deletes exactly these from Qdrant.
- **Unchanged rows are still rewritten** (cheap in SQLite), keeping their embedding state, because pages, cross-references and Defined Terms can change without the vector changing. A row's embedding state is cleared when its hash changes or it is no longer embeddable.
- A full run (no `--doc`) deletes documents that are no longer in the manifest. A `--doc` run never does.
- `QdrantSettings.sparse_model = "Qdrant/bm25"` was brought forward from C4, because `ingestion_runs.sparse_model` is NOT NULL.
- `EmbeddingSettings.api_key` is now optional: chunking, loading and dry runs only need the model name and dimension. `EmbeddingClient` (and `check_openai_embeddings`) refuse to start without it.

**Checkpoint results** (`uv run python -m sqlite3 data/app.db < scripts/sql/c3_checkpoint.sql`):
- The dry run matched the plan exactly: 4,060 rows, 3,397 to embed, 587,492 tokens, about $0.0117. GLO has 665 rows and 663 embeddable.
- `ingest --no-embed`: run 1 had 4,060 new rows; run 2 had 0 new, 0 changed and 4,060 unchanged. Both runs are recorded in `ingestion_runs`.
- `Regulatory Law Art. 76` is referenced by `Regulatory Law Art. 77` and `COB 4.4.2`.
- `Retail Client` is used by 88 COB chunks, 14 GEN, 3 FPR, 1 GLO and 1 in the Collective Investment Law.
- The children of `COB 3.3`, in order: 3.3.1 (the scope rule), 3.3.2, 3.3.2 Guidance, 3.3.3, 3.3.4.
- The split rule `GEN 2.3.2` (653 tokens) has parts `#p1` (463) and `#p2` (247).
- 25 deleted provisions: 20 Law articles and 5 rules (AML 2, GEN 2, COB 1).
- `chunk_refs` holds 2,495 rows, the same as the JSONL. `chunk_terms` holds 16,469 rows across 3,425 chunks.

### ⬜ C4: Qdrant collection, sparse encoder, embed GLO only
**Build:**
- `src/embeddings/sparse.py`: `SparseEncoder` wrapping fastembed `Qdrant/bm25`, **the one sparse encoder for documents and queries**. The model name is in config.
- `QdrantSettings.sparse_model = "Qdrant/bm25"`; `EmbeddingSettings.max_retries` for the OpenAI client.
- `src/storage/qdrant_store.py`:
  - `ensure_collection()` creates the collection if missing; otherwise it **checks the collection metadata** (embedding fingerprint, sparse model, schema version) and raises on a mismatch.
  - The collection has a named dense vector `dense` (1536, cosine) and a named sparse vector `bm25` (`Modifier.IDF`).
  - Payload indexes:
    - KEYWORD: regulator, doc_type, module, doc_id, chunk_type, citation, key, parent_key, term, defined_terms;
    - BOOL: binding, scope_rule.
  - Methods: `upsert(points)`, `delete(ids)`, `count()`, `scroll_hashes()` for reconcile, `get_by_citation(citation)`.
- **Payload per point:**
  - identity and filters: key, citation, chunk_type, binding, regulator, doc_type, module, doc_id, version, effective_date;
  - structure: section, parent_key, part_of, rule_number, term, scope_rule;
  - links: defined_terms, cross_refs;
  - content: header_text, text, page_start, page_end;
  - provenance: content_hash, embedding_fingerprint.
- **Ingest order per document:**
  1. embed new + changed (dense in batches of 100; sparse locally);
  2. Qdrant upsert (`wait=True`), then delete removed points;
  3. one SQLite transaction (rows, references, terms, deletions, `is_embedded`, `embedding_model`/`dimensions`/`embedded_at`);
  4. the `ingestion_runs` row.

  A crash between steps 2 and 3 heals itself: SQLite still holds the old hash, so the next run re-embeds those chunks.
- **Notes from C3:**
  - `diff.to_embed` already includes unchanged-but-never-embedded rows (from `--no-embed`).
  - `diff.stale_point_ids` is the delete list.
  - Unchanged chunks can still have changed payload fields (pages, references, terms), so either `set_payload` for them or compare a payload hash.
  - `OPENAI_API_KEY` is **not in `.env` yet**; add it before C4.
- `scripts/inspect_store.py "<key>"` shows one chunk as stored in SQLite and in Qdrant, side by side: vector sizes, the top BM25 terms, the payload.

**Checkpoint:** `ingest --doc dfsa-rulebook-glo` (660 chunks, about $0.001), then browse points in the Qdrant dashboard and inspect `GLO Retail Client`.

### ⬜ C5: Full ingest, idempotency, reconcile
1. A full `ingest` gives 3,397 points, 4,060 rows and about $0.012, recorded in `ingestion_runs`.
2. Running it **again**: 0 new, 0 changed, $0.00.
3. `--reconcile`: every row marked `is_embedded` has a point with the same `content_hash`, and there are no orphan points.

**Checkpoint:** run `check_qdrant` (shows the collection, its points and model), then compare the two `ingestion_runs` rows.

### ⬜ C6: Tests
- **Unit tests:** temporary SQLite, `QdrantClient(":memory:")` and a fake embedding client:
  - the diff classification;
  - an unchanged re-ingest embeds nothing;
  - editing one chunk re-embeds only that chunk;
  - removed chunks are deleted from both stores;
  - a fingerprint or sparse-model mismatch raises;
  - the payload indexes exist;
  - `get_by_citation`;
  - the repository lookups.
- **Corpus tests** (`-m corpus`; need Docker and keys): the counts, a clean reconcile, citation lookup of `COB 3.3.2`.

---

## Phase D: Retrieval ⬜
**Goal:** question → the best rule-level chunks, with scores and a confidence signal, plus the context the LLM needs. Everything is configurable, so evals can compare setups.

| Step | What | Files |
|---|---|---|
| D1 | `RetrievalSettings`: `mode` (`dense`/`hybrid`), `candidate_k=40`, `rerank`, `reranker_model=BAAI/bge-reranker-v2-m3`, `reranker_max_length=512`, `top_n=6`, `refusal_threshold`, `parent_mode` (`none`/`section`/`window`), `parent_max_tokens=1500`, `context_budget_tokens=6000`, `expand_refs`, filters | `src/config/settings.py` |
| D2 | **Citation router**: find `COB 3.3.2` / `Regulatory Law Art. 20` / `GLO Retail Client` in the question (reusing the `refs.py` patterns) and look it up through the payload `citation` index. Exact hits come first. This is needed because BM25 splits `3.3.2` into `3 3 2` | `src/retrieval/citations.py` |
| D3 | Hybrid candidates: embed the query (the same `EmbeddingClient`), sparse-encode it (the same `SparseEncoder`), Qdrant `query_points` with dense and BM25 prefetches fused by RRF, metadata filters (regulator/module/binding), de-duplicated | `src/retrieval/hybrid.py` |
| D4 | Cross-encoder rerank: the model is loaded once, scores (question, header+text) pairs in batches, **logs truncation using the reranker's own tokenizer**, keeps the top N; the top score is the confidence signal | `src/retrieval/rerank.py` |
| D5 | Context assembly from SQLite: part → whole rule; parents under the size policy (the full section if ≤ 1,500 tokens, otherwise a window); **scope rules always included**; Guidance attached to its rule; whole section included if ≥3 of its children were hit; optional one-hop reference expansion; **Defined Term definitions** (skipping very common terms such as "Person" and "Rule"); overall token budget; strongest source first | `src/retrieval/context.py` |
| D6 | `retrieve(question, filters) → RetrievalResult`: candidates, reranked results, confidence, context blocks, latency per stage, all ready for the trace | `src/retrieval/pipeline.py` |
| D7 | Tests with a fake Qdrant and fake models; manual check: the Market Counterparty exemption question surfaces COB 3.3.1 | `tests/retrieval/` |

## Phase E: Langfuse, self-hosted ⬜
**Goal:** versioned prompts, request traces and eval datasets, all running locally.

| Step | What | Files |
|---|---|---|
| E1 | Langfuse v3 in Docker (web, worker, Postgres, ClickHouse, Redis, MinIO) in the same compose file; localhost only; pinned images; headless setup so the project keys match `.env`; telemetry off. Needs about 4–6 GB of RAM | `infra/docker-compose.yml`, `infra/.env.example` |
| E2 | `LangfuseSettings` and `get_langfuse()` (does nothing when disabled, so tests run offline); `scripts/check_langfuse.py` | `src/observability/langfuse.py` |
| E3 | Prompts as files in git (`prompts/*.yaml`: answer, query-rewrite, refusal, judge-faithfulness, judge-relevance); `sync_prompts.py` pushes changed prompts as a `staging` version; `promote_prompt.py` moves `production` only after the eval gate passes (Phase H) | `prompts/`, `src/prompts/registry.py`, `scripts/` |
| E4 | `PromptRegistry.get(name, label)`: if Langfuse is unreachable, **falls back to the git copy**, recorded as `local-fallback` with a hash; a check at startup reports when git and production differ | `src/prompts/registry.py` |

## Phase F: Generation, grounding and refusal ⬜
| Step | What | Files |
|---|---|---|
| F1 | Azure GPT-4o wrapper; each call is a Langfuse generation linked to its prompt version, with usage and cost passed explicitly (Azure deployment names don't map to Langfuse's price table) | `src/generation/llm.py` |
| F2 | **Confidence gate**: below `refusal_threshold`, return the refusal without calling the LLM | `src/generation/answer.py` |
| F3 | The answer prompt gets numbered context blocks labelled binding Rule / non-binding Guidance / definition / Law article, and returns JSON `{answer, citations:[{context_id, citation}], confident}` | `prompts/answer.yaml` |
| F4 | **Citation validator**: every cited block must be in the context; every rule-like number in the answer must be among the cited chunks; an answer supported only by Guidance must say so; any violation means refusal, never a patched answer | `src/generation/citations.py` |
| F5 | Tests: refusal path, fabricated citation rejected, Guidance-only answer labelled | `tests/generation/` |

## Phase G: Chat history, query rewriting, API ⬜
| Step | What | Files |
|---|---|---|
| G1 | Alembic 0002: `conversations`, `messages` (rewritten query, cited chunk ids, trace_id, tokens, cost, latency) and a slim `traces` row (Langfuse trace id, candidates and reranked results with scores, prompt name/version/source, tokens and cost per stage, refused flag) | `src/storage/models.py`, migration |
| G2 | Rewriting follow-up questions using the last 6 turns, loaded **from SQLite, never from the client**; the original and rewritten queries are both logged | `src/chat/rewrite.py` |
| G3 | `ChatService`: rewrite → retrieve → answer, one Langfuse trace per request (session = conversation), writes messages and the trace row | `src/chat/service.py` |
| G4 | FastAPI: `POST /conversations`, `POST /conversations/{id}/messages`, `GET /conversations/{id}`, `GET /traces/{id}`, `GET /health`, `GET /stats/cost`. At startup: load the reranker, check embedding and sparse compatibility, check prompt drift | `src/api/` |
| G5 | `scripts/ask.py` command and API tests | `scripts/`, `tests/api/` |

## Phase H: Evaluation ⬜
| Step | What | Files |
|---|---|---|
| H1 | Golden set: 40–60 hand-written questions covering every module, Laws-term questions, multi-turn follow-ups, exemption-dependent questions (parent expansion) and **about 10 questions it can't answer** (refusal) | `data/eval/golden.yaml` |
| H2 | Sync the golden set to a Langfuse dataset | `scripts/sync_dataset.py` |
| H3 | Retrieval metrics (recall@k, MRR, nDCG on **citations**, so they survive re-ingestion), refusal precision and recall, p50/p95 latency | `src/eval/metrics.py` |
| H4 | LLM judges for faithfulness and relevance (prompts versioned in Langfuse; same-model bias noted as a limitation) | `src/eval/judges.py` |
| H5 | `eval.run --config {dense, hybrid, hybrid_rerank} [--parent-mode …] [--prompt-label staging]`, linked to Langfuse dataset runs; Alembic 0003 `eval_runs` summary | `src/eval/run.py` |
| H6 | Calibrate `refusal_threshold`; **release gate** against `eval/thresholds.yaml` and the last passing run, which is required before `promote_prompt` | `src/eval/calibrate.py`, `src/eval/gate.py` |

## Phase I: Cost, docs, polish ⬜
- Cost report: total, per query, embedding vs generation, refusal rate over time (`scripts/cost_report.py` and `/stats/cost`).
- README quickstart: compose up → migrate → ingest → ask → eval.
- `docs/architecture.md`.
- Update CLAUDE.md: Langfuse in the stack, the chunk types, SQLite vs Qdrant responsibilities, the prompt workflow.
- Portfolio write-up: the Docling findings, the evaluation comparison table, the source typos found.

## Phase J: VARA ⏸
Manual download → manifest → a VARA chunker (a different numbering scheme) → ingest into the same collection with `regulator=VARA` → filters and eval questions.

---

## Decisions log
| Decision | Why |
|---|---|
| Rule-level chunks, section parents, retrieve the small chunk and send its parent to the LLM | Precise citations plus the surrounding context (e.g. COB 3.3.1's exemption) |
| Guidance as separate, non-binding chunks | Legal weight, precision, fewer than 512 tokens |
| Hierarchy taken from rule numbers, not Docling's labels; sequence guards | Docling's labels and reading order proved unreliable |
| 512-token limit on header + text, split at sub-rules | Matches the reranker's input; only 2.6% of chunks split |
| Images ignored | Figures restate rules that are already in the text |
| SQLite is the system of record, Qdrant the index; same id in both | Auditing, joins, parent and reference lookups |
| Change detection per chunk (`content_hash`) | Re-embeds only what changed; chunker improvements cost cents |
| Exact citations through a payload index, not BM25 | BM25 splits `3.3.2` into `3 3 2` |
| Text stored in the Qdrant payload | One lookup to the reranker, a readable dashboard; `content_hash` guards against drift |
| Langfuse together with a slim SQLite record; prompts in git, synced to Langfuse; git copy as fallback | Chosen by you at the start of the project |

## Open items / known limitations
- Source typos: COB "Rule 16.6.4" and GEN "A2.3.2" (reported, not "fixed").
- The ~9 Insolvency Law references point outside the corpus.
- `Chapter 6, section 6.2, of the General (GEN)`: the "Chapter 6" part is attributed to AML (rare, harmless).
- Corpus updates are a manual re-download and re-run (by design, per CLAUDE.md).

## Commands
```
docker compose -f infra/docker-compose.yml --env-file .env up -d
uv run python -m scripts.check_qdrant
uv run alembic upgrade head
uv run python -m src.ingestion.chunk_corpus            # chunk to JSONL + reports
uv run python -m src.ingestion.ingest --dry-run         # what would change + cost; writes nothing
uv run python -m src.ingestion.ingest --no-embed        # load SQLite only (C3)
uv run python -m sqlite3 data/app.db < scripts/sql/c3_checkpoint.sql
uv run python -m src.ingestion.ingest                   # C4+
uv run pytest            # all tests;  -m "not corpus" skips the PDF-dependent ones
uv run ruff check
```

## Verification for the next steps (C4–C6)
- ✅ (C3) `ingest --dry-run` shows 4,060 new rows, 3,397 to embed, 587,492 tokens, about $0.012.
- ✅ (C3) `ingest --no-embed`: the C3 SQL checkpoint queries return the expected rows, and a re-run reports 0 new and 0 changed.
- `OPENAI_API_KEY` is set in `.env` and `check_openai_embeddings` passes.
- `ingest --doc dfsa-rulebook-glo` gives 663 points (660 definitions + GLO's 2 rules + 1 Guidance), visible in the dashboard and in `inspect_store`.
- A full `ingest`, then a re-run with $0.00, then `--reconcile` reports clean.
- `pytest` and `ruff check` are green.
