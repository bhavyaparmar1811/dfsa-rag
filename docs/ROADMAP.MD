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
| C | Storage (SQLite + Qdrant) and embedding | ✅ |
| D | Retrieval: hybrid search, citation lookup, rerank, parent expansion | ✅ |
| E | Langfuse: self-hosted tracing and prompt store | ✅ |
| F | Generation, grounding and refusal | ✅ |
| G | Chat history, query rewriting, FastAPI | ✅ |
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

**Tests:** 138 at the end of Phase B, 168 after C3, 196 after C4, 203 after C5, 212 at the end of Phase C, including corpus tests against the real PDFs (`-m corpus`).

---

## Phase C: Storage and embedding ✅
**Goal:** persist everything with SQLite as the system of record and Qdrant as the search index, through an idempotent `ingest` command that only re-embeds chunks whose content changed.

### ✅ C1: Qdrant connectivity
`scripts/check_qdrant.py` reports the server version, that requests without the key get HTTP 401, that the key works, and lists collections with their point counts and embedding model.

### ✅ C2: SQLite schema (Alembic 0001)
- `src/storage/models.py` defines `documents`, `chunks` (30 columns; `id` is the Qdrant point id; `key` is unique across the corpus), `chunk_refs` (indexed on `target_key` for reverse lookups), `chunk_terms` (indexed on `term`) and `ingestion_runs` (counts, tokens, $).
- `src/storage/db.py` sets WAL, `foreign_keys=ON` and `synchronous=NORMAL`; `session_scope()` commits on success and rolls back on error.
- `alembic.ini` and `src/storage/migrations/`: the database URL comes from settings; batch mode is on; `0001_ingestion_tables.py` has been applied to `data/app.db`.
- `tests/storage/test_schema.py` runs the migration on a temporary database and checks cascading deletes, the unique key, the database settings, and that the models and migrations agree.

### ✅ C3: Load SQLite, no embedding yet
**Run it:** `uv run python -m src.ingestion.ingest --dry-run` (writes nothing), then `--no-embed`.

| What | Where |
|---|---|
| `content_hash(chunk, fingerprint)` = sha256(fingerprint + `embedding_text()`), defined once next to `embedding_text()` | `src/ingestion/models.py` |
| `diff_chunks()`, a pure function: new / changed / unchanged / removed, plus `to_embed` and estimated tokens | `src/ingestion/diff.py` |
| `DocumentRepository`, `ChunkRepository` (`existing`, `upsert`, `delete`, `mark_embedded`, `by_key`, `children_of`, `parts_of`, `referenced_by`, `with_term`, `refs_of`, `terms_of`), `IngestionRunRepository` + `RunStats` | `src/storage/repositories.py` |
| `assert_migrated(engine)`: `ingest` fails at once with "run `alembic upgrade head`" if the database is behind | `src/storage/db.py` |
| `chunk_document()` returns `ChunkedDocument(chunks, report)` and still writes the JSONL and the report | `src/ingestion/chunk_corpus.py` |
| `ingest()` (takes a chunker callable, so tests use fakes) and its CLI: GLO first, one transaction per document, an `ingestion_runs` row per run (failed runs too, with the error) | `src/ingestion/ingest.py` |
| Checkpoint queries | `scripts/sql/c3_checkpoint.sql` |
| 25 new tests: diff classification, repositories, `ingest` end to end, migration check | `tests/ingestion/test_diff.py`, `tests/ingestion/test_ingest.py`, `tests/storage/test_repositories.py` |

**Decisions made while building it:**
- `existing(doc_id)` returns `{key: StoredChunk(id, content_hash, is_embedded)}`, not just the hash. A row loaded with `--no-embed` is "unchanged" but still needs a vector, and C4 must know which old points to delete.
- **Changed** means a different `content_hash` **or a different id**. A new document version changes every id (`uuid5(doc_id|version|key)`), so the whole document is re-embedded under new ids and the old row with the same key is replaced.
- *(Superseded in C4.)* `stale_point_ids` worked out which Qdrant points to delete from SQLite's state. C4 replaced it with `plan_index()`, which reads the points actually in Qdrant.
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

### ✅ C4: Qdrant collection, sparse encoder, embed GLO only
**Run it:** `uv run python -m src.ingestion.ingest --doc dfsa-rulebook-glo`, then `uv run python -m scripts.inspect_store "GLO Retail Client"
uv run python -m scripts.retrieve "Do we need a Client Agreement with a Market Counterparty?"`.

| What | Where |
|---|---|
| `SparseEncoder`: fastembed `Qdrant/bm25`, the one sparse encoder for documents and queries (`encode_documents` gives TF weights, `encode_query` gives binary weights), plus `token_count` and `vocabulary` for inspection | `src/embeddings/sparse.py` |
| `QdrantSettings`: `sparse_k`, `sparse_b`, `sparse_avg_len=124`, `sparse_fingerprint`, `timeout_s`, `upsert_batch_size`, default URL `127.0.0.1`; `collection_name()`; `EmbeddingSettings.max_retries=5` and `timeout_s` (passed to the OpenAI SDK) | `src/config/settings.py`, `src/embeddings/client.py` |
| `QdrantStore`: `ensure_collection`, `check`, `upsert`, `overwrite_payloads`, `delete`, `delete_document`, `count`, `point_states`, `scroll_hashes`, `get_by_citation`, `retrieve`, `payload_indexes`; `build_payload()` with a `payload_hash`; `make_client()` | `src/storage/qdrant_store.py` |
| `plan_index()`: the embeddable chunks against the points in Qdrant give `to_embed`, `payload_only`, `stale_ids` and `current` | `src/ingestion/diff.py` |
| `ingest(..., index=Index(store, embedder, sparse))`; with no flag the CLI embeds; `--no-embed` writes SQLite only; `--dry-run` reads Qdrant, never creates the collection, and works without an OpenAI key | `src/ingestion/ingest.py` |
| `inspect_store "<key>"`: the SQLite row next to the Qdrant point (dense size and norm, the strongest BM25 terms, whether `content_hash` matches, the payload); a split rule also shows its parts | `scripts/inspect_store.py` |
| `check_qdrant` also shows the sparse fingerprint and schema version | `scripts/check_qdrant.py` |
| 28 new tests: the store on in-memory Qdrant, `plan_index`, `ingest` with an index (fakes), the real BM25 encoder | `tests/storage/test_qdrant_store.py`, `tests/ingestion/test_ingest_index.py`, `tests/ingestion/test_diff.py`, `tests/test_sparse.py` |

**Collection:**
- `dfsa_rules__text-embedding-3-small-1536` has a dense vector `dense` (1536, cosine) and a sparse vector `bm25` (`Modifier.IDF`).
- Its metadata records `fingerprint`, `embedding_model`, `dimensions`, `sparse_fingerprint` (`Qdrant/bm25-k1.2-b0.75-avg124`) and `schema_version` (1).
- The 12 payload indexes are the 10 KEYWORD fields plus 2 BOOL fields, as planned. All 12 were confirmed on the real server.

**Payload:** the planned fields, plus `part`, `guidance_for` and `payload_hash`. `payload_hash` is the sha256 of every other payload field.

**Ingest order per document (as built):**
1. Diff against SQLite. Plan against Qdrant from `point_states(doc_id)`, a scroll that returns only `content_hash` and `payload_hash`.
2. Embed `to_embed`: dense in batches of 100, sparse locally. Upsert with `wait=True`, rewrite the `payload_only` payloads, and delete `stale_ids`.
3. One SQLite transaction: rows, references, terms and deletions; mark `to_embed` as embedded; mark `current` and `payload_only` too, but only where they aren't marked yet.
4. Write the `ingestion_runs` row with the token count and cost reported by the API.

**Decisions made while building it:**
- **The index plan reads Qdrant, not SQLite's `is_embedded` flag.** This heals three cases:
  - a crash between the Qdrant write and the SQLite commit: the next run sees the points are current and only marks them, embedding nothing;
  - a lost or deleted collection: it is rebuilt from scratch;
  - SQLite rebuilt with `--no-embed`: existing points are marked, not re-embedded.
- Each case has a test.
- **`payload_hash`**: a change that leaves the text alone (pages, references, terms) gets the payload rewritten, with no re-embedding. This answers the C3 note.
- **BM25 `avg_len = 124`, not fastembed's default of 256.** It is the corpus's real mean: 419,845 BM25 tokens over 3,397 chunks. The document weights depend on it, so it is part of the sparse fingerprint and a change is refused by `check()`. A full run warns if the corpus drifts more than 15% from it.
- **`127.0.0.1`, not `localhost`.** On Windows, `localhost` tries IPv6 first and the container only listens on IPv4, which cost about 2.2 s per connection; an unchanged GLO re-run went from 15 s to under 1 s. The API-key-over-HTTP warning is silenced only for loopback hosts.
- Mismatches fail loudly:
  - a different embedding model raises `EmbeddingModelMismatch`;
  - a different sparse encoder, schema version or vector setup raises `CollectionMismatch`;
  - `ingest()` refuses an `Index` whose fingerprint differs from the one it was given.
- Qdrant's in-memory mode ignores payload indexes, so the unit tests check which indexes were *requested*. The real indexes are checked against the server (and will be in the C6 corpus tests).

**Checkpoint results:**
- Dry run: 663 to embed, about 37,031 tokens, about $0.0007. Real run: **663 points, 37,031 tokens (the API's count matched the tiktoken estimate exactly), $0.000741**, `ingestion_runs.id = 4`.
- Re-run: 0 embedded, $0.00.
- `inspect_store "GLO Retail Client"`:
  - dense vector: 1536 dims, norm 1.0;
  - BM25: 12 terms;
  - `content_hash` matches SQLite;
  - cross-reference: `COB 2.3.2`.
- `get_by_citation("GLO Retail Client")` finds the definition.
- The query "who counts as a professional client?" ranks `GLO Professional Client` first with dense, BM25 and RRF hybrid alike.

### ✅ C5: Full ingest, idempotency, reconcile
**Run it:** `uv run python -m src.ingestion.ingest`, run it again, then `uv run python -m src.ingestion.ingest --reconcile`.

| What | Where |
|---|---|
| `reconcile()` reports `unembedded`, `missing_points`, `hash_mismatch`, `wrong_model`, `marked_not_embeddable` and `orphan_points`, and checks the collection's configuration first; `fix()` deletes orphan points and un-marks rows with a bad point | `src/ingestion/reconcile.py` |
| `ingest --reconcile [--fix]`: exits 0 when clean and 1 when not, so it can gate scripts and CI | `src/ingestion/ingest.py` |
| `ChunkRepository.index_states()`, `clear_embedded()` | `src/storage/repositories.py` |
| 7 tests: clean after ingest, rows loaded with `--no-embed`, a missing point found, fixed and re-embedded (only that one), a stale hash, the wrong model, orphan points deleted by `--fix`, a mismatched collection refused | `tests/ingestion/test_reconcile.py` |

**Decisions:**
- `--fix` only repairs what `ingest` can't: orphan points (e.g. from a document that left SQLite) and wrong `is_embedded` flags. Re-embedding stays the job of `ingest`, which plans against Qdrant and embeds only what is missing.
- Rows loaded with `--no-embed` count as **not clean** (`unembedded`): a clean state means every embeddable chunk is searchable.

**Checkpoint results:**

| | Run 8 (full ingest) | Run 9 (re-run) |
|---|---|---|
| documents | 11 | 11 |
| new / changed / removed rows | 0 / 0 / 0 (rows came from C3) | 0 / 0 / 0 |
| unchanged rows | 4,060 | 4,060 |
| chunks embedded | 2,734 (GLO was already current from C4) | 0 |
| tokens | 550,461 (the API's count matched tiktoken exactly) | 0 |
| cost | $0.011009 | $0.00 |
| time | 57 s | 9 s |

- The whole corpus cost **$0.01175** to embed: GLO $0.000741 in C4, plus $0.011009 here.
- **3,397 points**, matching the 3,397 embeddable rows out of 4,060.
  - By module: GLO 663, GEN 644, COB 731, AML 186, MKT 461, FPR 97. The Laws: 615.
  - `documents.embedded_count` adds up to 3,397.
- `--reconcile` is **CLEAN** (exit 0), and `check_qdrant` shows 3,397 points, status green, with the embedding model and BM25 settings.
- The mean BM25 length measured on the full corpus is 123.6, against the configured 124 (ok).
- `inspect_store "GEN 2.3.2"`: the whole rule has no point; both parts have one, and both hashes match SQLite.
- `get_by_citation("COB 3.3.2")` finds the binding rule and its non-binding Guidance.

### ✅ C6: Tests
- ✅ **Unit tests** (written with C3–C5): temporary SQLite, `QdrantClient(":memory:")`, a fake embedding client and a fake BM25 encoder:
  - the diff classification and `plan_index`;
  - an unchanged re-ingest embeds nothing;
  - editing one chunk re-embeds only that chunk;
  - a metadata-only change rewrites the payload only;
  - removed chunks are deleted from both stores;
  - a fingerprint or sparse-model mismatch raises;
  - payload indexes are requested;
  - `get_by_citation`;
  - the repository lookups;
  - the self-healing cases;
  - every reconcile problem type, and `--fix`.
- ✅ **Corpus tests**: `tests/storage/test_corpus_store.py`, 9 tests against the live `data/app.db` and Qdrant.
  - **Read-only:** no `ingestion_runs` row and no point is written.
  - **Skips:** cleanly when Qdrant is unreachable, the collection doesn't exist, or SQLite is empty.
  - **What they check:**
    - the collection metadata matches the configuration;
    - all 12 payload indexes exist on the server with the right types;
    - the counts: 4,060 rows, 3,397 points, and the points per document equal to `documents.embedded_count`;
    - reconcile is clean;
    - `COB 3.3.2` returns the rule and its Guidance, with the same id, hash and text as SQLite;
    - the split rule `GEN 2.3.2` is indexed through its parts only;
    - payload filters (`module=AML`, `binding=true`) run in Qdrant;
    - BM25 alone ranks `GLO Market Counterparty` first for "Market Counterparty", with no OpenAI call;
    - a dry-run re-ingest of the whole corpus plans 0 embeddings, 0 payload rewrites and 0 stale points.

**Phase C summary:**
- SQLite holds all 4,060 chunks, with references and Defined Terms.
- Qdrant holds the 3,397 embeddable chunks: dense vectors from `text-embedding-3-small` and BM25 vectors (avg_len 124), plus 12 payload indexes.
- A full embed cost $0.01175; an unchanged re-run costs $0.00.
- The index plan reads Qdrant itself, so crashes and lost collections repair themselves.
- `--reconcile` proves the two stores agree.
- **Ready for Phase D:**
  - `EmbeddingClient.embed_query` and `SparseEncoder.encode_query` produce query vectors;
  - `QdrantStore.get_by_citation` is the exact-citation path;
  - `ChunkRepository` (`children_of`, `parts_of`, `referenced_by`, `with_term`) supplies context assembly.

---

## Phase D: Retrieval ✅
**Goal:** question → the best rule-level chunks, with scores and a confidence signal, plus the context the LLM needs. Everything is configurable, so evals can compare setups.

**Run it:** `uv run python -m scripts.retrieve "Do we need a Client Agreement with a Market Counterparty?"`. It shows every stage; `--json` prints the trace, and `--no-rerank`, `--mode`, `--reranker`, `--module` and `--binding-only` change the setup.

| Step | What | Files |
|---|---|---|
| ✅ D1 | `RetrievalSettings` (env prefix `RETRIEVAL_`): `mode` (dense/sparse/hybrid), `candidate_k=40`, `rrf_k=60`, `citation_lookup`, `max_citation_hits=8`, `rerank`, **`rerank_k=40`**, `reranker_model`, `reranker_device` (auto), `reranker_fp16`, `reranker_max_length=512`, `top_n=6`, `refusal_threshold=None` (not calibrated), `parent_mode=section`, `parent_max_tokens=1500`, `parent_window=1`, `section_hit_min=3`, `include_scope_rules`, `include_guidance`, `expand_refs`, `include_definitions`, `max_definitions=6`, `max_hit_definitions=3`, `term_max_share=0.05`, `skip_terms`, `context_budget_tokens=6000`. Filters are per request (`Filters`: regulators, modules, doc_types, chunk_types, binding_only) | `src/config/settings.py`, `src/retrieval/models.py` |
| ✅ D2 | **Citation router.** Reads `COB 3.3.2`, `cob rule 3.3.2(1)(a)`, `Rule 3.3.2 of COB`, `COB section 3.3`, a bare `Rule 2.3.2` (looked up in every module), `Article 20 of the Regulatory Law`, `regulatory law art. 76 and article 77`, and the Law Regulating Islamic Financial Business. Resolves through SQLite to the embedded chunks (a rule with its parts and Guidance; a section with its children), applies the filters, and reports `unresolved` (e.g. the source typo `COB 16.6.4`), `too_broad` (chapters, appendices) and `filtered_out` | `src/retrieval/citations.py` |
| ✅ D3 | Hybrid candidates: the same `EmbeddingClient` and `SparseEncoder` as ingestion, both searches in **one batched Qdrant request** with the same filter, and **RRF fusion done client-side**, so each candidate keeps its dense rank, BM25 rank and fused score for the trace | `src/retrieval/hybrid.py` |
| ✅ D4 | Cross-encoder rerank, scores 0–1 (sigmoid), **on the GPU in fp16 when there is one**. Loaded once; header + text are scored, exactly what was embedded; **truncation counted with the reranker's own tokenizer** and logged per chunk. Only the exact hits plus the top `rerank_k` candidates are scored | `src/retrieval/rerank.py` |
| ✅ D5 | Context assembly from SQLite: a split rule shown whole; scope rules always; Guidance ↔ rule; the whole section if ≤ 1,500 tokens, otherwise ±1 sibling, and the whole section anyway on ≥ 3 hits; Defined Terms (the question's terms first, matched case-insensitively; then the rarest terms of the strongest hits; terms used in > 5% of chunks skipped; the hit's own Law's definition preferred over the Glossary's); optional one-hop references; a token budget that drops by priority and never drops a hit. **One block = one chunk**, labelled *Rule (binding)* / *Guidance (non-binding)* / *Definition* / *Law article* | `src/retrieval/context.py` |
| ✅ D6 | `Retriever.build()` checks the collection against the configuration, loads the reranker, and wires the stages; `retrieve(question, filters) → RetrievalResult` with the citations, candidates (all scores), hits, confidence, context blocks, timings per stage, embedding tokens and $, truncations and dropped blocks; `trace()` returns plain JSON for Phase G | `src/retrieval/pipeline.py` |
| ✅ D7 | 56 tests with fakes (temporary SQLite + in-memory Qdrant + fake embedder, BM25 and scorer) plus 5 corpus tests on the live index (BM25 only, free); **manual check passed** (below) | `tests/retrieval/` |

**Decisions made while building it:**
- **Exact citation hits are kept and placed first**, even beyond `top_n`. They are still reranked, so the trace has their scores. A question naming a provision always gets that provision.
- **The citation router resolves through SQLite, not the Qdrant payload index.** SQLite knows sections, parts and deleted provisions, so it can expand "COB 3.3" to its children and tell "not in the corpus" apart from "too broad". The points are then fetched by id, and the filters are applied the same way as in search.
- **Confidence** is the best rerank score among the hits. `confident` stays `None` until Phase H calibrates `refusal_threshold`; without a reranker there is no confidence signal at all, on purpose.
- **`rerank_k=40`** (all candidates) on the GPU. The setting exists because the cross-encoder scores about linearly in the number of pairs: on a CPU-only machine, lower it to about 20 or use `bge-reranker-base`. Candidates past it stay in the trace, unscored.
- **Definitions are chosen by the question first.** Ranking by how often the hits use a term gave noise (Fund Manager, Holding Company, Undertaking), so the rarest terms of the strongest hits now fill at most 3 slots.
- **fastembed's model cache moved to `data/cache/fastembed`** (gitignored). Its default is the system temp directory, which gets cleaned, and it then fails offline.
- **`sentence-transformers` 6.1 was added**, as CLAUDE.md specifies for the reranker.
- **PyTorch comes from the CUDA 12.6 index** (`torch 2.14.1+cu126`, `torchvision 0.29.1+cu126`).
  - They are declared as direct dependencies, because `[tool.uv.sources]` only applies to direct dependencies.
  - The CUDA wheels still run on CPU elsewhere; macOS keeps the PyPI build.
  - Docling shares this PyTorch, and its parse cache is unaffected (its key depends on the Docling version only).

**Manual check (D7): "Do we need a Client Agreement with a Market Counterparty?"**
- Hybrid search alone ranks the exemption **COB 3.3.1** 12th (dense 35th, BM25 4th), outside the top 6.
- With `bge-reranker-v2-m3` it is **1st, with a score of 0.860**; the next is 0.623.
- The context is the whole of COB 3.3 (3.3.1–3.3.4 with the Guidance), the Market Counterparty classification rule COB 2.3.9 with its Guidance, and the definitions of *Client Agreement* and *Market Counterparty*: 17 blocks, 5,984 tokens.

**Reranker: `BAAI/bge-reranker-v2-m3` on the GPU (decided).** The machine has an RTX 3050 Laptop GPU (4 GB, compute capability 8.6, driver 610.78). Measured on the same 40 hybrid candidates per question:

| Reranker | Parameters | CPU, fp32 | **GPU, fp16** | Peak GPU memory | Q1: COB 3.3.1 | Q2: COB 2.3.3 | Q3: COB 3.4.2 | Q4: AML 8.1.1 | Scores |
|---|---|---|---|---|---|---|---|---|---|
| none (hybrid RRF) | – | – | – | – | 12 | 2 | 3 | 1 | rank-based, no confidence |
| `cross-encoder/ms-marco-MiniLM-L6-v2` | 22M | 3.4–4.5 s | 0.12–0.5 s | 0.12 GiB | 4 | 2 | 1 | 1 | saturate at 0.99+, a weak refusal signal |
| `BAAI/bge-reranker-base` (CLAUDE.md fallback) | 278M | 23–28 s | 0.4–0.5 s | 0.67 GiB | **1** | 3 | 2 | 1 | 0.21 on Q1: low, but separated |
| **`BAAI/bge-reranker-v2-m3` (CLAUDE.md default)** | 568M | 80–97 s | **1.1–1.6 s** | 1.77 GiB | **1** | 3 | 1 | 1 | the best separation (0.86 vs 0.62) |

- fp16 on the GPU gives the **same rankings** as fp32 on the CPU, with scores equal to 2–3 decimals (COB 3.3.1: 0.860 vs 0.86).
- **End to end**, the Market Counterparty question takes **2.6 s**: rerank of all 40 candidates 1.8 s, query embedding 0.8 s, search 42 ms, context 22 ms. On CPU it took 57 s.
- 4 questions are not an evaluation. Phase H compares the configurations, including all three rerankers, on the golden set.

## Phase E: Langfuse, self-hosted ✅
**Goal:** versioned prompts, request traces and eval datasets, all running locally.

| Step | What | Files |
|---|---|---|
| ✅ E1 | Langfuse **v4** in Docker (web, worker, Postgres, ClickHouse, Redis, MinIO) in the same compose file; only `langfuse-web` published (`127.0.0.1:3000`), the data stores on the compose network only; pinned images; headless setup so the project keys match `.env`; signup disabled; telemetry off. Secrets generated into `.env` by `init_langfuse_env` (never printed, never overwritten). Walkthrough and operations: [`docs/langfuse-setup.md`](langfuse-setup.md) | `infra/docker-compose.yml`, `.env.example`, `scripts/init_langfuse_env.py`, `tests/test_init_langfuse_env.py` |
| ✅ E2 | `LangfuseSettings` and `get_langfuse()` (a no-op client when disabled or without keys; the tests force it off, so they run offline); `check_langfuse`: health, keys, a test trace **read back** through the worker, prompt drift | `src/config/settings.py`, `src/observability/langfuse.py`, `scripts/check_langfuse.py`, `tests/test_observability.py` |
| ✅ E3 | Prompts as files in git (`prompts/*.yaml`: answer, query-rewrite, refusal, judge-faithfulness, judge-relevance), validated (declared variables = used ones) and content-hashed; `sync_prompts` pushes changed prompts as a `staging` version; `promote_prompt` moves `production` only to a version equal to the **committed** git file, and needs `--skip-gate REASON` until Phase H's gate exists | `prompts/`, `src/prompts/files.py`, `src/prompts/sync.py`, `scripts/sync_prompts.py`, `scripts/promote_prompt.py` |
| ✅ E4 | `PromptRegistry.get(name, label)`: if Langfuse can't serve it (unreachable, no such label, variables that don't match the code), **falls back to the git copy**, recorded as `local-fallback` with the reason and the content hash; `drift()` / `report_drift()` say per prompt whether production is what's in git | `src/prompts/registry.py`, `tests/prompts/` |

**Run it:** `uv run python -m scripts.init_langfuse_env --email …` once, then `docker compose -f infra/docker-compose.yml --env-file .env up -d`; the UI is at http://127.0.0.1:3000 (use `127.0.0.1`, not `localhost`).

**Pinned images:**
- `langfuse/langfuse` and `langfuse/langfuse-worker` `4.57.0` (always bumped together)
- `clickhouse/clickhouse-server:26.4.5.143` (v4 needs ≥ 25.12; 26.4 is Langfuse's recommendation)
- `postgres:17.11`
- `redis:7.4.11`
- `cgr.dev/chainguard/minio@sha256:dc455c…` (pinned by digest; Chainguard's free tier has only `latest`)

**Decisions made while building it (E1):**
- **v4, not v3** (see the decisions log).
- **Fewer ports than upstream.** Upstream binds web and MinIO on every network interface and publishes the data stores on localhost; here only web is published, on `127.0.0.1`. The cost: SDK media uploads (images) can't reach MinIO. We trace text only.
- **A real Redis healthcheck.** Upstream's `redis-cli ping` exits 0 even on `NOAUTH`; ours authenticates (`REDISCLI_AUTH`) and greps for `PONG`.
- **Secrets stay out of the terminal.** `init_langfuse_env` never prints a value and never overwrites one: changing `SALT` or `ENCRYPTION_KEY` after data exists breaks Langfuse, and database passwords only apply when a volume is first created.
- **`NEXTAUTH_URL=http://127.0.0.1:3000`**, following the Phase C IPv6 finding.

**Checkpoint results (2026-10-13):**
- Docker Desktop 4.94.0 / Engine 29.8.2; the WSL2 VM has 7.6 GiB.
- First start:
  - all Prisma (Postgres) migrations and 51 ClickHouse migrations applied, with no permission errors;
  - the v3 → v4 background migrations found 0 partitions (nothing to migrate);
  - Langfuse detected ClickHouse 26.4.5.143 and needed neither of its workarounds for known ClickHouse bugs.
- `/api/public/health` → `{"status":"OK","version":"4.57.0"}`; `/api/public/projects` without a key → `401`.
- Headless setup:
  - org and project "DFSA RAG";
  - the UI's public key equals `LANGFUSE_PUBLIC_KEY` in `.env`;
  - sign-up disabled.
- Ports: only `127.0.0.1:3000` (Langfuse) and `127.0.0.1:6333-6334` (Qdrant) are published.
- **Memory, idle: ≈ 2.6 GiB in total.**

  | Container | Memory |
  |---|---|
  | web | 972 MiB |
  | worker | 798 MiB |
  | ClickHouse | 604 MiB |
  | Qdrant | 120 MiB |
  | Postgres | 93 MiB |
  | MinIO | 66 MiB |
  | Redis | 7 MiB |

  The two Node.js apps use about twice what was estimated.
- **Data survives `down` and `up`:** still one project and one key, and `check_qdrant` still shows 3,397 points, status green.
- One harmless start-up race: the worker logs `events_full table does not exist. Retrying in 10s` while web is still migrating, then validates fine.
- `pytest -m "not corpus"` (235) and `ruff check` are green.

**Decisions made while building it (E2–E4):**
- **Strict prompt variables.** Each file declares its variables, and they must equal the `{{…}}` it uses. `compile()` raises on a missing or an unknown variable (Langfuse's own compile leaves gaps silently), and fills in one pass, so a `{{x}}` inside a value is never substituted.
- **One content hash** (type + messages + config, not the description), computed the same way for a git file and for a Langfuse version. It drives sync, promote and drift, and goes into every trace.
- **Production is always in git.** Promote refuses a version whose content isn't the committed file. A rollback is a revert in git, a sync and a promote.
- **The fallback can serve an unpromoted prompt.** When git is ahead of production (synced to staging, not promoted) and Langfuse is down, the git copy is the newer text. The trace records `local-fallback` and the hash, and the drift check shows the state.
- **No retries on a prompt fetch, and a 60 s pause after a connection failure:** a refused connection costs ~2 s on Windows, and a cached prompt already survives a failed refresh.
- **The refusal is a prompt too** (a text prompt with a `reason`), so its wording is versioned like the others, though it never goes to an LLM.

**Checkpoint results (2026-10-14):** SDK `langfuse` 4.18.0 against server 4.57.0.
- `check_langfuse`: health OK, keys accepted, the test trace read back after 1.1 s.
- `sync_prompts`: the five prompts created as v1 with `staging`; a second run reports all five `unchanged`, so the hash survives the round trip through the server.
- Registry: `get("answer", "staging")` → `langfuse` v1; `get("answer")` → `local-fallback` (`HTTP 404`, nothing promoted yet); drift: all five `awaiting-promotion`.
- With `langfuse-web` stopped: the first `get` falls back in 2.05 s, the next ones in 0.00 s; the drift check reports all five `unreachable` in 2.0 s.
- `promote_prompt` refuses while the prompt file has uncommitted changes.
- **Finding:** v4 doesn't serve its traces from the legacy `api.trace.get` (404); the v2 observations API (`api.observations.get_many(trace_id=…)`) does. Phase G's `GET /traces/{id}` must use it.
- **Finding:** `update_prompt(new_labels=…)` adds labels rather than replacing them.
- `pytest` (260 offline + 36 corpus) and `ruff check` are green.

**Not done in Phase E, on purpose:** no prompt had a `production` version yet; they were promoted at the end of Phase F.

## Phase F: Generation, grounding and refusal ✅
**Flow:** `gate (no context / below threshold: refuse, no LLM call) → answer prompt over numbered blocks (JSON) → citation check → answer, or the refusal prompt with a reason`

| Step | What | Files |
|---|---|---|
| ✅ F1 | Azure GPT-4o wrapper: prompt `config` mapped through an allow-list (a typo fails loudly); each call is a Langfuse generation linked to its prompt version, with usage and cost passed explicitly (Azure deployment names don't map to Langfuse's price table); API errors are raised, never turned into refusals | `src/generation/llm.py` |
| ✅ F2 | **Confidence gate**: no context, or `confident is False`, refuses without calling the LLM; `None` (threshold not calibrated yet) runs the LLM and is recorded as gate `uncalibrated`. `AnswerResult.trace()` holds everything Phase G stores | `src/generation/answer.py` |
| ✅ F3 | Numbered context blocks `[n] CITATION \| Label (binding/non-binding)` + heading + text; Guidance shown as `COB 3.3.2 Guidance` (a rule and its Guidance share a citation). JSON `{answer, citations:[{context_id, citation}], confident}`; answer prompt v2 | `src/generation/context.py`, `prompts/answer.yaml` |
| ✅ F4 | **Citation validator**: at least one citation; every listed block exists and has that citation; every provision the answer names is a listed block **or a reference quoted from one**; model not confident, cut off, filtered or malformed → refusal, never a patched answer. All-Guidance answers get a fixed notice from the code | `src/generation/citations.py` |
| ✅ F5 | 41 tests: refusal paths (0 LLM calls at the gate), fabricated and unlisted citations, relayed references, Guidance-only label, LLM wrapper, end to end on the mini corpus | `tests/generation/` |
| ✅ | `scripts/ask.py`: one question, one Langfuse trace (retriever → embedding → answer chain → generation → citation check) | `scripts/ask.py` |

**Decisions made while building it:**
- **Guidance-only answers are labelled by the code** (fixed notice before the text), not by checking the answer's wording: deterministic, and never a refusal over phrasing.
- **Relayed references are allowed.** "What does COB 3.3.2 require?" was first refused: the draft repeated "App2" and "Rule 3.3.4", which COB 3.3.2's own text names but which weren't context blocks. A mention now passes if a listed block's text refers to it (resolved in that block's module, so "GEN 3.3.4" doesn't pass for COB's "Rule 3.3.4"). No number ever comes from the model itself; the trace lists such mentions as `relayed`.
- **An outage is an error, not a refusal.** "No confident answer" would misdescribe it; Phase G's API returns 503.
- **The refusal names unresolved citations** from the question ("COB 99.9.9 is not in the documents I can search").
- **JSON mode, not structured outputs**, for now. The deployment runs `gpt-4o-2024-11-20`, which supports strict `json_schema`; worth switching if malformed output shows up in evals.

**Checkpoint results (2026-10-14):** `answer` staging v2, deployment `gpt-4o-poc` (`gpt-4o-2024-11-20`).

| Question | Result | Cost | Answer step |
|---|---|---|---|
| Do we need a Client Agreement with a Market Counterparty? | Answered "No": COB 3.3.1 (scope), COB 3.3.2, two GLO definitions | $0.0194 | 4.5 s |
| What does COB 3.3.2 require? | Answered from COB 3.3.2, App2 and Rule 3.3.4 relayed (refused before the relay rule) | $0.0155 | 4.7 s |
| What is a Retail Client? | Answered: GLO Retail Client, COB 2.3.2 | $0.0128 | 3.8 s |
| What does Article 20 of the Regulatory Law say? | Answered: Regulatory Law Art. 20 | $0.0140 | 5.2 s |
| Can a firm give the key information in a separate form …? (`--chunk-type guidance`) | Answered with the Guidance-only notice: COB 3.3.2 Guidance | $0.0191 | 3.9 s |
| Should a Client Agreement be in writing? (Guidance only) | Refused, `model_not_confident`: the Guidance only partly covers it | $0.0186 | 4.0 s |
| What is the VAT rate in the UAE? | Refused, `model_not_confident` (retrieval confidence 0.003) | $0.0318 | 4.2 s |
| What does COB 99.9.9 require? | Refused, names COB 99.9.9 as not in the corpus | $0.0185 | 3.8 s |
| Client Agreement question, `RETRIEVAL_REFUSAL_THRESHOLD=0.99` | Refused at the gate, **no LLM call** | $0.0000 | 30 ms |

- Langfuse: the generation is linked to `answer` v2, with usage (7,008 in / 159 out) and cost ($0.0191) as sent.
- An answer costs about $0.013–0.020, almost all of it GPT-4o input (4.5–7k tokens of context).
- **Findings for Phase H:** the VAT question's top rerank score was 0.003 against 0.78–0.999 for answerable ones, so a calibrated threshold would refuse it before the LLM (it cost the most, $0.032: its 6 unrelated hits were 11.8k tokens, and hits are never cut by the budget). Retrieval takes 2–5 s, the answer 4–5 s.
- `pytest` (301 offline + 36 corpus) and `ruff check` are green.

**Prompts promoted (2026-10-14)**, after review of the checkpoint answers, with `--skip-gate "Phase F manual checkpoint"`: `answer` v2, `refusal`, `query-rewrite`, `judge-faithfulness` and `judge-relevance` v1 are `production`; the drift check reports all five `in-sync`. From Phase H on, a promotion needs a passing eval run instead.

## Phase G: Chat history, query rewriting, API ✅
**One turn:** `history (SQLite, last 6 messages) → rewrite (follow-ups only) → retrieve → answer → user + assistant message + trace row, in one transaction`

| Step | What | Files |
|---|---|---|
| ✅ G1 | Alembic 0002: `conversations`, `messages` (rewritten query, status and refusal code, cited chunks, trace_id, tokens, cost, latency) and the slim `traces` row (id = Langfuse trace id; retrieval and answer traces as JSON; tokens and $ per stage: embedding, rewrite, answer; latency per stage; `status` ok/error). Repositories for each, plus `cost_stats()` | `src/storage/models.py`, `src/storage/migrations/versions/0002_chat_tables.py`, `src/storage/repositories.py` |
| ✅ G2 | Follow-ups rewritten with the `query-rewrite` prompt over the last 6 messages, loaded **from SQLite, never from the client**; no LLM call on a first message; an empty or cut-off rewrite falls back to the original; both are stored | `src/chat/rewrite.py` |
| ✅ G3 | `ChatService.send`: one Langfuse trace per turn (`chat-turn` → `rewrite-query` → `retrieve`/`embed-query` → `answer`/`generate-answer`/`validate-citations`), session = conversation; the trace id is created up front, so SQLite and Langfuse share it even with tracing off; a failed turn stores an error trace and no messages | `src/chat/service.py` |
| ✅ G4 | FastAPI with `X-API-Key` on every endpoint but `/health`; startup checks the migration, loads the reranker, checks the index, logs prompt drift; 404 / 422 (a `history` field is rejected) / 503 for an upstream outage, with the trace id | `src/api/app.py`, `src/api/schemas.py` |
| ✅ G5 | `scripts/ask.py` goes through `ChatService` (`--conversation` to follow up); 30 tests: repositories via the service, rewrite, service, API | `scripts/ask.py`, `tests/chat/`, `tests/api/` |

**Decisions made while building it:**
- **The answer gets the rewritten question**, with the history beside it for context; retrieval and the citation router use the rewritten one too.
- **A turn is one message**: `CHAT_HISTORY_TURNS=6` is the last 6 messages, each cut to 1,500 characters.
- **`GET /traces/{id}` reads SQLite**, not Langfuse (v4's legacy trace endpoint answers 404), and adds the Langfuse URL.
- **Outage vs bug:** Azure OpenAI, OpenAI and Qdrant errors are 503 (retry); anything else is 500. Both store an error trace.
- **Retrieval runs one at a time** (a lock around the GPU reranker); FastAPI runs the sync endpoints in its thread pool.
- **The reranker is pinned and loaded offline:** `bge-reranker-v2-m3` at commit `953dc6f6…` (`bge-reranker-base` at `2cfc18c9…`), from the local Hugging Face cache only (`RETRIEVAL_RERANKER_LOCAL_ONLY=true`). Startup makes no Hub request (that was the "unauthenticated requests" warning), works offline, and eval results can't shift because the model changed upstream. The revision is in every retrieval trace and in `/health`; an uncached model fails with a hint to download it once.
- **Langfuse is optional at runtime:** `/health` reports it, but only the database, Qdrant and the reranker make the app `degraded`.

**Checkpoint results (2026-10-14):** `data/app.db` migrated 0001 → 0002 (backup taken first); the 4,060 chunks are untouched.
- `ask.py`, two turns:
  - turn 1, "Do we need a Client Agreement with a Market Counterparty?": answered "No" with COB 3.3.1 and 3.3.2, $0.0193, 8.1 s;
  - turn 2, "what about retail clients?": rewritten to "Is a Client Agreement required when dealing with Retail Clients under DFSA regulation?", answered from COB 3.3.2 and COB A2.1.2, $0.0170, 8.3 s (rewrite 2.1 s, retrieve 2.3 s, answer 3.8 s).
- Langfuse: both traces under one session; the rewrite is its own generation (276 in / 15 out, $0.0008), linked to `query-rewrite` v1.
- API (uvicorn, 127.0.0.1:8000):
  - `/health` 200 without a key (Qdrant 3,397 points, reranker on cuda, Langfuse 4.57.0); `POST /conversations` without a key → 401;
  - "What is a Retail Client?" → GLO Retail Client + COB 2.3.2; "and what is the alternative classification?" → rewritten, answered from COB 2.3.1 and 2.3.2;
  - a `history` field → 422; `GET /traces/{id}` returns 40 candidates with scores, the prompts used and the Langfuse URL;
  - `/stats/cost`: 4 requests, $0.0695 in total, $0.0174 per request (answer $0.0679, rewrite $0.0017, embeddings $0.000001).
- `pytest` (322 offline + 36 corpus) and `ruff check` are green.

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
| The index plan is diffed against Qdrant's own points, not SQLite's `is_embedded` flag | Crashes, lost collections and `--no-embed` reloads heal without re-embedding what exists |
| `payload_hash` in each point | Metadata-only changes rewrite the payload instead of re-embedding |
| BM25 `avg_len` measured on the corpus (124) and part of the sparse fingerprint | fastembed's default of 256 under-penalises long chunks; a change means re-encoding |
| RRF fusion client-side, both searches in one batched request | The trace keeps the dense and BM25 ranks of every candidate; the same latency |
| Exact citation hits pinned first, resolved through SQLite | A named provision is always in the answer's context; sections and "not in the corpus" are handled |
| One context block = one chunk, labelled binding / non-binding | Every sentence of an answer can cite exactly one provision (Phase F's validator) |
| The reranker runs on the GPU in fp16 (`bge-reranker-v2-m3`), all 40 candidates | ~65× faster than CPU with identical rankings; `rerank_k` stays configurable for CPU-only machines |
| Langfuse together with a slim SQLite record; prompts in git, synced to Langfuse; git copy as fallback | Chosen by you at the start of the project |
| Langfuse v4, not v3 | v4 GA since 2026-07-29; v3 gets security patches only until 2027-01; same components; a fresh install has nothing to migrate |

## Open items / known limitations
- Source typos: COB "Rule 16.6.4" and GEN "A2.3.2" (reported, not "fixed").
- The ~9 Insolvency Law references point outside the corpus.
- `Chapter 6, section 6.2, of the General (GEN)`: the "Chapter 6" part is attributed to AML (rare, harmless).
- Corpus updates are a manual re-download and re-run (by design, per CLAUDE.md).
- ClickHouse writes its own system log tables constantly (503 MB in its first 30 minutes, migrations included). Measure the volume's growth; if it matters, mount a `config.d` file that disables or shortens those logs.
- `langfuse-web` and `langfuse-worker` have no compose healthcheck yet.
- The prompt fallback can serve git text that hasn't been promoted (see Phase E decisions); a local cache of the last production version would close that gap.

## Commands
```
uv run python -m scripts.init_langfuse_env --email you@example.com  # Langfuse secrets into .env (once)
docker compose -f infra/docker-compose.yml --env-file .env up -d     # start Qdrant + Langfuse
docker compose -f infra/docker-compose.yml --env-file .env up -d qdrant   # start Qdrant only
docker compose -f infra/docker-compose.yml --env-file .env stop      # stop all, keep data
curl.exe -s http://127.0.0.1:3000/api/public/health    # Langfuse; UI at http://127.0.0.1:3000
uv run python -m scripts.check_langfuse                # keys, a trace round trip, prompt drift
uv run python -m scripts.sync_prompts [--dry-run] [name ...]   # git prompts -> Langfuse staging
uv run python -m scripts.promote_prompt answer --skip-gate "<reason>"   # staging -> production
uv run python -m scripts.ask "Do we need a Client Agreement with a Market Counterparty?" [--label staging] [--json]
uv run python -m scripts.ask "what about retail clients?" --conversation <id>   # a follow-up
uv run uvicorn src.api.app:app --host 127.0.0.1 --port 8000      # the API (needs API_KEY in .env)
uv run python -m scripts.check_qdrant
uv run alembic upgrade head
uv run python -m src.ingestion.chunk_corpus            # chunk to JSONL + reports
uv run python -m src.ingestion.ingest --dry-run         # what would change + cost; writes nothing
uv run python -m src.ingestion.ingest --no-embed        # load SQLite only (C3)
uv run python -m sqlite3 data/app.db < scripts/sql/c3_checkpoint.sql
uv run python -m src.ingestion.ingest                   # embed + index (only what changed)
uv run python -m src.ingestion.ingest --doc dfsa-rulebook-glo
uv run python -m src.ingestion.ingest --reconcile       # SQLite vs Qdrant; exit 1 if not clean
uv run python -m src.ingestion.ingest --reconcile --fix # delete orphan points, un-mark bad rows
uv run python -m scripts.inspect_store "GLO Retail Client"
uv run pytest            # all tests;  -m "not corpus" skips the ones needing PDFs, Docker and the ingested corpus
uv run ruff check
```

## Verification for Phase C (all done)
- ✅ (C3) `ingest --dry-run` shows 4,060 new rows, 3,397 to embed, 587,492 tokens, about $0.012.
- ✅ (C3) `ingest --no-embed`: the C3 SQL checkpoint queries return the expected rows, and a re-run reports 0 new and 0 changed.
- ✅ (C4) `OPENAI_API_KEY` is set in `.env` and `check_openai_embeddings` passes.
- ✅ (C4) `ingest --doc dfsa-rulebook-glo` gives 663 points (660 definitions + GLO's 2 rules + 1 Guidance), visible in `inspect_store`; a re-run costs $0.00.
- ✅ (C5) A full `ingest` ($0.011009, 3,397 points), then a re-run at $0.00, then `--reconcile` reports CLEAN.
- ✅ (C6) `uv run pytest -m corpus` passes against the live Qdrant (31 corpus tests, 9 of them on the store).
- ✅ `pytest` (212) and `ruff check` are green.

## Verification for Phase D
- ✅ `pytest`: 268 tests (36 corpus) and `ruff check` are green.
- ✅ The manual check: the Market Counterparty question surfaces COB 3.3.1 first (with a reranker).
- ✅ The reranker: `bge-reranker-v2-m3` on the GPU (fp16); 2.6 s end to end.
