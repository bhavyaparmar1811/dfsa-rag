# DFSA/VARA Regulatory Compliance RAG Assistant

## Project story

A retrieval-augmented Q&A system over UAE financial regulatory rulebooks (DFSA, and later VARA). A compliance officer, legal counsel, or fintech founder asks a plain-language question (e.g. "do we need client suitability assessments before recommending this investment?") and gets back a grounded answer with an **exact rule-level citation** (e.g. "COB 3.4.2"), not a vague paraphrase.

This is a compliance-context tool: a wrong or unsupported answer is worse than no answer. Precision of retrieval and citation is the actual product requirement, not a nice-to-have. When retrieval confidence is low, the system must say so rather than guess.

Portfolio positioning: this is project 1 of a multi-project portfolio (production RAG first, then a true tool-calling agent, then multi-agent systems). This project's job is to demonstrate production RAG practices end to end, not agentic behavior.

## Tech stack (decided, do not swap without discussion)

- **LLM (generation)**: Azure OpenAI GPT-4o (existing deployment)
- **Embeddings**: OpenAI API (not Azure), `text-embedding-3-small`
- **Vector store**: Qdrant (hybrid dense + sparse search, payload filtering)
- **Relational store**: SQLite (single file DB, see "SQLite" section below)
- **Parsing**: Docling (layout-aware PDF to structured Markdown)
- **Serving**: FastAPI
- **Dev tool**: Claude Code

Generation and embeddings use two different providers, so they need separate clients and separate credentials (`AZURE_OPENAI_*` for GPT-4o, `OPENAI_API_KEY` for embeddings). Keep them in separate config sections; never mix them up.

## Embeddings: one model for documents AND queries (hard rule)

- Queries **must** be embedded with the exact same model (and same dimensions setting) used to embed the document chunks. Mismatched models produce vectors in different spaces and retrieval silently breaks.
- The embedding model name and dimension live in **one config value**, imported by both the ingestion pipeline and the query path. Never hardcode the model name in two places.
- Store the embedding model name and dimension with every chunk (Qdrant payload + SQLite). At query time, check the configured model matches what the collection was built with; fail loudly if not.
- Changing the embedding model means a **full re-embed** of the corpus into a new collection. Never mix vectors from two models in one collection.
- Log embedding token usage and cost for both ingestion and queries.

## Data acquisition: read this before writing any ingestion code

**No web scraping. No crawler.** VARA's `robots.txt` disallows all crawlers; DFSA's Thomson Reuters-hosted rulebook also disallows automated fetching. Source documents are downloaded **manually, one time**, by the project owner, and live locally under `data/raw/`. Do not write scraper/crawler code against these sites. Do not fetch these sites programmatically at all.

`data/manifest.yaml` is the source of truth for the corpus: regulator, module code/name, version, effective date, source URL, local file path, hash. Every ingestion step reads from local files listed in the manifest, never from a live URL.

Current corpus (v1):
- **DFSA Rulebook modules (5)**: GEN (General), COB (Conduct of Business), MKT (Markets), AML (Anti-Money Laundering, Counter-Terrorist Financing and Sanctions), FPR (Fund Protocol Rules)
- **DFSA Laws (5)**: foundational DIFC legislation, used mainly to ground cross-references like "as defined in the Regulatory Law"; not expected to be the primary source of demo answers
- **VARA**: not yet downloaded; planned as a second regulator once the DFSA pipeline works

Updating the corpus (new amendments, new modules) is a **manual re-download + re-run**, not an automated job. This is a documented limitation, not a gap to engineer around.

## Ingestion pipeline

`data/raw/*.pdf` → Docling parse → structure-aware chunk → embed → store

- **Parsing**: Docling, layout-aware. Rule numbers, section headers, and cross-references must be reconstructed from PDF text (no HTML structure available).
- **Chunking**: structure-aware, one chunk per rule/sub-rule (e.g. "COB 3.4.2"), not fixed-size splitting. Parent-child retrieval: small rule-level chunk for precision, parent section passed to the LLM for context.
- **Contextual headers**: prepend the module/chapter/rule path (e.g. "Module: COB > Chapter 3 > Rule 3.4.2") to each chunk before embedding.
- **Metadata per chunk**: regulator, module, rule number, heading path, effective date, version, source document, defined terms used, cross-references to other rules (extracted via regex on known rule-number formats), embedding model + dimension.
- **Versioning**: hash-based. Store a hash per source document; only re-parse and re-embed when a document changes.

## Retrieval

Two-stage retrieval:

1. **Candidate retrieval (recall)**: hybrid search (dense + sparse/BM25-style) via Qdrant, with metadata filtering (e.g. "DFSA only", "AML only"). Fetch a wide candidate set (configurable, start with top 30-50), deduplicated by chunk id.
2. **Cross-encoder reranking (precision)**: score every (query, chunk) pair jointly with a cross-encoder and keep the top N (configurable, start with 5-8) for the LLM.

Reranker rules:
- Runs locally via `sentence-transformers` `CrossEncoder`. Default model: `BAAI/bge-reranker-v2-m3` (fall back to `BAAI/bge-reranker-base` if CPU latency is too high). Model name lives in config, not hardcoded.
- Rerank the rule-level child chunk text **including its contextual header**. The parent section is only attached after reranking, for LLM context.
- Watch the model's max input length (~512 tokens for many cross-encoders). Log when a chunk is truncated; long rules getting cut off is a real failure mode for legal text.
- Batch the scoring calls; load the model once at app startup, not per request.
- The top reranker score is the primary **confidence signal** for refusal: if it's below a configurable threshold, the system says it has no confident answer instead of generating one. Calibrate this threshold on the eval set, don't guess it.
- Log to the trace: candidate ids with hybrid scores, reranked ids with reranker scores, rerank latency.
- Reranking must be switchable in config (`rerank: on/off`, model choice) so evals can compare configurations.

For follow-up questions in a conversation, rewrite the user's message into a standalone query using recent chat history **before** embedding it (e.g. "what about for retail clients?" becomes "COB suitability requirements for retail clients"). Embed the rewritten query, and log both the original and rewritten versions.

## Grounding and refusal behavior

Every answer must cite the specific rule(s) it's based on. If retrieval confidence is low or no relevant rule is found, the system must say it doesn't have a confident answer rather than generate one. This is a first-class requirement, not error handling.

## SQLite

One SQLite file (e.g. `data/app.db`, gitignored) holds all relational data:

- **documents**: one row per manifest entry (regulator, module, version, effective date, hash, ingested_at)
- **chunks**: chunk id, document id, rule number, heading path, text, metadata JSON, embedding model; the Qdrant point id links back to this row
- **conversations**: id, created_at, updated_at, title
- **messages**: id, conversation_id, role (user/assistant), content, rewritten_query (user turns), cited chunk ids (JSON, assistant turns), trace_id, tokens, cost, latency, created_at
- **traces**: per-request trace (retrieved chunk ids + scores, prompt, response, latency, tokens, cost per step)
- **eval_runs** / **eval_results**: eval run history and per-question scores

Operational notes:
- Enable WAL mode (`PRAGMA journal_mode=WAL`) so FastAPI reads don't block on writes.
- Use migrations (Alembic or simple numbered SQL files) from the start, not ad hoc `CREATE TABLE` calls.
- Keep all DB access behind a storage layer in `src/storage/` so swapping to Postgres later is a config change, not a rewrite.

## Chat history

- Stored in the SQLite `conversations` and `messages` tables above. The client sends a `conversation_id`; the server loads history from SQLite, never trusts history sent by the client.
- Only the last N turns (configurable, start with ~6) are used for query rewriting and answer context, to bound token cost.
- Assistant messages store their cited chunk ids and trace_id, so any past answer can be audited back to the exact rules and retrieval results that produced it.
- Chat history is **not** embedded into Qdrant and is never treated as a source of regulatory truth. Answers cite rulebook chunks only, never prior conversation turns.

## Evaluation

- A hand-built golden Q&A test set covering all modules in the corpus, including a few questions that require resolving a term from a Laws document, and a few multi-turn follow-up cases to test query rewriting.
- Retrieval metrics: recall@k, MRR, nDCG. Report them for three configurations: dense only, hybrid, and hybrid + rerank, so the value of each stage is measured rather than assumed. Also report p50/p95 retrieval latency per configuration.
- Answer metrics: faithfulness/groundedness, relevance (LLM-as-judge).
- **Eval gate**: a prompt or retrieval-config change is not a valid release if it regresses eval scores below threshold.

## Observability

Every request logs: original and rewritten query, retrieved chunks with scores, prompt, response, latency, token counts, and cost, tied together in a per-request trace in SQLite. Log from day one, not bolted on later.

## Cost tracking

Log token usage and $ cost per request, split by embeddings (OpenAI) and generation (Azure OpenAI GPT-4o). Track semantic cache hit rate once caching is implemented. A simple dashboard or summary query should show total cost, cost per query, and cache hit rate over time.

## Repo structure

```
data/
  raw/{dfsa,vara}/         # downloaded PDFs, gitignored (copyright)
  manifest.yaml
  app.db                   # SQLite, gitignored
src/
  config/                  # settings, incl. the single embedding model config
  ingestion/               # Docling parsing, chunking
  embeddings/              # one embedding client used by ingestion AND queries
  storage/                 # SQLite + Qdrant clients, schema, migrations
  retrieval/
  chat/                    # conversation handling, history, query rewriting
  eval/
  api/                     # FastAPI
docs/
CLAUDE.md
```

## Explicit constraints: do not do these things

- No scraping/crawling of DFSA or VARA sites, ever.
- No committing raw PDFs or the SQLite DB to the repo.
- No fabricated citations: an answer without a real supporting chunk must refuse, not invent a rule number.
- No embedding queries with a different model or dimension than the document chunks.
- No citing chat history as a source.
- No swapping core stack choices (Qdrant, SQLite, Docling, OpenAI embeddings, Azure OpenAI GPT-4o) without flagging it first.
