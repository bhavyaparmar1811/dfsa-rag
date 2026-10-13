-- C3 checkpoint: what SQLite holds after `ingest --no-embed`.
--   uv run python -m sqlite3 data/app.db < scripts/sql/c3_checkpoint.sql

-- 1. Which chunks reference Regulatory Law Art. 76?
SELECT c.key, c.chunk_type
FROM chunk_refs r JOIN chunks c ON c.id = r.chunk_id
WHERE r.target_key = 'Regulatory Law Art. 76' AND c.part_of IS NULL
ORDER BY c.document_id, c.order_idx;

-- 2. How many chunks use the Defined Term "Retail Client", per module?
SELECT d.module_code, count(*) AS chunks
FROM chunk_terms t JOIN chunks c ON c.id = t.chunk_id JOIN documents d ON d.id = c.document_id
WHERE t.term = 'Retail Client' AND c.chunk_type != 'section'
GROUP BY d.module_code ORDER BY chunks DESC;

-- 3. The children of COB 3.3, in document order (the scope rule comes first).
SELECT key, chunk_type, binding, scope_rule, token_count
FROM chunks WHERE parent_key = 'COB 3.3' AND part_of IS NULL
ORDER BY order_idx;

-- 4. A split rule: the whole rule (kept for LLM context) and its embeddable parts.
SELECT key, part, token_count, substr(text, 1, 60) AS starts_with
FROM chunks WHERE key = 'GEN 2.3.2' OR part_of = 'GEN 2.3.2'
ORDER BY coalesce(part, 0);

-- 5. Deleted provisions, per document and type.
SELECT document_id, chunk_type, count(*) AS deleted
FROM chunks WHERE deleted GROUP BY 1, 2 ORDER BY 1;

-- 6. Per document: rows, embeddable chunks, embedded so far.
SELECT d.id, d.version, d.chunk_count, d.embedded_count,
       sum(c.chunk_type != 'section' AND NOT c.deleted AND c.parts = '[]') AS embeddable
FROM documents d JOIN chunks c ON c.document_id = d.id
GROUP BY d.id ORDER BY d.id;

-- 7. The ingestion runs.
SELECT id, status, embedding_fingerprint, sparse_model, docs_processed,
       chunks_new, chunks_changed, chunks_unchanged, chunks_removed, embed_tokens,
       embed_cost_usd, finished_at
FROM ingestion_runs ORDER BY id;
