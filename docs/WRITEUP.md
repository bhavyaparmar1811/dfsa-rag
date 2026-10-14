# Building a RAG system that refuses: a case study

*A compliance Q&A system over the DFSA Rulebook and the DIFC Laws. Every number here was measured, and the build log behind it is in [ROADMAP.MD](ROADMAP.MD).*

---

## 1. The problem

Compliance officers, in-house lawyers and fintech founders in the DIFC ask questions like *"Do we need a Client Agreement with a Market Counterparty?"* The answer lives in about 4,000 provisions spread across six Rulebook modules, the Glossary and five Laws, in PDFs.

A chatbot that answers fluently is easy to build. One that is safe to rely on has to meet three requirements:

1. **Cite the exact provision** (COB 3.3.1, not "the conduct rules").
2. **Never invent one.** A plausible but nonexistent rule number is the worst possible output.
3. **Refuse when the evidence isn't there.** In this domain, a wrong answer is worse than no answer.

Those requirements shaped every decision. Retrieval precision isn't a nice-to-have here: it's the product.

**The stack:** LLM GPT-4o for generation, OpenAI `text-embedding-3-small` for embeddings, Qdrant for hybrid search, SQLite as the system of record, Docling for PDF parsing, FastAPI, and a self-hosted Langfuse for tracing and prompt versions.

---

## 2. Ingestion: the PDFs fought back

There is no HTML version to scrape, and the regulators' `robots.txt` forbids crawling anyway. So the corpus is 11 PDFs, downloaded once by hand, each listed with its SHA-256 hash.

Docling is a good layout-aware parser, but legal PDFs found its limits:
- **Reading order was wrong on 89 of 244 COB pages.** Side-by-side columns were interleaved.
- **It labelled real FPR rule pages as a table of contents,** which silently dropped 21 rules.
- **It moved rule numbers into a separate `marker` field,** away from their text.
- **It read column-aligned rule text as tables.**

**The fix:** don't trust the parser's structure. The chunker rebuilds the hierarchy from the **rule numbers themselves**, with sequence guards (3.3.2 must follow 3.3.1). Every module was brought to zero validation issues.

The output is one chunk per rule, Guidance note, definition or Law article: 4,060 chunks, 3,397 of them embedded. A rule that doesn't fit the reranker's 512-token window is split only at sub-rule boundaries (2.6% of chunks).

**A side effect: the reference checker found two typos in the regulator's own PDFs.** COB cites a non-existent "Rule 16.6.4", and GEN cites "A2.3.2" where it means A2.3.1. They are reported as findings, not "fixed".

**Ingestion is idempotent.** Each chunk carries a content hash, so an unchanged re-run costs $0.00, and a chunker improvement re-embeds only the chunks it changed. The whole corpus cost **$0.012** to embed.

---

## 3. Retrieval: measuring each stage instead of assuming it

**The pipeline:**
- an exact-citation router ("COB 3.3.2" resolved through SQLite, since BM25 splits it into `3 3 2`);
- hybrid search (dense + BM25, fused with reciprocal rank fusion);
- a cross-encoder reranker over the top 40 candidates;
- context assembly that adds scope rules, Guidance, parent sections and Defined Terms.

On the 59-conversation golden set, each stage was measured on its own:

| Configuration | Recall@1 | Recall@6 | MRR | Context recall |
|---|---|---|---|---|
| Dense only | 0.630 | 0.852 | 0.740 | 0.963 |
| BM25 only | 0.435 | 0.778 | 0.578 | 0.880 |
| Hybrid (RRF) | 0.565 | 0.852 | 0.692 | 0.907 |
| **Hybrid + bge-reranker-v2-m3** | **0.796** | **0.944** | **0.877** | **0.982** |

**Three results I didn't expect:**
- **Hybrid fusion alone ranked *worse* than dense at the top** (MRR 0.69 vs 0.74). Its value is a wider candidate pool, which only pays off once a reranker re-scores it.
- **The biggest reranker was worth its latency.** It beat `bge-reranker-base` and MiniLM by 0.085 and 0.15 MRR, for about 1 s on a laptop GPU (fp16 gives the same rankings as fp32 on CPU, about 65× faster).
- **Some answers depend on provisions that never rank.** "Do we need a suitability assessment for an execution-only trade?" turns on COB 3.4.1, a scope rule ("this section does not apply to…") that is never retrieved by search. It reaches the model only because scope rules of a hit's section are always added to the context.

---

## 4. Grounding: a validator, and refusing on purpose

The model answers from numbered context blocks, in JSON, and then code checks the answer:
- every cited block exists and carries that citation;
- every provision named in the answer text is either a cited block or a reference quoted inside one.

Any failure turns the answer into a refusal. A failed answer is never patched.

**The refusal threshold was calibrated, not guessed.** The top reranker score is the confidence signal: 10 unanswerable golden questions score 0.002–0.374, and answerable ones mostly 0.62 and up. The automatic calibration rule (≤ 5% false refusals, threshold placed mid-gap) suggested 0.525. That would also have refused a correct exemption answer (0.430) without catching anything more. The chosen value is **0.40**: all 10 unanswerable questions are refused before any LLM call, and 1 of 54 answerable ones is a false refusal.

**"What is the VAT rate in the UAE?" scores 0.003** and costs a single query embedding.

---

## 5. Evaluation that gates releases

The golden set has 59 conversations and 64 turns. It covers every module, Laws definitions, exemptions, multi-turn follow-ups and unanswerable questions. Gold labels are provision ids, so they survive re-ingestion.

**What a run measures:**
- retrieval metrics per configuration;
- refusal behaviour;
- citation precision and recall;
- two LLM judges (faithfulness, relevance).

**The judges caught what the deterministic checks can't:**
- **A reversed rule.** An answer cited AML 6.1.1 correctly but reversed it, putting CDD *before* the risk assessment. The citation validator passes it, since the citation is real; the faithfulness judge flagged it as contradicted.
- **The wrong regime.** An answer about inside-information disclosure used MKT 6.5 (listed *funds*) for a question about listed *companies*.
- **A judge that couldn't discriminate.** The first relevance judge gave 4/5 to every single answer. Its prompt's example reply was `{"score": 4}`, which anchored it. With a placeholder example it now scores the wrong-regime answer 2 and the rest 5.

**Gating.** A prompt can only move to production through `promote_prompt`. The version must be the committed git file, and an end-to-end run of exactly that content must pass the floors in `eval/thresholds.yaml` with at most 3% regression. Run #14, the current baseline, scored faithfulness 0.995, relevance 4.94, citation recall 0.981 and refusal recall 10/10.

---

## 6. The answer cache: the data said no

I planned a semantic cache: if a new question's embedding is close enough to a cached question's, serve the stored answer. LLM gateways and libraries offer this off the shelf.

**The danger, measured.** In regulatory text, one word can flip an answer. Before trusting the cache, I built a test set from the golden questions:
- **32 paraphrases** (same answer);
- **32 hard negatives** (nearly the same words, different answer: another client type, entity, threshold, regime or module).

| Threshold | Similarity only: wrong answers served | Paraphrases served |
|---|---|---|
| 0.80 (LiteLLM's documented example) | **17 / 32** | 100% |
| 0.90 (RedisVL's default distance of 0.1) | **6 / 32** | 75% |
| 0.95 | **4 / 32** | 22% |

**The fix I tried: deterministic guards.** A hit had to name the same provisions, the same Defined Terms and the same numbers as the cached question. This stops "Market Counterparty" vs "Retail Client" and "30 investors" vs "80 investors". It stopped 17 of the 32 negatives.

**Why it still failed.** The guards can't see an ordinary word that decides the answer. "Minimum margin for a CFD on a **major** currency pair" vs "… a **minor** currency pair" have different answers, and their cosine similarity is **0.978**. Meanwhile real paraphrases ("execution-only trade" vs "execution-only transaction") score as low as 0.83. A single embedding captures the topic of a question, not its answer.

**The decision.** Any threshold that serves no wrong answer serves no paraphrase, so the cache matches **exact repeats only**. Even that is useful:
- **Normalised repeats are free.** Case, spacing and punctuation are ignored, so a repeat is served in about 5 ms at $0.
- **Follow-ups benefit too.** Their rewritten question often matches a cached one word for word. In the live check, "Do we need a Client Agreement with one?" (after "What is a Market Counterparty?") was served for the cost of the rewrite alone, in 1.5 s instead of about 5 s.

**Safety by design.** Each entry is namespaced by a hash of everything that shapes an answer: the corpus content hashes, the retrieval config, the answer prompt's hash, the model and the filters. A re-ingest, a prompt promotion or a config change makes old answers unreachable at once. I checked this live by changing one retrieval setting. Each hit also re-checks that the cited chunks still have the content they were answered from.

**What it would take to go further.** A semantic layer would need something that reads both questions word by word: an LLM or cross-encoder equivalence check, gated at 0 false hits on the same pairs. The eval is in the code (`src/eval/cache_eval.py`), so that change would have to prove itself.

**The broader lesson:** off-the-shelf semantic caches report hit rates; they don't report what they served wrongly. In this domain, the second number is the one that matters.

---

## 7. Cost

| | Cost |
|---|---|
| Embedding the whole corpus (3,397 chunks, 587k tokens) | $0.012 |
| An answered question | $0.013–0.02 (almost all GPT-4o input: 4.5–7k tokens of context) |
| A gate refusal | < $0.000001 (one query embedding) |
| An exact cache hit | $0 |
| An end-to-end eval run (64 turns, answers + two judges) | ~$1.70 (judges are half of it) |
| A retrieval-only eval run (all configs) | < $0.01 |
| The cache eval (64 pairs) | $0.00003 |

`scripts/cost_report.py` splits all spend (production, evals, ingestion) by provider, with refusal and cache-hit rates per day. **Evals dominate the total**, which is the right shape for a system whose correctness matters more than its traffic.

---

## 8. Limitations

- **The judges share a model with the generator.** GPT-4o judges GPT-4o, so same-model bias is possible.
- **The eval data is small, and LLM-drafted.** The golden set (64 turns) and the cache pairs (64) were drafted by an LLM from the corpus and reviewed by the owner. The refusal threshold is calibrated on the same set it is evaluated on, with margins of about 0.03.
- **Some answer issues remain:**
  - the AML 6.1.1 inversion recurs;
  - the listed-company rule (MKT 4.2.1) is outranked by the listed-fund rule (MKT 6.5).
- **Follow-up rewrites vary between runs** even at temperature 0. This is the main source of run-to-run noise in multi-turn evals.
- **The corpus is a snapshot.** Updates are a manual re-download and re-run, by design.

---

## 9. What project 2 builds on

This project is deliberately not an agent: one question, one retrieval, one grounded answer.

**What carries forward to a tool-calling agent:**
- the citation router and the retriever, as tools;
- the refusal signal;
- the trace and cost model;
- the eval harness and its release gate.

The lesson that matters most here, measuring what can go wrong before trusting a component, applies even more there.
