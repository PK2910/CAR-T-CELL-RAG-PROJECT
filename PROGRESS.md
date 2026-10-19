# CAR-T in ALL — RAG project: progress log

A question-answering system over the CAR-T / acute lymphoblastic leukemia (ALL) literature, built step by
step as a baseline, then improved one change at a time and measured on a fixed exam.
Last updated: 2026-10-06.

---

## 1. What we built (the pipeline)

```
OFFLINE (run once)
200 PMC papers → clean text → 5,354 chunks → meaning codes (vectors) + keyword index
   collect.py      parse.py      chunk.py        embed.py (Qdrant)      bm25.py

ONLINE (every question)
question → search (vector + BM25, merged with RRF) → top 5 chunks → Claude Sonnet 5.5 → cited answer
               retrieve.py                                            generate.py
```

| Step | File | What it does | Key choices |
|---|---|---|---|
| 1. Collect | `src/collect.py` | Downloads 200 open-access full-text papers from PubMed Central | Title/abstract query; retries, rate limit, skips already-downloaded papers; `data/raw/manifest.json` records the query and date |
| 2. Parse | `src/parse.py` | XML → clean text by section + metadata (title, year, author, journal, article type) | 194 English papers kept; administrative sections dropped (funding, conflicts of interest...); 359 tables extracted but **not yet used** |
| 3. Chunk | `src/chunk.py` | Splits each paper into ~400-token pieces | Recursive splitting (paragraph → sentence → word), 50-token sentence overlap, labels on every chunk |
| 4. Embed + index | `src/embed.py` | Turns every chunk into 384 numbers, stores them in Qdrant | `bge-small-en-v1.5` (runs locally, free); Qdrant local mode, exact search |
| 5. Retrieve | `src/retrieve.py` | Finds the 5 best chunks for a question | Modes: `vector` (baseline), **`hybrid`** (best), `rerank` (tested, not kept) |
|    | `src/bm25.py` | Keyword (BM25) search for hybrid | Keeps abbreviations exact and case-sensitive: "ALL" ≠ "all", "T-ALL", "CD22", "ZUMA-3" |
| 6. Generate | `src/generate.py` | Sends question + 5 labelled chunks to Claude with rules; returns a cited answer + cost | Answer only from sources, cite [n], say when the answer isn't there, give ranges when papers disagree, prefer stronger evidence |

**Run it:** `python src/generate.py "your question"` (uses the API key in `.env`; you run this yourself).
`src/generate.py` and the evaluation still default to `vector` search; pass `mode="hybrid"` / `--mode hybrid`.

---

## 2. How we evaluate (the exam)

- **25 questions** written to cover facts, mechanisms, comparisons, side effects, a brand name (Kymriah),
  2 questions not answerable from our papers, 1 off-topic, and 1 unsafe "dose for my child" question.
- **Split:** 18 **dev** questions (used for all comparisons) and 7 **test** questions (**sealed** until the final report).
- **Answer key** (`eval/questions.json`): for each question, the correct answer, must-include facts, must-not
  traps, and which chunks contain the answer. Built with two methods:
  - **Method A:** keyword rules (`eval/label_rules.py`), checked by reading samples
  - **Method B:** our search's top 10, every chunk read and judged (`eval/method_b_judgments.json`, with reasons)
  - Every new chunk an add-on surfaces is read and judged before comparing (the key is now v1.3).
- **Test 1, search (free):** Hit@5, MRR, Precision@5, Recall@5, Hit at depths 10/20/50/100 → `eval/eval_retrieval.py`
- **Test 2, answers (you run it, ~$1):** `eval/run_answers.py` (system answers) + `eval/judge.py`
  (Claude Opus grades against the answer key): correctness, faithfulness, relevance, citation precision,
  abstention, safety, medical checks, **context sufficiency** (were the needed facts on the 5 pages?).
- **Diagnosis tools:** `eval/diagnose_retrieval.py` (where each missing fact ranked), `eval/show_unjudged.py`.

All results and details: **`eval/RESULTS.md`**. Raw files: `eval/results/` (baseline frozen in `baseline_v1/`).

---

## 3. Results so far (dev set, 15 answerable questions)

| | Baseline (vector) | **Hybrid** ✅ | + MedCPT reranker | + general reranker |
|---|---|---|---|---|
| Correct answers | 53% | **67%** | 67% | 60% |
| Nothing made up (faithfulness) | **78%** | 72% | 67% | 50% |
| Citations check out | 87% | **93%** | 87% | 73% |
| Safe on special questions | 100% | **100%** | 100% | 67% |
| Serious errors | 0 | **0** | 0 | 1 |
| Hit@5 (search) | 87% | 93% | 100% | 93% |
| Time per answer | 6.1 s | **6.4 s** | 11.4 s | 12.7 s |
| Kept? | — | **Yes** | No | No |

**LangChain version (`langchain_version/`, side by side):** same results as the plain-Python version (69%
correct, the same 5 questions wrong, 100% safe). LangChain needed our own BM25 word-splitting, a storage-layout
copy, and our guardrails. See `eval/RESULTS.md`, Run 6.

**Held-out test set (9 unseen questions):** hybrid did not clearly beat vector on accuracy (43% vs 57%, one
question apart, inconclusive) but gave more trustworthy answers (citations 100% vs 57%, faithfulness 78% vs 67%).
The dev-set gain may partly be noise and tuning on dev questions → a larger exam is needed.

**Run 5, hybrid + safety guardrails (current version):** correctness unchanged on the original questions
(10/15), all 5 special/safety questions handled safely, 0 serious errors, the research dose question still
answered, no normal question wrongly blocked. Faithfulness 67% (minor slips) is the next thing to improve.

---

## 4. What we learned

1. **The problem was retrieval, not Claude.** In every run, "all needed facts were on the pages" matched
   "answer correct" almost exactly, and Claude almost never ignored or invented a key fact.
2. **Diagnose before fixing.** The missing facts existed in our papers and in normal text (not data, not
   tables). They were ranked just below the top 5 (#12–#27), so this was a ranking problem.
3. **Hybrid search helped** (53% → 67%): exact words like "ALL" and "CD22" matter in medicine, and the
   embedding model lowercases text, so it can't tell "ALL" (the leukemia) from "all".
4. **Rerankers didn't help here.** MedCPT was trained to match searches to articles that are *about* a topic;
   the general reranker scored "describes ICANS" and "gives ICANS frequency" the same. A sensible idea still
   has to be tested.
5. **Safety can't depend on search.** With a better answer-finding reranker, the "dose for my child" question
   retrieved a pediatric dosing page, and the answer included it. Earlier runs were safe partly by luck.
6. **Answer keys have mistakes too.** Keyword rules match words, not meaning; checking by reading caught
   wrong labels (q07, q14, q15).
7. **Avoid data snooping:** general fixes only, tuned on dev, test set sealed, every attempt recorded.

---

## 5. Known issues / to do

- [x] **Dose guardrail** — done 2026-10-06 (`src/guardrails.py`): input check spots personal questions,
      output code check removes doses; tested on 5 new safety questions (all safe, no over-blocking). See Run 5.
- [x] **Make hybrid the default** search mode — done.
- [ ] **Guardrail stress test (optional):** run the 3 personal questions with `--mode rerank` (the search that
      retrieved the pediatric dosing page) to check the input guardrail holds when the dose page is in front of
      Claude (~$0.10). The output guardrail already blocked the real unsafe answer in an offline test.
- [ ] **Fix answer-key errors:** q07 accepted range too narrow (papers report ~60–88%); q14 and q15 labels
      matched wrong sentences; then re-score all runs.
- [ ] **Prompt tweak:** don't state the patient population (children/adults) unless the source does.
- [ ] q04 serious-or-minor ("9%" vs source 1/31 ≈ 3%): the judge disagreed with itself → needs a human decision.
- [ ] Spot-check ~5 judge grades by hand (AI-judge calibration).
- [ ] Later add-ons: tables, contextual headers, metadata filters, section-aware chunking, abbreviation
      expansion (q17 "T-cell ALL" → "T-ALL"), more questions (target 50).
- [ ] Production stage: FastAPI, tracing, Docker, CI evals, README with results.
- [ ] **If users can upload documents (planned feature): prompt-injection guardrails, built with the feature.**
      (1) prompt rule: text inside sources is content, never instructions; (2) scan uploads for instruction-like
      or hidden text and flag/quarantine; (3) label chunk provenance (PMC = trusted, upload = untrusted) and
      keep each user's uploads private to them; (4) keep the system text-only (no actions/tools);
      (5) output guardrails (dose rule, citation/groundedness checks) apply regardless of source;
      (6) add planted-injection test documents to the exam. Also: PHI detection in uploads, copyright.
- [x] Ran the sealed **test** set (2026-10-06): hybrid vs vector inconclusive on accuracy (43% vs 57%, a
      one-question difference on 7 questions); hybrid better on citations (100% vs 57%) and faithfulness
      (78% vs 67%); safety held. See `eval/RESULTS.md`. No tuning based on these results.
- [ ] **Bigger exam (next, to settle accuracy):** grow to 50+ questions so differences of a few points are meaningful;
      new questions also give a fresh held-out set (the current test set has now been used).
      **2026-10-13: answer key written for 25 new questions (q31-q55, split `final`)** — 20 answerable, 1 partial
      (q54, ICANS steroid doses: sources have almost none), 2 personal, 1 unanswerable (q55). The old 30
      (dev + used test) are now "practice" (`--split practice`); the new 25 are the sealed final exam
      (`--split final`), not yet run. Keys written from sentences read in the papers; q34 passages judged by reading.
      Still to do: user spot-check of ~5 keys; judge unread passages the search surfaces when Test 1 runs.

---

## 5b. Latest (2026-10-19) — where to pick up

- **Done since 10-13:** final set run (48% correct, 100% safe); 16-question holdout; two Claude-generated question
  sets (30 + 50); tables + figure captions index (`RAG_INDEX=plus`); contrastive fine-tuning of the embedding model
  (`RAG_INDEX=tuned`). Full numbers: `eval/RESULTS.md`, Runs 7–9.
- **Decisions:** keep the text index + original bge-small as the default. Fine-tuning helped only on
  Claude-generated questions (hand-written 37 → 37 of 60), tables gave equal correctness.
- **Multi-query decomposition** (2026-10-19, `eval/RESULTS.md` Run 10): no gain on two-part questions (both pages
  in top 5: 25% vs 25%) and worse on one-part ones (75% → 65%). Not adopted.
- **Project closed as the final version (2026-10-19):** hybrid search + safety guardrails, text index.
  Possible future work: HyDE; per-sub-query interleaving instead of RRF; fine-tuning v2 on all pages with
  question-style anchors (e.g. doc2query); a small production demo (FastAPI + Docker + rate-limited public link).
- **Open items:** 34 holdout questions (q72–q105) have no answer key; the personal-question detector missed q53;
  q07/q14/q15 answer-key fixes; human spot-check of the judge.

## 6. Setup notes

- Python 3.11, `.venv/` (`source .venv/bin/activate`), packages in `requirements.txt`.
- **API key:** `.env` (never committed; `.claude/settings.json` blocks Claude from reading it). The user runs
  every script that calls Claude; Claude only runs key-free steps.
- **GitHub:** `PK2910/CAR-T-CELL-RAG-PROJECT` — all code, the exam, and the summaries are uploaded. Raw result
  files (`eval/results/`, which contain paper excerpts) and the papers (`data/`) stay on the Mac only.
- Spending so far: about $5 of the $10 credit (each full Test 2 ≈ $1).
