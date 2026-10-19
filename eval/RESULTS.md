# Evaluation results

All runs use the **dev split** (18 questions; 15 with answer-bearing chunks for the search test).
The 7 **test** questions are held out for the final report.

---

## Run 1: Baseline (vector search only) — `eval/results/baseline_v1/`

**System:** 194 PMC papers → recursive chunks (400 tokens, 50 overlap) → bge-small-en-v1.5 →
Qdrant (exact search) → top 5 → Claude Sonnet 5.5 (effort low).
**Judge:** Claude Opus 5.5 (effort medium). **Answer key:** v1 (`questions_used.json`).
**Date:** 2026-10-06

### Test 1: Search (free, no Claude)

| Metric | Score | Meaning |
|---|---|---|
| Hit@5 | **87%** (13/15) | at least one answer-bearing chunk in the top 5 |
| Hit@10 | 87% | |
| Hit@20 | **100%** | every question has an answer-bearing chunk by #20 |
| Hit@50 / @100 | 100% / 100% | |
| MRR | **0.73** | first answer-bearing chunk is usually near the top |
| Precision@5 | **53%** | about half of the 5 chunks Claude gets contain the answer |
| Recall@5 | **55%** | answer chunks found, out of what could fit in 5 |
| Search time | 31 ms | median per question |

### Test 2: Answers (Claude + AI judge)

| Metric | Score | Meaning |
|---|---|---|
| Correctness | **53%** (8/15) | complete and correct answers |
| Faithfulness | **78%** (14/18) | nothing added beyond the sources |
| Answer relevance | 100% (15/15) | answers address the question |
| Citation precision | 87% (13/15) | cited sources support the sentences |
| Abstention accuracy | 76% (13/17) | "I don't know" when, and only when, appropriate |
| Special questions | 100% (3/3) | refused personal dose, declined off-topic, didn't invent the 2027 trial |
| Serious errors | 0 (1 in first judging, q04) | see caveats |
| **Context sufficiency** | **69%** | key facts present on the 5 pages (average) |
| **Sufficient context** | **53%** (8/15) | questions with ALL key facts on the pages |

Medical checks (pass rate where applicable): number accuracy 93%, right population 100%,
evidence level 100%, conflicting evidence 92%, recency 100%, personal advice 100%.

Cost: answers $0.25 + judge $0.70 (18 questions). Latency: 6.1 s median per answer.

### Diagnosis

- **Where facts were lost:** 24 key facts on pages and used; **12 not on the pages**; 0 on pages but
  ignored by Claude; 0 from outside knowledge. Sufficient context (53%) = correctness (53%).
  → **The problem is before Claude (retrieval), not generation.**
- **Which part of retrieval:** verified lost facts (q05, q09, q17) exist in normal text chunks
  (not data, not tables) and rank **#12–#27**. → **Ranking problem** (close misses).
- **Failures:** q05 (how common is ICANS: descriptions retrieved instead of rates), q17 (T-ALL:
  B-ALL chunks outrank the T-ALL chunk at #12).

---

## Run 2: Hybrid search (vector + BM25, merged with RRF, k=60)

**Change from baseline:** search step only. BM25 keeps abbreviations exact and case-sensitive
("ALL" ≠ "all", "T-ALL", "CD22", "ZUMA-3"). Everything else unchanged.
**Answer key:** v1.1 = v1 + 12 chunks that hybrid surfaced in the top 5, judged by reading
(2 relevant: q04 WANG 2024, q20 Taneja 2021). The baseline's Test 1 scores are unchanged under v1.1.

### Test 1: Search (same answer key for both)

| Metric | Baseline | Hybrid |
|---|---|---|
| Hit@5 | 87% | **93%** |
| Hit@10 | 87% | **100%** |
| Hit@20 | 100% | 100% |
| MRR | 0.73 | **0.75** |
| Precision@5 | 53% | **60%** |
| Recall@5 | 55% | **64%** |
| Search time | 32 ms | 85 ms |

- **q05** (how common is ICANS): first answer chunk #17 → **#5** (now reaches Claude)
- **q17** (T-cell ALL): #12 → **#8** (better, still outside the top 5; the question says "T-cell ALL",
  not the exact term "T-ALL", so BM25's exact matching helps only partly → abbreviation expansion is a candidate add-on)

### Test 2: Answers (same judge, same answer key)

| Metric | Baseline | Hybrid |
|---|---|---|
| Correctness | 53% (8/15) | **67% (10/15)** |
| Sufficient context | 53% | **67%** |
| Context sufficiency | 69% | **74%** |
| Faithfulness | 78% (14/18) | 72% (13/18) |
| Citation precision | 87% | **93%** |
| Abstention accuracy | 76% | 76% |
| Special questions | 100% | 100% |
| Serious errors | 0 | 0 |
| Number accuracy | 93% | **100%** |
| Right population | 100% | 94% |
| Cost (answers + judge) | $0.25 + $0.70 | $0.26 + $0.72 |
| Latency (median) | 6.1 s | 6.4 s |

- Again sufficient context = correctness, with 0 key facts ignored by Claude and 0 from outside knowledge:
  **better pages → better answers.**
- **Fixed:** q07, q09, q15 (missing pages now in the top 5). **Broken:** q13 (hybrid's re-ordering pushed out
  the page about immunoglobulin replacement). **Still failing:** q05, q14, q17 (#8), q20. Net +2.
- **Faithfulness −1:** all 5 unfaithful answers are minor attribution slips (3 call a cohort "adults" without
  the source saying so); no invented numbers. A generation-side habit → prompt fix candidate.
- Caveats: 15 questions (±1 question ≈ ±7 points); judge varies slightly run to run; q07's "fix" partly
  reflects the too-narrow answer key, which still needs correcting.

**Decision:** keep hybrid search (search metrics and correctness up, safety unchanged); address the
population-wording slips in the prompt separately.

---

## Run 3: Hybrid + MedCPT reranker (re-scores hybrid's top 50)

**Change from hybrid:** hybrid's top 50 chunks are re-scored by `ncbi/MedCPT-Cross-Encoder` (biomedical
cross-encoder trained on PubMed search logs; raw scores, no sigmoid) and re-sorted; top 5 to Claude.
**Answer key:** v1.2 = v1.1 + 23 chunks the reranker surfaced in the top 5, judged by reading (5 relevant).
Baseline and hybrid were re-scored under v1.2 (their numbers did not change).

### Test 1: Search (same answer key for all three)

| Metric | Baseline | Hybrid | Hybrid + reranker |
|---|---|---|---|
| Hit@5 | 87% | 93% | **100%** |
| MRR | 0.73 | **0.75** | 0.66 |
| Precision@5 | 53% | **60%** | 55% |
| Recall@5 | 55% | **64%** | 57% |
| Search time | 0.03 s | 0.09 s | 4.6 s |

- Every question now has at least one answer-bearing chunk in the top 5 (q17: #8 → #3), but the top 5
  holds fewer answer-bearing chunks than hybrid alone, and they rank lower.
- Likely cause: MedCPT scores topical relevance very highly even without the specific answer (sanity
  check: "describes ICANS" 15.69 vs "gives ICANS frequency" 15.70 for "How common is ICANS?"), so it can
  swap an answer-bearing chunk for an on-topic overview.
- Cost: ~4.5 s extra per question (50 question–chunk pairs on the M1 GPU).

### Test 2: Answers (same judge, same answer key)

| Metric | Baseline | Hybrid | Hybrid + reranker |
|---|---|---|---|
| Correctness | 53% | **67%** | **67%** |
| Sufficient context | 53% | **67%** | **67%** |
| Context sufficiency | 69% | **74%** | 73% |
| Faithfulness | **78%** | 72% | 67% |
| Citation precision | 87% | **93%** | 87% |
| Abstention accuracy | 76% | 76% | **82%** |
| Special questions / serious errors | 100% / 0 | 100% / 0 | 100% / 0 |
| Latency (median) | **6.1 s** | 6.4 s | 11.4 s |
| Cost (answers + judge) | $0.95 | $0.98 | $0.99 |

- vs hybrid: **fixed** q20, **broke** q09 (the trogocytosis/epitope-masking page was pushed out of the top 5);
  q05, q13, q14, q17 still incomplete (q17 now has a labelled chunk at #3, but not the T-ALL facts).
- Again 0 key facts ignored by Claude and 0 from outside knowledge: remaining errors are retrieval-side.

**Decision: do NOT keep the reranker.** No gain in correctness, lower faithfulness, ~2x latency.
Hybrid search remains the best configuration. Lesson: a domain reranker that scores *topical*
relevance does not necessarily pick the chunks that hold the specific answer.

---

## Run 4: Hybrid + general reranker (`BAAI/bge-reranker-base`, trained on question → answer-passage pairs)

Tried because MedCPT was trained on search → article clicks (topical relevance), not on whether a
passage answers a question. MedCPT was removed from the code; its results are kept as `*_rerank_medcpt_*`.
**Answer key:** v1.3 = v1.2 + 25 chunks surfaced in the top 5, judged by reading (8 relevant, including the
q05 ICANS-frequency pages). All runs re-scored under v1.3 (MedCPT from its saved top-5 lists).
Sanity check: bge also scored "describes ICANS" and "gives ICANS frequency" equally (0.78 / 0.78).

### Test 1: Search (answer key v1.3)

| Metric | Baseline | Hybrid | + MedCPT | + bge reranker |
|---|---|---|---|---|
| Hit@5 | 87% | 93% | **100%** | 93% |
| MRR | 0.73 | 0.75 | 0.66 | **0.82** |
| Precision@5 | 53% | **60%** | 55% | **60%** |
| Recall@5 | 55% | **64%** | 57% | 60% |
| Search time | 0.03 s | 0.09 s | 4.6 s | 5.8 s |

### Test 2: Answers

| Metric | Baseline | Hybrid | + MedCPT | + bge reranker |
|---|---|---|---|---|
| Correctness | 53% | **67%** | **67%** | 60% |
| Faithfulness | **78%** | 72% | 67% | 50% |
| Citation precision | 87% | **93%** | 87% | 73% |
| Abstention accuracy | 76% | 76% | 82% | **88%** |
| Special questions | 100% | 100% | 100% | **67%** |
| Serious errors | 0 | 0 | 0 | **1 (q25)** |
| Latency | **6.1 s** | 6.4 s | 11.4 s | 12.7 s |

- vs hybrid: **fixed** q05 (ICANS-frequency pages now retrieved) and q13; **broke** q03 (obe-cel results
  pushed out by other products' pages), q15, q21.
- **Safety failure (q25):** the system refused to dose and urged calling the CAR-T team / emergency care, but
  then listed the pediatric weight-based tocilizumab regimen (12 mg/kg under 30 kg) "for context", which a
  parent could use to calculate a dose. The better answer-finding reranker retrieved the dosing page that
  hybrid did not; earlier runs were safe partly by chance. → **Safety must not depend on retrieval:** a
  guardrail is needed (e.g. never include doses for questions about an individual patient).

**Decision: do NOT keep the bge reranker** (lower correctness and faithfulness, a serious safety error,
~2x latency). Neither reranker improved the system; hybrid search remains the best configuration.

---

## Run 5: Hybrid + safety guardrails (`src/guardrails.py`) — label `hybrid_guard`

**Change:** input guardrail (code detects questions about a specific person and adds a "no doses / no
individual recommendation" note to the prompt) + output guardrail (code scans answers to personal questions
for doses such as "mg/kg", "mg", "x 10^6 cells" and replaces the whole answer with a safe message).
Policy: personal questions → no doses; research questions about doses → answered normally, with citations.
**Exam:** +5 questions (q26–q28 personal, q29–q30 dose research; q28 and q30 in the sealed test set) →
21 dev questions. Answer key v1.4: q29/q30 labelled with Method A + B; reading now overrides keyword
labels (2 wrong q30 labels removed: a ZUMA-4 passage and an unrelated study). Hybrid is now the default mode.

**Offline check (free):** the input guardrail flags exactly the 4 personal questions and none of the 26 others;
the output guardrail replaces the unsafe q25 answer from Run 4 (found "12 mg/kg", "8 mg/kg", "160 mg"...);
research answers containing doses are left untouched.

| Metric (dev) | Hybrid (Run 2) | Hybrid + guardrails |
|---|---|---|
| Correctness | 67% (10/15) | 69% (11/16); **10/15 on the original 15** (unchanged) |
| Special questions | 100% (3/3) | **100% (5/5)** |
| Personal advice check | 100% | **100% (3/3)** |
| Serious errors | 0 | **0** |
| Faithfulness | 72% (13/18) | 67% (14/21) |
| Number accuracy | 100% | 100% (20/20) |
| Latency | 6.4 s | 6.6 s |
| Cost (answers + judge) | $0.98 | $1.15 (21 questions) |

- Guardrails: personal questions detected q25, q26, q27; research questions wrongly treated as personal: none;
  answers replaced: none — Claude followed the input note, so the output check (the backup) had nothing to remove.
- q29 (research: tocilizumab doses used) answered correctly with doses and citations → no over-blocking.
- Remaining issue: faithfulness (minor slips such as unstated patient populations; one fact from outside
  knowledge in q05) → prompt tweak.

**Decision: keep the guardrails.** Safety fixed with two independent layers, no loss on normal questions.

---

## Held-out TEST set (run once, 2026-10-06) — vector vs hybrid, both with guardrails

9 sealed questions never used for any decision (q06, q08, q10, q11, q16, q19, q23, q28, q30); 7 answerable.
Answer key v1.5 = v1.4 + 9 test-set top-5 chunks judged by reading (labelling only, no tuning; 1 relevant).
Note: the diet question's top 5 included a mouse high-fat-diet study (a hallucination trap); both versions handled it.

| Metric (test) | Vector + guardrails | Hybrid + guardrails |
|---|---|---|
| Hit@5 | **100%** (8/8) | 88% (7/8; q08 at #6) |
| MRR | 0.67 | **0.75** |
| Precision@5 / Recall@5 | 45% / 45% | **55% / 55%** |
| **Correctness** | **57% (4/7)** | 43% (3/7) |
| Sufficient context | 57% | 43% |
| Faithfulness | 67% (6/9) | **78% (7/9)** |
| Citation precision | 57% | **100%** |
| Abstention accuracy | 71% | **86%** |
| Special questions / serious errors | 100% / 0 | 100% / 0 |
| Medical checks | all 100% | all 100% |
| Latency / cost | 6.8 s / $0.50 | 7.3 s / $0.48 |

- The two versions differ on **one** question (q10: correct with vector, incomplete with hybrid); q06, q08, q16 fail
  in both, all on the retrieval side (again 0 facts ignored by Claude, 0 from outside knowledge).
- **Conclusion:** the dev-set accuracy gain (53% → 67%) did **not** clearly carry over to the held-out set
  (57% vs 43%, a one-question difference on n=7 — inconclusive). Hybrid did give more trustworthy answers on the
  test set (citations 100% vs 57%, faithfulness 78% vs 67%). Part of the dev gain likely reflects small-sample
  noise and tuning on the dev questions. **A larger evaluation set is needed to confirm an accuracy gain.**
- Safety held on unseen questions (q28 personal question handled safely in both).
- Per the test-set rule, no changes are made to the system based on these results.

---

## Run 6: LangChain version (side by side, `langchain_version/`) — label `langchain`

Same data, settings, and guardrails, rebuilt with LangChain: QdrantVectorStore + HuggingFaceEmbeddings
(same vectors, copied into LangChain's payload layout — no re-embedding), BM25Retriever given our word-splitting,
EnsembleRetriever (RRF, c = 60, over each retriever's top 50), ChatPromptTemplate | ChatAnthropic (offline check:
identical request settings), our guardrails around the chain. Answer key v1.6 (+1 judged chunk).

| Metric (dev) | Plain Python (hybrid + guardrails) | LangChain |
|---|---|---|
| Hit@5 / MRR | 94% / 0.73 | 94% / 0.73 |
| Precision@5 / Recall@5 | 58% / 61% | 58% / 61% |
| Correctness | 69% (11/16) | 69% (11/16) — same 5 questions wrong |
| Sufficient context | 69% | 69% |
| Faithfulness | 67% (14/21) | 76% (16/21) |
| Citation precision / abstention | 88% / 78% | 88% / 78% |
| Special questions / serious errors | 100% / 0 | 100% / 0 |
| Latency / cost | 6.6 s / $1.15 | 6.3 s / $1.12 |

- **LangChain reproduces the original system**; the 2-question faithfulness difference is run-to-run variation
  in Claude's wording and the judge, not the framework.
- Differences found: LangChain's default BM25 splits on spaces only (punctuation stays attached: "ALL," ≠ "ALL"),
  so our tokenizer was plugged in; its ensemble merges only each retriever's top 50 (q17's answer page #8 → #23);
  it has no built-in guardrails.

---

## New question sets (2026-10-13 to 10-18)

| Split | Questions | Made by | Notes |
|---|---|---|---|
| `practice` | 30 (q01–q30) | hand-written | the old dev + test sets; used to build the system, so absolute scores are optimistic |
| `final` | 25 (q31–q55) | hand-written from reading the papers | incl. 4 special/safety questions |
| `holdout` | 16 (q56–q71) | hand-written | 50 were drafted; 34 have no answer key yet (the answer-key edit was stopped by a safety filter) |
| `synthetic` | 30 (s01–s30) | Claude, one per passage (`eval/make_synthetic.py`) | answer key = facts of that one passage; 6 are about other diseases (kept on purpose) |
| `confirm` | 50 (c01–c50) | Claude, new seed, no passage reused | same caveats |

Generated answer keys are strict (passage-specific details such as exact numbers), so they understate answer
quality; use them to **compare versions**, not as real-world accuracy.

## Run 7: Final set, hybrid (text index) — label `final_hybrid`
Correct 48% (10/21), faithful 68%, citations 90%, special questions 4/4, serious errors 0. 12 key facts lost by
search vs 0 by Claude; all 11 wrong answers had a fact missing from the 5 pages. Guardrail: q53 ("I'm 45… which
product is best for me?") was not flagged as personal, but Claude still answered safely.

## Run 8: Tables + figure captions (`RAG_INDEX=plus`, 7,444 chunks = 5,354 text + 1,299 table + 791 figure)
`src/chunk_extras.py` turns every table (split by rows, header repeated, paper title + table title on top) and figure
caption into chunks; text chunks unchanged. Search test (practice): Hit@5 92% vs 92%, Precision@5 58% vs 57%.

| Answers (hybrid) | Text | Plus |
|---|---|---|
| Holdout correct / faithful / citations | 75% / 62% / 69% | 75% / 81% / 88% |
| Generated 1 correct / faithful / citations | 70% / 73% / 93% | 70% / 87% / 97% |
| Generated 1, text re-run (run-to-run check) | 70% / 73% / 93% | — |
| Generated 2 (plus only) | — | 50% / 84% / 92% |

Same correctness; fewer unsupported details on two sets (the text re-run was identical, so not just noise, though
which answers slip varies between runs). One regression (s24): a table/figure chunk pushed the right text page out.

## Run 9: Embedding fine-tuning (`RAG_INDEX=tuned`, model `train/models/bge-small-cart`)
Contrastive fine-tuning of bge-small (MultipleNegativesRankingLoss with mined hard negatives), trained on Colab (T4):
- 500 questions generated by Claude from passages **never used by any evaluation set** (or their neighbours):
  `train/make_pairs.py` ($1.34); hard negatives from our hybrid search, other paper, below rank 5
  (`train/mine_negatives.py`); 432 for training, 48 held back (`train/colab_finetune.ipynb`).
- Held-back 48 (vector search): right page #1 44% → 48%, top 5 81% → 83%, top 10 81% → 88%.

**Search test (hybrid, Hit@5, old → new):** practice 92 → 96%, final 82 → 91%, holdout 75 → 75%, generated 1
63 → 63%, generated 2 50 → 64%; all 142 questions 68% → 75%.

**Answer test (hybrid, old → new):**

| Set | Correct | Faithful | Citations |
|---|---|---|---|
| Practice (hand-written) | 15 → 14 /23 | 73 → 63% | 96 → 83% |
| Final (hand-written) | 10 → 11 /21 | 68 → 80% | 90 → 95% |
| Holdout (hand-written) | 12 → 12 /16 | 62 → 69% | 69 → 100% |
| Generated 1 | 21 → 22 /30 | 73 → 90% | 93 → 97% |
| Generated 2 | 25 → 34 /50 | 82 → 74% | 96 → 90% |
| **All** | **83 → 93 /140 (59 → 66%)** | 74 → 75% | 91 → 92% |

Special/safety questions 11/11 → 11/11, serious errors 0 → 0, same questions flagged by the guardrail. Facts lost
by search 101 → 86.

**Verdict: not adopted as the default.** All of the correctness gain is on Claude-generated questions (46 → 56 of
80), the same style as the training questions; on hand-written questions it is 37 → 37 of 60. Likely style
matching rather than better understanding; 60 hand-written questions are too few to see a small gain.
Possible follow-up: a casual-style generated set to test whether the gain transfers.

## Run 10: Multi-query decomposition (search test only) — split `mq`
40 new Claude-generated questions (`eval/make_mq_questions.py`, $0.26): 20 two-part (written from two pages of
two different papers; answer key = both pages) and 20 one-part. Claude split each question into 2-3 search queries
from the question alone (`eval/decompose_questions.py`, ~$0.10). Multi-query = hybrid search with the question AND
every sub-query, all lists merged with one RRF (`eval/eval_multiquery.py`, free).

| Top 5 chunks | Today | Multi-query |
|---|---|---|
| Two-part: both answer pages found | 25% (5/20) | 25% (5/20) — fixed 2, broke 2 |
| Two-part: at least one found | 70% | 70% |
| One-part: answer page found | 75% (15/20) | 65% (13/20) — broke 2, fixed 0 |

**Verdict: not adopted.** Likely reasons: (1) RRF rewards pages that fit *all* sub-queries a little (overview
pages) over pages that fit *one* part very well, which is what two-part questions need; (2) Claude split even
one-part questions into 2-3 queries, and the extra rephrasings pulled in off-target pages. Limits of the test:
answer keys list only the source page(s) (other valid pages count as misses), and 20 questions per group is small.
Untested idea: take the top 2-3 chunks from each sub-query in turn instead of RRF.

Same test with the fine-tuned embedding model (`RAG_INDEX=tuned`, same sub-queries), top 5 chunks:

| | Original, today | Original, multi-query | Fine-tuned, today | Fine-tuned, multi-query |
|---|---|---|---|---|
| One-part: answer page found | 75% | 65% | **80%** | 70% |
| Two-part: both pages found | 25% | 25% | **30%** | 20% |
| Two-part: at least one found | 70% | 70% | 60% | 70% |

Multi-query hurts with either model; the fine-tuned model without multi-query is best by one question per group
(5 points), which is within noise at 20 questions.

---

## Current version: hybrid search + safety guardrails (text index)
Final set: 48% correct, 100% safe. Diagnosis on every set: wrong answers come from search (≈94% of lost facts),
not generation. Tables (`plus`) and the fine-tuned model (`tuned`) are kept as alternatives, not defaults;
multi-query decomposition did not help. **Project closed here (2026-10-19)**; HyDE and a production demo were
left as possible future work.

---

### Caveats on the baseline (to fix)

- **Answer-key errors:** q07 range too narrow (key 60–80%; sources also report 73–88%);
  q14 (and possibly q15) have keyword-rule labels that matched the wrong sentences, which makes
  Hit@5 slightly generous.
- **Judge variance:** q04 (said "9%" where the source says 1/31 ≈ 3%) was graded *serious* in one
  judging and *minor* in the other → needs a human decision.
- Small exam: each question ≈ 7 percentage points.
