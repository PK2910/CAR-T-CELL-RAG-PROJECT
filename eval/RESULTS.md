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
  Hit@5 slightly generous. **Fixed 2026-10-26** (details in PROGRESS.md section 5). Free search re-test, practice
  set: Hit@5 unchanged (vector 92%, hybrid 92%); MRR vector 0.70 → 0.68, hybrid 0.74 → 0.71; P@5 and R@5 down
  about 1 point. Answer grades in earlier runs still use the old q07 key.
- **Judge variance:** q04 (said "9%" where the source says 1/31 ≈ 3%) was graded *serious* in one
  judging and *minor* in the other → needs a human decision.
- Small exam: each question ≈ 7 percentage points.

---

## Search scores by slice (2026-10-26, free)

`eval/slice_report.py`: the search test split into groups, 142 questions with answer pages (practice, final,
holdout, synthetic, confirm). "Hand-written" = written in conversation while reading the papers, with every answer
page searched for and marked; "Claude-generated" = written by a script from one page, with only that page marked.
Hybrid search, Hit@5 / MRR:

| Question type | Hand-written (all answer pages marked) | Claude-generated (one page marked) |
|---|---|---|
| Mechanism | 82% / 0.78 (n=17) | 70% / 0.55 (n=30) |
| Fact / number | 83% / 0.59 (n=23) | 20% / 0.17 (n=15) |
| Comparison | 78% / 0.52 (n=9) | 60% / 0.44 (n=20) |
| Side effect | 100% / 0.84 (n=8) | 50% / 0.28 (n=6) |

Vector search is lower in almost every cell (overall Hit@5 61% vs hybrid 68%).

- **Hand-written number questions are found (83%) but ranked lower** than mechanism questions (MRR 0.59 vs 0.78): the
  "about the topic outranks the specific number" pattern seen in q05.
- **Claude-generated number questions look much worse (20%), but mostly because of the answer key**: each lists only
  the page it was written from. Reading 6 misses: in c04, c15 and c32 the top pages also answer (another trial page with
  the CD22 dose response, the same review's cytopenia pages, the same paper's abstract); c30 and c31 are real misses
  (one study's specific number). So the overall "fact questions are weakest" result is mostly a mix-of-questions effect.
- Pooling hand-written and generated questions hides this, so slices are reported by writer as well as by type.

---

## Would 10 pages instead of 5 help? (2026-10-26, free check)

For every key fact the judge marked as missing from the 5 pages (current version: final, holdout and practice runs),
pages #6–10 of the same hybrid search were read for that fact. The top 5 today were checked to be identical to the
pages used in the runs (25 of 25 on the final set).

| Set | Missing facts | Found on pages #6–10 | Answers that would become fully correct (best case) |
|---|---|---|---|
| Final | 13 | 4 (q37, q43, q44, q50) | 10 → 14 of 21 |
| Holdout | 6 | 2 (both q64 facts) | 12 → 13 of 16 |
| Practice | 15 | 7 (q05 ×2, q13, q16, q17 ×3) | 15 → 17 of 23 |
| **Total** | **34** | **13 (38%)** | **37 → 44 of 60** |

- Best case only: it assumes Claude uses each found fact and is not misled by the extra pages, which are mostly "about
  the topic" pages (e.g. q45: Ph+ case reports; q14: chemotherapy survival). 21 facts are still further down or not in
  the papers.
- These three sets were used to make this check, so a confirming answer test should use other questions (e.g. the
  generated `confirm` set) before switching the default.

### Answer test with 10 pages, final set (2026-10-26, user-run, $0.57 answers + $1.42 judge)

Run `final_hybrid_k10` vs `final_hybrid` (same 25 questions, same prompt, same judge):

| | 5 pages | 10 pages |
|---|---|---|
| Correct (21 answerable) | 10 | **14** |
| Faithful | 13 | 11 |
| Citations check out | 19 | 18 |
| Key facts missing from the pages | 13 | 8 |
| Special questions handled correctly | 4 / 4 | 4 / 4 |
| Harm: none / minor / serious | 12 / 13 / 0 | 10 / 15 / 0 |
| Median time per answer | 6.9 s | 7.6 s |

- Newly correct: q37, q43, q44, q50, exactly the four predicted by the free check. No answer went from right to wrong.
- Faithfulness changed in both directions (5 better, 7 worse; net −2). Every new unsupported claim is a minor
  attribution slip: journal names not in the sources (q43), "adults" for an unlabeled cohort (q32), mixing up which
  source said what (q38, q50, q51), an over-broad summary (q46). Same habit as Run 4; candidate prompt fix.
- q53 (personal, missed by the input detector) still declined to choose a product; q55 still said the sources do not
  answer. Both "minor" grades are misattributions inside otherwise correct answers.
- Caveat: the final set was used to choose 10 pages, so this confirms the prediction; 21 questions (±1 ≈ ±5 points).

**Decision:** keep 10 pages.

---

## Section-based chunking and contextual chunk headers (2026-10-27, free search test)

- `python src/chunk.py --by-section`: each section (abstract included) is split on its own, 5,532 chunks, none crosses
  a section boundary. Each chunk also gets a header "Paper: <title> (<author>, <year>) / Section: <name>".
- Two indexes: `section` (section-based, header not indexed) and `section_hdr` (header indexed by both the vector and
  the keyword search). Claude is shown the same chunk text either way.
- The answer key was translated to the new chunk ids (`eval/section_labels.py`): a new chunk is relevant if it holds a
  sentence that matches the question's answer rules, or, for questions without rules (reading-judged labels and
  generated questions), at least 30% of the original chunk's sentences. Translating onto the original chunks gives back
  the original key (100% kept, 50 extra neighbouring chunks over 182 questions).

Hybrid search, 142 questions (`eval/slice_report.py`), Hit@5 / Hit@10 / MRR:

| | Original chunks | Section-based | Section-based + header |
|---|---|---|---|
| All | 68% / 80% / 0.52 | 68% / 76% / 0.50 | **74% / 85% / 0.55** |
| Hand-written (all answer pages marked) | 84% / 94% / 0.66 | 85% / 95% / 0.65 | 85% / 90% / 0.67 |
| Claude-generated (one page, approximate translation) | 55% / 69% / 0.41 | 55% / 61% / 0.38 | **65% / 80% / 0.46** |
| Asks for a number (21) | 62% / 76% / 0.38 | 71% / 76% / 0.46 | 76% / 86% / 0.54 |

- **Section-based alone: no change.**
- **Headers help questions about a specific study** (generated questions: Hit@10 61% → 80% vs section-based alone),
  most likely because the paper title in the header matches the study the question names.
- **On the hand-written questions, headers are mixed:** top-10 lost q08 (#6 → #16), q17 (#8 → #26), q67 (#5 → #20),
  q71 (#6 → #15) and gained q33 (#14 → #5), q69 (#11 → #6). q17 is one of the answers the switch to 10 pages fixed.
- Not adopted yet: the trusted (hand-written) set does not improve; the generated-set gain rests on approximate labels.

---

## Parent-child chunking (2026-10-27, free search test)

`python src/chunk_children.py`: every original chunk (the parent, which Claude reads) is cut into children of 1-3
sentences, up to 100 tokens, never crossing a paragraph (25,657 children, median 69 tokens, ~5 per parent). The
sentence splitter also cuts after citation numbers ("...severe).10,24 Risk..."). The search fuses the vector and BM25
rankings of the children, and each parent ranks by its best child (`to_parents` in `src/retrieve.py`). The parents are
the original chunks, so the answer key is used as is (no translation). Three versions: no header (`child`),
"Section: <name>" (`child_sec`), "Paper: <title> (<author>, <year>) / Section: <name>" (`child_hdr`).

Hybrid search, Hit@5 / Hit@10 / MRR:

| | All (142) | Hand-written (62) | Claude-generated (80) |
|---|---|---|---|
| Original chunks | 68% / 80% / 0.52 | 84% / 94% / 0.66 | 55% / 69% / 0.41 |
| Section-based + header | 74% / 85% / 0.55 | 85% / 90% / 0.67 | 65% / 80% / 0.46 |
| Parent-child, no header | 68% / 77% / 0.49 | 87% / 94% / 0.63 | 52% / 64% / 0.38 |
| Parent-child, section header | 68% / 77% / 0.51 | 85% / 92% / 0.65 | 55% / 65% / 0.39 |
| **Parent-child, full header** | 73% / 84% / **0.57** | **87% / 94% / 0.69** | 61% / 76% / **0.48** |

- Parent-child + full header has the best MRR on both kinds of questions. On hand-written questions it adds 2 questions
  to the top 5 and keeps the top 10 (94%), which is what Claude reads now.
- Rank of the first correct chunk, hand-written, original → parent-child + full header: 31 same, 17 better, 14 worse.
  Large drops: q17 (#8 → #46) and q67 (#5 → #92); q33 (#14 → #4) and q69 (#11 → #3) entered the top 10.
- **It does not fix the motivating example.** The chunk with "ICANS ... in 40% of children" ranks #13 (original), #29
  (children, no header) and #45 (children + header) for q05. Single sentences that are only about ICANS, without a
  number, now compete directly with the answer sentence. Smaller units sharpen "about the topic" matching; they do not
  teach the search that the question wants a frequency.
- Headers: the paper title boosts every child of a paper whose title shares the question's words. For q17 (T-cell ALL vs
  B-ALL) the top 3 are all from a review titled "Challenges and advances in CAR-T cell therapy for B-ALL".
- Not adopted (default stays "text"): no top-10 gain on the trusted set, and the remaining weakness is the
  "about vs answers" gap, which needs a step that reads the question and the chunk together.

---

## LLM relevance check, Step A (2026-10-27, user-run, $0.42)

A Corrective-RAG-style check (not the full CRAG method: no web search, no sentence-level refinement).
`eval/check_chunks.py`: Claude Haiku 5.5 (effort low, structured output) reads each question with its top 30 hybrid
chunks and marks every chunk yes / partly / no; questions with no "yes" but some "partly" also get chunks #31-60 checked.
`eval/checker_report.py` (free) compares the "yes" chunks (search order, at most 10) with today's top 10.
151 questions, $0.0028 each.

| | Today: correct chunk in Claude's 10 | With checker | Share of Claude's chunks that are correct | Chunks Claude reads (median) |
|---|---|---|---|---|
| Hand-written (62) | 58 (94%) | 57 (92%) | 47% → **70%** | 10 → 8 |
| Claude-generated (80) | 55 (69%) | **67 (84%)** | 7% → **34%** | 10 → 4 |

- q05 example: the "ICANS ... in 40% of children" chunk (#13) is marked yes; topic-only ICANS chunks (#1, #2) are
  marked partly ("describes ICANS ... but gives no actual rate"). q02: the pediatric ZUMA-4 chunk and the mantle cell
  lymphoma chunk are marked no.
- Hand-written questions lost: q65 (the checker's "yes" chunks state the mechanism but are not in the answer key: a
  label gap), q40 (exhaustion evidence from CLL/NHL marked partly as "not ALL": the checker is too strict for mechanism
  questions; it still finds two "yes" chunks at #13 and #19), q08 (12-month EFS of a dual-target product marked partly;
  a long-term remission chunk at #15 marked yes), q23 (diet: "no specific diet recommendation", so unsure). Gained:
  q33, q54, q69.
- 94% of answer-key chunks in the top 30 are marked yes or partly, but only 55% yes: the hand-written keys include broad
  topic chunks, and the checker is stricter. Generated questions: 96% of their source chunks are marked yes.
- Outcomes: answerable questions 140 good, 2 unsure, 0 nothing (the guardrail would block no real question). Special
  questions: off-topic q24 → nothing; unanswerable q22 and q55 → unsure (answer with a warning); personal questions →
  good (they are handled by the existing personal-question guardrail).
- **Decision: go to Step B** (answer test on the final set) without changing the checker prompt, so the test measures
  the checker as tested here. Known weakness to revisit: evidence from other diseases for mechanism questions.

### LLM relevance check, Step B: answer test, final set (2026-10-27, user-run, $0.50 answers + $1.26 judge)

`python eval/run_answers.py --mode checked --split final --label final_checked`: Claude answers from the checker's
"yes" chunks (saved in Step A; `src/relevance_check.py`, `generate.answer_checked`). Compared with `final_hybrid_k10`
(the current version: top 10 chunks), same prompt and judge:

| | Top 10 chunks (current) | With the relevance check |
|---|---|---|
| Correct (21 answerable) | 14 | **17** |
| Faithful | 11 | 11 |
| Citations check out | 18 | 15 |
| Key facts missing from Claude's chunks | 8 | **4** |
| Special questions handled correctly | 4 / 4 | 4 / 4 |
| Serious errors | 0 | 0 |
| Chunks Claude reads (median) | 10 | 9 |
| Answer cost (25 questions) | $0.57 | $0.50 (+ checker ≈ $0.07) |

- Wrong → right: q33, q36, q41, q45, q47. Right → wrong: q31, q40.
- q40 (T-cell exhaustion): the checker kept only 2 chunks and marked the naive/memory T-cell chunks "partly"
  (evidence from other diseases), so that key point was missing. The "too strict for mechanism questions" weakness
  from Step A, now confirmed on an answer.
- q31 (what MRD-negative means, and how often): a two-part question; the checker marked the definition chunks
  "partly" because they give no rate, and kept the rate chunks. Neither run's chunks had the numeric threshold; the judge
  accepted the definition in one run and not the other, so this flip is partly judge variance.
- The 5 new citation/faithfulness slips are all minor attribution slips (study type, population label), the same habit
  as before.
- Caveats: 21 questions (±1 ≈ ±5 points); 5 gained vs 2 lost. The checker prompt was not tuned on this set. The checker
  adds one Haiku call per question (not timed here: its verdicts were saved in Step A).
- **Next:** fix the checker prompt for multi-part and mechanism questions, re-check, and confirm on other questions
  before making it the default.
