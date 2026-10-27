# CAR-T Literature Q&A (RAG)

A retrieval-augmented question-answering system over the published literature on **CAR-T cell therapy for
acute lymphoblastic leukemia (ALL)**. It answers questions from 194 open-access papers and cites the
source of every fact.

The project focuses on **measurement**: every change was tested on fixed question sets and kept only if it helped.
Seven improvements were tried; two were kept.

> **Research and education prototype — not medical advice.** It summarizes published studies and is not
> designed or validated to guide the care of any individual patient.

---

## How it works

```mermaid
flowchart LR
    subgraph Offline["Built once"]
        A[PubMed Central<br/>200 papers] --> B[Parse<br/>194 papers]
        B --> C[Chunk<br/>5,354 pieces]
        C --> D[(Vector index<br/>Qdrant · bge-small)]
        C --> E[(Keyword index<br/>BM25)]
    end
    subgraph Online["Every question"]
        Q[Question] --> G1[Input guardrail<br/>personal question?]
        G1 --> V[Vector search]
        G1 --> K[BM25 search]
        V --> R[Merge<br/>reciprocal rank fusion]
        K --> R
        R --> T[Top 10 pieces]
        T --> L[Claude Sonnet 5.5<br/>answers only from sources]
        L --> G2[Output guardrail<br/>no doses for personal questions]
        G2 --> OUT[Answer + PubMed links]
    end
    D -.-> V
    E -.-> K
```

| Step | File | What it does |
|---|---|---|
| Collect | `src/collect.py` | Downloads open-access papers from PubMed Central (NCBI E-utilities) |
| Parse | `src/parse.py` | Converts XML to clean text by section, with title, year, author, article type, tables and figure captions |
| Chunk | `src/chunk.py` | Recursive splitting into ~400-token pieces with sentence overlap |
| Embed | `src/embed.py` | `BAAI/bge-small-en-v1.5` vectors stored in Qdrant |
| Keyword index | `src/bm25.py` | BM25 with tokenization that keeps medical abbreviations exact ("ALL" ≠ "all", "CD22", "ZUMA-3") |
| Search | `src/retrieve.py` | Hybrid search: vector + BM25 rankings merged with reciprocal rank fusion |
| Safety | `src/guardrails.py` | Detects questions about a specific person; a code check removes doses from those answers |
| Answer | `src/generate.py` | Claude Sonnet 5.5 answers only from the 10 retrieved pieces, citing each fact |
| Index versions | `src/index_config.py` | Switch between the text index, text + tables/figures, and the fine-tuned model (`RAG_INDEX`) |
| LangChain version | `langchain_version/` | The same system rebuilt with LangChain, for comparison |
| Fine-tuning | `train/` | Contrastive fine-tuning of the embedding model (experiment) |
| Evaluation | `eval/` | Question sets, answer keys, search and answer tests, LLM judge |

---

## Evaluation

**Question sets** (each with an answer key: the correct answer, facts it must include, mistakes to avoid,
and which passages contain the answer):

| Set | Questions | Written by | Used for |
|---|---|---|---|
| Practice | 30 | hand-written | building the system (hybrid search, guardrails) |
| Final | 25 | hand-written from reading the papers | a sealed test, incl. safety questions |
| Holdout | 16 | hand-written | comparing later versions |
| Generated | 30 + 50 + 40 | Claude, from single passages (and passage pairs) | comparing versions on fresh questions |

- Every set covers facts, mechanisms, comparisons and side effects; the hand-written sets also include
  unanswerable, off-topic and personal medical-advice questions.
- **Search test (free, local):** Hit@k, MRR, precision@5, recall@5.
- **Answer test:** Claude Opus grades each answer against the answer key for correctness, faithfulness, citation
  precision, abstention, medical checks (numbers, patient population, evidence level), harm, and
  **context sufficiency** — whether the needed facts were in the retrieved pages at all, which separates
  retrieval failures from generation failures.
- Evaluation scripts save after every answer and grade, and resume after an interruption.

### Results

| Version (practice questions) | Correct | Faithful | Safe on special questions | Time per answer | Decision |
|---|---|---|---|---|---|
| Vector search only | 53% | 78% | 100% | 6.1 s | baseline |
| Hybrid search (vector + BM25) | 67% | 72% | 100% | 6.4 s | kept |
| + MedCPT reranker | 67% | 67% | 100% | 11.4 s | dropped |
| + BGE reranker | 60% | 50% | 67% | 12.7 s | dropped |
| Hybrid + safety guardrails | 67% | 67% | 100% | 6.6 s | **current** |
| LangChain version | 69%* | 76% | 100% | 6.3 s | same as current |

\*16 answerable questions after two dose questions were added. "Correct" requires every key fact in the answer key.

**Sealed final set (25 new hand-written questions), current version:** 48% correct, 100% safe on the special
questions, 0 serious errors. The drop from the practice set is expected: the system was tuned on the practice
questions, so 48% is the more honest estimate for unseen questions.

### Findings

1. **Errors come from retrieval, not generation.** Across every test, about 94% of the key facts missing from
   answers were never on the retrieved pages; when the facts were there, Claude used them.
2. **Most misses are close or partial**: the right page ranks just below the top 5, or a two-part question gets
   pages for only one part.
3. **Hybrid search helped**: exact terms like drug, trial and gene names matter, and the embedding model lowercases
   text ("ALL" the leukemia becomes "all").
4. **Rerankers did not help.** Both scored "describes the side effect" and "gives how often it happens" almost
   equally, and one surfaced a pediatric dosing page that leaked into an answer to a parent's question.
5. **Safety must not depend on retrieval.** The guardrails are code, so they hold whichever pages are found.
6. **The framework did not change the results.** The LangChain version reproduced the plain-Python results.
7. **Scores split by question type and by how questions were written** (`eval/slice_report.py`): on questions with
   every answer page marked, the search finds number answers (83%) but ranks them lower than mechanism answers.
   Script-generated questions mark only one answer page, so their low scores partly reflect the answer key.

### Change after the final results: 10 pages instead of 5

The results above used 5 pages. A free check read pages #6–10 for each key fact the judge found missing: 13 of 34
were there, which could fix up to 7 of 60 answers (37 → 44 correct) if Claude uses them. Claude now gets 10 pages.
**This is a best-case estimate, not yet confirmed with the answer test**: the extra pages also add pages that are
about the topic without answering the question. Details: [`eval/RESULTS.md`](eval/RESULTS.md).

---

## Experiments that were tried and not adopted

Each was tested against the current version on the same questions. Full numbers: [`eval/RESULTS.md`](eval/RESULTS.md).

**Tables and figure captions as extra chunks** (`src/chunk_extras.py`; 1,299 table + 791 figure chunks)
- Same correctness (75% vs 75% holdout, 70% vs 70% generated); fewer unsupported details (faithful 62% → 81% and
  73% → 87%), confirmed with a repeat run of the baseline. Occasionally a table chunk pushed out the right text page.
- Kept as an option (`RAG_INDEX=plus`), not the default.

**Contrastive fine-tuning of the embedding model** (`train/`)
- bge-small fine-tuned with MultipleNegativesRankingLoss (in-batch + hard negatives mined with hybrid search) on 432
  Claude-written questions from pages no test uses; trained on a free Colab GPU.
- Search: right page in the top 5 for 68% → 75% of 142 questions. Answers: 59% → 66% correct over 140 questions,
  same safety.
- **But the whole gain was on Claude-generated questions (46 → 56 of 80); on hand-written ones it was 37 → 37 of 60.**
  The model learned the style of its training questions more than the content. Excluding every test page from
  training (to avoid leakage) also meant it never trained on the core topics.

**Multi-query decomposition** (`eval/decompose_questions.py`, `eval/eval_multiquery.py`)
- Claude splits a question into 2–3 searches; all result lists are merged with reciprocal rank fusion.
- Two-part questions, both answer pages in the top 5: 25% → 25%. One-part questions: 75% → 65% (worse).
- Rank fusion favours overview pages that fit every sub-query a little over the specific page each part needs.

---

## Running it

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
echo "ANTHROPIC_API_KEY=your-key" > .env     # never commit this file

python src/collect.py      # download papers (data/ is not in the repository)
python src/parse.py
python src/chunk.py
python src/embed.py
python src/generate.py "What drugs are used to manage cytokine release syndrome after CAR-T?"
```

Evaluation (`--split`: practice, final, holdout, synthetic, confirm, mq):

```bash
python eval/eval_retrieval.py --mode hybrid --split final         # search test (free)
python eval/run_answers.py --mode hybrid --split final --label hybrid
python eval/judge.py --split final --label hybrid                 # answer test (~$1 per 25 questions with 5 pages; more with 10)
```

Other index versions: `RAG_INDEX=plus python src/embed.py` (text + tables and figures), then prefix any
evaluation command with `RAG_INDEX=plus`.

The downloaded papers (`data/`), raw evaluation outputs (`eval/results/`, which contain excerpts of the papers),
and the fine-tuned model are not included in this repository.

---

## Limitations

- Small corpus: 194 open-access papers; several key original trials are not open access.
- Small hand-written evaluation sets: one question moves a score by 4–7 points.
- Generated question sets list only the passage each question came from, so other valid pages count as misses.
- The answer keys and grading were produced with AI assistance and spot-checked, not reviewed by a clinician.
- Guardrails use pattern matching and may miss unusually phrased personal questions (one such case on the final set
  was still answered safely).

## Possible next steps

- Confirm the switch to 10 pages with the answer test on questions not used to choose it (about $1).
- An LLM check of whether each retrieved page actually contains the answer (the idea behind Corrective RAG), since
  retrieval of the specific fact is the main weakness.
- Fine-tuning again on all pages with question-style training data, and HyDE (searching with a hypothetical answer).
- A small production demo: API, container, and a rate-limited public link.
