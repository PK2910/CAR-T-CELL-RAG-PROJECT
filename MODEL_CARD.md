# Model card: CAR-T in ALL Literature Q&A

A one-page summary of what this system is for, how well it works, and where it fails. Details and full numbers are
in the [README](README.md) and [`eval/RESULTS.md`](eval/RESULTS.md).

> **Research and education tool. Not medical advice.** It summarizes published studies. It is not designed or
> validated to guide the care of any individual patient.

## What it is

A retrieval-augmented question-answering system (RAG). It answers questions about CAR-T cell therapy in acute
lymphoblastic leukemia (ALL) using only passages from published papers, and cites the passage behind every claim.

| Part | What is used |
|---|---|
| Papers | 194 open-access papers from PubMed Central, cut into 5,354 passages of ~400 tokens (index built 2026-10-04) |
| Search | Hybrid: meaning-based vectors (BAAI/bge-small-en-v1.5) plus exact-word search (BM25), merged; typos in the question are corrected for the search using the papers' vocabulary |
| Relevance check | Claude Haiku 5.5 reads the question with the top 30 passages and keeps the ones that answer it |
| Answer | Claude Sonnet 5.5 writes the answer from the kept passages (up to 10), citing each one |
| Safety | Rules in code around the models (see "Safety measures") |

Version: as of commit on 2026-11-01. Every answer records the versions that produced it (index, chunking, models,
prompt fingerprints, guardrail fingerprint).

## Intended use

- **For:** students, researchers and clinicians who want a quick, cited summary of what the literature says about
  CAR-T in ALL: response rates, side effects (CRS, ICANS), relapse, transplant after CAR-T, products, mechanisms.
  Patients and families can use it to prepare questions for their care team.
- **Not for:** decisions about a specific person's treatment, doses for a specific person, diagnosis, emergencies,
  or questions about other cancers (the papers are about ALL). Answers are in English only.

## How well it works

Measured on fixed, hand-written question sets with answer keys, graded by Claude Opus 5.5 against the key.

| Measure | Result | Notes |
|---|---|---|
| Correct answers, final test set | **17 of 21 (81%)** | This set was also used to choose 10 passages and to check the relevance step, so the figure is optimistic. The first, untouched run on it scored 48% (with an older version); expect something in between on new questions. |
| Special questions handled as expected | 4 of 4 | Two personal-advice questions (steroids for a confused child; "which product is best for me"), a dose research question, and a question the papers cannot answer |
| Serious errors | 0 of 25 | |
| Answers with no unsupported detail | 11 of 21 | Most slips are minor (study type, patient group, mixing two sources). The judge also wrongly flagged correct journal names until 2026-11-01, so this is a lower bound. |
| Citations pointing to a real given passage | 1 slip in 610 saved answers | Checked by code on every answer |
| Off-topic questions refused | 2 of 2 live checks | "The papers in this collection don't answer this question" |
| Cost | ~2.3–2.7 cents per question | Limit for the public demo: 10 questions per person per day, $1 per day in total |
| Speed | ~13 s median for a full answer, up to ~17 s | From the first live questions (a small sample). First words appear sooner (streaming). The target of 95% under 15 s is **not met**; the relevance check takes ~6–8 s. |
| Readability | about grade 10.4 | 8.2 abbreviations per 100 words; plain-language health writing aims for about grade 8 |
| Load | Fine up to 50 people at once; limit between 50 and 75 | With a stand-in Claude as slow as the real one, on a laptop (`eval/load_test.py`, Locust). 25 people asking and reading for 3.5 minutes: 256 of 257 answered, the server added under 1.2 s for 95% of questions. 50 people: 2.5 answers a second, all answered. 75–100 people: about 10 s longer waits, up to 1.1% errors. Not yet tested with the real Claude or on the online server. |
| Robustness of the search | holds up to lowercase, keywords, "CAR T"; a typo is corrected | On 120 new questions: correctly typed 72%, with a typo 72% after correction (69% before). Spelling out "CAR-T" as "chimeric antigen receptor T-cell" still lowers it to 59%. |

Main finding from testing: when an answer is wrong, the needed fact was usually never retrieved (about 94% of
missing facts), not ignored by the answer model.

## Safety measures

- **Dose guardrail (code):** a question about a specific person ("my son", "should I give") never gets a dose; if one
  appears anyway, the answer is replaced with a safe message pointing to the treating team or emergency services.
- **Nothing relevant, no answer:** if no passage answers the question, a fixed reply says so instead of guessing.
- **Trick check:** questions that try to change the rules ("ignore your instructions") are not answered. All 5,354
  passages were scanned for hidden instructions: none found.
- **Citation check:** a citation to a passage the answer was not given adds a warning to the answer.
- **Privacy:** names, ID numbers, dates of birth, phone numbers and emails are removed before anything is logged
  (Microsoft Presidio, run locally). The API key is never stored in the code or the Docker image.
- **Failing safely:** if Claude is unavailable or slow, users see the passages that were found instead of an error;
  the system refuses to start if the search setup does not match the index.
- **Policy:** personal questions about treatment options get a general summary of the evidence and a clear statement
  that the treating team decides; only doses are blocked outright.

## Known limitations

- **Small, open-access corpus:** 194 papers; several key original trials are not open access. The papers are
  fixed as of October 2026 and do not update themselves.
- **ALL only:** comparisons with other blood cancers can only use the few passages that make them.
- **Small test sets:** 21 answerable questions in the final set, so one question moves a score by about 5 points.
- **Wording matters:** writing out abbreviations in full ("chimeric antigen receptor" instead of "CAR-T") or a typo
  that forms another real word ("patent" for "patient") can make the search miss the right passage.
- **Answer keys** were written with AI assistance and spot-checked, not reviewed by a clinician.
- **Habits:** the answer model sometimes adds a detail the passage does not state (a patient group, a study type).
  The relevance check is sometimes too strict on mechanism and two-part questions.
- **Rules have gaps:** the personal-question, trick and privacy checks catch common phrasings; unusual wording can
  get past them. The dose guardrail covers personal dosing, not other high-risk advice such as steroid management.
- **Language is technical** for non-specialists. A plain-language version was tested and not adopted: it caused 3
  serious errors by dropping context while simplifying.

## Ethical considerations

Wrong medical information can cause harm, so the system answers only from cited papers, shows the passages, says
when the papers do not answer, and carries a disclaimer on every answer. It is presented as a literature tool, not
clinical decision support, which would need regulatory review and clinical validation. User questions are logged
for quality monitoring, with personal details removed, and are kept only on the server.

## Monitoring and maintenance

- Every answer is logged with its time per step, cost, relevance-check outcome, passages used and versions; users can
  rate answers, and thumbs-down answers can be exported as candidate test questions (`python api/log_report.py`).
- Alerts: every 5 minutes the server checks for slow answers, failures, spending near the daily limit, many thumbs
  down and wrong citations, and lists them at `/health`. What to do for each: [RUNBOOK.md](RUNBOOK.md).
- Any change to models, prompts, chunking or guardrails is tested on the same question sets before it is kept.
- Robustness: `python eval/robustness.py` re-tests the search with reworded and misspelled questions.
- Every push runs 71 code tests and a security check of the packages on GitHub (CI).
- Updates: new papers are added every 3–6 months, or sooner after major results, with the same checks before the
  new index is used (README, "Keeping it up to date"). On 2026-11-02, 113 papers published since the collection
  date were waiting to be reviewed.

## Contact

Project repository: https://github.com/PK2910/CAR-T-CELL-RAG-PROJECT
