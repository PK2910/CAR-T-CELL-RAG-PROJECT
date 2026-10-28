"""
Test 2, step 1: run the FULL RAG system on the exam questions and save the answers.
Uses the Claude API key from .env (run this yourself; it costs ~1-2 cents per question).

Saves: eval/results/answers_<label>_<split>.json
  question, answer, the sources Claude was given (10 by default; 5 before 2026-10-26), tokens, cost, latency
Re-running with the same label and split keeps the saved answers and only answers the missing ones
(e.g. after the API credit ran out), so nothing is paid for twice.

Run:  python eval/run_answers.py [--split dev|test|all] [--label baseline]
      python eval/run_answers.py --mode checked --split final --label final_checked
        (LLM relevance check: answers from the checker's saved verdicts in eval/results/chunk_checks.json,
         made by eval/check_chunks.py; see src/relevance_check.py)
"""

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))  # so the LangChain version (langchain_version/) can be imported
import anthropic  # noqa: E402
from generate import EFFORT, MODEL, SYSTEM_PROMPT, answer, answer_checked  # noqa: E402

QUESTIONS_FILE = ROOT / "eval/questions.json"
from question_sets import SPLITS, load_questions  # noqa: E402  (+ synthetic set if made)
from index_config import INFO_FILE as INDEX_INFO_FILE  # noqa: E402  (the index actually used: text, plus or tuned)
RESULTS_DIR = ROOT / "eval/results"
CHECKS_FILE = RESULTS_DIR / "chunk_checks.json"  # the relevance checker's verdicts (--mode checked)
SOURCE_FIELDS = ["chunk_id", "pmcid", "pmid", "title", "first_author", "journal", "year",
                 "article_type", "sections", "score", "text"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="dev", choices=SPLITS)
    parser.add_argument("--mode", default="vector", choices=["vector", "hybrid", "rerank", "langchain", "checked"])
    parser.add_argument("--label", default=None, help="name of this run (default: baseline for vector, else the mode)")
    args = parser.parse_args()
    args.label = args.label or ("baseline" if args.mode == "vector" else args.mode)

    questions = [q for q in load_questions()
                 if args.split == "all" or q["split"] == args.split
                 or (args.split == "practice" and q["split"] in ("dev", "test"))]
    checks = {}
    if args.mode == "checked":
        checks = json.loads(CHECKS_FILE.read_text()) if CHECKS_FILE.exists() else {}
        missing = [q["id"] for q in questions if q["id"] not in checks]
        if missing:
            print(f"No checker verdicts for {missing}: run python eval/check_chunks.py first.")
            return
    print(f"Running {len(questions)} {args.split} questions through the RAG system "
          f"({MODEL}, {args.mode} search) -> label '{args.label}'...\n")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_DIR / f"answers_{args.label}_{args.split}.json"
    # Resume: keep answers already saved under this label; only the missing (or failed) ones are re-run
    saved = {}
    if out.exists():
        saved = {r["id"]: r for r in json.loads(out.read_text())["answers"] if "error" not in r}
        if saved:
            print(f"Resuming: {len(saved)} answers already saved under this label are kept.\n")

    def save(rows):
        out.write_text(json.dumps({
            "run": args.label, "split": args.split, "date": datetime.now().isoformat(timespec="seconds"),
            "model": MODEL, "effort": EFFORT, "search_mode": args.mode, "system_prompt": SYSTEM_PROMPT,
            "index": json.loads(INDEX_INFO_FILE.read_text()), "answers": rows,
        }, indent=2, ensure_ascii=False))

    rows, total_cost = [], 0.0
    for i, q in enumerate(questions, 1):
        if q["id"] in saved:
            rows.append(saved[q["id"]])
            continue
        start = time.perf_counter()
        try:
            if args.mode == "langchain":  # the LangChain version in langchain_version/
                from langchain_version import rag
                result = rag.answer(q["question"])
            elif args.mode == "checked":
                result = answer_checked(q["question"], checks[q["id"]]["chunks"])
            else:
                result = answer(q["question"], mode=args.mode)
        except anthropic.APIError as e:  # keep going; one failed call shouldn't lose the whole run
            if "credit balance" in str(e):  # no point continuing: every further call would fail too
                print(f"\nStopped at {q['id']}: the API credit ran out. Add credit and re-run the same command;"
                      f" the {len(rows)} answers so far are saved and will be kept.")
                save(rows)
                return
            print(f"[{i:>2}/{len(questions)}] {q['id']}  ERROR: {type(e).__name__}")
            rows.append({"id": q["id"], "question": q["question"], "error": f"{type(e).__name__}: {e}"})
            continue
        latency = time.perf_counter() - start
        total_cost += result["cost_usd"]
        rows.append({
            "id": q["id"],
            "question": q["question"],
            "answer": result["answer"],
            "original_answer": result["original_answer"],
            "guardrail": result["guardrail"],
            "stop_reason": result["stop_reason"],
            "model": result["model"],
            "sources": [{k: s.get(k) for k in SOURCE_FIELDS} for s in result["sources"]],
            "input_tokens": result["input_tokens"],
            "output_tokens": result["output_tokens"],
            "cost_usd": result["cost_usd"],
            "latency_s": round(latency, 2),
            **({"checker_outcome": result["checker_outcome"]} if "checker_outcome" in result else {}),
        })
        print(f"[{i:>2}/{len(questions)}] {q['id']}  {latency:5.1f}s  ${result['cost_usd']:.4f}  {q['question'][:60]}")
        save(rows)  # after every answer, so a stopped run loses nothing

    save(rows)
    errors = sum(1 for r in rows if "error" in r)
    print(f"\nDone: {len(rows) - errors} answered, {errors} errors, total cost ${total_cost:.3f}")
    print(f"Saved -> {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
