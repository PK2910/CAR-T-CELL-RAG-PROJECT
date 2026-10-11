"""
Test 2, step 1: run the FULL RAG system on the exam questions and save the answers.
Uses the Claude API key from .env (run this yourself; it costs ~1-2 cents per question).

Saves: eval/results/answers_<label>_<split>.json
  question, answer, the 5 sources Claude was given, tokens, cost, latency

Run:  python eval/run_answers.py [--split dev|test|all] [--label baseline]
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
from generate import EFFORT, MODEL, SYSTEM_PROMPT, answer  # noqa: E402

QUESTIONS_FILE = ROOT / "eval/questions.json"
INDEX_INFO_FILE = ROOT / "data/processed/index_info.json"
RESULTS_DIR = ROOT / "eval/results"
SOURCE_FIELDS = ["chunk_id", "pmcid", "pmid", "title", "first_author", "journal", "year",
                 "article_type", "sections", "score", "text"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="dev", choices=["dev", "test", "all"])
    parser.add_argument("--mode", default="vector", choices=["vector", "hybrid", "rerank", "langchain"])
    parser.add_argument("--label", default=None, help="name of this run (default: baseline for vector, else the mode)")
    args = parser.parse_args()
    args.label = args.label or ("baseline" if args.mode == "vector" else args.mode)

    questions = [q for q in json.loads(QUESTIONS_FILE.read_text())
                 if args.split == "all" or q["split"] == args.split]
    print(f"Running {len(questions)} {args.split} questions through the RAG system "
          f"({MODEL}, {args.mode} search) -> label '{args.label}'...\n")

    rows, total_cost = [], 0.0
    for i, q in enumerate(questions, 1):
        start = time.perf_counter()
        try:
            if args.mode == "langchain":  # the LangChain version in langchain_version/
                from langchain_version import rag
                result = rag.answer(q["question"])
            else:
                result = answer(q["question"], mode=args.mode)
        except anthropic.APIError as e:  # keep going; one failed call shouldn't lose the whole run
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
        })
        print(f"[{i:>2}/{len(questions)}] {q['id']}  {latency:5.1f}s  ${result['cost_usd']:.4f}  {q['question'][:60]}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_DIR / f"answers_{args.label}_{args.split}.json"
    out.write_text(json.dumps({
        "run": args.label, "split": args.split, "date": datetime.now().isoformat(timespec="seconds"),
        "model": MODEL, "effort": EFFORT, "search_mode": args.mode, "system_prompt": SYSTEM_PROMPT,
        "index": json.loads(INDEX_INFO_FILE.read_text()), "answers": rows,
    }, indent=2, ensure_ascii=False))
    errors = sum(1 for r in rows if "error" in r)
    print(f"\nDone: {len(rows) - errors} answered, {errors} errors, total cost ${total_cost:.3f}")
    print(f"Saved -> {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
