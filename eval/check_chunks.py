"""
LLM relevance check (Corrective-RAG-style), Step A: does a checker pick the chunks that answer the question?

For each test question:
  1. hybrid search -> top 30 chunks (the same search as today, free)
  2. Claude Haiku 5.5 reads the question and all 30 chunks and marks each one:
       yes     states information that answers the question (or one part of it)
       partly  same topic, but not the specific information asked for
       no      not about what was asked
  3. "unsure" (no chunk marked yes, at least one partly): check the next 30 chunks (#31-60) the same way
Nothing is answered here; eval/checker_report.py then compares, for free, the chunks the checker keeps with the
10 chunks Claude reads today, using the answer key.

Uses the Claude API key from .env (run this yourself). About 0.3 cents per question; the run prints its real cost.
Saves after every question to eval/results/chunk_checks.json, so a stopped run continues where it left off.

Run:  python eval/check_chunks.py --limit 5     (try 5 questions first, then look at the cost)
      python eval/check_chunks.py               (all questions in practice, final, holdout, synthetic, confirm)
Then: python eval/checker_report.py             (free)
"""

import argparse
import json
import sys
from pathlib import Path

import anthropic
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "eval"))
from generate import format_sources  # noqa: E402  (chunks labelled exactly as Claude sees them when answering)
from question_sets import load_questions  # noqa: E402
from relevance_check import BATCH, EFFORT, MODEL, cost, run_checker  # noqa: E402  (the same checker the system uses)
from retrieve import load_chunks, ranking  # noqa: E402

RESULTS_FILE = ROOT / "eval/results/chunk_checks.json"
SPLITS = {"dev", "test", "final", "holdout", "synthetic", "confirm"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None, help="only check this many new questions")
    args = parser.parse_args()

    questions = [q for q in load_questions() if q["split"] in SPLITS]
    chunks = load_chunks()
    RESULTS_FILE.parent.mkdir(parents=True, exist_ok=True)
    done = json.loads(RESULTS_FILE.read_text()) if RESULTS_FILE.exists() else {}

    client = anthropic.Anthropic()
    total, new = 0.0, 0
    for q in questions:
        if q["id"] in done:
            continue
        if args.limit is not None and new >= args.limit:
            break
        ranked = [chunks[pos] for pos in ranking(q["question"], "hybrid")[:2 * BATCH]]
        try:
            rows, rounds, usage_in, usage_out = run_checker(client, q["question"], ranked, format_sources)
        except anthropic.APIError as e:
            if "credit balance" in str(e):
                print(f"\nStopped at {q['id']}: the API credit ran out. Add credit and re-run the same command; "
                      f"{len(done)} questions so far are saved and will be kept.")
                break
            print(f"{q['id']}: API error, skipped (re-run to retry): {e}")
            continue
        except RuntimeError as e:
            print(f"{q['id']}: {e}, skipped (re-run to retry)")
            continue

        q_cost = cost(usage_in, usage_out)
        total += q_cost
        new += 1
        done[q["id"]] = {"question": q["question"], "split": q["split"], "model": MODEL, "effort": EFFORT,
                         "rounds": rounds, "input_tokens": usage_in, "output_tokens": usage_out,
                         "cost_usd": round(q_cost, 5), "chunks": rows}
        RESULTS_FILE.write_text(json.dumps(done, indent=2, ensure_ascii=False))
        n_yes = sum(r["verdict"] == "yes" for r in rows)
        n_partly = sum(r["verdict"] == "partly" for r in rows)
        print(f"{q['id']:4} yes {n_yes:2} | partly {n_partly:2} | rounds {rounds} | ${q_cost:.4f} | "
              f"{q['question'][:60]}")

    print(f"\n{new} questions checked this run, ${total:.3f} "
          f"(${total / max(new, 1):.4f} per question). {len(done)} saved in total -> "
          f"{RESULTS_FILE.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
