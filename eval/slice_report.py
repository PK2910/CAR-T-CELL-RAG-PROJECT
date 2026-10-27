"""
Search scores by slice (groups of questions), to see WHERE the search is weak, not just how good it is overall.

Groups:
  - question type     fact_number, mechanism, comparison, side_effect, ...
  - written by        hand-written (practice, final, holdout: every answer page marked) vs
                      Claude-generated (synthetic, confirm: written by a script from one page, only that page marked)
  - asks for a number questions worded "how often", "what rate", "what percentage", ... vs the rest

Each question is graded exactly as in eval_retrieval.py (Hit@5, Hit@10, MRR). Only questions with answer pages
can be scored, so personal, off-topic and unanswerable questions are left out (their safety is checked by the answer
test). The "mq" set is left out because its types describe one-part/two-part questions, not the topic.

Free: local search only, no API calls.
Run:  python eval/slice_report.py [--mode hybrid|vector]
"""

import argparse
import json
import re
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "eval"))
from eval_retrieval import RESULTS_DIR, grade, search  # noqa: E402
from question_sets import load_questions  # noqa: E402

HAND_WRITTEN = {"dev", "test", "final", "holdout"}
GENERATED = {"synthetic", "confirm"}
NUMBER_WORDS = r"how often|how many|how common|how frequent|what (rate|rates|percentage|proportion|share|fraction)|incidence|frequency|\brates?\b"
FEW = 10  # slices with fewer questions than this are flagged: one question moves the score a lot


def slices(q: dict) -> dict[str, str]:
    return {
        "question type": q["type"],
        "written by": "hand-written" if q["split"] in HAND_WRITTEN else "Claude-generated",
        "asks for a number": "yes" if re.search(NUMBER_WORDS, q["question"], re.I) else "no",
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", default="hybrid", choices=["vector", "hybrid"])
    args = parser.parse_args()

    questions = [q for q in load_questions() if q["relevant_chunks"] and q["split"] in HAND_WRITTEN | GENERATED]
    judged = json.loads((ROOT / "eval/method_b_judgments.json").read_text())
    search("warm-up", 1, args.mode)
    rows = []
    for q in questions:
        r = grade(q, args.mode, judged)
        rows.append({"id": q["id"], "split": q["split"], "question": q["question"], **slices(q),
                     "hit": r["hit"], "hit_at_10": r["hit_at_10"], "mrr": r["mrr"], "best_rank": r["best_rank"]})

    overall = (statistics.mean(r["hit"] for r in rows), statistics.mean(r["mrr"] for r in rows))
    print(f"\nSearch scores by slice | mode={args.mode} | {len(rows)} questions with answer pages")
    print(f"Overall: Hit@5 {overall[0]:.0%}, MRR {overall[1]:.2f}\n")
    report = {}
    for group in ("question type", "written by", "asks for a number"):
        print(f"{group}:")
        print(f"  {'slice':18} {'n':>4} {'Hit@5':>6} {'Hit@10':>7} {'MRR':>5}")
        report[group] = {}
        for name in sorted({r[group] for r in rows}, key=lambda n: -sum(r[group] == n for r in rows)):
            g = [r for r in rows if r[group] == name]
            s = {"n": len(g), "hit_at_5": statistics.mean(r["hit"] for r in g),
                 "hit_at_10": statistics.mean(r["hit_at_10"] for r in g), "mrr": statistics.mean(r["mrr"] for r in g),
                 "missed_top5": [r["id"] for r in g if not r["hit"]]}
            report[group][name] = s
            flag = "  (few questions)" if len(g) < FEW else ""
            print(f"  {name:18} {s['n']:>4} {s['hit_at_5']:>6.0%} {s['hit_at_10']:>7.0%} {s['mrr']:>5.2f}{flag}")
        print()

    print("Missed (no answer page in the top 5), by question type:")
    for name, s in report["question type"].items():
        if s["missed_top5"]:
            print(f"  {name:18} {', '.join(s['missed_top5'])}")

    out = RESULTS_DIR / f"slices_{args.mode}.json"
    out.write_text(json.dumps({"mode": args.mode, "slices": report, "questions": rows}, indent=2, ensure_ascii=False))
    print(f"\nSaved -> {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
