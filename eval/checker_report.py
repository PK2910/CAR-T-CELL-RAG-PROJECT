"""
LLM relevance check, Step A report (free, no API calls): compare the chunks the checker keeps with the 10 chunks
Claude reads today, using the answer key.

  today         the search's top 10 chunks
  with checker  the chunks marked "yes" (in search order, at most 10), from #1-30, or #31-60 for unsure questions

For each question the checker's result is one of:
  good     at least one chunk marked "yes"           -> Claude would answer from the "yes" chunks
  unsure   no "yes" in #1-60, but some "partly"      -> answer with a "papers only partly address this" warning
  nothing  every chunk marked "no"                   -> guardrail: "the papers don't answer this question"

Run:  python eval/checker_report.py     (after python eval/check_chunks.py)
"""

import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "eval"))
from question_sets import load_questions  # noqa: E402

CHECKS_FILE = ROOT / "eval/results/chunk_checks.json"
TOP_K = 10
HAND_WRITTEN = {"dev", "test", "final", "holdout"}


def outcome(rows: list[dict]) -> str:
    labels = {r["verdict"] for r in rows}
    return "good" if "yes" in labels else "unsure" if "partly" in labels else "nothing"


def pct(part: int, whole: int) -> str:
    return f"{part}/{whole} ({part / whole:.0%})" if whole else "-"


def main():
    checks = json.loads(CHECKS_FILE.read_text())
    questions = {q["id"]: q for q in load_questions()}
    rows = []
    for qid, c in checks.items():
        q = questions[qid]
        key = set(q["relevant_chunks"])
        today = [r for r in c["chunks"] if r["rank"] <= TOP_K]
        kept = [r for r in c["chunks"] if r["verdict"] == "yes"][:TOP_K]
        first30 = [r for r in c["chunks"] if r["rank"] <= 30]
        correct30 = [r for r in first30 if r["chunk_id"] in key]
        rows.append({
            "id": qid, "group": "hand-written" if q["split"] in HAND_WRITTEN else "Claude-generated",
            "type": q["type"], "answerable": q["answerable"], "has_key": bool(key), "outcome": outcome(c["chunks"]),
            "today_hit": any(r["chunk_id"] in key for r in today),
            "kept_hit": any(r["chunk_id"] in key for r in kept),
            "today_share": sum(r["chunk_id"] in key for r in today) / len(today),
            "kept_share": sum(r["chunk_id"] in key for r in kept) / len(kept) if kept else None,
            "n_kept": len(kept),
            "correct_in_30": len(correct30),
            "correct_marked_yes": sum(r["verdict"] == "yes" for r in correct30),
            "correct_marked_yes_or_partly": sum(r["verdict"] in ("yes", "partly") for r in correct30),
            "cost": c["cost_usd"],
        })

    print(f"\nLLM relevance check, Step A | {len(rows)} questions checked | "
          f"checker cost ${sum(r['cost'] for r in rows):.3f} (${statistics.mean(r['cost'] for r in rows):.4f} each)\n")

    for group in ("hand-written", "Claude-generated"):
        g = [r for r in rows if r["group"] == group and r["has_key"]]
        if not g:
            continue
        shares_today = [r["today_share"] for r in g]
        shares_kept = [r["kept_share"] for r in g if r["kept_share"] is not None]
        print(f"{group} questions with an answer key ({len(g)}):")
        print(f"  correct chunk among the chunks Claude reads   today {pct(sum(r['today_hit'] for r in g), len(g)):>12}"
              f"   with checker {pct(sum(r['kept_hit'] for r in g), len(g))}")
        print(f"  share of those chunks that are correct        today {statistics.mean(shares_today):>12.0%}"
              f"   with checker {statistics.mean(shares_kept) if shares_kept else 0:.0%}")
        print(f"  chunks Claude reads (median)                  today {TOP_K:>12}"
              f"   with checker {statistics.median(r['n_kept'] for r in g):.0f}")
        lost = [r["id"] for r in g if r["today_hit"] and not r["kept_hit"]]
        gained = [r["id"] for r in g if not r["today_hit"] and r["kept_hit"]]
        print(f"  correct chunk gained by the checker: {gained or 'none'}")
        print(f"  correct chunk lost by the checker:   {lost or 'none'}")
        in30 = sum(r["correct_in_30"] for r in g)
        print(f"  correct chunks in the top 30 that the checker marked yes: "
              f"{pct(sum(r['correct_marked_yes'] for r in g), in30)} (yes or partly: "
              f"{pct(sum(r['correct_marked_yes_or_partly'] for r in g), in30)})\n")

    print("Checker outcome per question (good = answer, unsure = answer with a warning, nothing = guardrail):")
    for label, sel in (("answerable, with answer key", lambda r: r["has_key"]),
                       ("unanswerable / off-topic / personal", lambda r: not r["has_key"])):
        g = [r for r in rows if sel(r)]
        if g:
            counts = {o: sum(r["outcome"] == o for r in g) for o in ("good", "unsure", "nothing")}
            print(f"  {label:38} " + " | ".join(f"{o} {pct(n, len(g))}" for o, n in counts.items()))
            if not sel({"has_key": True}):
                for r in g:
                    print(f"      {r['id']} ({r['type']}): {r['outcome']}")

    wrongly_blocked = [r["id"] for r in rows if r["has_key"] and r["outcome"] == "nothing"]
    print(f"\nAnswerable questions the guardrail would block (checker found nothing): {wrongly_blocked or 'none'}")

    if "q05" in checks:  # the example used to explain the problem: the ICANS-rate chunk ranked #13
        hit = next((r for r in checks["q05"]["chunks"] if r["chunk_id"] == "PMC9528944_0028"), None)
        if hit:
            print(f"\nq05 example, chunk PMC9528944_0028 (\"ICANS ... in 40% of children\", rank #{hit['rank']}): "
                  f"{hit['verdict']} - {hit['reason']}")


if __name__ == "__main__":
    main()
