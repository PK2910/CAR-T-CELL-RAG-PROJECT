"""
Summarize the logbook (api/logbook.py): how many questions, how fast, how much they cost, how many came from the
cache and what that saved, the most asked questions, what the relevance check decided, how often the fallback was
used, and the thumbs up/down, compared with the targets in the README (Requirements).
Speed and cost targets are checked on answers that were worked out, not on saved answers from the cache.
Free: it only reads data/api_log.sqlite.

Run:  python api/log_report.py                 the last 7 days
      python api/log_report.py --days 30
      python api/log_report.py --export-down   also save the thumbs-down answers to eval/results/thumbs_down.json,
                                               as candidates for new test questions (answer keys still to be written)
"""

import argparse
import json
import math
from collections import Counter
from datetime import datetime, timedelta, timezone

from cache import tidy
from logbook import ROOT, entries

TARGETS = {"first_text_s": 3.0, "total_p95_s": 15.0, "cost_usd": 0.05}  # README, Requirements
EXPORT_FILE = ROOT / "eval/results/thumbs_down.json"


def percentile(values: list[float], p: float) -> float | None:
    """Nearest-rank percentile: p=95 gives the value 95% of answers are at or under."""
    values = sorted(v for v in values if v is not None)
    return values[math.ceil(p / 100 * len(values)) - 1] if values else None


def fmt(seconds: float | None) -> str:
    return "-" if seconds is None else f"{seconds:.1f} s"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--days", type=int, default=7)
    parser.add_argument("--export-down", action="store_true")
    args = parser.parse_args()

    since = (datetime.now(timezone.utc) - timedelta(days=args.days)).isoformat(timespec="seconds")
    rows = entries(since)
    print(f"Logbook, last {args.days} days: {len(rows)} questions")
    if not rows:
        print("Nothing logged yet: ask a question through the API (uvicorn api.main:app) first.")
        return

    print("\nQuestions per day (UTC):")
    for day, n in sorted(Counter(r["time"][:10] for r in rows).items()):
        print(f"  {day}  {n}")
    users = Counter(r["user"] for r in rows)
    print(f"  users: {len(users) - ('owner' in users)} public" + (f" + owner ({users['owner']} questions)"
                                                                  if "owner" in users else ""))

    answered = [r for r in rows if r["answer"] is not None]
    from_cache = [r for r in answered if r["cached"]]
    answered = [r for r in answered if not r["cached"]]  # worked out: what the speed and cost targets are about
    totals = [r["seconds_total"] for r in answered]
    first = [r["seconds_first_text"] for r in answered if r["seconds_first_text"] is not None]
    timing = [json.loads(r["timing"] or "{}") for r in answered]
    p95, first_median = percentile(totals, 95), percentile(first, 50)
    print("\nSpeed (answers worked out, not from the cache):")
    if totals:
        print(f"  full answer      median {fmt(percentile(totals, 50))}, 95% under {fmt(p95)}"
              f"   (target: 95% under {TARGETS['total_p95_s']:.0f} s"
              + (", MET)" if p95 <= TARGETS["total_p95_s"] else ", NOT met)"))
    if first:
        print(f"  first words      median {fmt(first_median)} (streamed answers only)"
              f"   (target: under {TARGETS['first_text_s']:.0f} s"
              + (", MET)" if first_median <= TARGETS["first_text_s"] else ", NOT met)"))
    for stage, label in (("search_s", "search"), ("check_s", "relevance check"), ("answer_s", "writing")):
        values = [t[stage] for t in timing if stage in t]
        if values:
            print(f"  {label:16} median {fmt(percentile(values, 50))}, slowest {fmt(max(values))}")

    costs = [r["cost_usd"] for r in answered if r["cost_usd"] is not None]
    if costs:
        average = sum(costs) / len(costs)
        print(f"\nCost: ${sum(costs):.3f} in total, {100 * average:.2f} cents per worked-out answer on average"
              f"   (target: under {100 * TARGETS['cost_usd']:.0f} cents" + (", MET)" if average <= TARGETS["cost_usd"]
                                                                          else ", NOT met)"))

    print(f"\nCache: {len(from_cache)} of {len(from_cache) + len(answered)} answers came from the cache"
          + (f" (median {fmt(percentile([r['seconds_total'] for r in from_cache], 50))}), "
             f"saving ${sum(r['saved_usd'] or 0 for r in from_cache):.3f}" if from_cache else ""))
    asked = Counter(tidy(r["question"]) for r in rows)
    repeated = [(q, n) for q, n in asked.most_common(10) if n > 1]
    if repeated:
        print("Most asked questions:")
        for q, n in repeated:
            print(f"  {n:>3}x  {q[:80]}")

    print("\nRelevance check:  " + " · ".join(f"{k or 'not run'} {v}"
                                              for k, v in Counter(r["outcome"] for r in rows).most_common()))
    print(f"Fallback used:    {sum(r['fallback'] for r in rows)} times (no summary written; passages shown)")
    print(f"No answer at all: {len(rows) - len(answered) - len(from_cache)}")
    guard = Counter(r["guardrail"] for r in rows if r["guardrail"] not in (None, "none"))  # replaced, not_found, ...
    if guard:
        print("Guardrails:       " + " · ".join(f"{k} {v}" for k, v in guard.items()))
    errors = Counter((r["error"] or "").split(":")[0] for r in rows if r["error"])
    if errors:
        print("Errors:           " + " · ".join(f"{k} {v}" for k, v in errors.most_common()))

    rated = [r for r in rows if r["rating"] is not None]
    down = [r for r in rated if r["rating"] < 0]
    print(f"\nFeedback: {len(rated) - len(down)} thumbs up, {len(down)} thumbs down "
          f"({len(rated)} of {len(rows)} answers rated)")
    for r in down:
        print(f"  {r['id']}  [{r['outcome'] or '-'}]  {r['question'][:70]}"
              + (f"\n      comment: {r['comment']}" if r["comment"] else ""))

    setups = Counter(r["versions"] for r in rows)
    if len(setups) > 1:
        print(f"\nNote: {len(setups)} different setups answered in this period (index, models or prompts changed):")
        for v, n in setups.items():
            print(f"  {n} questions: {v}")

    if args.export_down:
        EXPORT_FILE.parent.mkdir(parents=True, exist_ok=True)
        EXPORT_FILE.write_text(json.dumps([
            {"answer_id": r["id"], "time": r["time"], "question": r["question"], "answer": r["answer"],
             "comment": r["comment"], "relevance_check": r["outcome"], "chunks": json.loads(r["chunks"]),
             "versions": json.loads(r["versions"])} for r in down], indent=2, ensure_ascii=False))
        print(f"\nSaved {len(down)} thumbs-down answers -> {EXPORT_FILE}")


if __name__ == "__main__":
    main()
