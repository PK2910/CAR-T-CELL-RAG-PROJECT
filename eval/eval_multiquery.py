"""
Multi-query decomposition, step 2: free search test, today's hybrid search vs multi-query.

  today        hybrid search with the question            (2 ranked lists: meaning + keywords, merged with RRF)
  multi-query  hybrid search with the question AND each of Claude's sub-queries; all lists merged with one RRF
               (the original question always stays in, as a safety net)
               score(page) = sum over texts t in [question, sub-queries]:
                             1/(60 + meaning rank for t) + 1/(60 + keyword rank for t)

Scores, reported separately for one-part and two-part questions:
  any@k   at least one answer page in the top k
  all@k   EVERY answer page in the top k (for two-part questions: both parts found)
  MRR     how high the first answer page ranks (1.0 = always #1)

Free: local search only, no API calls.
Run:  python eval/eval_multiquery.py [--split mq]                    (original embedding model)
      RAG_INDEX=tuned python eval/eval_multiquery.py [--split mq]    (fine-tuned embedding model)
"""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "eval"))
from bm25 import bm25_ranking  # noqa: E402
from index_config import INDEX  # noqa: E402  (text, plus or tuned: set with RAG_INDEX)
from question_sets import load_questions  # noqa: E402
from retrieve import RRF_K, load_chunks, vector_ranking  # noqa: E402

RESULTS_DIR = ROOT / "eval/results"
KS = [5, 10]


def fused(texts: list[str]) -> list[int]:
    """One RRF over the meaning and keyword rankings of every text; chunk positions, best first."""
    scores = {}
    for t in texts:
        for ranking in (vector_ranking(t), bm25_ranking(t)):
            for rank, pos in enumerate(ranking, 1):
                scores[pos] = scores.get(pos, 0.0) + 1 / (RRF_K + rank)
    return [pos for pos, _ in sorted(scores.items(), key=lambda x: -x[1])]


def score(ranking_ids: list[str], relevant: list[str]) -> dict:
    ranks = {cid: ranking_ids.index(cid) + 1 for cid in relevant if cid in ranking_ids}
    first = min(ranks.values()) if ranks else None
    out = {"ranks": [ranks.get(cid) for cid in relevant], "mrr": 1 / first if first else 0.0}
    for k in KS:
        hits = [r for r in ranks.values() if r <= k]
        out[f"any@{k}"] = int(bool(hits))
        out[f"all@{k}"] = int(len(hits) == len(relevant))
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="mq")
    args = parser.parse_args()

    questions = [q for q in load_questions() if q["split"] == args.split]
    subs = json.loads((RESULTS_DIR / f"decompositions_{args.split}.json").read_text())
    ids = [c["chunk_id"] for c in load_chunks()]

    rows = []
    for q in questions:
        today = [ids[p] for p in fused([q["question"]])]
        multi = [ids[p] for p in fused([q["question"]] + subs.get(q["id"], []))]
        rows.append({"id": q["id"], "type": q["type"], "question": q["question"], "sub_queries": subs.get(q["id"], []),
                     "today": score(today, q["relevant_chunks"]), "multi": score(multi, q["relevant_chunks"])})

    metrics = [f"any@{k}" for k in KS] + [f"all@{k}" for k in KS] + ["mrr"]
    print(f"Multi-query search test | split={args.split} | index={INDEX} | {len(rows)} questions\n")
    for group in ("one_part", "two_part"):
        g = [r for r in rows if r["type"] == group]
        if not g:
            continue
        print(f"{group.replace('_', '-')} questions ({len(g)}):")
        print(f"  {'metric':8} {'today':>8} {'multi-query':>12}")
        for m in metrics:
            t = sum(r["today"][m] for r in g) / len(g)
            mq = sum(r["multi"][m] for r in g) / len(g)
            fmt = (lambda v: f"{v:.2f}") if m == "mrr" else (lambda v: f"{v:.0%}")
            print(f"  {m:8} {fmt(t):>8} {fmt(mq):>12}")
        better = [r["id"] for r in g if r["multi"]["all@5"] > r["today"]["all@5"]]
        worse = [r["id"] for r in g if r["multi"]["all@5"] < r["today"]["all@5"]]
        print(f"  all@5 fixed by multi-query: {better or 'none'} | broken by multi-query: {worse or 'none'}\n")

    out = RESULTS_DIR / f"multiquery_{args.split}_{INDEX}.json"
    out.write_text(json.dumps(rows, indent=2, ensure_ascii=False))
    print(f"Saved -> {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
