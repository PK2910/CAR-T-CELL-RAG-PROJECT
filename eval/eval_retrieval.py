"""
Test 1: grade the SEARCH step against the answer key (no Claude, no API key).

For each question that has correct passages in eval/questions.json:
  - Hit@5        did any of the top 5 passages contain the answer?        (1 or 0)
  - MRR          1 / rank of the first answer-bearing passage (0 if none)
  - Precision@5  share of the top 5 passages that contain the answer
  - Recall@5     answer-bearing passages found / min(5, number that exist)
A passage "contains the answer" if it is in the question's relevant_chunks.

Only the dev split is graded by default; the test split is kept for the final report.

Run:  python eval/eval_retrieval.py [--split dev|test|all] [--label baseline]
"""

import argparse
import json
import statistics
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))  # so the LangChain version (langchain_version/) can be imported
from retrieve import ranking, retrieve  # noqa: E402

QUESTIONS_FILE = ROOT / "eval/questions.json"
CHUNKS_FILE = ROOT / "data/processed/chunks.json"
JUDGMENTS_FILE = ROOT / "eval/method_b_judgments.json"
INDEX_INFO_FILE = ROOT / "data/processed/index_info.json"
RESULTS_DIR = ROOT / "eval/results"
K = 5  # same number of passages that generate.py gives Claude
DEPTHS = [5, 10, 20, 50, 100]  # Hit@k at several depths, to tell close misses from far misses
CHUNK_IDS = [c["chunk_id"] for c in json.loads(CHUNKS_FILE.read_text())]  # list position = Qdrant point id


def search(question: str, k: int, mode: str) -> list[dict]:
    if mode == "langchain":  # the LangChain version in langchain_version/
        from langchain_version import rag
        return rag.retrieve(question, k=k)
    return retrieve(question, k=k, mode=mode)


def full_ranking(question: str, mode: str) -> list[int]:
    if mode == "langchain":
        from langchain_version import rag
        return rag.ranking(question)
    return ranking(question, mode)


def first_relevant_rank(question: str, relevant: set[str], mode: str) -> int:
    """Rank of the best answer-bearing chunk when chunks are ranked for this question
    (10**6 if none appears, e.g. LangChain only ranks its merged top-50 lists)."""
    return next((rank for rank, pos in enumerate(full_ranking(question, mode), 1) if CHUNK_IDS[pos] in relevant), 10**6)


def grade(question: dict, mode: str, judged: dict) -> dict:
    relevant = set(question["relevant_chunks"])
    start = time.perf_counter()
    results = search(question["question"], K, mode)
    latency = time.perf_counter() - start
    best_rank = first_relevant_rank(question["question"], relevant, mode)
    # Chunks in the top 5 that nobody has judged yet (new results this search surfaced)
    unjudged = [r["chunk_id"] for r in results
                if r["chunk_id"] not in relevant and r["chunk_id"] not in judged.get(question["id"], {})]

    ranks = [rank for rank, r in enumerate(results, 1) if r["chunk_id"] in relevant]
    return {
        "best_rank": best_rank,
        **{f"hit_at_{d}": int(best_rank <= d) for d in DEPTHS},
        "id": question["id"],
        "type": question["type"],
        "label_quality": question["label_quality"],
        "hit": int(bool(ranks)),
        "mrr": 1 / ranks[0] if ranks else 0.0,
        "precision": len(ranks) / K,
        "recall": len(ranks) / min(K, len(relevant)),
        "first_hit_rank": ranks[0] if ranks else None,
        "latency_s": round(latency, 3),
        "unjudged_in_top5": unjudged,
        "retrieved": [{"rank": i, "chunk_id": r["chunk_id"], "score": r["score"],
                       "relevant": r["chunk_id"] in relevant} for i, r in enumerate(results, 1)],
    }


def mean(rows: list[dict], key: str) -> float:
    return statistics.mean(r[key] for r in rows) if rows else float("nan")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="dev", choices=["dev", "test", "all"])
    parser.add_argument("--mode", default="vector", choices=["vector", "hybrid", "rerank", "langchain"])
    parser.add_argument("--label", default=None, help="name of this run (default: baseline for vector, else the mode)")
    args = parser.parse_args()
    label = args.label or ("baseline" if args.mode == "vector" else args.mode)

    questions = [q for q in json.loads(QUESTIONS_FILE.read_text())
                 if q["relevant_chunks"] and (args.split == "all" or q["split"] == args.split)]
    judged = json.loads(JUDGMENTS_FILE.read_text())
    search("warm-up", 1, args.mode)  # load models/indexes once so they don't count toward latency
    rows = [grade(q, args.mode, judged) for q in questions]

    print(f"\nTest 1 (search) | mode={args.mode} | split={args.split} | {len(rows)} questions | top {K}\n")
    print(f"{'id':4} {'type':13} {'labels':11} {'best rank':>9} " + " ".join(f"{'@' + str(d):>4}" for d in DEPTHS)
          + f" {'prec':>5} {'recall':>6}")
    for r in rows:
        print(f"{r['id']:4} {r['type']:13} {r['label_quality']:11} {'#' + str(r['best_rank']):>9} "
              + " ".join(f"{'✅' if r[f'hit_at_{d}'] else '❌':>3}" for d in DEPTHS)
              + f" {r['precision']:>5.1f} {r['recall']:>6.2f}")

    summary = {
        "hit_at_5": mean(rows, "hit"),
        "mrr": mean(rows, "mrr"),
        "precision_at_5": mean(rows, "precision"),
        "recall_at_5": mean(rows, "recall"),
        **{f"hit_at_{d}": mean(rows, f"hit_at_{d}") for d in DEPTHS},
        "median_latency_s": statistics.median(r["latency_s"] for r in rows),
        "n_questions": len(rows),
        "hit_precise_labels": mean([r for r in rows if r["label_quality"] == "precise"], "hit"),
        "hit_approximate_labels": mean([r for r in rows if r["label_quality"] == "approximate"], "hit"),
    }
    print(f"\nHit@5        {summary['hit_at_5']:.0%}   ({sum(r['hit'] for r in rows)} of {len(rows)} questions had an answer in the top 5)")
    print(f"MRR          {summary['mrr']:.2f}   (1.0 = the answer is always ranked #1)")
    print(f"Precision@5  {summary['precision_at_5']:.0%}   (share of the 5 passages that contain the answer)")
    print(f"Recall@5     {summary['recall_at_5']:.0%}   (answer passages found, out of what could fit in 5)")
    print(f"Search time  {summary['median_latency_s']*1000:.0f} ms per question (median)")
    print("\nHit at deeper depths (is an answer-bearing chunk within the top k?):")
    for d in DEPTHS:
        print(f"  Hit@{d:<4} {summary[f'hit_at_{d}']:.0%}")
    print(f"\nHit@5 on precise labels: {summary['hit_precise_labels']:.0%} | on approximate labels: {summary['hit_approximate_labels']:.0%}")

    unjudged = {r["id"]: r["unjudged_in_top5"] for r in rows if r["unjudged_in_top5"]}
    if unjudged:
        print(f"\n⚠️  New chunks in the top 5 that have never been judged (judge them, then re-run): {unjudged}")

    # Save everything, tagged with the index version, so runs can be compared later
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_DIR / f"retrieval_{label}_{args.split}.json"
    out.write_text(json.dumps({
        "run": label, "mode": args.mode, "split": args.split, "date": datetime.now().isoformat(timespec="seconds"),
        "index": json.loads(INDEX_INFO_FILE.read_text()), "summary": summary, "questions": rows,
    }, indent=2))
    print(f"\nSaved -> {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
