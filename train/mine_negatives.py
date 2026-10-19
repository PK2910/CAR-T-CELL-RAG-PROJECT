"""
Step 2 of embedding fine-tuning: add a CLOSE-BUT-WRONG page ("hard negative") to every training pair.

For each training question we run our current hybrid search and pick a page that ranks high (so it looks
similar to the question) but is very unlikely to contain the answer:
  - not the right page, and from a DIFFERENT paper (papers repeat their own results across pages)
  - ranked below the top 5 (the very top ranks are the most likely to secretly answer the question too)
  - never an evaluation page or its neighbour (training must not move the exam pages)

Free: runs our local search, no API calls.
Input:  train/pairs.jsonl     Output: train/triplets.jsonl  (question, positive page, negative page)

Run:  python train/mine_negatives.py
"""

import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from retrieve import hybrid_ranking, load_chunks  # noqa: E402  (text index: the model we are improving)

spec = importlib.util.spec_from_file_location("make_pairs", ROOT / "train/make_pairs.py")
make_pairs = importlib.util.module_from_spec(spec)
spec.loader.exec_module(make_pairs)

PAIRS_FILE = ROOT / "train/pairs.jsonl"
OUT_FILE = ROOT / "train/triplets.jsonl"
SKIP_TOP = 5         # don't take a negative from the top 5
SEARCH_DEPTH = 50    # look for a negative within the top 50


def main():
    chunks = load_chunks()
    by_pos = {i: c for i, c in enumerate(chunks)}
    excluded = make_pairs.with_neighbours(make_pairs.eval_chunks())
    pairs = [json.loads(line) for line in PAIRS_FILE.read_text().splitlines()]
    pairs = [p for p in pairs if p["question"]]
    text_of = {c["chunk_id"]: c["text"] for c in chunks}

    found, ranks = 0, []
    with OUT_FILE.open("w", encoding="utf-8") as out:
        for p in pairs:
            ranking = [pos for pos, _ in hybrid_ranking(p["question"])[:SEARCH_DEPTH]]
            ids = [by_pos[pos]["chunk_id"] for pos in ranking]
            pos_rank = ids.index(p["chunk_id"]) + 1 if p["chunk_id"] in ids else None
            negative = None
            for rank, pos in enumerate(ranking, 1):
                c = by_pos[pos]
                if (rank > SKIP_TOP and c["pmcid"] != p["pmcid"] and c["chunk_id"] not in excluded):
                    negative = c["chunk_id"]
                    break
            if negative:
                found += 1
            ranks.append(pos_rank)
            out.write(json.dumps({"question": p["question"], "positive": text_of[p["chunk_id"]],
                                  "negative": text_of[negative] if negative else None,
                                  "positive_id": p["chunk_id"], "negative_id": negative,
                                  "positive_rank_before": pos_rank}, ensure_ascii=False) + "\n")

    in5 = sum(1 for r in ranks if r and r <= 5)
    print(f"{len(pairs)} training pairs -> {OUT_FILE}")
    print(f"  close-but-wrong page found for {found} ({found / len(pairs):.0%})")
    print(f"  current search already ranks the right page in the top 5 for {in5} ({in5 / len(pairs):.0%}), "
          f"top 50 for {sum(1 for r in ranks if r)} ({sum(1 for r in ranks if r) / len(pairs):.0%})")


if __name__ == "__main__":
    main()
