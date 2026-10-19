"""
Step 4 of embedding fine-tuning: compare the original and the fine-tuned model on the 48 held-back
training questions (never trained on). Each question is ranked against all 5,354 text pages.

This is a quick sanity check only. The real verdict is the search test on our evaluation sets,
after re-embedding the pages with the fine-tuned model.

Free: local, no API calls.
Run:  python train/evaluate_heldout.py
"""

import json
import sys
from pathlib import Path

import torch
from sentence_transformers import SentenceTransformer
from sentence_transformers.evaluation import InformationRetrievalEvaluator

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from embed import CHUNKS_FILE, EMBEDDING_MODEL, QUERY_PREFIX  # noqa: E402

HELDOUT_FILE = ROOT / "train/heldout_split.jsonl"
TUNED_MODEL = ROOT / "train/models/bge-small-cart"


def main():
    held = [json.loads(line) for line in HELDOUT_FILE.read_text().splitlines()]
    corpus = {c["chunk_id"]: c["text"] for c in json.loads(CHUNKS_FILE.read_text(encoding="utf-8"))}
    check = InformationRetrievalEvaluator(
        queries={str(i): QUERY_PREFIX + r["question"] for i, r in enumerate(held)},
        corpus=corpus,
        relevant_docs={str(i): {r["positive_id"]} for i, r in enumerate(held)},
        accuracy_at_k=[1, 5, 10], precision_recall_at_k=[5], mrr_at_k=[10], ndcg_at_k=[10], map_at_k=[10],
        name="heldout", show_progress_bar=False,
    )
    device = "mps" if torch.backends.mps.is_available() else "cpu"  # reading only (no training): fits in memory
    print(f"{len(held)} held-back questions vs all {len(corpus):,} pages\n")
    print(f"{'model':12} {'#1':>6} {'top 5':>7} {'top 10':>7} {'MRR@10':>8}")
    results = {}
    for name, path in [("original", EMBEDDING_MODEL), ("fine-tuned", str(TUNED_MODEL))]:
        s = check(SentenceTransformer(path, device=device))
        results[name] = s
        print(f"{name:12} {s['heldout_cosine_accuracy@1']:>6.0%} {s['heldout_cosine_accuracy@5']:>7.0%} "
              f"{s['heldout_cosine_accuracy@10']:>7.0%} {s['heldout_cosine_mrr@10']:>8.3f}")
    (TUNED_MODEL / "heldout_eval.json").write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
