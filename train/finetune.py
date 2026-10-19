"""
Step 3 of embedding fine-tuning: contrastive training of bge-small on our (question, right page,
close-but-wrong page) triplets.

Loss: (Cached)MultipleNegativesRankingLoss. For each question, the right page must score higher than
  - its close-but-wrong page (the hard negative), and
  - every other page in the same batch (in-batch negatives, free).
The "Cached" version computes the same loss in mini-batches of 8, so a batch of 32 fits in the Mac GPU memory.
Questions get the same bge query prefix that search uses, so training matches how the model is used.

10% of the triplets are held back (never trained on) to measure the change on unseen questions.
The real verdict comes later, from the search test on our evaluation sets with a re-embedded index.

Free: runs on the Mac (M1 GPU if available), no API calls.
Input:  train/triplets.jsonl      Output: train/models/bge-small-cart/

Run:  python train/finetune.py
"""

import json
import random
import sys
from pathlib import Path

from datasets import Dataset
from sentence_transformers import (SentenceTransformer, SentenceTransformerTrainer,
                                   SentenceTransformerTrainingArguments)
from sentence_transformers.sentence_transformer import losses
from sentence_transformers.evaluation import InformationRetrievalEvaluator
from sentence_transformers.training_args import BatchSamplers

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from embed import CHUNKS_FILE, EMBEDDING_MODEL, QUERY_PREFIX  # noqa: E402  (text index; base model)

TRIPLETS_FILE = ROOT / "train/triplets.jsonl"
OUT_DIR = ROOT / "train/models/bge-small-cart"
SEED = 42
HOLDOUT_SHARE = 0.10
EPOCHS = 3
BATCH_SIZE = 32       # questions compared against each other per step
MINI_BATCH = 8        # rows processed at once inside the cached loss (memory only; same result)
LEARNING_RATE = 2e-5  # small: adjust the model gently instead of overwriting what it already knows


def evaluator(rows: list[dict], corpus: dict[str, str]) -> InformationRetrievalEvaluator:
    """Rank the held-back questions against ALL 5,354 text pages; report how high the right page lands."""
    return InformationRetrievalEvaluator(
        queries={str(i): QUERY_PREFIX + r["question"] for i, r in enumerate(rows)},
        corpus=corpus,
        relevant_docs={str(i): {r["positive_id"]} for i, r in enumerate(rows)},
        accuracy_at_k=[1, 5, 10], precision_recall_at_k=[5], mrr_at_k=[10], ndcg_at_k=[10], map_at_k=[10],
        name="heldout", show_progress_bar=False,
    )


def summary(scores: dict) -> str:
    return (f"right page at #1: {scores['heldout_cosine_accuracy@1']:.0%} | in top 5: "
            f"{scores['heldout_cosine_accuracy@5']:.0%} | in top 10: {scores['heldout_cosine_accuracy@10']:.0%} | "
            f"MRR@10: {scores['heldout_cosine_mrr@10']:.3f}")


def main():
    rows = [json.loads(line) for line in TRIPLETS_FILE.read_text().splitlines()]
    rows = [r for r in rows if r["negative"]]
    random.Random(SEED).shuffle(rows)
    n_hold = round(len(rows) * HOLDOUT_SHARE)
    held, train = rows[:n_hold], rows[n_hold:]
    corpus = {c["chunk_id"]: c["text"] for c in json.loads(CHUNKS_FILE.read_text(encoding="utf-8"))}
    print(f"Training on {len(train)} triplets, checking on {len(held)} held-back questions "
          f"against all {len(corpus):,} pages")

    # CPU on purpose: with 8 GB of memory the Mac GPU path ran out of memory and stalled while swapping.
    # bge-small is small enough to train on the CPU in minutes.
    device = "cpu"
    # "eager" attention: the faster SDPA routine has no dropout support on the Mac GPU during training
    model = SentenceTransformer(EMBEDDING_MODEL, device=device, model_kwargs={"attn_implementation": "eager"})
    check = evaluator(held, corpus)
    before = check(model)
    print(f"BEFORE: {summary(before)}")

    dataset = Dataset.from_dict({
        "anchor": [QUERY_PREFIX + r["question"] for r in train],
        "positive": [r["positive"] for r in train],
        "negative": [r["negative"] for r in train],
    })
    args = SentenceTransformerTrainingArguments(
        output_dir=str(ROOT / "train/models/checkpoints"),
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        warmup_ratio=0.1,
        batch_sampler=BatchSamplers.NO_DUPLICATES,  # the same page twice in a batch would be a false "wrong" answer
        save_strategy="no",
        logging_steps=5,
        report_to="none",
        seed=SEED,
    )
    trainer = SentenceTransformerTrainer(model=model, args=args, train_dataset=dataset,
                                         loss=losses.CachedMultipleNegativesRankingLoss(model, mini_batch_size=MINI_BATCH))
    trainer.train()

    after = check(model)
    print(f"\nBEFORE: {summary(before)}")
    print(f"AFTER:  {summary(after)}")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    model.save(str(OUT_DIR))
    (OUT_DIR / "training_info.json").write_text(json.dumps({
        "base_model": EMBEDDING_MODEL, "triplets": len(train), "heldout": len(held), "epochs": EPOCHS,
        "batch_size": BATCH_SIZE, "learning_rate": LEARNING_RATE, "loss": "CachedMultipleNegativesRankingLoss", "mini_batch": MINI_BATCH,
        "heldout_before": before, "heldout_after": after,
    }, indent=2))
    print(f"Saved fine-tuned model -> {OUT_DIR}")


if __name__ == "__main__":
    main()
