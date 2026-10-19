"""
Which search index to use. Several versions exist side by side, so they can be compared:

  "text"  the baseline: paper text only            (chunks.json      -> Qdrant "cart_all_baseline")
  "plus"  text + tables + figure captions           (chunks_plus.json -> Qdrant "cart_all_plus")
  "tuned" text only, embedded with the FINE-TUNED   (chunks.json      -> Qdrant "cart_all_tuned")
          bge-small from train/ (experiment: does contrastive fine-tuning improve search?)

Choose with an environment variable (default "text"), e.g.
  RAG_INDEX=plus python src/embed.py
  RAG_INDEX=plus python eval/eval_retrieval.py --mode hybrid --split practice --label plus_hybrid
"""

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BASE_MODEL = "BAAI/bge-small-en-v1.5"

INDEXES = {
    "text": {"chunks": ROOT / "data/processed/chunks.json", "collection": "cart_all_baseline",
             "info": ROOT / "data/processed/index_info.json", "model": BASE_MODEL},
    "plus": {"chunks": ROOT / "data/processed/chunks_plus.json", "collection": "cart_all_plus",
             "info": ROOT / "data/processed/index_info_plus.json", "model": BASE_MODEL},
    "tuned": {"chunks": ROOT / "data/processed/chunks.json", "collection": "cart_all_tuned",
              "info": ROOT / "data/processed/index_info_tuned.json",
              "model": str(ROOT / "train/models/bge-small-cart")},
}

INDEX = os.environ.get("RAG_INDEX", "text")
if INDEX not in INDEXES:
    raise ValueError(f"RAG_INDEX must be one of {list(INDEXES)}, got {INDEX!r}")

CHUNKS_FILE = INDEXES[INDEX]["chunks"]
COLLECTION = INDEXES[INDEX]["collection"]
INFO_FILE = INDEXES[INDEX]["info"]
MODEL = INDEXES[INDEX]["model"]  # the embedding model this index was built with (must match at search time)
