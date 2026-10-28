"""
Which search index to use. Several versions exist side by side, so they can be compared:

  "text"  the baseline: paper text only            (chunks.json      -> Qdrant "cart_all_baseline")
  "plus"  text + tables + figure captions           (chunks_plus.json -> Qdrant "cart_all_plus")
  "tuned" text only, embedded with the FINE-TUNED   (chunks.json      -> Qdrant "cart_all_tuned")
          bge-small from train/ (experiment: does contrastive fine-tuning improve search?)
  "section"      section-based chunks              (chunks_section.json -> Qdrant "cart_all_section")
  "section_hdr"  section-based chunks, searched WITH their contextual header
                                                   (chunks_section.json -> Qdrant "cart_all_section_hdr")
  "child"      parent-child: the search reads small CHILD chunks (1-3 sentences, chunks_children.json) and returns
               their PARENT chunks (chunks.json), which Claude reads. No header.
  "child_sec"  same, each child searched with "Section: <name>"
  "child_hdr"  same, each child searched with "Paper: <title> (<author>, <year>) / Section: <name>"

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
    "section": {"chunks": ROOT / "data/processed/chunks_section.json", "collection": "cart_all_section",
                "info": ROOT / "data/processed/index_info_section.json", "model": BASE_MODEL},
    "section_hdr": {"chunks": ROOT / "data/processed/chunks_section.json", "collection": "cart_all_section_hdr",
                    "info": ROOT / "data/processed/index_info_section_hdr.json", "model": BASE_MODEL,
                    "header": "header"},
    **{name: {"chunks": ROOT / "data/processed/chunks.json", "units": ROOT / "data/processed/chunks_children.json",
              "collection": f"cart_all_{name}", "info": ROOT / f"data/processed/index_info_{name}.json",
              "model": BASE_MODEL, "header": header}
       for name, header in (("child", None), ("child_sec", "section_header"), ("child_hdr", "header"))},
}

INDEX = os.environ.get("RAG_INDEX", "text")
if INDEX not in INDEXES:
    raise ValueError(f"RAG_INDEX must be one of {list(INDEXES)}, got {INDEX!r}")

CHUNKS_FILE = INDEXES[INDEX]["chunks"]
COLLECTION = INDEXES[INDEX]["collection"]
INFO_FILE = INDEXES[INDEX]["info"]
MODEL = INDEXES[INDEX]["model"]  # the embedding model this index was built with (must match at search time)
USE_HEADER = INDEXES[INDEX].get("header")  # None, or the field holding the header the search reads
# What the search indexes: the chunks themselves, or (parent-child) their children
UNITS_FILE = INDEXES[INDEX].get("units", CHUNKS_FILE)
PARENT_CHILD = UNITS_FILE != CHUNKS_FILE


def search_text(unit: dict) -> str:
    """The text the vector and keyword indexes read. Claude is always shown chunk["text"] (plus its labels)."""
    return unit[USE_HEADER] + unit["text"] if USE_HEADER else unit["text"]
