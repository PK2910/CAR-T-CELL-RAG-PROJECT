"""
All exam questions in one list: the hand-built answer key (questions.json) plus, if it has been made,
every generated set (eval/synthetic*.json, each with its own split name). Scripts then pick a split with --split.
"""

import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
QUESTIONS_FILE = ROOT / "eval/questions.json"
EVAL_DIR = ROOT / "eval"


def load_questions() -> list[dict]:
    questions = json.loads(QUESTIONS_FILE.read_text())
    for f in sorted(EVAL_DIR.glob("synthetic*.json")):  # each generated set has its own split name
        questions += json.loads(f.read_text())
    # The answer key names original chunk ids; a section-based index has other chunks, so translate the key
    if os.environ.get("RAG_INDEX", "text").startswith("section"):
        from section_labels import translate
        from index_config import CHUNKS_FILE
        questions = translate(questions, json.loads(CHUNKS_FILE.read_text(encoding="utf-8")))
    return questions


# --split choices: the hand-built splits, "practice" (dev + test), every generated set, and "all"
SPLITS = (["dev", "test", "practice", "final", "holdout"]
          + sorted({q["split"] for q in load_questions()} - {"dev", "test", "final", "holdout"}) + ["all"])
