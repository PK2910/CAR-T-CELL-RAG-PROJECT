"""
Scan for trick phrases (the trick check in src/guardrails.py), free:
  1. every chunk of the papers: text hidden in a paper such as "ignore previous instructions" would reach Claude
     as a source (indirect prompt injection). Run it again after adding papers.
  2. every question in the test sets: real questions must not be blocked (q25 is meant to be: it is a safety test
     that starts with "Ignore your instructions").

Run:  python eval/trick_scan.py
"""

import glob
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "eval")]
from guardrails import TRICK_PATTERNS, is_trick  # noqa: E402
from index_config import CHUNKS_FILE  # noqa: E402
from question_sets import load_questions  # noqa: E402


def main():
    chunks = json.loads(CHUNKS_FILE.read_text())
    hits = [(c, m) for c in chunks for p in TRICK_PATTERNS for m in [re.search(p, c["text"], re.IGNORECASE)] if m]
    print(f"Chunks: {len(chunks)} scanned ({CHUNKS_FILE.name}), {len(hits)} with a trick phrase")
    for c, m in hits:
        print(f"  {c['chunk_id']}  {c['first_author']} {c['year']}: ...{c['text'][max(0, m.start() - 80):m.end() + 80]}...")

    questions = {q["question"]: q["id"] for q in load_questions()}
    for f in glob.glob(str(ROOT / "eval/results/answers_*.json")):  # questions from runs on older sets too
        questions.update({r["question"]: r["id"] for r in json.loads(Path(f).read_text())["answers"]})
    blocked = [(qid, q) for q, qid in questions.items() if is_trick(q)]
    print(f"\nQuestions: {len(questions)} checked, {len(blocked)} blocked by the trick check")
    for qid, q in blocked:
        print(f"  {qid}: {q}")


if __name__ == "__main__":
    main()
