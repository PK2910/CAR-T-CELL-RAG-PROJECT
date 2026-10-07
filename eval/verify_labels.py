"""
Answer-key check: show the chunks that the keyword rules (Method A) marked relevant
AND that our search ranks in the top 20, so each can be confirmed by reading.
These are the labels that actually affect the search scores (Hit@k, MRR, precision).
Corrections are saved by hand in eval/label_corrections.json.

Run:  python eval/verify_labels.py q04 q05 ...
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "eval"))
from method_b import key_sentences  # noqa: E402
from retrieve import retrieve  # noqa: E402

QUESTIONS_FILE = ROOT / "eval/questions.json"
JUDGMENTS_FILE = ROOT / "eval/method_b_judgments.json"
DEPTH = 20


def main():
    questions = {q["id"]: q for q in json.loads(QUESTIONS_FILE.read_text())}
    judged_by_b = json.loads(JUDGMENTS_FILE.read_text())
    for qid in sys.argv[1:]:
        q = questions[qid]
        relevant = set(q["relevant_chunks"])
        confirmed = {cid for cid, j in judged_by_b.get(qid, {}).items() if j["relevant"]}  # already read in Method B
        print(f"\n{'=' * 100}\n{qid}: {q['question']}")
        for rank, r in enumerate(retrieve(q["question"], k=DEPTH), 1):
            if r["chunk_id"] not in relevant:
                continue
            if r["chunk_id"] in confirmed:
                print(f"  #{rank:<2} {r['chunk_id']}  (already confirmed by reading in Method B)")
                continue
            print(f"  #{rank:<2} {r['chunk_id']} {r['first_author']} {r['year']} ({r['article_type']})")
            print(f"       {key_sentences(r['text'], q['question'])}")


if __name__ == "__main__":
    main()
