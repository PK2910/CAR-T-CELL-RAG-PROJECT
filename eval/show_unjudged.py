"""
After a new search run, show the top-5 chunks that have never been judged, so each can be
read and recorded in eval/method_b_judgments.json before trusting precision/recall/MRR.

Run:  python eval/show_unjudged.py eval/results/retrieval_<label>_dev.json
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "eval"))
sys.path.insert(0, str(ROOT / "src"))
from index_config import CHUNKS_FILE  # noqa: E402
from method_b import key_sentences  # noqa: E402


def main():
    run = json.loads(Path(sys.argv[1]).read_text())
    chunks = {c["chunk_id"]: c for c in json.loads(CHUNKS_FILE.read_text())}
    questions = {q["id"]: q for q in json.loads((ROOT / "eval/questions.json").read_text())}
    for row in run["questions"]:
        if not row["unjudged_in_top5"]:
            continue
        q = questions[row["id"]]
        print(f"\n{'=' * 90}\n{row['id']}: {q['question']}")
        for cid in row["unjudged_in_top5"]:
            c = chunks[cid]
            print(f"  {cid} {c['first_author']} {c['year']} ({c['article_type']}) {c['sections']}")
            print(f"     {key_sentences(c['text'], q['question'])}")


if __name__ == "__main__":
    main()
