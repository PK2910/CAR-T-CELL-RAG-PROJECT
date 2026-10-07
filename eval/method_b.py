"""
Method B for labelling: show our search's top-k passages that the keyword rules
(Method A) did NOT already mark relevant, so each can be judged by reading.
Judgments are saved by hand in eval/method_b_judgments.json and merged into the
answer key by build_questions.py.

Run:  python eval/method_b.py q01 q02 ...
"""

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from retrieve import retrieve  # noqa: E402

QUESTIONS_FILE = ROOT / "eval/questions.json"
TOP_K = 10
SHOW_SENTENCES = 3  # most question-related sentences shown per passage
STOPWORDS = set("what which how why who does do did is are was were the a an of in on for to with and or after "
                "before than their they some have has been be it its this that from by as at what's".split())
SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9(\[])|\n+")


def key_sentences(text: str, question: str) -> str:
    """The sentences sharing the most words with the question, in their original order."""
    q_words = {w for w in re.findall(r"[a-z0-9-]+", question.lower()) if w not in STOPWORDS and len(w) > 2}
    sentences = [s for s in SENTENCE_SPLIT.split(text) if s.strip()]
    scored = sorted(range(len(sentences)),
                    key=lambda i: -len(q_words & set(re.findall(r"[a-z0-9-]+", sentences[i].lower()))))
    keep = sorted(scored[:SHOW_SENTENCES])
    return " … ".join(sentences[i].strip()[:260] for i in keep)


def main():
    questions = {q["id"]: q for q in json.loads(QUESTIONS_FILE.read_text())}
    for qid in sys.argv[1:]:
        q = questions[qid]
        already = set(q["relevant_chunks"])
        print(f"\n{'=' * 100}\n{qid}: {q['question']}")
        for rank, r in enumerate(retrieve(q["question"], k=TOP_K), 1):
            if r["chunk_id"] in already:
                print(f"  #{rank:<2} {r['chunk_id']}  (already relevant via Method A)")
                continue
            print(f"  #{rank:<2} {r['chunk_id']} {r['first_author']} {r['year']} ({r['article_type']})")
            print(f"       {key_sentences(r['text'], q['question'])}")


if __name__ == "__main__":
    main()
