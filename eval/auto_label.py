"""
Apply the answer rules (label_rules.py) to every chunk and report, per question,
which chunks/papers contain an answer, plus a few matching sentences to read.

Run:  python eval/auto_label.py [q01 q02 ...] [--samples N]
"""

import json
import random
import re
import sys
from pathlib import Path

from label_rules import RULES

ROOT = Path(__file__).resolve().parent.parent
CHUNKS_FILE = ROOT / "data/processed/chunks.json"
SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9(\[])|\n+")


def matching_sentence(chunk: dict, rules: list[dict]) -> str | None:
    """The first sentence that satisfies one of the question's rules, or None."""
    text = chunk["text"]
    for rule in rules:
        if not all(re.search(rx, text, re.I) for rx in rule.get("context", [])):
            continue
        for sentence in SENTENCE_SPLIT.split(text):
            if all(re.search(rx, sentence, re.I) for rx in rule["sentence"]):
                return sentence
    return None


def label(chunks: list[dict], qid: str) -> dict[str, str]:
    """chunk_id -> matching sentence, for every chunk that answers question qid."""
    return {c["chunk_id"]: s for c in chunks if (s := matching_sentence(c, RULES[qid]))}


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    samples = int(sys.argv[sys.argv.index("--samples") + 1]) if "--samples" in sys.argv else 4
    args = [a for a in args if not a.isdigit()]
    chunks = json.loads(CHUNKS_FILE.read_text())
    by_id = {c["chunk_id"]: c for c in chunks}
    random.seed(0)

    for qid in args or RULES:
        hits = label(chunks, qid)
        papers = sorted({by_id[cid]["pmcid"] for cid in hits})
        print(f"\n{qid}: {len(hits)} chunks in {len(papers)} papers")
        for cid in random.sample(sorted(hits), min(samples, len(hits))):
            c = by_id[cid]
            print(f"   {cid} {c['first_author']} {c['year']} ({c['article_type']}): {hits[cid][:240]}")


if __name__ == "__main__":
    main()
