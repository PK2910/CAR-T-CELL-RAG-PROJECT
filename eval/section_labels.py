"""
Translate the answer key (relevant_chunks, written for the original chunks) to another chunking,
e.g. the section-based chunks, so the same search tests can run on a new index.

For every original answer chunk of a question:
  - if the question has answer rules (label_rules.py): find the sentences in that chunk that match a rule;
    a new chunk counts as relevant if it contains one of those sentences             (precise)
  - otherwise (labels judged by reading, or generated questions with one source chunk): a new chunk counts as
    relevant if it contains at least 30% of the original chunk's sentences            (approximate)
Only new chunks from the same paper are considered. The original text is the same in both chunkings, only the cut
points differ, so sentences can be matched exactly.

Check the translation (free):  python eval/section_labels.py
  translates the key onto the ORIGINAL chunks (should give back the original labels) and onto the section chunks.
"""

import json
import re
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "eval"))
from label_rules import RULES  # noqa: E402

ORIGINAL_CHUNKS = ROOT / "data/processed/chunks.json"
SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9(\[])|\n+")
MIN_SENTENCE = 15    # ignore fragments shorter than this (headings, "Table 1.")
MIN_SHARE = 0.30     # approximate mapping: share of the original chunk's sentences a new chunk must contain


def norm(text: str) -> str:
    return " ".join(text.split())


def sentences(text: str) -> list[str]:
    return [norm(s) for s in SENTENCE_SPLIT.split(text) if len(s.strip()) >= MIN_SENTENCE]


def answer_sentences(chunk_text: str, rules: list[dict]) -> list[str]:
    """Sentences of the chunk that satisfy one of the question's answer rules."""
    found = []
    for rule in rules:
        if all(re.search(rx, chunk_text, re.I) for rx in rule.get("context", [])):
            found += [norm(s) for s in SENTENCE_SPLIT.split(chunk_text)
                      if all(re.search(rx, s, re.I) for rx in rule["sentence"])]
    return [s for s in found if len(s) >= MIN_SENTENCE]


def translate(questions: list[dict], new_chunks: list[dict]) -> list[dict]:
    """Copies of the questions with relevant_chunks rewritten to new chunk ids."""
    original = {c["chunk_id"]: c for c in json.loads(ORIGINAL_CHUNKS.read_text(encoding="utf-8"))}
    by_paper = {}
    for c in new_chunks:
        by_paper.setdefault(c["pmcid"], []).append((c["chunk_id"], norm(c["text"])))

    out = []
    for q in questions:
        new_ids, approximate = set(), False
        for cid in q["relevant_chunks"]:
            old = original.get(cid)
            if old is None:  # e.g. a table chunk from the "plus" index: no text-only equivalent
                continue
            candidates = by_paper.get(old["pmcid"], [])
            key = answer_sentences(old["text"], RULES[q["id"]]) if q["id"] in RULES else []
            if key:
                hits = {nid for nid, text in candidates if any(s in text for s in key)}
            else:
                approximate = True
                sents = sentences(old["text"])
                shared = {nid: sum(s in text for s in sents) for nid, text in candidates}
                hits = {nid for nid, n in shared.items() if sents and n / len(sents) >= MIN_SHARE}
                if not hits and shared and max(shared.values()) > 0:  # tiny original chunk: take the best overlap
                    hits = {max(shared, key=shared.get)}
            new_ids |= hits
        out.append({**q, "relevant_chunks": sorted(new_ids),
                    "label_quality": "approximate" if approximate and q["label_quality"] == "precise"
                    else q["label_quality"]})
    return out


def main():
    from question_sets import load_questions
    questions = [q for q in load_questions() if q["relevant_chunks"]]
    original = json.loads(ORIGINAL_CHUNKS.read_text(encoding="utf-8"))
    section = json.loads((ROOT / "data/processed/chunks_section.json").read_text(encoding="utf-8"))

    for name, chunks in (("original chunks (should match the key)", original), ("section chunks", section)):
        mapped = {q["id"]: set(q["relevant_chunks"]) for q in translate(questions, chunks)}
        print(f"\nTranslated onto {name}:")
        if chunks is original:
            kept = [len(mapped[q["id"]] & set(q["relevant_chunks"])) / len(q["relevant_chunks"]) for q in questions]
            extra = [len(mapped[q["id"]] - set(q["relevant_chunks"])) for q in questions]
            print(f"  original labels kept: {statistics.mean(kept):.0%} on average "
                  f"(all kept for {sum(k == 1 for k in kept)} of {len(questions)} questions)")
            print(f"  extra labels added: {sum(extra)} in total (neighbouring chunks that share the answer sentence)")
        lost = [q["id"] for q in questions if not mapped[q["id"]]]
        ratio = [len(mapped[q["id"]]) / len(q["relevant_chunks"]) for q in questions]
        print(f"  questions left with no answer chunk: {len(lost)} {lost[:10]}")
        print(f"  answer chunks per question, new / old: median {statistics.median(ratio):.2f}")


if __name__ == "__main__":
    main()
