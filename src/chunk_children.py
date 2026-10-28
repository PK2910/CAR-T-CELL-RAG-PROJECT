"""
Parent-child chunking, step 1: cut every chunk (the PARENT, which Claude reads) into small CHILD chunks of 1-3
sentences (which the search reads). A child never crosses a paragraph, so it stays on one topic.

Each child keeps:
  parent_id / parent_pos   which parent it came from (its position in chunks.json)
  section                  the section its sentences come from
  header                   contextual header: "Paper: <title> (<author>, <year>) / Section: <name>"
  section_header           shorter header: "Section: <name>"
Which header (if any) the search reads is set per index in index_config.py.

Input:  data/processed/chunks.json, data/processed/papers.json
Output: data/processed/chunks_children.json

Run:  python src/chunk_children.py
Then: RAG_INDEX=child python src/embed.py   (also child_sec, child_hdr)
"""

import json
import re
import statistics

from chunk import OUT_FILE as PARENTS_FILE, PAPERS_FILE, ROOT, n_tokens

OUT_FILE = ROOT / "data/processed/chunks_children.json"
# Sentence end, also when citation numbers follow the period: "...(50% severe).10,24 Risk predictors..."
SENTENCE_END = re.compile(r"[.!?](?:\d+(?:[,–-]\d+)*)?\s+(?=[A-Z(\[])")
MAX_SENTENCES = 3
MAX_TOKENS = 100       # a child grows sentence by sentence up to this size (a longer single sentence stays alone)
MIN_CHARS = 25         # fragments shorter than this ("Table 1.", a heading) join the previous child


def norm(text: str) -> str:
    return " ".join(text.split())


def sentence_groups(paragraph: str) -> list[str]:
    """Consecutive sentences of one paragraph, packed into children of up to MAX_SENTENCES / MAX_TOKENS."""
    cuts = [0] + [m.end() for m in SENTENCE_END.finditer(paragraph)] + [len(paragraph)]
    sentences = [s for a, b in zip(cuts, cuts[1:]) if (s := paragraph[a:b].strip())]
    groups, current = [], []
    for s in sentences:
        if current and (len(current) == MAX_SENTENCES or n_tokens(" ".join(current + [s])) > MAX_TOKENS):
            groups.append(" ".join(current))
            current = []
        current.append(s)
    if current:
        groups.append(" ".join(current))
    merged = []
    for g in groups:
        if merged and len(g) < MIN_CHARS:
            merged[-1] += " " + g
        else:
            merged.append(g)
    return merged


def main():
    parents = json.loads(PARENTS_FILE.read_text(encoding="utf-8"))
    papers = {p["pmcid"]: p for p in json.loads(PAPERS_FILE.read_text(encoding="utf-8"))}
    # Each paper's paragraphs with their section, to find the section of every child
    sections_of = {}
    for pmcid, p in papers.items():
        paras = ([("Abstract", p["abstract"])] if p["abstract"] else []) + [
            (sec["title"], para) for sec in p["sections"] for para in sec["paragraphs"]]
        sections_of[pmcid] = [(title, norm(para)) for title, para in paras]

    children = []
    for pos, parent in enumerate(parents):
        first, last = parent["paragraph_range"]
        candidates = sections_of[parent["pmcid"]][first:last + 1]
        n = 0
        for paragraph in parent["text"].split("\n\n"):
            for text in sentence_groups(paragraph):
                section = next((title for title, para in candidates if norm(text)[:80] in para),
                               parent["sections"][0])
                children.append({
                    "chunk_id": f"{parent['chunk_id']}_c{n:02d}",
                    "parent_id": parent["chunk_id"],
                    "parent_pos": pos,
                    "text": text,
                    "n_tokens": n_tokens(text),
                    "section": section,
                    "header": f"Paper: {parent['title']} ({parent['first_author']}, {parent['year']})\n"
                              f"Section: {section}\n\n",
                    "section_header": f"Section: {section}\n\n",
                    "pmcid": parent["pmcid"],
                })
                n += 1
    OUT_FILE.write_text(json.dumps(children, indent=2, ensure_ascii=False), encoding="utf-8")

    sizes = [c["n_tokens"] for c in children]
    per_parent = [0] * len(parents)
    for c in children:
        per_parent[c["parent_pos"]] += 1
    print(f"Children: {len(children):,} from {len(parents):,} parents -> {OUT_FILE}")
    print(f"Tokens per child:    median {statistics.median(sizes):.0f}, min {min(sizes)}, max {max(sizes)}")
    print(f"Children per parent: median {statistics.median(per_parent):.0f}, max {max(per_parent)}")
    print(f"Parents without children: {sum(n == 0 for n in per_parent)} (must be 0)")


if __name__ == "__main__":
    main()
