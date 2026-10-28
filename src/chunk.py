"""
Box 3: CHUNK
Split each paper into ~400-token chunks with recursive splitting,
and attach labels (metadata) to every chunk for citations.

Input:  data/processed/papers.json
Output: data/processed/chunks.json           whole paper split as one text (chunks can cross sections)
        data/processed/chunks_section.json   --by-section: each section split on its own, plus a
                                             contextual header ("Paper: ..., Section: ...") per chunk

Run:  python src/chunk.py [--by-section]
"""

import bisect
import json
import re
import statistics
import sys
from pathlib import Path

from tokenizers import Tokenizer

ROOT = Path(__file__).resolve().parent.parent  # project folder, wherever we run from
PAPERS_FILE = ROOT / "data/processed/papers.json"
OUT_FILE = ROOT / "data/processed/chunks.json"
SECTION_OUT_FILE = ROOT / "data/processed/chunks_section.json"  # --by-section

# Count tokens with the SAME tokenizer the embedding model uses
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
MODEL_MAX_TOKENS = 512  # the model ignores anything beyond this
CHUNK_TOKENS = 400      # our target: leaves headroom under the limit
OVERLAP_TOKENS = 50     # repeat the end of one chunk at the start of the next

tokenizer = Tokenizer.from_pretrained(EMBEDDING_MODEL)

# Recursive splitting tries these in order: paragraph -> sentence -> word
SEPARATORS = [
    re.compile(r"\n\n"),                         # paragraph break
    re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9(\[])"),  # sentence end (approximate: "et al. 2018" can fool it)
    re.compile(r"\s+"),                          # word break
]


def n_tokens(text: str) -> int:
    return len(tokenizer.encode(text, add_special_tokens=False).ids)


def split_at(text: str, start: int, end: int, pattern: re.Pattern) -> list[tuple[int, int]]:
    """Cut text[start:end] after every match of the pattern; returns (start, end) positions."""
    cuts = [m.end() for m in pattern.finditer(text, start, end)]
    points = [start] + cuts + [end]
    return [(a, b) for a, b in zip(points, points[1:]) if b > a]


def hard_split(text: str, start: int, end: int) -> list[tuple[int, int]]:
    """Last resort for a single 'word' longer than a chunk: cut every CHUNK_TOKENS tokens."""
    offsets = tokenizer.encode(text[start:end], add_special_tokens=False).offsets
    return [(start + offsets[i][0], start + offsets[min(i + CHUNK_TOKENS, len(offsets)) - 1][1])
            for i in range(0, len(offsets), CHUNK_TOKENS)]


def pieces(text: str, start: int, end: int, level: int = 0) -> list[tuple[int, int]]:
    """RECURSIVE step: break text into pieces that each fit in a chunk,
    using the gentlest separator that works (paragraph, then sentence, then word)."""
    if n_tokens(text[start:end]) <= CHUNK_TOKENS:
        return [(start, end)]
    if level == len(SEPARATORS):
        return hard_split(text, start, end)
    # Split with this level's separator; any part still too big goes one level finer
    result = []
    for s, e in split_at(text, start, end, SEPARATORS[level]):
        result.extend(pieces(text, s, e, level + 1))
    return result


def trailing_sentences(text: str, start: int, end: int, budget: int) -> tuple[int, int, int] | None:
    """The last whole sentences of text[start:end] that fit in `budget` tokens."""
    taken_start, taken_tokens = None, 0
    for s, _ in reversed(split_at(text, start, end, SEPARATORS[1])):
        size = n_tokens(text[s:end])
        if size > budget:
            break
        taken_start, taken_tokens = s, size
    return (taken_start, end, taken_tokens) if taken_start is not None else None


def merge(text: str, spans: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """Pack consecutive pieces into chunks of up to CHUNK_TOKENS,
    starting each new chunk with ~OVERLAP_TOKENS from the end of the previous one."""
    sized = [(s, e, n_tokens(text[s:e])) for s, e in spans]
    chunks, current, total = [], [], 0
    for piece in sized:
        if current and total + piece[2] > CHUNK_TOKENS:
            chunks.append((current[0][0], current[-1][1]))
            # Overlap: carry over the last pieces; if a piece is too big to carry whole
            # (usually a full paragraph), carry just its last sentence(s) instead
            carried, carried_total = [], 0
            for prev in reversed(current):
                if carried_total + prev[2] <= OVERLAP_TOKENS:
                    carried.insert(0, prev)
                    carried_total += prev[2]
                    continue
                tail = trailing_sentences(text, prev[0], prev[1], OVERLAP_TOKENS - carried_total)
                if tail:
                    carried.insert(0, tail)
                    carried_total += tail[2]
                break
            current, total = carried, carried_total
            if total + piece[2] > CHUNK_TOKENS:  # no room for overlap next to a big piece
                current, total = [], 0
        current.append(piece)
        total += piece[2]
    if current:
        chunks.append((current[0][0], current[-1][1]))
    return chunks


def chunk_paper(paper: dict) -> list[dict]:
    # Join abstract + all section paragraphs into one text, remembering where
    # each paragraph starts so every chunk can be labelled with its section(s)
    paragraphs = ([("Abstract", paper["abstract"])] if paper["abstract"] else []) + [
        (sec["title"], p) for sec in paper["sections"] for p in sec["paragraphs"]
    ]
    starts, pos = [], 0
    for _, p in paragraphs:
        starts.append(pos)
        pos += len(p) + 2  # +2 for the "\n\n" joiner
    text = "\n\n".join(p for _, p in paragraphs)

    chunks = []
    for i, (s, e) in enumerate(merge(text, pieces(text, 0, len(text)))):
        first = bisect.bisect_right(starts, s) - 1
        last = bisect.bisect_right(starts, e - 1) - 1
        sections = list(dict.fromkeys(paragraphs[k][0] for k in range(first, last + 1)))
        chunk_text = text[s:e].strip()
        chunks.append({
            "chunk_id": f"{paper['pmcid']}_{i:04d}",
            "text": chunk_text,
            "n_tokens": n_tokens(chunk_text),
            # ----- labels (metadata) -----
            "pmcid": paper["pmcid"],
            "pmid": paper["pmid"],
            "doi": paper["doi"],
            "title": paper["title"],
            "first_author": paper["first_author"],
            "journal": paper["journal"],
            "year": paper["year"],
            "article_type": paper["article_type"],
            "sections": sections,  # more than one if the chunk crosses a section boundary
            "paragraph_range": [first, last],
        })
    return chunks


def header(paper: dict, section: str) -> str:
    """Contextual chunk header: tells the SEARCH which paper and section a chunk comes from.
    Claude already sees these labels, so the header is only added to the text that gets indexed."""
    return f"Paper: {paper['title']} ({paper['first_author']}, {paper['year']})\nSection: {section}\n\n"


def chunk_paper_by_section(paper: dict) -> list[dict]:
    """Section-based chunking: each section (the abstract counts as one) is split on its own,
    so no chunk mixes two sections. Same recursive splitting and overlap inside a section."""
    sections = ([("Abstract", [paper["abstract"]])] if paper["abstract"] else []) + [
        (sec["title"], sec["paragraphs"]) for sec in paper["sections"]
    ]
    chunks = []
    for title, paras in sections:
        text = "\n\n".join(paras)
        for s, e in merge(text, pieces(text, 0, len(text))):
            chunk_text = text[s:e].strip()
            chunks.append({
                "chunk_id": f"{paper['pmcid']}_s{len(chunks):04d}",
                "text": chunk_text,
                "n_tokens": n_tokens(chunk_text),
                "header": header(paper, title),
                # ----- labels (metadata) -----
                "pmcid": paper["pmcid"],
                "pmid": paper["pmid"],
                "doi": paper["doi"],
                "title": paper["title"],
                "first_author": paper["first_author"],
                "journal": paper["journal"],
                "year": paper["year"],
                "article_type": paper["article_type"],
                "sections": [title],
            })
    return chunks


def main():
    by_section = "--by-section" in sys.argv
    out_file = SECTION_OUT_FILE if by_section else OUT_FILE
    papers = json.loads(PAPERS_FILE.read_text(encoding="utf-8"))
    chunk_fn = chunk_paper_by_section if by_section else chunk_paper
    chunks = [c for paper in papers for c in chunk_fn(paper)]
    out_file.write_text(json.dumps(chunks, indent=2, ensure_ascii=False), encoding="utf-8")

    # ---------- Report ----------
    sizes = [c["n_tokens"] for c in chunks]
    per_paper = [sum(1 for c in chunks if c["pmcid"] == p["pmcid"]) for p in papers]
    crossing = sum(1 for c in chunks if len(c["sections"]) > 1)
    print(f"Chunks:            {len(chunks):,} from {len(papers)} papers -> {out_file}")
    if by_section:
        with_header = [n_tokens(c["header"] + c["text"]) for c in chunks]
        print(f"With header:       median {statistics.median(with_header):.0f} tokens, "
              f"over model limit: {sum(1 for n in with_header if n > MODEL_MAX_TOKENS - 2)} (must be 0)")
    print(f"Tokens per chunk:  median {statistics.median(sizes):.0f}, min {min(sizes)}, max {max(sizes)}")
    print(f"Over model limit:  {sum(1 for n in sizes if n > MODEL_MAX_TOKENS - 2)} chunks (must be 0)")
    print(f"Tiny (<50 tokens): {sum(1 for n in sizes if n < 50)} chunks")
    print(f"Chunks per paper:  median {statistics.median(per_paper):.0f}, min {min(per_paper)}, max {max(per_paper)}")
    print(f"Cross a section boundary: {crossing:,} ({crossing / len(chunks):.0%})")


if __name__ == "__main__":
    main()
