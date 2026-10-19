"""
Box 3b: CHUNK TABLES + FIGURE CAPTIONS (the "plus" index)
Turn every table and every figure caption into chunks, and save them AFTER the unchanged text
chunks, so text chunks keep their positions and ids (and the answer key stays valid).

  table  -> "<paper title>\n<Table 2>: <caption>\n<header row>\n<rows...>\n<footnote>"
            big tables are split by rows; every piece repeats the title, label and header row,
            so each piece still says what its numbers mean
  figure -> "<paper title>\n<Fig. 3>: <caption>"
            the XML has only the caption and the image's file name (kept as a label);
            the image itself is not downloaded or read

Input:  data/processed/papers.json, data/processed/chunks.json
Output: data/processed/chunks_plus.json  (text chunks + table chunks + figure chunks)

Run:  python src/chunk_extras.py   then   RAG_INDEX=plus python src/embed.py
"""

import json
import statistics
from collections import Counter

from chunk import CHUNK_TOKENS, MODEL_MAX_TOKENS, merge, n_tokens, pieces
from index_config import INDEXES

PAPERS_FILE = INDEXES["text"]["chunks"].parent / "papers.json"
TEXT_CHUNKS_FILE = INDEXES["text"]["chunks"]
OUT_FILE = INDEXES["plus"]["chunks"]


def labels(paper: dict, chunk_type: str, section: str) -> dict:
    """The same labels text chunks carry, plus the chunk type."""
    return {
        "pmcid": paper["pmcid"], "pmid": paper["pmid"], "doi": paper["doi"], "title": paper["title"],
        "first_author": paper["first_author"], "journal": paper["journal"], "year": paper["year"],
        "article_type": paper["article_type"],
        "sections": [section],
        "paragraph_range": None,
        "chunk_type": chunk_type,
    }


def fit(text: str) -> list[str]:
    """Split text that is still too long (a huge caption or a single giant row) with the same
    recursive splitter as the paper text."""
    if n_tokens(text) <= CHUNK_TOKENS:
        return [text]
    return [text[s:e].strip() for s, e in merge(text, pieces(text, 0, len(text)))]


def table_texts(paper: dict, table: dict) -> list[str]:
    label = table["label"] or "Table"
    head = f"{paper['title']}\n{label}: {table['caption']}".strip()
    lines = table["markdown"].split("\n")
    header, rows = lines[:2], lines[2:]
    texts, current = [], []
    for row in rows:
        if current and n_tokens("\n".join([head, *header, *current, row])) > CHUNK_TOKENS:
            texts.append("\n".join([head, *header, *current]))
            current = []
        current.append(row)
    last = "\n".join([head, *header, *current])
    if table["footnote"]:
        with_note = f"{last}\nNote: {table['footnote']}"
        if n_tokens(with_note) <= CHUNK_TOKENS:
            last = with_note
        else:
            texts.append(last)
            last = f"{head}\nNote: {table['footnote']}"
    texts.append(last)
    return [piece for text in texts for piece in fit(text)]


def figure_texts(paper: dict, figure: dict) -> list[str]:
    label = figure["label"] or "Figure"
    return fit(f"{paper['title']}\n{label}: {figure['caption']}")


def main():
    papers = json.loads(PAPERS_FILE.read_text(encoding="utf-8"))
    text_chunks = json.loads(TEXT_CHUNKS_FILE.read_text(encoding="utf-8"))

    extras = []
    for paper in papers:
        for t_i, table in enumerate(t for t in paper["tables"] if t["markdown"]):  # image-only tables have no text
            for part, text in enumerate(table_texts(paper, table)):
                extras.append({"chunk_id": f"{paper['pmcid']}_T{t_i + 1:02d}_{part}", "text": text,
                               "n_tokens": n_tokens(text), **labels(paper, "table", table["label"] or "Table")})
        for f_i, figure in enumerate(f for f in paper["figures"] if f["caption"]):
            for part, text in enumerate(figure_texts(paper, figure)):
                extras.append({"chunk_id": f"{paper['pmcid']}_F{f_i + 1:02d}_{part}", "text": text,
                               "n_tokens": n_tokens(text), **labels(paper, "figure", figure["label"] or "Figure"),
                               "image_file": figure["graphic"]})

    # Text chunks first, unchanged: list position = Qdrant point id = BM25 position, as before
    chunks = [{**c, "chunk_type": "text"} for c in text_chunks] + extras
    OUT_FILE.write_text(json.dumps(chunks, indent=2, ensure_ascii=False), encoding="utf-8")

    # ---------- Report ----------
    types = Counter(c["chunk_type"] for c in chunks)
    print(f"Chunks: {len(chunks):,} -> {OUT_FILE}  ({', '.join(f'{n:,} {t}' for t, n in types.items())})")
    for t in ("table", "figure"):
        sizes = [c["n_tokens"] for c in extras if c["chunk_type"] == t]
        print(f"  {t:6} tokens per chunk: median {statistics.median(sizes):.0f}, min {min(sizes)}, max {max(sizes)}")
    print(f"Over model limit: {sum(1 for c in extras if c['n_tokens'] > MODEL_MAX_TOKENS - 2)} chunks (must be 0)")
    print(f"Duplicate ids:    {len(chunks) - len({c['chunk_id'] for c in chunks})} (must be 0)")


if __name__ == "__main__":
    main()
