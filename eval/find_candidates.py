"""
Labelling helper: for each exam question, find candidate passages with two
independent methods, so a human (or Claude, reading them) can judge which
papers really answer it.

  A) keyword matching: a chunk matches if it contains at least one term
     from EVERY keyword group (case-insensitive)
  B) our own meaning search, but wider (top 30 instead of top 5)

Using both avoids labelling only what our search system already finds.

Run:  python eval/find_candidates.py q01 q02 ...   (no ids = all questions)
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from retrieve import retrieve  # noqa: E402

QUESTIONS_FILE = ROOT / "eval/questions_draft.json"
CHUNKS_FILE = ROOT / "data/processed/chunks.json"
WIDE_K = 30         # how many results method B looks at
SHOW_MAX = 12       # candidates printed per question
SNIPPET_CHARS = 260


def keyword_hits(chunk: dict, groups: list[list[str]]) -> int | None:
    """Position of the first term of the first group if every group matches, else None."""
    text = chunk["text"].lower()
    first_pos = None
    for g, group in enumerate(groups):
        positions = [text.find(term.lower()) for term in group if term.lower() in text]
        if not positions:
            return None
        if g == 0:
            first_pos = min(positions)
    return first_pos


def snippet(text: str, pos: int) -> str:
    start = max(0, pos - SNIPPET_CHARS // 3)
    return text[start:start + SNIPPET_CHARS].replace("\n", " ")


def main():
    questions = json.loads(QUESTIONS_FILE.read_text())
    chunks = json.loads(CHUNKS_FILE.read_text())
    wanted = set(sys.argv[1:])

    for q in questions:
        if wanted and q["id"] not in wanted:
            continue
        kw = {c["chunk_id"]: (c, pos) for c in chunks
              if (pos := keyword_hits(c, q["keywords"])) is not None}
        sem = {r["chunk_id"]: (rank, r) for rank, r in enumerate(retrieve(q["question"], k=WIDE_K), 1)}

        print(f"\n{'=' * 100}\n{q['id']} [{q['type']}] {q['question']}")
        print(f"  A) keyword: {len(kw)} chunks in {len({c['pmcid'] for c, _ in kw.values()})} papers"
              f" | B) meaning search top {WIDE_K}: {len({r['pmcid'] for _, r in sem.values()})} papers")

        # Show chunks found by BOTH methods first, then by meaning rank, then keyword-only
        ids = ([i for i in sem if i in kw] + [i for i in sem if i not in kw] + [i for i in kw if i not in sem])
        for cid in ids[:SHOW_MAX]:
            c = kw[cid][0] if cid in kw else sem[cid][1]
            tag = ("K+" if cid in kw else "  ") + (f"S{sem[cid][0]:<2}" if cid in sem else "K  ")
            pos = kw[cid][1] if cid in kw else 0
            print(f"  [{tag}] {cid} {c['first_author']} {c['year']} ({c['article_type']})")
            print(f"        {snippet(c['text'], pos)}")
        extra = [i for i in ids[SHOW_MAX:] if i in kw]
        if extra:
            print(f"  ...+{len(extra)} more keyword chunks in papers: "
                  f"{sorted({kw[i][0]['pmcid'] for i in extra})}")


if __name__ == "__main__":
    main()
