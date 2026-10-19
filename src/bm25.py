"""
BM25 keyword search over the same chunks as the vector index.

Word splitting is ours to control (unlike the embedding model, which lowercases everything):
  - abbreviations stay EXACT and in capitals:  "ALL", "T-ALL", "CD22", "ZUMA-3", "ICANS"
    so the leukemia "ALL" never matches the ordinary word "all"
  - other words are lowercased:                "Remission" -> "remission"
  - hyphenated terms are kept whole AND split:  "T-ALL" -> "T-ALL", "T", "ALL"
    so "T-cell ALL" in a question still matches "T-ALL" in a chunk through "ALL"
  - very common English words are dropped ("the", "of", ...)
"""

import json
import re
from functools import lru_cache

from rank_bm25 import BM25Okapi

from index_config import CHUNKS_FILE  # same chunks as the vector index (text, or text + tables + figures)

TOKEN = re.compile(r"[A-Za-z0-9]+(?:[-/][A-Za-z0-9]+)*")
ABBREVIATION = re.compile(r"^(?=.*[A-Z].*[A-Z]|.*[A-Z].*\d|.*\d.*[A-Z])[A-Za-z0-9/-]+$")  # 2+ capitals, or capitals + digits
STOPWORDS = set("""a an and are as at be been but by can could did do does for from had has have how if in into is it
its may might of on or our such than that the their them then there these they this those to was we were what when
where which while who why will with would also after before between both each more most other some""".split())


def normalize(token: str) -> str:
    return token if ABBREVIATION.match(token) else token.lower()


def tokenize(text: str) -> list[str]:
    tokens = []
    for raw in TOKEN.findall(text):
        parts = [raw] + (re.split(r"[-/]", raw) if re.search(r"[-/]", raw) else [])
        for part in parts:
            word = normalize(part)
            if len(word) > 1 and word.lower() not in STOPWORDS:  # single letters ("t" from CAR-T) are noise
                tokens.append(word)
    return tokens


@lru_cache(maxsize=1)
def load_bm25() -> BM25Okapi:
    """Build the BM25 index from chunks.json (a second or two; done once per run).
    k1=1.5 (how fast repeated words stop adding score), b=0.75 (how much long chunks are evened out)."""
    chunks = json.loads(CHUNKS_FILE.read_text(encoding="utf-8"))
    return BM25Okapi([tokenize(c["text"]) for c in chunks], k1=1.5, b=0.75)


def bm25_ranking(question: str) -> list[int]:
    """All chunk positions (= Qdrant point ids), best BM25 match first."""
    scores = load_bm25().get_scores(tokenize(question))
    return sorted(range(len(scores)), key=lambda i: -scores[i])
