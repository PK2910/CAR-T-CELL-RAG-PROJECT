"""
Box 5: RETRIEVE
Given a question, find the top-k most relevant chunks.
Other steps (generate, evaluate) import retrieve() from here.

Three search modes:
  "vector"  meaning only (bge-small + Qdrant) — the baseline
  "hybrid"  meaning + exact words: vector ranking and BM25 ranking merged with
            Reciprocal Rank Fusion, score = 1/(60 + vector rank) + 1/(60 + BM25 rank)
  "rerank"  hybrid's top 50, re-scored by a reranker (cross-encoder) that reads the
            question and each chunk TOGETHER and judges whether the chunk answers the question

Run:  python src/retrieve.py [--hybrid | --rerank] "your question here"
"""

import atexit
import json
import sys
from functools import lru_cache

import torch
from qdrant_client import QdrantClient
from sentence_transformers import CrossEncoder, SentenceTransformer

from bm25 import bm25_ranking
# Reuse the exact settings the index was built with
from embed import CHUNKS_FILE, COLLECTION, EMBEDDING_MODEL, QDRANT_PATH, QUERY_PREFIX
from index_config import PARENT_CHILD, UNITS_FILE  # parent-child: the indexes hold children, results are parents

TOP_K = 10  # pages given to Claude; was 5. 13 of 34 missing facts were on pages #6-10 (eval/RESULTS.md)
SEARCH_MODE = "hybrid"  # best configuration (see eval/RESULTS.md); "vector" = baseline, "rerank" = tested, not kept
RRF_K = 60              # the standard RRF constant: keeps #1 from counting far more than #2, #3...
# General reranker trained on question -> answer-passage pairs (does this passage ANSWER the question?).
# Replaced ncbi/MedCPT-Cross-Encoder, which was trained on search -> article clicks (is this article ABOUT
# the topic?) and did not improve answers (see eval/RESULTS.md, Run 3).
RERANKER_MODEL = "BAAI/bge-reranker-base"
RERANK_CANDIDATES = 50  # how many hybrid results the reranker re-reads (more = slower, wider net)


def device() -> str:
    return "mps" if torch.backends.mps.is_available() else "cpu"


@lru_cache(maxsize=1)  # load once, then reuse (loading takes a few seconds)
def load_model() -> SentenceTransformer:
    return SentenceTransformer(EMBEDDING_MODEL, device=device())


@lru_cache(maxsize=1)
def load_reranker() -> CrossEncoder:
    # Identity = keep the raw scores. The default squashes them into 0-1 with a sigmoid; large scores
    # all become 1.00 and can no longer be ranked (this happened with MedCPT).
    return CrossEncoder(RERANKER_MODEL, max_length=512, device=device(), activation_fn=torch.nn.Identity())


@lru_cache(maxsize=1)
def load_client() -> QdrantClient:
    client = QdrantClient(path=str(QDRANT_PATH))
    atexit.register(client.close)  # close cleanly before Python shuts down
    return client


@lru_cache(maxsize=1)
def load_chunks() -> list[dict]:
    """All chunks (the parents, in parent-child); list position = what the rankings return."""
    return json.loads(CHUNKS_FILE.read_text(encoding="utf-8"))


@lru_cache(maxsize=1)
def load_units() -> list[dict]:
    """What the indexes hold: the chunks, or their children. List position = Qdrant point id = BM25 position."""
    return load_chunks() if not PARENT_CHILD else json.loads(UNITS_FILE.read_text(encoding="utf-8"))


def to_parents(scored: list[tuple[int, float]]) -> list[tuple[int, float]]:
    """Parent-child: turn a ranking of children into a ranking of parents.
    Each parent takes the place (and score) of its best child; its other children are skipped."""
    if not PARENT_CHILD:
        return scored
    units, seen, parents = load_units(), set(), []
    for pos, score in scored:
        parent = units[pos]["parent_pos"]
        if parent not in seen:
            seen.add(parent)
            parents.append((parent, score))
    return parents


def vector_hits(question: str, limit: int) -> list[tuple[int, float]]:
    """(unit position, cosine similarity), closest in meaning first."""
    q_vec = load_model().encode(QUERY_PREFIX + question, normalize_embeddings=True)
    hits = load_client().query_points(COLLECTION, query=q_vec.tolist(), limit=limit, with_payload=False).points
    return [(h.id, h.score) for h in hits]


def vector_ranking(question: str) -> list[int]:
    """All unit positions (chunks, or children in parent-child), closest in meaning first."""
    return [pos for pos, _ in vector_hits(question, len(load_units()))]


def hybrid_ranking(question: str) -> list[tuple[int, float]]:
    """Reciprocal Rank Fusion of the vector and BM25 rankings: (chunk position, RRF score), best first.
    Both full rankings are fused; at our size (5,354 chunks, 25,657 children) that takes milliseconds.
    In parent-child the fusion runs over children, then each parent ranks by its best child."""
    scores = {}
    for ranking in (vector_ranking(question), bm25_ranking(question)):
        for rank, pos in enumerate(ranking, 1):
            scores[pos] = scores.get(pos, 0.0) + 1 / (RRF_K + rank)
    return to_parents(sorted(scores.items(), key=lambda x: -x[1]))


def reranked(question: str) -> list[tuple[int, float]]:
    """Hybrid's top RERANK_CANDIDATES chunks, re-sorted by the reranker's score (best first).
    The reranker scores each (question, chunk) pair by reading both together."""
    candidates = [pos for pos, _ in hybrid_ranking(question)[:RERANK_CANDIDATES]]
    chunks = load_chunks()
    scores = load_reranker().predict([(question, chunks[pos]["text"]) for pos in candidates], batch_size=16)
    return sorted(zip(candidates, (float(s) for s in scores)), key=lambda x: -x[1])


def ranking(question: str, mode: str = SEARCH_MODE) -> list[int]:
    """All chunk positions in the order the given search mode ranks them (used by the evaluation).
    For "rerank": the re-sorted top 50, followed by the rest in hybrid order."""
    if mode == "vector":
        return [pos for pos, _ in to_parents(vector_hits(question, len(load_units())))]
    hybrid = [pos for pos, _ in hybrid_ranking(question)]
    if mode == "hybrid":
        return hybrid
    top = [pos for pos, _ in reranked(question)]
    return top + hybrid[RERANK_CANDIDATES:]


def retrieve(question: str, k: int = TOP_K, mode: str = SEARCH_MODE) -> list[dict]:
    """Return the k best chunks for the question, best first.
    Each result is the chunk's text + labels, plus its score
    (cosine similarity for "vector", RRF score for "hybrid", reranker score for "rerank")."""
    if mode == "vector" and not PARENT_CHILD:
        q_vec = load_model().encode(QUERY_PREFIX + question, normalize_embeddings=True)
        hits = load_client().query_points(COLLECTION, query=q_vec.tolist(), limit=k).points
        return [{"score": round(h.score, 3), **h.payload} for h in hits]
    chunks = load_chunks()
    if mode == "vector":
        ranked = to_parents(vector_hits(question, len(load_units())))
    else:
        ranked = hybrid_ranking(question) if mode == "hybrid" else reranked(question)
    return [{"score": round(score, 4), **chunks[pos]} for pos, score in ranked[:k]]


def show(question: str, results: list[dict]) -> None:
    print(f"\nQ: {question}")
    for rank, r in enumerate(results, 1):
        print(f"  [{rank}] {r['score']:.3f}  {r['first_author']} {r['year']} · {r['article_type']} · "
              f"{', '.join(r['sections'])}  ({r['chunk_id']})")
        print(f"      {r['text'][:200]}...")


if __name__ == "__main__":
    mode = "rerank" if "--rerank" in sys.argv else "hybrid" if "--hybrid" in sys.argv else "vector"
    q = " ".join(a for a in sys.argv[1:] if a not in ("--hybrid", "--rerank")) or \
        "What is the remission rate of tisagenlecleucel in children with B-ALL?"
    show(q, retrieve(q, mode=mode))
