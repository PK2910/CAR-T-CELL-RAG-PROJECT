"""
Box 5: RETRIEVE
Given a question, find the top-k most relevant chunks in Qdrant.
Other steps (generate, evaluate) import retrieve() from here.

Run:  python src/retrieve.py "your question here"
"""

import atexit
import sys
from functools import lru_cache

import torch
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

# Reuse the exact settings the index was built with
from embed import COLLECTION, EMBEDDING_MODEL, QDRANT_PATH, QUERY_PREFIX

TOP_K = 5


@lru_cache(maxsize=1)  # load once, then reuse (loading takes a few seconds)
def load_model() -> SentenceTransformer:
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    return SentenceTransformer(EMBEDDING_MODEL, device=device)


@lru_cache(maxsize=1)
def load_client() -> QdrantClient:
    client = QdrantClient(path=str(QDRANT_PATH))
    atexit.register(client.close)  # close cleanly before Python shuts down
    return client


def retrieve(question: str, k: int = TOP_K) -> list[dict]:
    """Return the k chunks closest in meaning to the question, best first.
    Each result is the chunk's text + labels, plus its similarity score."""
    q_vec = load_model().encode(QUERY_PREFIX + question, normalize_embeddings=True)
    hits = load_client().query_points(COLLECTION, query=q_vec.tolist(), limit=k).points
    return [{"score": round(h.score, 3), **h.payload} for h in hits]


def show(question: str, results: list[dict]) -> None:
    print(f"\nQ: {question}")
    for rank, r in enumerate(results, 1):
        print(f"  [{rank}] {r['score']:.3f}  {r['first_author']} {r['year']} · {r['article_type']} · "
              f"{', '.join(r['sections'])}  ({r['chunk_id']})")
        print(f"      {r['text'][:200]}...")


if __name__ == "__main__":
    q = " ".join(sys.argv[1:]) or "What is the remission rate of tisagenlecleucel in children with B-ALL?"
    show(q, retrieve(q))
