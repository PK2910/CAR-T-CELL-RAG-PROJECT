"""
Box 4: EMBED + INDEX
Turn every chunk into a vector with bge-small, and store the vectors
together with the chunk text and labels in a Qdrant collection.

Input:  data/processed/chunks.json       (or chunks_plus.json with RAG_INDEX=plus, see index_config.py)
Output: data/qdrant/              (Qdrant local-mode storage)
        data/processed/index_info.json  (how this index was built)

Run:  python src/embed.py                 (text index)
      RAG_INDEX=plus python src/embed.py  (text + tables + figure captions)
"""

import json
from datetime import date
from pathlib import Path

import torch
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams
from sentence_transformers import SentenceTransformer

from index_config import (CHUNKS_FILE, COLLECTION, INDEX, INFO_FILE, MODEL, PARENT_CHILD, UNITS_FILE, USE_HEADER,
                          search_text)

ROOT = Path(__file__).resolve().parent.parent  # project folder, wherever we run from
QDRANT_PATH = ROOT / "data/qdrant"

EMBEDDING_MODEL = MODEL  # bge-small (or its fine-tuned copy for RAG_INDEX=tuned); same tokenizer as chunk.py
BATCH_SIZE = 32  # chunks embedded at once; lower it if the Mac runs out of memory

# bge models work best when QUESTIONS (not chunks) get this prefix
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


def main():
    chunks = json.loads(UNITS_FILE.read_text(encoding="utf-8"))  # the chunks, or their children (parent-child)

    # ---------- Embed ----------
    device = "mps" if torch.backends.mps.is_available() else "cpu"  # M1 GPU if available
    model = SentenceTransformer(EMBEDDING_MODEL, device=device)
    print(f"Embedding {len(chunks):,} chunks with {EMBEDDING_MODEL} on {device}...")
    vectors = model.encode(
        [search_text(c) for c in chunks],
        batch_size=BATCH_SIZE,
        normalize_embeddings=True,  # length 1, so cosine similarity = a simple dot product
        show_progress_bar=True,
    )

    # ---------- Index ----------
    client = QdrantClient(path=str(QDRANT_PATH))
    # Rebuild from scratch each run, so the index always matches chunks.json exactly
    if client.collection_exists(COLLECTION):
        client.delete_collection(COLLECTION)
    client.create_collection(
        COLLECTION,
        vectors_config=VectorParams(size=vectors.shape[1], distance=Distance.COSINE),
    )
    points = [PointStruct(id=i, vector=v.tolist(), payload=c)  # payload = text + labels
              for i, (v, c) in enumerate(zip(vectors, chunks))]
    for start in range(0, len(points), 256):
        client.upsert(COLLECTION, points=points[start:start + 256])
    print(f"Stored {client.count(COLLECTION).count:,} vectors in collection '{COLLECTION}'")

    # Record how this index was built (the index "version")
    INFO_FILE.write_text(json.dumps({
        "collection": COLLECTION,
        "embedding_model": EMBEDDING_MODEL,
        "dimensions": int(vectors.shape[1]),
        "n_chunks": len(chunks),
        "index": INDEX,
        "chunking": ("section-based, " if INDEX.startswith("section") else "")
                    + "recursive, 400 tokens, 50 overlap" + ("; + tables and figure captions" if INDEX == "plus" else "")
                    + ("; parent-child: children of 1-3 sentences are indexed, parents returned" if PARENT_CHILD
                       else "")
                    + (f"; contextual header indexed ({USE_HEADER})" if USE_HEADER else ""),
        "fine_tuned": INDEX == "tuned",
        "chunk_types": {t: sum(1 for c in chunks if c.get("chunk_type", "text") == t)
                        for t in sorted({c.get("chunk_type", "text") for c in chunks})},
        "date_built": date.today().isoformat(),
    }, indent=2))

    # ---------- Sanity check: one test question ----------
    question = "What is the remission rate of tisagenlecleucel in children with B-ALL?"
    q_vec = model.encode(QUERY_PREFIX + question, normalize_embeddings=True)
    hits = client.query_points(COLLECTION, query=q_vec.tolist(), limit=3).points
    print(f"\nTest question: {question}")
    for h in hits:
        p = h.payload
        label = p.get("parent_id") or f"{p['first_author']} {p['year']} ({p['article_type']}) {p['sections']}"
        print(f"  {h.score:.3f}  {label}")
        print(f"         {p['text'][:150]}...")
    client.close()


if __name__ == "__main__":
    main()
