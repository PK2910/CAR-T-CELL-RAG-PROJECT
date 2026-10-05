"""
Box 4: EMBED + INDEX
Turn every chunk into a vector with bge-small, and store the vectors
together with the chunk text and labels in a Qdrant collection.

Input:  data/processed/chunks.json
Output: data/qdrant/              (Qdrant local-mode storage)
        data/processed/index_info.json  (how this index was built)

Run:  python src/embed.py
"""

import json
from datetime import date
from pathlib import Path

import torch
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams
from sentence_transformers import SentenceTransformer

CHUNKS_FILE = Path("data/processed/chunks.json")
QDRANT_PATH = Path("data/qdrant")
INFO_FILE = Path("data/processed/index_info.json")

EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"  # must match the tokenizer used in chunk.py
COLLECTION = "cart_all_baseline"
BATCH_SIZE = 32  # chunks embedded at once; lower it if the Mac runs out of memory

# bge models work best when QUESTIONS (not chunks) get this prefix
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


def main():
    chunks = json.loads(CHUNKS_FILE.read_text(encoding="utf-8"))

    # ---------- Embed ----------
    device = "mps" if torch.backends.mps.is_available() else "cpu"  # M1 GPU if available
    model = SentenceTransformer(EMBEDDING_MODEL, device=device)
    print(f"Embedding {len(chunks):,} chunks with {EMBEDDING_MODEL} on {device}...")
    vectors = model.encode(
        [c["text"] for c in chunks],
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
        "chunking": "recursive, 400 tokens, 50 overlap",
        "date_built": date.today().isoformat(),
    }, indent=2))

    # ---------- Sanity check: one test question ----------
    question = "What is the remission rate of tisagenlecleucel in children with B-ALL?"
    q_vec = model.encode(QUERY_PREFIX + question, normalize_embeddings=True)
    hits = client.query_points(COLLECTION, query=q_vec.tolist(), limit=3).points
    print(f"\nTest question: {question}")
    for h in hits:
        p = h.payload
        print(f"  {h.score:.3f}  {p['first_author']} {p['year']} ({p['article_type']}) {p['sections']}")
        print(f"         {p['text'][:150]}...")
    client.close()


if __name__ == "__main__":
    main()
