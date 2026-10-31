"""
Make a slim copy of the search index for the Docker image: only the collection the live system uses (the index
chosen by RAG_INDEX, "text" by default), not the experiments kept in data/qdrant (section, parent-child, tuned...).
The vectors are copied as they are (no re-embedding), and the copy is checked: the same questions must return the
same chunks in the same order from both indexes.

Run:  python src/slim_index.py       -> data/docker/qdrant (about 30 MB instead of 536 MB)
Stop the API first: only one process can open the local index at a time.
"""

import shutil
import sys

from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, VectorParams

from embed import COLLECTION, QDRANT_PATH, ROOT
from retrieve import load_model, QUERY_PREFIX

SLIM_PATH = ROOT / "data/docker/qdrant"
CHECK_QUESTIONS = ["How common is ICANS after CAR-T in ALL?", "What drugs are used to manage cytokine release syndrome?",
                   "What is the relapse rate after tisagenlecleucel?", "CD22 CAR-T after CD19 failure"]


def main():
    full = QdrantClient(path=str(QDRANT_PATH))
    if SLIM_PATH.exists():
        shutil.rmtree(SLIM_PATH)
    slim = QdrantClient(path=str(SLIM_PATH))
    params = full.get_collection(COLLECTION).config.params.vectors
    slim.create_collection(COLLECTION, vectors_config=VectorParams(size=params.size, distance=params.distance))

    offset, copied = None, 0
    while True:  # copy every point (id, vector, payload) in pages
        points, offset = full.scroll(COLLECTION, limit=1000, offset=offset, with_vectors=True, with_payload=True)
        slim.upsert(COLLECTION, [PointStruct(id=p.id, vector=p.vector, payload=p.payload) for p in points])
        copied += len(points)
        if offset is None:
            break
    slim.close()  # check the copy as saved on disk (as the API will open it), not the one still in memory
    slim = QdrantClient(path=str(SLIM_PATH))
    n_full, n_slim = full.count(COLLECTION).count, slim.count(COLLECTION).count
    print(f"{COLLECTION}: copied {copied} points ({n_slim} in the copy, {n_full} in the full index)")

    model = load_model()
    for q in CHECK_QUESTIONS:  # the copy must search exactly like the original
        vec = model.encode(QUERY_PREFIX + q, normalize_embeddings=True).tolist()
        a = [(h.id, round(h.score, 6)) for h in full.query_points(COLLECTION, query=vec, limit=60).points]
        b = [(h.id, round(h.score, 6)) for h in slim.query_points(COLLECTION, query=vec, limit=60).points]
        if a != b:
            sys.exit(f"MISMATCH for {q!r}: the copy does not search like the original")
    print(f"Search check: {len(CHECK_QUESTIONS)} questions return the same 60 chunks, same order and scores: OK")
    full.close()
    slim.close()
    size = sum(f.stat().st_size for f in SLIM_PATH.rglob("*") if f.is_file()) / 1e6
    print(f"Saved -> {SLIM_PATH.relative_to(ROOT)} ({size:.0f} MB)")


if __name__ == "__main__":
    main()
