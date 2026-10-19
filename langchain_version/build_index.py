"""
LangChain version, one-time setup: copy our existing vectors into a Qdrant collection laid
out the way LangChain's QdrantVectorStore expects ({"page_content": ..., "metadata": {...}}).
The vectors are copied as they are (same bge-small numbers), so nothing is re-embedded and the
comparison with the plain-Python version stays fair.

Run:  python langchain_version/build_index.py
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from qdrant_client.models import Distance, PointStruct, VectorParams  # noqa: E402
from index_config import INDEXES  # noqa: E402
from retrieve import load_client  # noqa: E402

COLLECTION = INDEXES["text"]["collection"]  # the LangChain copy is of the text index

LC_COLLECTION = "cart_all_langchain"


def main():
    client = load_client()
    size = client.get_collection(COLLECTION).config.params.vectors.size
    if client.collection_exists(LC_COLLECTION):
        client.delete_collection(LC_COLLECTION)
    client.create_collection(LC_COLLECTION, vectors_config=VectorParams(size=size, distance=Distance.COSINE))

    offset, copied = None, 0
    while True:
        points, offset = client.scroll(COLLECTION, limit=512, offset=offset, with_vectors=True, with_payload=True)
        client.upsert(LC_COLLECTION, points=[
            PointStruct(id=p.id, vector=p.vector, payload={"page_content": p.payload["text"], "metadata": p.payload})
            for p in points])
        copied += len(points)
        if offset is None:
            break
    print(f"Copied {copied:,} vectors from '{COLLECTION}' into '{LC_COLLECTION}' "
          f"(now {client.count(LC_COLLECTION).count:,} points)")


if __name__ == "__main__":
    main()
