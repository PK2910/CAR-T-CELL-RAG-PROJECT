"""
Startup check: before the system answers anything, confirm that what it searches with matches what the index was
built with (a mismatch is what "Designing Machine Learning Systems" calls train-serving skew: no error, just wrong
results).

  embedding model   the model that turns questions into vectors = the model the index was built with
  vector size       the model's vector size = the index's vector size
  chunk count       chunks in the chunks file (or children, for parent-child) = vectors in the index = the count
                    recorded when the index was built (else results would point to the wrong text)
  API key           a key is set (else every answer falls back to showing the found passages)

The first three are fatal: the API refuses to start. A missing API key is a warning.
Run on its own:  python src/startup_check.py
"""

import json
import os

from embed import CHUNKS_FILE, COLLECTION, EMBEDDING_MODEL
from index_config import INDEX, INFO_FILE, UNITS_FILE
from retrieve import load_client, load_model, load_units


def run_checks() -> list[dict]:
    """Each check: name, ok (True/False), fatal (True/False), detail."""
    checks = []

    def add(name: str, ok: bool, fatal: bool, detail: str) -> None:
        checks.append({"name": name, "ok": ok, "fatal": fatal, "detail": detail})

    if not INFO_FILE.exists():
        add("index built", False, True, f"{INFO_FILE.name} not found: build the index with "
            f"RAG_INDEX={INDEX} python src/embed.py")
        return checks
    info = json.loads(INFO_FILE.read_text())

    add("embedding model", info.get("embedding_model") == EMBEDDING_MODEL, True,
        f"index built with {info.get('embedding_model')}, searching with {EMBEDDING_MODEL}")

    model_dim = load_model().get_embedding_dimension()
    add("vector size", model_dim == info.get("dimensions"), True,
        f"model makes {model_dim} numbers per text, index holds {info.get('dimensions')}")

    n_units = len(load_units())
    n_vectors = load_client().count(COLLECTION).count
    what = "children" if UNITS_FILE != CHUNKS_FILE else "chunks"
    add("chunk count", n_units == n_vectors == info.get("n_chunks"), True,
        f"{n_units} {what} in {UNITS_FILE.name}, {n_vectors} vectors in the index, "
        f"{info.get('n_chunks')} recorded when it was built ({info.get('date_built')})")

    add("API key", bool(os.environ.get("ANTHROPIC_API_KEY")), False,
        "ANTHROPIC_API_KEY is set" if os.environ.get("ANTHROPIC_API_KEY")
        else "ANTHROPIC_API_KEY is not set: answers will fall back to showing the found passages")
    return checks


def fatal_problems(checks: list[dict]) -> list[str]:
    return [f"{c['name']}: {c['detail']}" for c in checks if c["fatal"] and not c["ok"]]


if __name__ == "__main__":
    from dotenv import load_dotenv
    from embed import ROOT
    load_dotenv(ROOT / ".env")  # only to see whether a key is set; the key itself is never printed
    for c in run_checks():
        print(f"{'OK  ' if c['ok'] else ('FAIL' if c['fatal'] else 'WARN')}  {c['name']:16} {c['detail']}")
