"""
Shared setup for the code tests (run: pytest). They check that the code works, with a pretend Claude, pretend search
results and temporary storage: no API key, no cost, and no real papers needed (the papers and the index are not in the
repository). Whether the answers are good is a different question, tested on the real papers with the scripts in eval/.

On a fresh checkout (CI), made-up passages (tests/fixtures/chunks.json) stand in for data/processed/chunks.json; an
existing chunks file is never touched.
"""

import shutil
import sys
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT / "api"), str(ROOT / "src")]

from index_config import CHUNKS_FILE  # noqa: E402

if not CHUNKS_FILE.exists():
    CHUNKS_FILE.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(ROOT / "tests/fixtures/chunks.json", CHUNKS_FILE)

import anthropic  # noqa: E402
import httpx  # noqa: E402

import cache  # noqa: E402
import generate  # noqa: E402
import logbook  # noqa: E402
import usage  # noqa: E402

PASSAGE = {"chunk_id": "c1", "score": 0.9, "title": "A trial", "first_author": "Doe", "journal": "Test Journal",
           "year": 2024, "article_type": "research-article", "sections": ["Results"], "text": "ICANS 40%",
           "pmid": "1"}


class FakeClaude:
    """Stands in for the Claude client. Answers with `text`, raises a timeout when `fail` is set, counts calls.
    `api_key = None` acts like a server without a key."""

    def __init__(self):
        self.text, self.fail, self.calls, self.api_key, self.auth_token = "ICANS occurred in 40% [1].", False, 0, "k", None
        self.beta = types.SimpleNamespace(messages=self)

    def with_options(self, **kwargs):
        return self

    def _call(self):
        self.calls += 1
        if self.fail:
            raise anthropic.APITimeoutError(request=httpx.Request("POST", "https://api.anthropic.com"))
        return types.SimpleNamespace(
            stop_reason="end_turn", stop_details=None, model="claude-sonnet-5-5",
            content=[types.SimpleNamespace(type="text", text=self.text)],
            usage=types.SimpleNamespace(input_tokens=1000, output_tokens=100))

    def create(self, **kwargs):
        return self._call()

    def stream(self, **kwargs):
        response = self._call()
        words = response.content[0].text.split(" ")

        class Stream:
            text_stream = iter(w + " " for w in words)
            def __enter__(s): return s
            def __exit__(s, *exc): return False
            def get_final_message(s): return response
        return Stream()


class Pipeline:
    """The answer pipeline with everything outside our code replaced: Claude, the search, the relevance check."""

    def __init__(self):
        self.claude = FakeClaude()
        self.searches = 0
        self.outcome = "good"  # what the pretend relevance check decides: good / unsure / nothing

    def retrieve(self, question, k=10, mode=None):
        self.searches += 1
        return [dict(PASSAGE, chunk_id=f"c{i}") for i in range(k)]

    def select(self, rows, by_id, k=10):
        return (self.outcome, [] if self.outcome == "nothing" else [by_id["c1"]])


@pytest.fixture
def pipeline(monkeypatch, tmp_path):
    """A pretend pipeline, and the logbook, cache and usage counts in a temporary folder."""
    p = Pipeline()
    monkeypatch.setattr(generate.anthropic, "Anthropic", lambda *a, **k: p.claude)
    monkeypatch.setattr(generate, "retrieve", p.retrieve)
    monkeypatch.setattr(generate, "run_checker", lambda client, q, ranked, fmt: ([], 1, 9000, 300))
    monkeypatch.setattr(generate, "select", p.select)
    monkeypatch.setattr(logbook, "DB_FILE", tmp_path / "log.sqlite")
    monkeypatch.setattr(cache, "DB_FILE", tmp_path / "cache.sqlite")
    monkeypatch.setattr(usage, "DB_FILE", tmp_path / "usage.sqlite")
    monkeypatch.setattr(usage, "SALT_FILE", tmp_path / "salt.txt")
    monkeypatch.delenv("RAG_OWNER_KEY", raising=False)
    return p


@pytest.fixture
def client(pipeline):
    """The API, without its startup step (which opens the real index)."""
    from fastapi.testclient import TestClient
    import main
    return TestClient(main.app)
