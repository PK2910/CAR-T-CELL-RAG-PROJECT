"""
The logbook (README, Logging): one entry per question the API answers, so a wrong or slow answer can be traced back
to the stage that caused it, and users' thumbs up/down are kept next to the answer they rate.

Each entry: when, which user (the same salted hash as the usage limits, or "owner"; never the IP address), the
question and answer, the relevance check's outcome, the chunks the answer came from (ids and search scores), whether
the fallback was used, seconds per stage, tokens and cost, the versions that produced it (index, chunking,
embedding model, Claude models, prompts), whether it came from the cache (api/cache.py), and any citations to
sources it was not given (the citation check, src/guardrails.py). Kept in data/api_log.sqlite (not in the repository).
Summarize it with:  python api/log_report.py

Writing an entry never stops an answer: if the logbook cannot be written, the user still gets the answer.
"""

import hashlib
import json
import secrets
import sqlite3
import threading
from datetime import datetime, timezone
from functools import cache
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DB_FILE = ROOT / "data/api_log.sqlite"

COLUMNS = """
    id TEXT PRIMARY KEY, time TEXT, user TEXT, endpoint TEXT, question TEXT, answer TEXT,
    outcome TEXT, fallback INTEGER, guardrail TEXT, error TEXT, chunks TEXT, n_sources INTEGER,
    seconds_total REAL, seconds_first_text REAL, timing TEXT,
    tokens_in INTEGER, tokens_out INTEGER, cost_usd REAL, versions TEXT,
    rating INTEGER, comment TEXT, rated_at TEXT, cached INTEGER, saved_usd REAL, bad_citations TEXT
"""
# Columns added after the first version; an older logbook file gets them when it is opened
ADDED_COLUMNS = {"cached": "INTEGER", "saved_usd": "REAL",  # the cache (api/cache.py)
                 "bad_citations": "TEXT"}  # the citation check (src/guardrails.py)

_lock = threading.Lock()


def _short_hash(text: str) -> str:
    """A prompt's fingerprint: changes whenever the prompt's wording changes."""
    return hashlib.sha256(text.encode()).hexdigest()[:8]


@cache
def versions() -> dict:
    """Everything that decides what an answer looks like, so each answer can be traced to the setup that made it."""
    import guardrails
    from generate import MODEL, SYSTEM_PROMPT
    from index_config import INDEX, INFO_FILE
    from relevance_check import INSTRUCTIONS
    from relevance_check import MODEL as CHECKER_MODEL
    from retrieve import SEARCH_MODE, TOP_K
    info = json.loads(INFO_FILE.read_text()) if INFO_FILE.exists() else {}
    return {"index": INDEX, "index_built": info.get("date_built"), "n_chunks": info.get("n_chunks"),
            "chunking": info.get("chunking"), "embedding_model": info.get("embedding_model"),
            "search": f"{SEARCH_MODE}, top {TOP_K}", "answer_model": MODEL, "checker_model": CHECKER_MODEL,
            "answer_prompt": _short_hash(SYSTEM_PROMPT), "checker_prompt": _short_hash(INSTRUCTIONS),
            "guardrails": _short_hash(Path(guardrails.__file__).read_text())}  # any change to the safety rules


def _db() -> sqlite3.Connection:
    DB_FILE.parent.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(DB_FILE)
    db.execute(f"CREATE TABLE IF NOT EXISTS answers ({COLUMNS})")
    have = {row[1] for row in db.execute("PRAGMA table_info(answers)")}
    for name, kind in ADDED_COLUMNS.items():  # a logbook made before these columns existed keeps its entries
        if name not in have:
            db.execute(f"ALTER TABLE answers ADD COLUMN {name} {kind}")
    return db


def record(question: str, result: dict | None, user: str | None, endpoint: str, seconds: float,
           first_text_s: float | None = None, error: Exception | None = None) -> str | None:
    """Save one entry and return its id (sent to the user, who can rate the answer with it).
    result is None when no answer at all could be made (error says why). user None = the owner."""
    result = result or {}
    error_text = f"{type(error).__name__}: {error}" if error else result.get("error")
    entry = {
        "id": secrets.token_hex(8),  # random, so nobody can guess another user's answer id and rate it
        "time": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "user": user or "owner",
        "endpoint": endpoint,
        "question": question,
        "answer": result.get("answer"),
        "outcome": result.get("checker_outcome"),
        "fallback": int(bool(result.get("degraded"))),
        "guardrail": (result.get("guardrail") or {}).get("action"),
        "error": error_text or result.get("checker_error"),
        "chunks": json.dumps([{"id": s.get("chunk_id"), "score": s.get("score")} for s in result.get("sources", [])]),
        "n_sources": len(result.get("sources", [])),
        "seconds_total": seconds,
        "seconds_first_text": first_text_s,
        "timing": json.dumps(result.get("timing", {})),
        "tokens_in": result.get("input_tokens"),
        "tokens_out": result.get("output_tokens"),
        "cost_usd": result.get("cost_usd"),
        "versions": json.dumps(versions()),
        "cached": int(bool(result.get("cached"))),
        "saved_usd": result.get("saved_usd"),  # a cached answer: what it cost the first time
        "bad_citations": json.dumps((result.get("citations") or {}).get("invalid") or []),
    }
    try:
        with _lock, _db() as db:
            db.execute(f"INSERT INTO answers ({', '.join(entry)}) VALUES ({', '.join('?' * len(entry))})",
                       list(entry.values()))
    except (sqlite3.Error, OSError) as e:
        print(f"WARNING (logbook) entry not saved: {type(e).__name__}: {e}")
        return None
    return entry["id"]


def add_feedback(answer_id: str, rating: str, comment: str | None = None) -> bool:
    """Thumbs up ("up") or down ("down") for one answer. Rating again replaces the earlier rating.
    Returns False if there is no answer with this id."""
    with _lock, _db() as db:
        changed = db.execute("UPDATE answers SET rating = ?, comment = ?, rated_at = ? WHERE id = ?",
                             (1 if rating == "up" else -1, comment,
                              datetime.now(timezone.utc).isoformat(timespec="seconds"), answer_id)).rowcount
    return changed > 0


def entries(since: str | None = None) -> list[dict]:
    """All entries (from the time `since` on, if given), oldest first."""
    with _lock, _db() as db:
        db.row_factory = sqlite3.Row
        rows = db.execute("SELECT * FROM answers WHERE time >= ? ORDER BY time", (since or "",)).fetchall()
    return [dict(r) for r in rows]
