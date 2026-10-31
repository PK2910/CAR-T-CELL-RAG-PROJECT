"""
The answer cache (README, Caching): an answer is saved, and the same question asked again gets the saved answer
at once, for free, instead of searching, checking and writing again (~17 s, ~3 cents).

"The same question" means the same words, ignoring capital letters, extra spaces and the final "?" (tidy()).
Questions that only mean something similar are NOT matched: our search model scores different questions as similar
as rewordings (children vs adults 0.943, before vs after transplant 0.950, remission vs relapse 0.934, against 0.925
for a real rewording), so a "semantic cache" would hand out answers to different questions.

Each saved answer is tied to the versions that made it (api/logbook.py: index, chunking, models, prompts): after
any change, old answers no longer match. Answers expire after MAX_AGE_DAYS. Not saved: fallback answers (the
next user should get a real answer), personal questions, questions containing personal details such as a name
(api/redact.py), and declined answers.
Kept in data/api_cache.sqlite (not in the repository); delete the file to clear the cache.
A cache that cannot be read or written never stops an answer: the question is answered as if it were not cached.
"""

import hashlib
import json
import re
import sqlite3
import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path

from logbook import versions
from redact import redact

ROOT = Path(__file__).resolve().parent.parent
DB_FILE = ROOT / "data/api_cache.sqlite"
MAX_AGE_DAYS = 30

_lock = threading.Lock()


def tidy(question: str) -> str:
    """'  How common is ICANS after CAR-T?  ' -> 'how common is icans after car-t'"""
    return re.sub(r"\s+", " ", question).strip().lower().rstrip("?!. ")


def _key(question: str) -> str:
    return hashlib.sha256(f"{tidy(question)}|{json.dumps(versions(), sort_keys=True)}".encode()).hexdigest()


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _db() -> sqlite3.Connection:
    DB_FILE.parent.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(DB_FILE)
    db.execute("CREATE TABLE IF NOT EXISTS answers (key TEXT PRIMARY KEY, question TEXT, result TEXT, "
               "created TEXT, hits INTEGER DEFAULT 0)")
    return db


def get(question: str) -> dict | None:
    """The saved answer to this question, or None. A saved answer costs nothing: cost and tokens are 0, and
    saved_usd is what it cost the first time."""
    oldest = (_now() - timedelta(days=MAX_AGE_DAYS)).isoformat(timespec="seconds")
    try:
        with _lock, _db() as db:
            row = db.execute("SELECT result FROM answers WHERE key = ? AND created >= ?",
                             (_key(question), oldest)).fetchone()
            if not row:
                return None
            db.execute("UPDATE answers SET hits = hits + 1 WHERE key = ?", (_key(question),))
    except (sqlite3.Error, OSError) as e:
        print(f"WARNING (cache) not read: {type(e).__name__}: {e}")
        return None
    result = json.loads(row[0])
    return {**result, "question": question, "cached": True, "saved_usd": result.get("cost_usd", 0.0),
            "cost_usd": 0.0, "input_tokens": 0, "output_tokens": 0, "checker_cost_usd": 0.0, "timing": {}}


def put(question: str, result: dict) -> bool:
    """Save an answer for next time, unless it should not be reused. Returns True if it was saved."""
    if (result.get("degraded") or result.get("error") or result.get("stop_reason") == "refusal"
            or (result.get("guardrail") or {}).get("personal")
            or redact(question) != question):  # contains personal details: keep it out of the cache
        return False
    oldest = (_now() - timedelta(days=MAX_AGE_DAYS)).isoformat(timespec="seconds")
    try:
        with _lock, _db() as db:
            db.execute("DELETE FROM answers WHERE created < ?", (oldest,))  # expired answers
            db.execute("INSERT OR REPLACE INTO answers (key, question, result, created) VALUES (?, ?, ?, ?)",
                       (_key(question), tidy(question), json.dumps(result, ensure_ascii=False),
                        _now().isoformat(timespec="seconds")))
    except (sqlite3.Error, OSError, TypeError, ValueError) as e:
        print(f"WARNING (cache) not saved: {type(e).__name__}: {e}")
        return False
    return True
