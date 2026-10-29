"""
Usage limits for the public demo (README, Requirements): at most QUESTIONS_PER_DAY questions per user per day, and
at most DAILY_BUDGET_USD spent on public questions per day. The owner's key skips both.

Counts are kept in a small SQLite file (data/api_usage.sqlite, not in the repository), so they survive a restart.
A user is identified by their IP address, which is stored only as a salted hash: the counts can tell users apart
but cannot be traced back to an address. Days are UTC.
"""

import hashlib
import secrets
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DB_FILE = ROOT / "data/api_usage.sqlite"
SALT_FILE = ROOT / "data/api_usage_salt.txt"
QUESTIONS_PER_DAY = 10
DAILY_BUDGET_USD = 1.00

_lock = threading.Lock()


def today() -> str:
    return datetime.now(timezone.utc).date().isoformat()


def _salt() -> str:
    if not SALT_FILE.exists():
        SALT_FILE.parent.mkdir(parents=True, exist_ok=True)
        SALT_FILE.write_text(secrets.token_hex(16))
    return SALT_FILE.read_text().strip()


def user_id(ip: str) -> str:
    """A salted hash of the IP address: the same user gets the same id, but the address is not stored."""
    return hashlib.sha256(f"{_salt()}:{ip}".encode()).hexdigest()[:16]


def _db() -> sqlite3.Connection:
    DB_FILE.parent.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(DB_FILE)
    db.execute("CREATE TABLE IF NOT EXISTS questions (day TEXT, user TEXT, n INTEGER, PRIMARY KEY (day, user))")
    db.execute("CREATE TABLE IF NOT EXISTS spend (day TEXT PRIMARY KEY, usd REAL)")
    return db


def check_and_count(user: str) -> str | None:
    """Count one question for this user today. Returns None if it is allowed, or the reason it is not."""
    day = today()
    with _lock, _db() as db:
        spent = db.execute("SELECT usd FROM spend WHERE day = ?", (day,)).fetchone()
        if spent and spent[0] >= DAILY_BUDGET_USD:
            return "The demo has reached today's limit. Please try again tomorrow."
        asked = db.execute("SELECT n FROM questions WHERE day = ? AND user = ?", (day, user)).fetchone()
        if asked and asked[0] >= QUESTIONS_PER_DAY:
            return f"You have asked {QUESTIONS_PER_DAY} questions today, the daily limit. Please try again tomorrow."
        db.execute("INSERT INTO questions VALUES (?, ?, 1) ON CONFLICT (day, user) DO UPDATE SET n = n + 1",
                   (day, user))
    return None


def refund(user: str) -> None:
    """Give the question back when it could not be answered (e.g. the model service was down)."""
    with _lock, _db() as db:
        db.execute("UPDATE questions SET n = MAX(n - 1, 0) WHERE day = ? AND user = ?", (today(), user))


def add_spend(usd: float) -> None:
    """Add the cost of one answered public question to today's total."""
    with _lock, _db() as db:
        db.execute("INSERT INTO spend VALUES (?, ?) ON CONFLICT (day) DO UPDATE SET usd = usd + excluded.usd",
                   (today(), usd))


def spent_today() -> float:
    with _lock, _db() as db:
        row = db.execute("SELECT usd FROM spend WHERE day = ?", (today(),)).fetchone()
    return row[0] if row else 0.0
