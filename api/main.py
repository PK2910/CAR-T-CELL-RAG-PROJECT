"""
The API: the front door to the RAG system (README, Requirements).

  GET  /health       is the server running? (also shows which index version it uses)
  POST /ask          {"question": "..."} -> the full answer, with its sources
  POST /ask/stream   the same answer, sent piece by piece (Server-Sent Events): first progress messages, then the
                     answer as Claude writes it, then a final "done" event with the complete answer and sources.
                     Personal questions are not streamed (the dose guardrail checks the whole answer first).
  POST /feedback     {"answer_id": "...", "rating": "up" or "down", "comment": "..."} -> thumbs up/down for an answer

Every answer is saved in the logbook (api/logbook.py) with an id, which the user can send to /feedback.

Public demo: anyone can ask, at most 10 questions per user per day and $1 per day in total (api/usage.py).
The owner skips both limits by sending their key in the X-Owner-Key header; the key is set as RAG_OWNER_KEY in .env.

Run:  uvicorn api.main:app            then open http://localhost:8000/docs to try it in the browser
"""

import hmac
import json
import os
import sys
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

import anthropic
from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.responses import RedirectResponse, StreamingResponse
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "api"))
from generate import answer, answer_stream, seconds_since  # noqa: E402  (also loads .env)
from index_config import INFO_FILE  # noqa: E402
from retrieve import retrieve  # noqa: E402
from startup_check import fatal_problems, run_checks  # noqa: E402
import logbook  # noqa: E402
import usage  # noqa: E402

DISCLAIMER = "Research and education tool that summarizes published studies. Not medical advice."
UNAVAILABLE = "The answer service is unavailable right now. Please try again later."


CHECKS: list[dict] = []


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup check (src/startup_check.py): refuse to start if the search would not match the index
    CHECKS[:] = run_checks()
    problems = fatal_problems(CHECKS)
    if problems:
        raise RuntimeError("Startup check failed, not starting:\n  " + "\n  ".join(problems))
    for c in CHECKS:
        if not c["ok"]:
            print(f"WARNING (startup check) {c['name']}: {c['detail']}")
    retrieve("warm-up", k=1)  # load the search model and indexes once, before the first question
    yield


app = FastAPI(title="CAR-T Literature Q&A", description=DISCLAIMER, lifespan=lifespan)


class Question(BaseModel):
    question: str = Field(min_length=3, max_length=1000, examples=["How common is ICANS after CAR-T in ALL?"])


class Feedback(BaseModel):
    answer_id: str = Field(min_length=1, max_length=64)
    rating: Literal["up", "down"]
    comment: str | None = Field(default=None, max_length=1000)


def is_owner(key: str | None) -> bool:
    owner_key = os.environ.get("RAG_OWNER_KEY", "")
    return bool(owner_key and key) and hmac.compare_digest(key, owner_key)


def client_ip(request: Request) -> str:
    """The user's address. Behind a hosting service's proxy, set TRUST_PROXY=1 to use the address it forwards."""
    forwarded = request.headers.get("x-forwarded-for")
    if os.environ.get("TRUST_PROXY") == "1" and forwarded:
        return forwarded.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


def admit(request: Request, owner_key: str | None) -> str | None:
    """Apply the usage limits. Returns the user's id, or None for the owner (whose questions are not counted)."""
    if is_owner(owner_key):
        return None
    user = usage.user_id(client_ip(request))
    refused = usage.check_and_count(user)
    if refused:
        raise HTTPException(status_code=429, detail=refused)
    return user


def public(result: dict, answer_id: str | None) -> dict:
    """What a user sees: the answer and its sources (no internal fields such as prompts or token counts)."""
    return {
        "answer_id": answer_id,  # send it to /feedback to rate this answer
        "answer": result["answer"],
        "sources": [{"n": n, "title": s["title"], "first_author": s["first_author"], "year": s["year"],
                     "journal": s["journal"], "section": ", ".join(s.get("sections", [])),
                     "excerpt": s["text"][:400] + ("..." if len(s["text"]) > 400 else ""),
                     "link": f"https://pubmed.ncbi.nlm.nih.gov/{s['pmid']}/" if s.get("pmid") else None}
                    for n, s in enumerate(result["sources"], 1)],
        "relevance_check": result.get("checker_outcome"),
        "fallback": bool(result.get("degraded")),  # True: no summary could be written; the passages are shown
        "disclaimer": DISCLAIMER,
    }


@app.get("/", include_in_schema=False)
def home():
    return RedirectResponse("/docs")  # the main address opens the test page


@app.get("/health")
def health() -> dict:
    return {"status": "ok" if all(c["ok"] for c in CHECKS) else "degraded", "checks": CHECKS,
            "index": json.loads(INFO_FILE.read_text()) if INFO_FILE.exists() else None,
            "versions": logbook.versions()}


@app.post("/ask")
def ask(body: Question, request: Request, x_owner_key: str | None = Header(default=None)) -> dict:
    user = admit(request, x_owner_key)
    start = time.perf_counter()
    try:
        result = answer(body.question)
    except anthropic.APIError as e:
        if user:
            usage.refund(user)
        logbook.record(body.question, None, user, "ask", seconds_since(start), error=e)
        raise HTTPException(status_code=503, detail=UNAVAILABLE)
    if user:
        usage.add_spend(result["cost_usd"])
        if result.get("degraded"):
            usage.refund(user)  # no summary was written, so the question is not counted
    answer_id = logbook.record(body.question, result, user, "ask", seconds_since(start))
    return public(result, answer_id)


@app.post("/ask/stream")
def ask_stream(body: Question, request: Request, x_owner_key: str | None = Header(default=None)):
    user = admit(request, x_owner_key)

    def events():
        start, first_text = time.perf_counter(), None  # first_text: seconds until the first words appear
        try:
            for event in answer_stream(body.question):
                if event["type"] == "text" and first_text is None:
                    first_text = seconds_since(start)
                if event["type"] == "done":
                    result = event["result"]
                    if user:
                        usage.add_spend(result["cost_usd"])
                        if result.get("degraded"):
                            usage.refund(user)
                    answer_id = logbook.record(body.question, result, user, "ask/stream", seconds_since(start),
                                               first_text)
                    event = {"type": "done", **public(result, answer_id)}
                yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
        except anthropic.APIError as e:
            if user:
                usage.refund(user)
            logbook.record(body.question, None, user, "ask/stream", seconds_since(start), first_text, error=e)
            yield f"data: {json.dumps({'type': 'error', 'text': UNAVAILABLE})}\n\n"

    return StreamingResponse(events(), media_type="text/event-stream")


@app.post("/feedback")
def feedback(body: Feedback) -> dict:
    if not logbook.add_feedback(body.answer_id, body.rating, body.comment):
        raise HTTPException(status_code=404, detail="No answer with this id.")
    return {"status": "saved", "answer_id": body.answer_id, "rating": body.rating}
