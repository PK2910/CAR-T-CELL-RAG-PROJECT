"""
The API: the front door to the RAG system (README, Requirements).

  GET  /health       is the server running? (also shows which index version it uses)
  POST /ask          {"question": "..."} -> the full answer, with its sources
  POST /ask/stream   the same answer, sent piece by piece (Server-Sent Events): first progress messages, then the
                     answer as Claude writes it, then a final "done" event with the complete answer and sources.
                     Personal questions are not streamed (the dose guardrail checks the whole answer first).

Public demo: anyone can ask, at most 10 questions per user per day and $1 per day in total (api/usage.py).
The owner skips both limits by sending their key in the X-Owner-Key header; the key is set as RAG_OWNER_KEY in .env.

Run:  uvicorn api.main:app            then open http://localhost:8000/docs to try it in the browser
"""

import hmac
import json
import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path

import anthropic
from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.responses import RedirectResponse, StreamingResponse
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "api"))
from generate import answer, answer_stream  # noqa: E402  (also loads .env)
from index_config import INFO_FILE  # noqa: E402
from retrieve import retrieve  # noqa: E402
import usage  # noqa: E402

DISCLAIMER = "Research and education tool that summarizes published studies. Not medical advice."
UNAVAILABLE = "The answer service is unavailable right now. Please try again later."


@asynccontextmanager
async def lifespan(app: FastAPI):
    retrieve("warm-up", k=1)  # load the search model and indexes once, before the first question
    yield


app = FastAPI(title="CAR-T Literature Q&A", description=DISCLAIMER, lifespan=lifespan)


class Question(BaseModel):
    question: str = Field(min_length=3, max_length=1000, examples=["How common is ICANS after CAR-T in ALL?"])


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


def public(result: dict) -> dict:
    """What a user sees: the answer and its sources (no internal fields such as prompts or token counts)."""
    return {
        "answer": result["answer"],
        "sources": [{"n": n, "title": s["title"], "first_author": s["first_author"], "year": s["year"],
                     "journal": s["journal"], "section": ", ".join(s.get("sections", [])),
                     "link": f"https://pubmed.ncbi.nlm.nih.gov/{s['pmid']}/" if s.get("pmid") else None}
                    for n, s in enumerate(result["sources"], 1)],
        "relevance_check": result.get("checker_outcome"),
        "disclaimer": DISCLAIMER,
    }


@app.get("/", include_in_schema=False)
def home():
    return RedirectResponse("/docs")  # the main address opens the test page


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "index": json.loads(INFO_FILE.read_text()) if INFO_FILE.exists() else None}


@app.post("/ask")
def ask(body: Question, request: Request, x_owner_key: str | None = Header(default=None)) -> dict:
    user = admit(request, x_owner_key)
    try:
        result = answer(body.question)
    except anthropic.APIError:
        if user:
            usage.refund(user)
        raise HTTPException(status_code=503, detail=UNAVAILABLE)
    if user:
        usage.add_spend(result["cost_usd"])
    return public(result)


@app.post("/ask/stream")
def ask_stream(body: Question, request: Request, x_owner_key: str | None = Header(default=None)):
    user = admit(request, x_owner_key)

    def events():
        try:
            for event in answer_stream(body.question):
                if event["type"] == "done":
                    if user:
                        usage.add_spend(event["result"]["cost_usd"])
                    event = {"type": "done", **public(event["result"])}
                yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
        except anthropic.APIError:
            if user:
                usage.refund(user)
            yield f"data: {json.dumps({'type': 'error', 'text': UNAVAILABLE})}\n\n"

    return StreamingResponse(events(), media_type="text/event-stream")
