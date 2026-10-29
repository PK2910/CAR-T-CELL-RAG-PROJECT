"""
LLM relevance check (Corrective-RAG-style): a checker reads the question with the top search results and keeps the
chunks that actually answer it, before Claude writes the answer.

The checker (Claude Haiku 5.5) marks each of the top 30 chunks yes / partly / no (and #31-60 when unsure). Then:
  good     at least one "yes"  -> Claude answers from the "yes" chunks (search order, at most TOP_K)
  unsure   only "partly"       -> Claude answers from the "partly" chunks, told they may only partly answer
  nothing  every chunk "no"    -> guardrail: no answer is written, a fixed reply is returned instead

Used live by src/generate.py (answer) and offline by eval/check_chunks.py (the tests). Not the full Corrective RAG
method: no web search (medical answers come only from the curated papers) and no sentence-level refinement.
Tested on the final set: 14 -> 17 of 21 answers correct (eval/RESULTS.md). Known weaknesses: too strict on
mechanism evidence from other diseases (q40), and two-part questions can lose the part without a number (q31).
"""

import json

from retrieve import TOP_K

MODEL = "claude-haiku-5-5"
EFFORT = "low"
PRICE_PER_MTOK = {"input": 0.10, "output": 0.50}  # Haiku 5.5, prompts up to 100K tokens; USD per million tokens
BATCH = 30          # chunks checked per call: #1-30, and #31-60 for "unsure" questions

SCHEMA = {
    "type": "object",
    "properties": {"chunks": {"type": "array", "items": {
        "type": "object",
        "properties": {"n": {"type": "integer"},
                       "verdict": {"type": "string", "enum": ["yes", "partly", "no"]},
                       "reason": {"type": "string"}},
        "required": ["n", "verdict", "reason"], "additionalProperties": False}}},
    "required": ["chunks"], "additionalProperties": False,
}

INSTRUCTIONS = """You check search results for a question-answering system about CAR-T cell therapy in acute \
lymphoblastic leukemia (ALL). You get a question and numbered excerpts from published papers. For EVERY excerpt, \
decide whether it helps answer the question:

- yes: the excerpt states information that answers the question, or one part of it. If the question asks for a \
number, rate or frequency ("how common", "how often", "what percentage"), the excerpt must give that number for the \
thing asked about. If the question has several parts, an excerpt that answers one part is "yes".
- partly: the excerpt is about the same topic, but does not contain the specific information asked for (for example, \
it describes a side effect but gives no rate).
- no: the excerpt is not about what was asked.

Judge only what the excerpt says, not what you know. An excerpt about a different disease (for example lymphoma \
instead of ALL) or a different product than the one asked about is not "yes". Give a reason of at most 12 words. \
Return one entry per excerpt, numbered as in the input."""

NOT_FOUND_ANSWER = ("The papers in this collection don't answer this question. "
                    "This is a research and education tool, not medical advice.")

PARTLY_NOTE = ("Note: an automatic check found that these sources are related to the question but may not contain "
               "the specific information asked for. If they do not fully answer it, say that the papers only partly "
               "address the question, and state what they do and do not cover.")


def check_batch(client, question: str, sources_text: str, n: int) -> tuple[list[dict], object]:
    """One checker call for n numbered chunks (already formatted): their verdicts in input order, and the usage."""
    response = client.messages.create(
        model=MODEL, max_tokens=8000, system=INSTRUCTIONS,
        messages=[{"role": "user", "content": f"Question: {question}\n\n<excerpts>\n{sources_text}\n</excerpts>"}],
        output_config={"effort": EFFORT, "format": {"type": "json_schema", "schema": SCHEMA}},
    )
    if response.stop_reason in ("refusal", "max_tokens"):
        raise RuntimeError(f"checker stop_reason={response.stop_reason}")
    entries = json.loads(next(b.text for b in response.content if b.type == "text"))["chunks"]
    by_n = {e["n"]: e for e in entries}
    verdicts = [by_n.get(i, {"verdict": "no", "reason": "(missing from the checker's reply)"}) for i in range(1, n + 1)]
    return verdicts, response.usage


def run_checker(client, question: str, ranked: list[dict], format_sources) -> tuple[list[dict], int, int, int]:
    """Check the top BATCH chunks, and the next BATCH if the first round is unsure (only "partly").
    ranked: chunks in search order (at least 2 * BATCH for the second round). Returns the verdict rows
    (rank, chunk_id, verdict, reason), the number of rounds, and the input / output tokens used."""
    rows, rounds, tokens_in, tokens_out = [], 0, 0, 0
    while rounds < 2:
        start = rounds * BATCH
        batch = ranked[start:start + BATCH]
        if not batch:
            break
        verdicts, usage = check_batch(client, question, format_sources(batch), len(batch))
        tokens_in += usage.input_tokens
        tokens_out += usage.output_tokens
        rows += [{"rank": start + i + 1, "chunk_id": c["chunk_id"], **v} for i, (c, v) in enumerate(zip(batch, verdicts))]
        rounds += 1
        if outcome(rows) != "unsure":  # found something, or nothing related: stop
            break
    return rows, rounds, tokens_in, tokens_out


def cost(tokens_in: int, tokens_out: int) -> float:
    return (tokens_in * PRICE_PER_MTOK["input"] + tokens_out * PRICE_PER_MTOK["output"]) / 1_000_000


def outcome(verdicts: list[dict]) -> str:
    labels = {v["verdict"] for v in verdicts}
    return "good" if "yes" in labels else "unsure" if "partly" in labels else "nothing"


def select(verdicts: list[dict], chunks_by_id: dict[str, dict], k: int = TOP_K) -> tuple[str, list[dict]]:
    """The checker's outcome and the chunks Claude reads (best search rank first)."""
    result = outcome(verdicts)
    wanted = {"good": "yes", "unsure": "partly"}.get(result)
    ranked = sorted(verdicts, key=lambda v: v["rank"])
    return result, [chunks_by_id[v["chunk_id"]] for v in ranked if v["verdict"] == wanted][:k]
