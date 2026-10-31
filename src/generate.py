"""
Box 6: GENERATE
Retrieve the top chunks for a question, keep the ones a relevance check says answer it (src/relevance_check.py),
put them into a prompt as numbered sources, and ask Claude to answer ONLY from those sources, with citations.

Run:  python src/generate.py "your question here"
Needs ANTHROPIC_API_KEY in the project's .env file.
"""

import os
import sys
import time

import anthropic
from dotenv import load_dotenv

from embed import ROOT
from guardrails import PERSONAL_NOTE, check_answer, is_personal
from relevance_check import BATCH, NOT_FOUND_ANSWER, PARTLY_NOTE, run_checker, select
from relevance_check import cost as checker_cost
from retrieve import SEARCH_MODE, TOP_K, load_chunks, retrieve

load_dotenv(ROOT / ".env")  # puts ANTHROPIC_API_KEY into the environment

MODEL = "claude-sonnet-5-5"
EFFORT = "low"  # how much the model thinks before answering; comparing levels is a later experiment
PRICE_PER_MTOK = {"input": 2.00, "output": 10.00}  # Sonnet 5.5, USD per million tokens
# Time limit for the answer call, in seconds, with one retry. If it runs out (or the model service is down, or the
# credit has run out), the user gets the passages that were found instead (unavailable()).
# ANSWER_TIMEOUT_S in the environment overrides it (e.g. a tiny value to test the fallback).
ANSWER_TIMEOUT_S = float(os.environ.get("ANSWER_TIMEOUT_S", 45))

UNAVAILABLE_SOURCES = ("A summary could not be written right now, so here are the passages most relevant to your "
                       "question, with links to the papers. This is a research and education tool, not medical advice.")
# Personal questions get no raw passages: they could contain doses, which the dose guardrail exists to block
UNAVAILABLE_PERSONAL = ("The answer service is unavailable right now. For questions about a specific person's care, "
                        "please contact the treating team. This is a research and education tool, not medical advice.")

SYSTEM_PROMPT = """You are a research assistant that answers questions about CAR-T cell therapy \
in acute lymphoblastic leukemia (ALL) using excerpts from published papers.

Rules:
- Answer ONLY from the numbered sources provided. Do not add facts from your own knowledge.
- Cite every factual claim with the source number in square brackets, e.g. [2] or [1][3].
- If the sources do not contain the answer, say so plainly instead of guessing.
- If sources disagree, give the range of findings, cite each one, and note differences in \
study type (e.g. trial vs. review vs. case report), patient population (pediatric vs. adult), and year.
- Prefer original studies and systematic reviews over narrative reviews and case reports, and say \
which kind of evidence each claim comes from when it matters.
- Read numbers carefully: make sure each percentage refers to the outcome being asked about.
- This is a literature tool, not medical advice for individual patients."""


def format_sources(chunks: list[dict]) -> str:
    """Number each chunk and label it, so the model can cite it as [n]."""
    blocks = []
    for n, c in enumerate(chunks, 1):
        label = (f"Title: {c['title']}\n"
                 f"{c['first_author']} et al., {c['journal']} ({c['year']}) · {c['article_type']} · "
                 f"section: {', '.join(c['sections'])}")
        blocks.append(f'<source id="{n}">\n{label}\n\n{c["text"]}\n</source>')
    return "\n\n".join(blocks)


def seconds_since(start: float) -> float:
    return round(time.perf_counter() - start, 2)


def not_found(question: str, checker: dict) -> dict:
    """Guardrail reply when the relevance check finds nothing related: no answer is written."""
    guard = check_answer(question, NOT_FOUND_ANSWER)
    return {"question": question, "answer": NOT_FOUND_ANSWER, "original_answer": None,
            "guardrail": {"personal": guard["personal"], "doses_found": [], "action": "not_found"},
            "sources": [], "model": None, "stop_reason": None, "input_tokens": 0, "output_tokens": 0,
            "cost_usd": checker.get("checker_cost_usd", 0.0), **checker}


def prepare(client, question: str, k: int = TOP_K, mode: str = SEARCH_MODE, chunks: list[dict] | None = None,
            extra_note: str = "", check: bool = True) -> dict:
    """Everything before Claude writes: search, relevance check, input guardrail, prompt.
    Returns {"done": <final result>} when no answer is written (nothing related was found),
    else {"chunks", "prompt", "checker", "timing"}. Shared by answer() and answer_stream().
    timing: seconds spent searching and checking (for the logbook, api/logbook.py)."""
    checker, timing = {}, {}
    if chunks is None and check:
        start = time.perf_counter()
        ranked = retrieve(question, k=2 * BATCH, mode=mode)
        timing["search_s"] = seconds_since(start)
        start = time.perf_counter()
        try:
            rows, rounds, tokens_in, tokens_out = run_checker(client, question, ranked, format_sources)
            result, chunks = select(rows, {c["chunk_id"]: c for c in ranked}, k)
            checker = {"checker_outcome": result, "checker_rounds": rounds,
                       "checker_cost_usd": round(checker_cost(tokens_in, tokens_out), 5)}
        except (anthropic.APIError, RuntimeError, ValueError, KeyError, StopIteration) as e:
            # Fail gracefully: without a working check, answer from the top k chunks as before
            result, chunks = "check_failed", ranked[:k]
            checker = {"checker_outcome": result, "checker_error": f"{type(e).__name__}: {e}", "checker_cost_usd": 0.0}
        timing["check_s"] = seconds_since(start)
        if result == "nothing":
            return {"done": {**not_found(question, checker), "timing": timing}}
        if result == "unsure":
            extra_note = PARTLY_NOTE
    elif chunks is None:
        start = time.perf_counter()
        chunks = retrieve(question, k=k, mode=mode)
        timing["search_s"] = seconds_since(start)
    personal = is_personal(question)  # input guardrail
    note = f"\n\n{PERSONAL_NOTE}" if personal else ""
    note += f"\n\n{extra_note}" if extra_note else ""
    prompt = f"<sources>\n{format_sources(chunks)}\n</sources>\n\nQuestion: {question}{note}"
    return {"chunks": chunks, "prompt": prompt, "checker": checker, "timing": timing}


def request(prompt: str) -> dict:
    """The answer request to Claude (the same for a normal and a streamed answer)."""
    return dict(
        model=MODEL,
        max_tokens=16000,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": prompt}],
        output_config={"effort": EFFORT},
        # If a safety filter declines, retry on a fallback model where Anthropic supports it
        betas=["server-side-fallback-2026-07-01"],
        fallbacks="default",
    )


def finish(question: str, response, prepared: dict) -> dict:
    """Everything after Claude writes: output guardrail, cost, and the result record."""
    checker, chunks = prepared["checker"], prepared["chunks"]
    if response.stop_reason == "refusal":  # a decline arrives as a normal response, not an error
        category = response.stop_details.category if response.stop_details else None
        text = f"[The model declined to answer this question (category: {category}).]"
    else:
        text = "".join(b.text for b in response.content if b.type == "text")

    guard = check_answer(question, text)  # output guardrail: replaces the answer if a dose slipped through

    usage = response.usage
    cost = (usage.input_tokens * PRICE_PER_MTOK["input"]
            + usage.output_tokens * PRICE_PER_MTOK["output"]) / 1_000_000 + checker.get("checker_cost_usd", 0.0)
    return {
        "question": question,
        "answer": guard["answer"],
        "original_answer": text if guard["action"] == "replaced" else None,  # kept for evaluation only
        "guardrail": {"personal": guard["personal"], "doses_found": guard["doses_found"], "action": guard["action"]},
        "sources": chunks,
        "model": response.model,
        "stop_reason": response.stop_reason,
        "input_tokens": usage.input_tokens,
        "output_tokens": usage.output_tokens,
        "cost_usd": round(cost, 5),  # answer + relevance check
        **checker,
    }


def unavailable(question: str, prepared: dict, error: Exception) -> dict:
    """Fail gracefully when no answer can be written (model service down, too slow, credit used up): return the
    passages that were found, so the user can still read them. Personal questions get a safe message instead."""
    personal = is_personal(question)
    checker = prepared["checker"]
    return {"question": question, "answer": UNAVAILABLE_PERSONAL if personal else UNAVAILABLE_SOURCES,
            "original_answer": None,
            "guardrail": {"personal": personal, "doses_found": [], "action": "unavailable"},
            "sources": [] if personal else prepared["chunks"], "model": None, "stop_reason": None,
            "input_tokens": 0, "output_tokens": 0, "cost_usd": checker.get("checker_cost_usd", 0.0),
            "degraded": True, "error": f"{type(error).__name__}: {error}", **checker}


def answer(question: str, k: int = TOP_K, mode: str = SEARCH_MODE, chunks: list[dict] | None = None,
           extra_note: str = "", check: bool = True, fallback: bool = True) -> dict:
    """Full RAG: retrieve -> relevance check -> build prompt -> ask Claude. Returns the answer plus everything
    needed to inspect or evaluate it (sources, token usage, cost).
    mode: "vector" (baseline), "hybrid" (vector + BM25, the default), or "rerank" (hybrid + reranker).
    check: run the LLM relevance check (src/relevance_check.py) on the top 30 chunks (and #31-60 when unsure),
           and answer only from the chunks it keeps. check=False: answer from the top k chunks (the tests before
           2026-10-27). If the check itself fails, the answer falls back to the top k chunks.
    chunks: answer from these chunks instead of searching (used by answer_checked).
    fallback: if the answer cannot be written, return the found passages instead (unavailable()).
              The evaluation scripts set it to False, so a failure shows up as an error, not as a weak answer.
    Safety guardrails (src/guardrails.py) run on the question before and on the answer after Claude."""
    client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY; retries 429/5xx errors automatically
    prepared = prepare(client, question, k, mode, chunks, extra_note, check)
    if "done" in prepared:
        return prepared["done"]
    start = time.perf_counter()
    try:
        response = client.with_options(timeout=ANSWER_TIMEOUT_S, max_retries=1).beta.messages.create(
            **request(prepared["prompt"]))
    except anthropic.APIError as e:  # includes timeouts, connection errors, and a used-up credit
        if not fallback:
            raise
        result = unavailable(question, prepared, e)
    else:
        result = finish(question, response, prepared)
    return {**result, "timing": {**prepared["timing"], "answer_s": seconds_since(start)}}


def answer_stream(question: str):
    """The same answer as answer(), sent piece by piece. Yields events:
      {"type": "status", "text": ...}   progress before the answer starts
      {"type": "text", "text": ...}     the next piece of the answer
      {"type": "done", "result": ...}   the full result record (as answer() returns it)
    Personal questions are NOT streamed: the output guardrail must check the whole answer for doses before
    anyone sees it, so their answer arrives in one piece after the check.
    The answer in the "done" event is the final one: it replaces the streamed pieces (e.g. if the model service
    failed halfway, it is the fallback message, and the sources are the found passages)."""
    client = anthropic.Anthropic()
    timed = client.with_options(timeout=ANSWER_TIMEOUT_S, max_retries=1)
    yield {"type": "status", "text": "Searching the papers and checking which passages answer the question..."}
    prepared = prepare(client, question)
    if "done" in prepared:
        yield {"type": "text", "text": prepared["done"]["answer"]}
        yield {"type": "done", "result": prepared["done"]}
        return
    yield {"type": "status", "text": f"Writing the answer from {len(prepared['chunks'])} passages..."}
    start = time.perf_counter()
    try:
        if is_personal(question):
            result = finish(question, timed.beta.messages.create(**request(prepared["prompt"])), prepared)
            yield {"type": "text", "text": result["answer"]}
        else:
            with timed.beta.messages.stream(**request(prepared["prompt"])) as stream:
                for piece in stream.text_stream:
                    yield {"type": "text", "text": piece}
                result = finish(question, stream.get_final_message(), prepared)
    except anthropic.APIError as e:  # fail gracefully: the found passages instead of an error
        result = unavailable(question, prepared, e)
    result["timing"] = {**prepared["timing"], "answer_s": seconds_since(start)}
    yield {"type": "done", "result": result}


def answer_checked(question: str, verdicts: list[dict]) -> dict:
    """Like answer(), but with the checker's verdicts already saved (eval/check_chunks.py), so the
    tests don't pay for the check twice. Same rules as the live check (src/relevance_check.py)."""
    result, chunks = select(verdicts, {c["chunk_id"]: c for c in load_chunks()})
    if result == "nothing":
        return not_found(question, {"checker_outcome": result})
    out = answer(question, chunks=chunks, extra_note=PARTLY_NOTE if result == "unsure" else "", fallback=False)
    return {**out, "checker_outcome": result}


def show(result: dict) -> None:
    print(f"\nQ: {result['question']}\n")
    print(result["answer"])
    print("\nSources:")
    for n, c in enumerate(result["sources"], 1):
        print(f"  [{n}] {c['first_author']} et al. ({c['year']}) {c['title'][:80]}... "
              f"https://pubmed.ncbi.nlm.nih.gov/{c['pmid']}/")
    print(f"\n({result['model']}, {result['input_tokens']:,} tokens in / {result['output_tokens']:,} out, "
          f"~${result['cost_usd']:.4f} incl. relevance check)")
    if result.get("degraded"):
        print(f"[fallback] no answer could be written ({result['error']}); the found passages are shown instead")
    if "checker_outcome" in result:
        print(f"[relevance check] {result['checker_outcome']}"
              + (f" ({result['checker_error']}; answered from the top chunks instead)" if "checker_error" in result
                 else f", {len(result['sources'])} chunks kept"))
    g = result["guardrail"]
    if g["personal"]:
        print(f"[guardrail] personal question detected; answer {'REPLACED (dose found: ' + ', '.join(g['doses_found']) + ')' if g['action'] == 'replaced' else 'passed'}")


if __name__ == "__main__":
    q = " ".join(sys.argv[1:]) or "What drugs are used to manage cytokine release syndrome after CAR-T?"
    try:
        show(answer(q))
    except anthropic.AuthenticationError:
        print("API key missing or invalid. Put your key in .env as ANTHROPIC_API_KEY=... and try again.")
