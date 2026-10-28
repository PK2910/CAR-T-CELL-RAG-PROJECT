"""
Box 6: GENERATE
Retrieve the top chunks for a question, put them into a prompt as numbered
sources, and ask Claude to answer ONLY from those sources, with citations.

Run:  python src/generate.py "your question here"
Needs ANTHROPIC_API_KEY in the project's .env file.
"""

import sys

import anthropic
from dotenv import load_dotenv

from embed import ROOT
from guardrails import PERSONAL_NOTE, check_answer, is_personal
from relevance_check import NOT_FOUND_ANSWER, PARTLY_NOTE, select
from retrieve import SEARCH_MODE, TOP_K, load_chunks, retrieve

load_dotenv(ROOT / ".env")  # puts ANTHROPIC_API_KEY into the environment

MODEL = "claude-sonnet-5-5"
EFFORT = "low"  # how much the model thinks before answering; comparing levels is a later experiment
PRICE_PER_MTOK = {"input": 2.00, "output": 10.00}  # Sonnet 5.5, USD per million tokens

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


def answer(question: str, k: int = TOP_K, mode: str = SEARCH_MODE, chunks: list[dict] | None = None,
           extra_note: str = "") -> dict:
    """Full RAG: retrieve -> build prompt -> ask Claude. Returns the answer plus everything
    needed to inspect or evaluate it (sources, token usage, cost).
    mode: "vector" (baseline), "hybrid" (vector + BM25, the default), or "rerank" (hybrid + reranker).
    chunks: answer from these chunks instead of searching (used by the relevance check, answer_checked).
    Safety guardrails (src/guardrails.py) run on the question before and on the answer after Claude."""
    if chunks is None:
        chunks = retrieve(question, k=k, mode=mode)
    personal = is_personal(question)  # input guardrail
    note = f"\n\n{PERSONAL_NOTE}" if personal else ""
    note += f"\n\n{extra_note}" if extra_note else ""
    prompt = f"<sources>\n{format_sources(chunks)}\n</sources>\n\nQuestion: {question}{note}"

    client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY; retries 429/5xx errors automatically
    response = client.beta.messages.create(
        model=MODEL,
        max_tokens=16000,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": prompt}],
        output_config={"effort": EFFORT},
        # If a safety filter declines, retry on a fallback model where Anthropic supports it
        betas=["server-side-fallback-2026-07-01"],
        fallbacks="default",
    )

    if response.stop_reason == "refusal":  # a decline arrives as a normal response, not an error
        category = response.stop_details.category if response.stop_details else None
        text = f"[The model declined to answer this question (category: {category}).]"
    else:
        text = "".join(b.text for b in response.content if b.type == "text")

    guard = check_answer(question, text)  # output guardrail: replaces the answer if a dose slipped through

    usage = response.usage
    cost = (usage.input_tokens * PRICE_PER_MTOK["input"]
            + usage.output_tokens * PRICE_PER_MTOK["output"]) / 1_000_000
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
        "cost_usd": round(cost, 5),
    }


def answer_checked(question: str, verdicts: list[dict]) -> dict:
    """Answer after the LLM relevance check (src/relevance_check.py), given the checker's verdicts:
    "yes" chunks -> normal answer; only "partly" -> answer with a may-only-partly-answer note;
    nothing related -> guardrail: a fixed reply, no answer is written (no API call)."""
    result, chunks = select(verdicts, {c["chunk_id"]: c for c in load_chunks()})
    if result == "nothing":
        guard = check_answer(question, NOT_FOUND_ANSWER)
        return {"question": question, "answer": NOT_FOUND_ANSWER, "original_answer": None,
                "guardrail": {"personal": guard["personal"], "doses_found": [], "action": "not_found"},
                "sources": [], "model": None, "stop_reason": None, "input_tokens": 0, "output_tokens": 0,
                "cost_usd": 0.0, "checker_outcome": result}
    out = answer(question, chunks=chunks, extra_note=PARTLY_NOTE if result == "unsure" else "")
    return {**out, "checker_outcome": result}


def show(result: dict) -> None:
    print(f"\nQ: {result['question']}\n")
    print(result["answer"])
    print("\nSources:")
    for n, c in enumerate(result["sources"], 1):
        print(f"  [{n}] {c['first_author']} et al. ({c['year']}) {c['title'][:80]}... "
              f"https://pubmed.ncbi.nlm.nih.gov/{c['pmid']}/")
    print(f"\n({result['model']}, {result['input_tokens']:,} tokens in / {result['output_tokens']:,} out, "
          f"~${result['cost_usd']:.4f})")
    g = result["guardrail"]
    if g["personal"]:
        print(f"[guardrail] personal question detected; answer {'REPLACED (dose found: ' + ', '.join(g['doses_found']) + ')' if g['action'] == 'replaced' else 'passed'}")


if __name__ == "__main__":
    q = " ".join(sys.argv[1:]) or "What drugs are used to manage cytokine release syndrome after CAR-T?"
    try:
        show(answer(q))
    except anthropic.AuthenticationError:
        print("API key missing or invalid. Put your key in .env as ANTHROPIC_API_KEY=... and try again.")
