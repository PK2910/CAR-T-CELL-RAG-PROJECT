"""
LLM relevance check (Corrective-RAG-style): turn the checker's verdicts into what Claude reads.

The checker (eval/check_chunks.py) marks each of the top 30 chunks (and #31-60 when unsure) as
  yes / partly / no. Then:
  good     at least one "yes"  -> Claude answers from the "yes" chunks (search order, at most TOP_K)
  unsure   only "partly"       -> Claude answers from the "partly" chunks, told they may only partly answer
  nothing  every chunk "no"    -> guardrail: no answer is written, a fixed reply is returned instead

Not the full Corrective RAG method: no web search (medical answers come only from the curated papers) and no
sentence-level refinement inside chunks.
"""

from retrieve import TOP_K

NOT_FOUND_ANSWER = ("The papers in this collection don't answer this question. "
                    "This is a research and education tool, not medical advice.")

PARTLY_NOTE = ("Note: an automatic check found that these sources are related to the question but may not contain "
               "the specific information asked for. If they do not fully answer it, say that the papers only partly "
               "address the question, and state what they do and do not cover.")


def outcome(verdicts: list[dict]) -> str:
    labels = {v["verdict"] for v in verdicts}
    return "good" if "yes" in labels else "unsure" if "partly" in labels else "nothing"


def select(verdicts: list[dict], chunks_by_id: dict[str, dict], k: int = TOP_K) -> tuple[str, list[dict]]:
    """The checker's outcome and the chunks Claude reads (best search rank first)."""
    result = outcome(verdicts)
    wanted = {"good": "yes", "unsure": "partly"}.get(result)
    ranked = sorted(verdicts, key=lambda v: v["rank"])
    return result, [chunks_by_id[v["chunk_id"]] for v in ranked if v["verdict"] == wanted][:k]
