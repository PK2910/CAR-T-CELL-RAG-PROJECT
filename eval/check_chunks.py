"""
LLM relevance check (Corrective-RAG-style), Step A: does a checker pick the chunks that answer the question?

For each test question:
  1. hybrid search -> top 30 chunks (the same search as today, free)
  2. Claude Haiku 5.5 reads the question and all 30 chunks and marks each one:
       yes     states information that answers the question (or one part of it)
       partly  same topic, but not the specific information asked for
       no      not about what was asked
  3. "unsure" (no chunk marked yes, at least one partly): check the next 30 chunks (#31-60) the same way
Nothing is answered here; eval/checker_report.py then compares, for free, the chunks the checker keeps with the
10 chunks Claude reads today, using the answer key.

Uses the Claude API key from .env (run this yourself). About 0.3 cents per question; the run prints its real cost.
Saves after every question to eval/results/chunk_checks.json, so a stopped run continues where it left off.

Run:  python eval/check_chunks.py --limit 5     (try 5 questions first, then look at the cost)
      python eval/check_chunks.py               (all questions in practice, final, holdout, synthetic, confirm)
Then: python eval/checker_report.py             (free)
"""

import argparse
import json
import sys
from pathlib import Path

import anthropic
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "eval"))
from generate import format_sources  # noqa: E402  (chunks labelled exactly as Claude sees them when answering)
from question_sets import load_questions  # noqa: E402
from retrieve import load_chunks, ranking  # noqa: E402

RESULTS_FILE = ROOT / "eval/results/chunk_checks.json"
MODEL = "claude-haiku-5-5"
EFFORT = "low"
PRICE_PER_MTOK = {"input": 0.10, "output": 0.50}  # Haiku 5.5, prompts up to 100K tokens; USD per million tokens
BATCH = 30          # chunks checked per call: #1-30, and #31-60 for "unsure" questions
SPLITS = {"dev", "test", "final", "holdout", "synthetic", "confirm"}

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


def check(client: anthropic.Anthropic, question: str, chunks: list[dict]) -> tuple[list[dict], object]:
    """One checker call for up to BATCH chunks: their verdicts (in input order) and the API usage."""
    response = client.messages.create(
        model=MODEL, max_tokens=8000, system=INSTRUCTIONS,
        messages=[{"role": "user",
                   "content": f"Question: {question}\n\n<excerpts>\n{format_sources(chunks)}\n</excerpts>"}],
        output_config={"effort": EFFORT, "format": {"type": "json_schema", "schema": SCHEMA}},
    )
    if response.stop_reason in ("refusal", "max_tokens"):
        raise RuntimeError(f"stop_reason={response.stop_reason}")
    entries = json.loads(next(b.text for b in response.content if b.type == "text"))["chunks"]
    by_n = {e["n"]: e for e in entries}
    verdicts = [by_n.get(n, {"verdict": "no", "reason": "(missing from the checker's reply)"})
                for n in range(1, len(chunks) + 1)]
    return verdicts, response.usage


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None, help="only check this many new questions")
    args = parser.parse_args()

    questions = [q for q in load_questions() if q["split"] in SPLITS]
    chunks = load_chunks()
    RESULTS_FILE.parent.mkdir(parents=True, exist_ok=True)
    done = json.loads(RESULTS_FILE.read_text()) if RESULTS_FILE.exists() else {}

    client = anthropic.Anthropic()
    cost, new = 0.0, 0
    for q in questions:
        if q["id"] in done:
            continue
        if args.limit is not None and new >= args.limit:
            break
        order = ranking(q["question"], "hybrid")
        rows, rounds, usage_in, usage_out = [], 0, 0, 0
        try:
            while rounds < 2:
                start = rounds * BATCH
                batch = [chunks[pos] for pos in order[start:start + BATCH]]
                verdicts, usage = check(client, q["question"], batch)
                usage_in += usage.input_tokens
                usage_out += usage.output_tokens
                rows += [{"rank": start + i + 1, "chunk_id": c["chunk_id"], **v}
                         for i, (c, v) in enumerate(zip(batch, verdicts))]
                rounds += 1
                labels = {r["verdict"] for r in rows}
                if "yes" in labels or "partly" not in labels:  # found something, or nothing related: stop
                    break                                      # unsure (only "partly"): check the next 30
        except anthropic.APIError as e:
            if "credit balance" in str(e):
                print(f"\nStopped at {q['id']}: the API credit ran out. Add credit and re-run the same command; "
                      f"{len(done)} questions so far are saved and will be kept.")
                break
            print(f"{q['id']}: API error, skipped (re-run to retry): {e}")
            continue
        except RuntimeError as e:
            print(f"{q['id']}: {e}, skipped (re-run to retry)")
            continue

        q_cost = (usage_in * PRICE_PER_MTOK["input"] + usage_out * PRICE_PER_MTOK["output"]) / 1_000_000
        cost += q_cost
        new += 1
        done[q["id"]] = {"question": q["question"], "split": q["split"], "model": MODEL, "effort": EFFORT,
                         "rounds": rounds, "input_tokens": usage_in, "output_tokens": usage_out,
                         "cost_usd": round(q_cost, 5), "chunks": rows}
        RESULTS_FILE.write_text(json.dumps(done, indent=2, ensure_ascii=False))
        n_yes = sum(r["verdict"] == "yes" for r in rows)
        n_partly = sum(r["verdict"] == "partly" for r in rows)
        print(f"{q['id']:4} yes {n_yes:2} | partly {n_partly:2} | rounds {rounds} | ${q_cost:.4f} | "
              f"{q['question'][:60]}")

    print(f"\n{new} questions checked this run, ${cost:.3f} "
          f"(${cost / max(new, 1):.4f} per question). {len(done)} saved in total -> "
          f"{RESULTS_FILE.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
