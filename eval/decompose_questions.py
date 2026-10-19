"""
Multi-query decomposition, step 1: Claude splits each test question into 1-3 short search queries.
Claude sees ONLY the question (never the papers), exactly as it would for a real user.
  - a two-part question -> one search query per part
  - a one-part question -> 1-2 rephrasings of the same thing (it should not invent extra parts)

Uses the Claude API key from .env (run this yourself; about $0.10 for 40 questions). Saves after every
question to eval/results/decompositions_<split>.json, so a stopped run continues where it left off.

Run:  python eval/decompose_questions.py [--split mq]
Then: python eval/eval_multiquery.py     (free search test: today's search vs multi-query)
"""

import argparse
import json
import sys
from pathlib import Path

import anthropic
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")
sys.path.insert(0, str(ROOT / "eval"))
from question_sets import load_questions  # noqa: E402

RESULTS_DIR = ROOT / "eval/results"
MODEL = "claude-sonnet-5-5"
PRICE_PER_MTOK = {"input": 2.00, "output": 10.00}  # Sonnet 5.5, USD per million tokens

SCHEMA = {
    "type": "object",
    "properties": {"sub_queries": {"type": "array", "items": {"type": "string"}}},
    "required": ["sub_queries"], "additionalProperties": False,
}

INSTRUCTIONS = """You prepare search queries for a search engine over research papers on CAR-T cell therapy.

Split the user's question into 1-3 short, self-contained search queries:
- If the question asks about several things (e.g. "How do X and Y compare?"), write one query per thing.
- If it asks about one thing, write 1-2 rephrasings of that same thing. Do not add new topics.
- Keep the user's important terms (drug, trial and gene names, abbreviations) exactly as written.
- Each query is a short phrase or question, under 15 words."""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="mq")
    args = parser.parse_args()

    questions = [q for q in load_questions() if q["split"] == args.split]
    out = RESULTS_DIR / f"decompositions_{args.split}.json"
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    done = json.loads(out.read_text()) if out.exists() else {}

    client = anthropic.Anthropic()
    cost = 0.0
    for q in questions:
        if q["id"] in done:
            continue
        response = client.messages.create(
            model=MODEL, max_tokens=2000, system=INSTRUCTIONS,
            messages=[{"role": "user", "content": f"Question: {q['question']}"}],
            output_config={"effort": "low", "format": {"type": "json_schema", "schema": SCHEMA}},
        )
        cost += (response.usage.input_tokens * PRICE_PER_MTOK["input"]
                 + response.usage.output_tokens * PRICE_PER_MTOK["output"]) / 1_000_000
        subs = ([] if response.stop_reason == "refusal" else
                json.loads(next(b.text for b in response.content if b.type == "text"))["sub_queries"])
        done[q["id"]] = [s.strip() for s in subs if s.strip()][:3]
        out.write_text(json.dumps(done, indent=2, ensure_ascii=False))
        print(f"{q['id']} ({q['type']}): {q['question'][:70]}")
        for s in done[q["id"]]:
            print(f"      -> {s}")

    print(f"\n{len(done)} questions decomposed -> {out}  (this run cost ${cost:.2f})")


if __name__ == "__main__":
    main()
