"""
Step 1 of embedding fine-tuning: make TRAINING PAIRS (question -> the passage that answers it).

Claude reads one passage at a time and writes one realistic question that the passage answers.
Only the question is generated (training needs no answer key), which keeps it cheap.

No leakage: passages used by ANY evaluation set (answer-key pages, judged pages, generated test
questions), and their neighbouring chunks (which overlap them), are never used for training.

Uses the Claude API key from .env (run this yourself). Saves after every question to
train/pairs.jsonl, so a stopped run (e.g. credit ran out) continues where it left off.

Run:  python train/make_pairs.py --dry-run      (counts only: no API call, no key needed)
      python train/make_pairs.py --n 500        (about $1.50-2.50)
"""

import argparse
import json
import random
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CHUNKS_FILE = ROOT / "data/processed/chunks.json"
EVAL_DIR = ROOT / "eval"
OUT_FILE = ROOT / "train/pairs.jsonl"
MODEL = "claude-sonnet-5-5"
PRICE_PER_MTOK = {"input": 2.00, "output": 10.00}  # Sonnet 5.5, USD per million tokens
MIN_TOKENS = 150
SKIP_SECTIONS = ("method", "material", "statistic", "patient selection", "study design", "procedure")
SEED = 4242

SCHEMA = {
    "type": "object",
    "properties": {"usable": {"type": "boolean"}, "question": {"type": "string"}},
    "required": ["usable", "question"],
    "additionalProperties": False,
}

INSTRUCTIONS = """You help build training data for a search engine over research papers on CAR-T cell \
therapy.

Read the passage and write ONE question that a student, researcher or clinician might type into a \
search box, which this passage answers. Rules:
- The question must make sense on its own: never say "this study", "the passage", "the authors" or "Table 2".
- Use everyday wording where possible; don't copy long phrases from the passage.
- It must be a research question about the literature, not advice for a specific person.
- If the passage has nothing worth asking about (references, acknowledgements, pure methods, lab \
protocol details), set usable to false and question to an empty string."""


def eval_chunks() -> set[str]:
    """Every chunk any evaluation set relies on: answer-key pages, reading judgments, generated test sets."""
    used = set()
    for q in json.loads((EVAL_DIR / "questions.json").read_text()):
        used.update(q["relevant_chunks"])
    for qid_judgments in json.loads((EVAL_DIR / "method_b_judgments.json").read_text()).values():
        used.update(qid_judgments)
    for f in EVAL_DIR.glob("synthetic*.json"):
        for q in json.loads(f.read_text()):
            used.update(q["relevant_chunks"])
    return used


def with_neighbours(chunk_ids: set[str]) -> set[str]:
    """Add the chunk before and after each text chunk: neighbours share ~50 overlapping tokens."""
    out = set(chunk_ids)
    for cid in chunk_ids:
        m = re.fullmatch(r"(PMC\d+)_(\d{4})", cid)
        if m:
            n = int(m.group(2))
            out.update({f"{m.group(1)}_{n - 1:04d}", f"{m.group(1)}_{n + 1:04d}"})
    return out


def eligible(c: dict) -> bool:
    sections = " ".join(c["sections"]).lower()
    return c["n_tokens"] >= MIN_TOKENS and not any(s in sections for s in SKIP_SECTIONS)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=500, help="target number of training pairs in pairs.jsonl")
    parser.add_argument("--dry-run", action="store_true", help="only count passages; no API call")
    args = parser.parse_args()

    chunks = json.loads(CHUNKS_FILE.read_text(encoding="utf-8"))
    excluded = with_neighbours(eval_chunks())
    pool = [c for c in chunks if eligible(c) and c["chunk_id"] not in excluded]
    random.Random(SEED).shuffle(pool)

    done = {}
    if OUT_FILE.exists():
        for line in OUT_FILE.read_text().splitlines():
            row = json.loads(line)
            done[row["chunk_id"]] = row
    tried = {cid for cid, row in done.items()}
    have = sum(1 for row in done.values() if row["question"])

    print(f"Eligible passages: {sum(eligible(c) for c in chunks):,}; excluded as evaluation pages "
          f"(+ neighbours): {len(excluded):,}; available for training: {len(pool):,} from "
          f"{len({c['pmcid'] for c in pool})} papers")
    print(f"Already in {OUT_FILE.name}: {have} pairs ({len(tried) - have} passages judged unusable)")
    if args.dry_run or have >= args.n:
        return

    import anthropic  # imported here so --dry-run needs neither the library call nor the API key
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
    client = anthropic.Anthropic()

    cost = 0.0
    with OUT_FILE.open("a", encoding="utf-8") as out:
        for c in pool:
            if have >= args.n:
                break
            if c["chunk_id"] in tried:
                continue
            passage = f"Paper: {c['title']}\n\n{c['text']}"
            response = client.messages.create(
                model=MODEL,
                max_tokens=2000,
                system=INSTRUCTIONS,
                messages=[{"role": "user", "content": f"<passage>\n{passage}\n</passage>"}],
                output_config={"effort": "low", "format": {"type": "json_schema", "schema": SCHEMA}},
            )
            cost += (response.usage.input_tokens * PRICE_PER_MTOK["input"]
                     + response.usage.output_tokens * PRICE_PER_MTOK["output"]) / 1_000_000
            result = (json.loads(next(b.text for b in response.content if b.type == "text"))
                      if response.stop_reason != "refusal" else {"usable": False, "question": ""})
            question = result["question"].strip() if result["usable"] else ""
            out.write(json.dumps({"question": question, "chunk_id": c["chunk_id"], "pmcid": c["pmcid"]},
                                 ensure_ascii=False) + "\n")
            out.flush()  # saved immediately: a stopped run loses nothing
            if question:
                have += 1
                if have % 25 == 0:
                    print(f"  {have}/{args.n} pairs  (cost so far ${cost:.2f})")

    print(f"\nDone: {have} training pairs in {OUT_FILE}  (this run cost ${cost:.2f})")


if __name__ == "__main__":
    main()
