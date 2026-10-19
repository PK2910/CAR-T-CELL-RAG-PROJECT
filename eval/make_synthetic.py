"""
Make a SYNTHETIC question set: Claude writes one question (plus its answer key) for each of N random
passages from our papers. The passage a question was written from is its known "correct page".

Why: hand-written answer keys are slow. Generated questions are new to both search versions, so they
give a fair comparison (e.g. text index vs plus index). Known weakness: generated questions tend to reuse
the passage's own words, so they are easier than real users' questions; compare versions, don't read
the absolute scores as real-world accuracy.

Uses the Claude API key from .env (run this yourself; about 1 cent per question).
Saves: eval/synthetic_questions.json (split "synthetic"), or eval/synthetic_<name>.json (split <name>)
       Passages already used by any earlier synthetic set are never picked again.

Run:  python eval/make_synthetic.py [--n 30] [--seed 2026]                      (the first set, s01-s30)
      python eval/make_synthetic.py --n 50 --seed 7 --name confirm --prefix c   (a second set, c01-c50)
Then: python eval/run_answers.py --mode hybrid --split synthetic --label synthetic_text_hybrid   (and judge.py)
"""

import argparse
import json
import random
from pathlib import Path

import anthropic
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

CHUNKS_FILE = ROOT / "data/processed/chunks.json"  # text chunks only: the correct page exists in BOTH indexes
EVAL_DIR = ROOT / "eval"
MODEL = "claude-sonnet-5-5"
PRICE_PER_MTOK = {"input": 2.00, "output": 10.00}  # Sonnet 5.5, USD per million tokens
MIN_TOKENS = 150  # skip short passages: too little to ask about
SKIP_SECTIONS = ("method", "material", "statistic", "patient selection", "study design", "procedure")

SCHEMA = {
    "type": "object",
    "properties": {
        "usable": {"type": "boolean"},
        "question": {"type": "string"},
        "reference_answer": {"type": "string"},
        "must_include": {"type": "array", "items": {"type": "string"}},
        "question_type": {"type": "string", "enum": ["fact_number", "mechanism", "comparison", "side_effect", "other"]},
    },
    "required": ["usable", "question", "reference_answer", "must_include", "question_type"],
    "additionalProperties": False,
}

INSTRUCTIONS = """You help build an evaluation set for a search tool over research papers on CAR-T cell \
therapy in acute lymphoblastic leukemia (ALL).

Read the passage and write ONE question that a student, researcher or clinician might ask, which this \
passage answers. Rules:
- The question must make sense on its own: never say "this study", "the passage", "the authors" or "Table 2".
- Ask about the topic in general words, the way a person would; don't copy long phrases from the passage.
- It must be a research question about the literature, not advice for a specific person.
- It must not repeat a question already in the list you are given.
- reference_answer: the correct answer in 1-3 sentences, using only what the passage says.
- must_include: 1-3 short key facts a correct answer must contain (e.g. "about 81% remission").
- If the passage has nothing worth asking about (references, acknowledgements, pure methods, lab \
protocol details), set usable to false and leave the other fields as empty strings/lists."""


def eligible(c: dict) -> bool:
    sections = " ".join(c["sections"]).lower()
    return c["n_tokens"] >= MIN_TOKENS and not any(s in sections for s in SKIP_SECTIONS)


def ask(client: anthropic.Anthropic, c: dict, earlier: list[str]) -> tuple[dict | None, float]:
    passage = f"Paper: {c['title']} ({c['first_author']} {c['year']}, {c['article_type']})\n\n{c['text']}"
    already = "\n".join(f"- {q}" for q in earlier) or "(none yet)"
    response = client.messages.create(
        model=MODEL,
        max_tokens=4000,
        system=INSTRUCTIONS,
        messages=[{"role": "user", "content": f"<questions_already_written>\n{already}\n</questions_already_written>\n\n"
                                              f"<passage>\n{passage}\n</passage>"}],
        output_config={"effort": "low", "format": {"type": "json_schema", "schema": SCHEMA}},
    )
    cost = (response.usage.input_tokens * PRICE_PER_MTOK["input"]
            + response.usage.output_tokens * PRICE_PER_MTOK["output"]) / 1_000_000
    if response.stop_reason == "refusal":
        return None, cost
    return json.loads(next(b.text for b in response.content if b.type == "text")), cost


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=30, help="how many usable questions to make")
    parser.add_argument("--seed", type=int, default=2026, help="fixed seed, so the same passages are picked every time")
    parser.add_argument("--name", default="synthetic", help="set name = its --split name (and file name)")
    parser.add_argument("--prefix", default="s", help="question id prefix, e.g. c -> c01, c02 ...")
    args = parser.parse_args()
    out_file = EVAL_DIR / ("synthetic_questions.json" if args.name == "synthetic" else f"synthetic_{args.name}.json")
    # Passages and ids already used by OTHER synthetic sets, so a new set is genuinely new
    used_chunks, used_ids = set(), set()
    for f in EVAL_DIR.glob("synthetic*.json"):
        if f != out_file:
            for q in json.loads(f.read_text()):
                used_chunks.update(q["relevant_chunks"])
                used_ids.add(q["id"])

    chunks = json.loads(CHUNKS_FILE.read_text(encoding="utf-8"))
    by_paper: dict[str, list[dict]] = {}
    for c in chunks:
        if eligible(c) and c["chunk_id"] not in used_chunks:
            by_paper.setdefault(c["pmcid"], []).append(c)
    rng = random.Random(args.seed)
    papers = sorted(by_paper)
    rng.shuffle(papers)  # one passage per paper, so questions spread across many papers

    client = anthropic.Anthropic()
    questions, cost, skipped = [], 0.0, 0
    for pmcid in papers:
        if len(questions) == args.n:
            break
        c = rng.choice(by_paper[pmcid])
        result, c_cost = ask(client, c, [q["question"] for q in questions])
        cost += c_cost
        if not result or not result["usable"] or not result["question"].strip():
            skipped += 1
            continue
        qid = f"{args.prefix}{len(questions) + 1:02d}"
        if qid in used_ids:
            raise SystemExit(f"Id {qid} is already used by another synthetic set; choose a different --prefix.")
        questions.append({
            "id": qid,
            "question": result["question"].strip(),
            "type": result["question_type"],
            "checks": ["number_accuracy"] if result["question_type"] == "fact_number" else [],
            "split": args.name,
            "answerable": True,
            "reference_answer": result["reference_answer"].strip(),
            "expected_behavior": None,
            "must_include": result["must_include"],
            "must_not": [],
            "relevant_papers": [c["pmcid"]],
            "relevant_chunks": [c["chunk_id"]],  # only the source passage is known; other papers may also answer
            "added_by_method_b": [],
            "label_quality": "approximate",
            "generated_by": MODEL,
        })
        print(f"[{qid}] {c['chunk_id']:20} {result['question'][:80]}")

    out_file.write_text(json.dumps(questions, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n{len(questions)} questions -> {out_file}  (skipped {skipped} unsuitable passages, cost ${cost:.3f})")
    if len(questions) < args.n:
        print(f"Only {len(questions)} usable questions were found; lower --n or change --seed.")


if __name__ == "__main__":
    main()
