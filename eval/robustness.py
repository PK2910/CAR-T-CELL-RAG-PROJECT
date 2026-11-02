"""
Robustness test for the search (free: no Claude, no API key). Real users do not type the test questions word for
word, so each hand-written question with an answer key is changed in small, realistic ways, and the search is graded
on every version with the same answer key. The answer to the question does not change, so the search result should
not either (an "invariance" test, in Designing Machine Learning Systems' terms).

  lowercase       "how common is icans after car-t in all"  (no capitals, no final "?")
  typo            one letter dropped from the longest word: "leukemia" -> "leukmia"
  spelled out     abbreviations written in full: "ICANS" -> "immune effector cell-associated neurotoxicity syndrome"
  abbreviated     full names shortened: "cytokine release syndrome" -> "CRS"
  keywords        question words and small words dropped: "common ICANS CAR-T ALL"
  no hyphen       "CAR-T" -> "CAR T"

For each change: how many questions it applies to, Hit@10 and MRR before and after (on those questions), and which
questions lost the answer passage from the top 10. Searches the live setup: hybrid search, the index in RAG_INDEX
(default "text"). The relevance check is not part of this test (it costs money); it reads the top 30, so a passage
that drops from #8 to #20 can still reach the answer.

Question sets (--sets): hand (all hand-written: practice, final, holdout; the default), practice (dev + test: used to
adjust changes to the search), generated (synthetic + confirm + mq: Claude-written questions, for judging such changes
on questions not used to make them), seen (final + holdout).
With RAG_QUERY_REWRITE set (src/query_rewrite.py), the search uses the rewritten question.

Run:  python eval/robustness.py [--sets hand|practice|generated|seen]   (stop the API first: one process per index)
      RAG_QUERY_REWRITE=names,spelling python eval/robustness.py --sets generated
Saves eval/results/robustness_<index>_<sets>_<rewrite>.json
"""

import argparse
import json
import re
import statistics
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from index_config import CHUNKS_FILE, INDEX  # noqa: E402
from query_rewrite import REWRITE  # noqa: E402
from question_sets import load_questions  # noqa: E402
from retrieve import ranking  # noqa: E402

MODE = "hybrid"
TOP = 10  # the answer model reads up to 10 passages
SETS = {"hand": {"dev", "test", "final", "holdout"}, "practice": {"dev", "test"},
        "generated": {"synthetic", "confirm", "mq"}, "seen": {"final", "holdout"}}
CHUNK_IDS = [c["chunk_id"] for c in json.loads(CHUNKS_FILE.read_text())]  # list position = index point id

# Abbreviation <-> full name, as users write them. Longest matches first, so "B-ALL" is replaced before "ALL".
TERMS = [
    ("allo-HSCT", "allogeneic stem cell transplant"),
    ("HSCT", "stem cell transplant"),
    ("ICANS", "immune effector cell-associated neurotoxicity syndrome"),
    ("CRS", "cytokine release syndrome"),
    ("B-ALL", "B-cell acute lymphoblastic leukemia"),
    ("ALL", "acute lymphoblastic leukemia"),
    ("MRD", "minimal residual disease"),
    ("GVHD", "graft-versus-host disease"),
    ("CNS", "central nervous system"),
    ("CAR-T cells", "chimeric antigen receptor T cells"),
    ("CAR-T", "chimeric antigen receptor T-cell"),
    ("tisa-cel", "tisagenlecleucel"),
    ("CR", "complete remission"),
]
SMALL_WORDS = set("""a an the of in on for to and or is are was were be been do does did can could should would will
what which who whom whose when where why how much many there their they it its this that these those with without
after before during between from by at as than then into about vs versus have has had any some doesn't don't isn't
aren't""".split())


def lowercase(q: str) -> str:
    return q.lower().rstrip("?!. ")


def typo(q: str) -> str:
    words = re.findall(r"[A-Za-z]{6,}", q)
    if not words:
        return q
    w = max(words, key=len)  # the longest word, usually the medical term that matters most
    return q.replace(w, w[:len(w) // 2] + w[len(w) // 2 + 1:], 1)


def spelled_out(q: str) -> str:
    for short, full in TERMS:
        q = re.sub(rf"(?<![\w-]){re.escape(short)}(?![\w-])", full, q)
    return q


def abbreviated(q: str) -> str:
    for short, full in TERMS:
        q = re.sub(re.escape(full), short, q, flags=re.I)
    return q


def keywords(q: str) -> str:
    return " ".join(w for w in re.findall(r"[\w'+/-]+", q) if w.lower() not in SMALL_WORDS)


def no_hyphen(q: str) -> str:
    return re.sub(r"\bCAR-T\b", "CAR T", q, flags=re.I)


CHANGES = {"lowercase": lowercase, "typo": typo, "spelled out": spelled_out, "abbreviated": abbreviated,
           "keywords": keywords, "no hyphen": no_hyphen}


def best_rank(question: str, relevant: set[str]) -> int:
    """Rank of the best answer-bearing passage (10**6 if none)."""
    return next((r for r, pos in enumerate(ranking(question, MODE), 1) if CHUNK_IDS[pos] in relevant), 10**6)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sets", default="hand", choices=SETS)
    args = parser.parse_args()
    rewrite_label = "+".join(sorted(REWRITE)) or "off"
    out = ROOT / f"eval/results/robustness_{INDEX}_{args.sets}_{rewrite_label}.json"
    questions = [q for q in load_questions() if q["relevant_chunks"] and q["split"] in SETS[args.sets]]
    ranking("warm-up", MODE)
    base = {q["id"]: best_rank(q["question"], set(q["relevant_chunks"])) for q in questions}
    hit = lambda r: r <= TOP
    mrr = lambda r: 1 / r if r <= TOP else 0.0
    print(f"Robustness of the search | {MODE}, index '{INDEX}', query rewrite: {rewrite_label} | {len(questions)} "
          f"{args.sets} questions | top {TOP}\n")
    print(f"Unchanged questions: Hit@{TOP} {100 * statistics.mean(hit(r) for r in base.values()):.0f}%, "
          f"Hit@30 {100 * statistics.mean(r <= 30 for r in base.values()):.0f}%, "
          f"MRR {statistics.mean(mrr(r) for r in base.values()):.2f}\n")
    print(f"{'change':12} {'applies to':>10} {'Hit@10 before':>13} {'after':>6} {'MRR before':>10} {'after':>6}  lost / gained")
    report = {"date": datetime.now().isoformat(timespec="seconds"), "mode": MODE, "index": INDEX, "top": TOP,
              "rewrite": rewrite_label, "sets": args.sets, "questions": len(questions), "baseline_ranks": base,
              "changes": {}}
    for name, change in CHANGES.items():
        rows = []
        for q in questions:
            new = change(q["question"])
            if new != q["question"]:
                rows.append({"id": q["id"], "question": new, "before": base[q["id"]],
                             "after": best_rank(new, set(q["relevant_chunks"]))})
        if not rows:
            continue
        lost = [r for r in rows if hit(r["before"]) and not hit(r["after"])]
        gained = [r for r in rows if not hit(r["before"]) and hit(r["after"])]
        b = 100 * statistics.mean(hit(r["before"]) for r in rows)
        a = 100 * statistics.mean(hit(r["after"]) for r in rows)
        print(f"{name:12} {len(rows):>10} {b:>12.0f}% {a:>5.0f}% {statistics.mean(mrr(r['before']) for r in rows):>10.2f} "
              f"{statistics.mean(mrr(r['after']) for r in rows):>6.2f}  {len(lost)} / {len(gained)}")
        report["changes"][name] = {"n": len(rows), "hit_before": b, "hit_after": a, "lost": [r["id"] for r in lost],
                                   "gained": [r["id"] for r in gained], "rows": rows}

    print("\nQuestions that lost the answer passage from the top 10 (rank before -> after):")
    for name, c in report["changes"].items():
        for r in c["rows"]:
            if r["id"] in c["lost"]:
                after = "not found" if r["after"] >= 10**6 else f"#{r['after']}"
                print(f"  {name:12} {r['id']}: #{r['before']} -> {after}   {r['question'][:80]}")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"\nSaved -> {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
