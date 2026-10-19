"""
Level-2 diagnosis: for every key fact that was NOT on the 5 pages Claude got,
find out WHERE in the pipeline it was lost. No Claude, no API key.

  in a chunk?          -> at what rank did the search put that chunk?
                            6-20 ranking (close miss) | 21-100 ranking (far miss) | 100+ chunking/matching
  only in a table?     -> parsing (tables are not chunked in the baseline)
  only in the raw XML? -> parsing (text dropped during cleaning)
  nowhere?             -> data (not in our papers)

Run:  python eval/diagnose_retrieval.py
"""

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from index_config import CHUNKS_FILE  # noqa: E402  (same chunks as the index in use)
from retrieve import COLLECTION, QUERY_PREFIX, load_client, load_model  # noqa: E402

PAPERS_FILE = ROOT / "data/processed/papers.json"
RAW_DIR = ROOT / "data/raw"
QUESTIONS_FILE = ROOT / "eval/questions.json"
SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9(\[])|\n+")

T_ALL = r"T-ALL|T.cell acute lymphoblastic|T.cell ALL"
ICANS = r"ICANS|neurotoxicit|neurologic(al)? (events|toxicit)"
ADULT = r"(?<!young )adults?\b"
KIDS = r"pediatric|paediatric|children"

# The 12 facts lost on the search side (from judged_baseline_dev.json).
# Each rule: every regex must match within ONE sentence ("context" = anywhere in the same chunk/table/paper).
FACTS = [
    ("q05", "ICANS frequency as a range", {"sentence": [ICANS, r"\d\s?%", r"occur|incidence|range|developed|experienced|reported"]}),
    ("q05", "severe ICANS less common", {"sentence": [r"(severe|grade\s*(≥|>=)?\s*3).{0,40}(ICANS|neurotox)|(ICANS|neurotox).{0,40}(severe|grade\s*(≥|>=)?\s*3)", r"\d\s?%"]}),
    ("q05", "higher ICANS rates/severity in adults", {"sentence": [ICANS, ADULT, KIDS]}),
    ("q07", "CD22 CAR-T remission rate after CD19 CAR-T", {"sentence": [r"CD22", r"\d\s?%", r"remission|\bCR\b|response"], "context": [r"CD19"]}),
    ("q09", "trogocytosis or epitope masking", {"sentence": [r"trogocytosis|epitope mask|mask(ed|ing)"]}),
    ("q14", "adults have more severe toxicity", {"sentence": [ADULT, KIDS, r"CRS|ICANS|toxicit|neurotox", r"higher|more|severe|whereas|compared|than"]}),
    ("q14", "children have more durable outcomes", {"sentence": [ADULT, KIDS, r"durab|sustain|relapse|long-term|survival"]}),
    ("q15", "transplant may benefit selected high-risk patients", {"sentence": [r"high.risk|selected patients|subgroup", r"transplant|HSCT|HCT|\bSCT\b"], "context": [r"CAR"]}),
    ("q17", "shared antigens (malignant vs normal/CAR T cells)", {"sentence": [r"shar(e|ed|ing)", r"antigen|CD7|CD5"], "context": [T_ALL]}),
    ("q17", "fratricide", {"sentence": [r"fratricide"], "context": [T_ALL]}),
    ("q17", "T-cell aplasia or product contamination", {"sentence": [r"t.cell aplasia|contaminat"], "context": [T_ALL]}),
    ("q20", "cytopenia management (transfusion/G-CSF/boost)", {"sentence": [r"cytopeni|neutropeni", r"G-CSF|transfus|boost|CD34|thrombopoietin|eltrombopag|growth factor"]}),
]


def first_match(text: str, rule: dict) -> str | None:
    if not all(re.search(rx, text, re.I) for rx in rule.get("context", [])):
        return None
    for sentence in SENTENCE_SPLIT.split(text):
        if all(re.search(rx, sentence, re.I) for rx in rule["sentence"]):
            return sentence.strip()
    return None


def full_ranking(question: str, n: int) -> dict[int, int]:
    """Rank of EVERY chunk for this question (point id -> rank), using the same search as retrieve()."""
    vec = load_model().encode(QUERY_PREFIX + question, normalize_embeddings=True)
    hits = load_client().query_points(COLLECTION, query=vec.tolist(), limit=n, with_payload=False).points
    return {h.id: rank for rank, h in enumerate(hits, 1)}


def raw_text(pmcid: str) -> str:
    path = RAW_DIR / f"{pmcid}.xml"
    return re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", path.read_text(encoding="utf-8"))) if path.exists() else ""


def verdict(rank: int) -> str:
    return ("🔍 RANKING (close miss)" if rank <= 20 else
            "🔍 RANKING (far miss)" if rank <= 100 else
            "✂️  CHUNKING / MATCHING (buried)")


def main():
    chunks = json.loads(CHUNKS_FILE.read_text())          # list order = Qdrant point id
    papers = json.loads(PAPERS_FILE.read_text())
    questions = {q["id"]: q for q in json.loads(QUESTIONS_FILE.read_text())}
    rankings, summary = {}, []

    for qid, fact, rule in FACTS:
        if qid not in rankings:
            rankings[qid] = full_ranking(questions[qid]["question"], len(chunks))
        ranks = rankings[qid]

        in_chunks = sorted(((ranks[i], c, s) for i, c in enumerate(chunks) if (s := first_match(c["text"], rule))),
                           key=lambda x: x[0])
        tables = [(p["pmcid"], t) for p in papers for t in p["tables"]
                  if first_match(" ".join(filter(None, [t["caption"], t["markdown"], t["footnote"]])), rule)]
        print(f"\n{qid} · {fact}")

        if in_chunks:
            best_rank, c, sentence = in_chunks[0]
            result = verdict(best_rank)
            print(f"   in {len(in_chunks)} chunks ({len({x[1]['pmcid'] for x in in_chunks})} papers) | best rank #{best_rank} "
                  f"| top-3 ranks {[x[0] for x in in_chunks[:3]]}")
            print(f"   best: {c['chunk_id']} {c['first_author']} {c['year']} {c['sections']}")
            print(f"   \"{sentence[:220]}\"")
        elif tables:
            best_rank, result = None, "🧹 PARSING (only in tables)"
            pmcid, t = tables[0]
            print(f"   not in any chunk; e.g. {pmcid} {t['label']}: {t['caption'][:150]}")
        elif raw_hits := [(p["pmcid"], s) for p in papers if (s := first_match(raw_text(p["pmcid"]), rule))]:
            best_rank, result = None, "🧹 PARSING (dropped while cleaning)"
            pmcid, s = raw_hits[0]
            print(f"   not in chunks or tables; in raw XML of {len(raw_hits)} papers, e.g. {pmcid}: \"{s[:200]}\"")
        else:
            best_rank, result = None, "📚 DATA (not in our papers)"
            print("   not in chunks, tables, or raw papers")
        if tables:
            print(f"   also in {len(tables)} tables")
        print(f"   → {result}")
        summary.append((qid, fact, best_rank, len(tables), result))

    print(f"\n{'=' * 110}\nSUMMARY: where each lost fact sits\n{'=' * 110}")
    print(f"{'fact':58} {'best rank':>9}  @5  @20 @100 tables  → problem")
    for qid, fact, rank, n_tables, result in summary:
        hits = " ".join(f"{'✅' if rank and rank <= d else '❌':>3}" for d in (5, 20, 100))
        print(f"{(qid + ' ' + fact)[:58]:58} {('#' + str(rank)) if rank else '-':>9} {hits} {n_tables:>6}  → {result}")


if __name__ == "__main__":
    main()
