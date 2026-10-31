"""
Run the citation check (src/guardrails.py) on saved answers: how often did an answer cite a source number it was
not given, and how often did it cite nothing at all? Free: it only reads eval/results/answers_*.json.

Run:  python eval/citation_report.py               every saved run
      python eval/citation_report.py final_live     only runs whose file name contains "final_live"
"""

import glob
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from guardrails import check_citations  # noqa: E402


def main():
    pattern = sys.argv[1] if len(sys.argv) > 1 else ""
    files = sorted(f for f in glob.glob(str(ROOT / "eval/results/answers_*.json")) if pattern in Path(f).name)
    print(f"{'run':52} {'answers':>7} {'citations':>9} {'bad':>4} {'uncited':>7}")
    all_bad = []
    totals = [0, 0, 0, 0]
    for f in files:
        rows = [r for r in json.loads(Path(f).read_text())["answers"] if r.get("answer")]
        n_cites = bad = uncited = 0
        for r in rows:
            check = check_citations(r["answer"], len(r.get("sources") or []))
            n_cites += len(check["cited"])
            if check["invalid"]:
                bad += 1
                all_bad.append((Path(f).name, r["id"], len(r.get("sources") or []), check["invalid"]))
            if not check["cited"] and r.get("sources"):
                uncited += 1
        name = Path(f).name.removeprefix("answers_").removesuffix(".json")
        print(f"{name:52} {len(rows):>7} {n_cites:>9} {bad:>4} {uncited:>7}")
        for i, v in enumerate((len(rows), n_cites, bad, uncited)):
            totals[i] += v
    print(f"{'TOTAL':52} {totals[0]:>7} {totals[1]:>9} {totals[2]:>4} {totals[3]:>7}")
    print("\nbad = answers citing a source number they were not given; uncited = answers with sources but no "
          "citation (often a correct 'the sources don't say')")
    for name, qid, n, invalid in all_bad:
        print(f"  {name}  {qid}: cites {invalid}, had {n} sources")


if __name__ == "__main__":
    main()
