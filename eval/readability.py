"""
How easy are the answers to read? Free: reads saved answers (eval/results/answers_*.json), no model calls.

  grade        Flesch-Kincaid grade level: the school grade needed to read the text, from sentence length and
               syllables per word (0.39 x words/sentence + 11.8 x syllables/word - 15.59). Plain-language health
               writing usually aims for about grade 8; scientific papers are often grade 15+.
  words/sent   average sentence length
  long words   share of words with 3+ syllables
  abbrev/100   abbreviations and codes per 100 words (ICANS, CRS, MRD, CD19, allo-HSCT, tisa-cel...). The grade formula
               counts syllables, so it treats "ICANS" as an easy one-syllable word; this measures that jargon instead
  words        answer length

Citations ([3]), Markdown symbols and URLs are removed first; each bullet point counts as a sentence. Syllables are
counted with a simple vowel-group rule, so scores are estimates: good for comparing versions, not exact.
Answers written plain-language first (from 2026-11-01: "Short answer", "Key points", then "Details for clinicians
and researchers") are also scored on the plain part alone (the part before the details line): "plain grade" and
"plain abbrev/100".

Run:  python eval/readability.py                every saved run
      python eval/readability.py final          only runs whose file name contains "final"
"""

import glob
import json
import re
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PLAIN_TARGET = 8


def syllables(word: str) -> int:
    word = word.lower()
    if len(word) <= 3:
        return 1
    word = re.sub(r"(?:[^laeiouy]es|ed|[^laeiouy]e)$", "", word)  # silent endings: "makes", "used", "make"
    return max(1, len(re.findall(r"[aeiouy]+", word)))


def clean(text: str) -> str:
    text = re.sub(r"\[\d+(?:\s*,\s*\d+)*\]", "", text)       # citations
    text = re.sub(r"https?://\S+", "", text)                 # links
    text = re.sub(r"[*#>`_|]", "", text)                     # Markdown symbols
    text = re.sub(r"^\s*(?:[-•]|\d+[.)])\s+", "", text, flags=re.M)  # bullet markers
    return text


# A token with 2+ capital letters or a capital+digit mix: ICANS, CRS, B-ALL, CD19, CAR-T, HSCT; also "tisa-cel" style
ABBREVIATION = re.compile(r"\b(?=[A-Za-z0-9-]*[A-Z][A-Za-z0-9-]*[A-Z0-9])[A-Za-z][A-Za-z0-9-]*\b|\b[a-z]+-cel\b")


def readability(text: str) -> dict:
    """Grade level and its parts for one text."""
    text = clean(text)
    sentences = [s for line in text.split("\n") for s in re.split(r"(?<=[.!?])\s+", line) if re.search(r"[A-Za-z]", s)]
    words = re.findall(r"[A-Za-z][A-Za-z'-]*", text)
    if not sentences or not words:
        return {"grade": None, "words_per_sentence": None, "long_words": None, "abbrev_per_100": None, "words": 0}
    counts = [syllables(w) for w in words]
    words_per_sentence = len(words) / len(sentences)
    grade = 0.39 * words_per_sentence + 11.8 * sum(counts) / len(words) - 15.59
    return {"grade": round(grade, 1), "words_per_sentence": round(words_per_sentence, 1),
            "long_words": round(sum(c >= 3 for c in counts) / len(words), 3),
            "abbrev_per_100": round(100 * len(ABBREVIATION.findall(text)) / len(words), 1), "words": len(words)}


DETAILS_LINE = re.compile(r"^[ \t]*(?:\*\*)?Details for clinicians and researchers", re.I | re.M)


def plain_part(answer: str) -> str | None:
    """The part of a plain-language-first answer before the technical details, or None if it has no details line."""
    m = DETAILS_LINE.search(answer)
    return answer[:m.start()] if m else None


def main():
    pattern = sys.argv[1] if len(sys.argv) > 1 else ""
    files = sorted(f for f in glob.glob(str(ROOT / "eval/results/answers_*.json")) if pattern in Path(f).name)
    print(f"{'run':40} {'answers':>7} {'grade':>6} {'words/sent':>10} {'long words':>10} {'abbrev/100':>10} "
          f"{'words':>6} {'plain grade':>11} {'plain abbrev':>12} {'plain words':>11}")
    for f in files:
        answers = [r["answer"] for r in json.loads(Path(f).read_text())["answers"] if r.get("answer")]
        scores = [s for s in map(readability, answers) if s["grade"] is not None and s["words"] >= 30]  # no refusals
        if not scores:
            continue
        med = {k: statistics.median(s[k] for s in scores)
               for k in ("grade", "words_per_sentence", "long_words", "abbrev_per_100", "words")}
        plain = [s for s in (readability(p) for p in map(plain_part, answers) if p) if s["grade"] is not None]
        plain_cols = (f"{statistics.median(s['grade'] for s in plain):>11.1f} "
                      f"{statistics.median(s['abbrev_per_100'] for s in plain):>12.1f} "
                      f"{statistics.median(s['words'] for s in plain):>11.0f}") if plain else f"{'-':>11} {'-':>12} {'-':>11}"
        name = Path(f).name.removeprefix("answers_").removesuffix(".json")
        print(f"{name:40} {len(scores):>7} {med['grade']:>6.1f} {med['words_per_sentence']:>10.1f} "
              f"{100 * med['long_words']:>9.0f}% {med['abbrev_per_100']:>10.1f} {med['words']:>6.0f} {plain_cols}")
    print(f"\nMedians per run. Plain-language health writing usually aims for about grade {PLAIN_TARGET}. "
          "'plain' = the short answer and key points only (answers written plain-language first).")


if __name__ == "__main__":
    main()
