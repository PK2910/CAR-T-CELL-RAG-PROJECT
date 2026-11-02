"""
Query rewriting for the search (README, Robustness). The search, not Claude, gets a tidied copy of the question; the
relevance check and the answer model always get the question as typed.

  names      Abbreviations and their full names, as the papers define them ("cytokine release syndrome (CRS)").
             If the question uses the form the papers use much less (MIN_RATIO), the common form is added:
             "...after chimeric antigen receptor T-cell therapy" -> "... CAR". (The exact-word search splits
             "CAR-T" in the papers into "CAR-T" and "CAR", so the added "CAR" matches them.)
  spelling   A word that never appears in the papers but is one letter away from a word that does is corrected to
             the most common such word: "sterids" -> "steroids". Abbreviations, words with digits and short words are
             left alone, and so is a word with two equally common candidates.

Both lists are built from the papers (the chunks file) when the search first runs, not from the test questions.
Switch: RAG_QUERY_REWRITE = "spelling" (the default), "names", "names,spelling", or "off".

Tested 2026-11-02 (eval/robustness.py; settings chosen on the practice questions, judged on 120 generated questions not
used to make them): spelling brought questions with a typo back to the level of correctly typed ones (Hit@10 69% ->
72%) and changed none of the 191 test questions, so it is on. Names helped on the practice questions (spelled-out
names 81% -> 90%) but not on the generated ones (59% -> 57%), so it is off.
Run:  python src/query_rewrite.py "your question"     (shows the rewritten question and the pairs found)
"""

import json
import os
import re
import sys
from collections import Counter
from functools import lru_cache

from index_config import CHUNKS_FILE

REWRITE = {s.strip() for s in os.environ.get("RAG_QUERY_REWRITE", "spelling").split(",")} - {"off", ""}
MIN_PAIR_COUNT = 3    # a definition must appear this often in the papers to be used
MIN_RATIO = 5         # add the other form only if the papers use it at least this many times more often. Set on the
                      # practice questions: adding "EFS" (4x "event-free survival") pushed an answer from #6 to #14
MIN_WORD_COUNT = 5    # a correction must be a word the papers use at least this often
MIN_TYPO_LENGTH = 5   # shorter words are not corrected (too many near neighbours)

SHORT_FORM = re.compile(r"\(([A-Za-z][A-Za-z0-9-]{1,9})\)")       # "(CRS)", "(CAR)", "(allo-HSCT)"
CAPITALS = re.compile(r"^(?=.*[A-Z].*[A-Z])")                      # a short form has 2+ capitals
WORD = re.compile(r"[A-Za-z]+")
QUESTION_WORD = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?")  # keeps "doesn't" whole (it was once "corrected" to "does't")
TOKEN = re.compile(r"[A-Za-z0-9]+(?:[-/][A-Za-z0-9]+)*")              # as the exact-word search splits words
LETTERS = "abcdefghijklmnopqrstuvwxyz"


@lru_cache(maxsize=1)
def _texts() -> list[str]:
    return [c["text"] for c in json.loads(CHUNKS_FILE.read_text(encoding="utf-8"))]


def _long_form(short: str, before: str) -> str | None:
    """The words just before "(SHORT)" that spell it out (the Schwartz-Hearst rule): every letter or digit of the
    short form appears in the long form in the same order, and its first letter starts a word."""
    candidate = " ".join(before.split()[-min(len(short) + 5, 2 * len(short)):])
    s, i = len(short) - 1, len(candidate) - 1
    while s >= 0:
        ch = short[s].lower()
        if not ch.isalnum():
            s -= 1
            continue
        while i >= 0 and (candidate[i].lower() != ch or (s == 0 and i > 0 and candidate[i - 1].isalnum())):
            i -= 1
        if i < 0:
            return None
        i, s = i - 1, s - 1
    long = candidate[candidate.rfind(" ", 0, i + 1) + 1:].strip(" ,;:")
    if len(long) <= len(short) or len(long.split()) > len(short) + 2:
        return None
    return long


@lru_cache(maxsize=1)
def name_pairs() -> list[tuple[str, str, str]]:
    """(short form, long form, the form the papers use at least MIN_RATIO times more, or None), for definitions found
    at least MIN_PAIR_COUNT times."""
    found = Counter()
    for text in _texts():
        for m in SHORT_FORM.finditer(text):
            short = m.group(1)
            if CAPITALS.match(short):
                long = _long_form(short, text[max(0, m.start() - 200):m.start()])
                if long:
                    found[(short, long.lower())] += 1
    best = {}  # each short form keeps its most frequent long form
    for (short, long), n in found.most_common():
        if n >= MIN_PAIR_COUNT and short not in best:
            best[short] = long
    lower = "\n".join(_texts()).lower()
    tokens = Counter(t for text in _texts() for t in TOKEN.findall(text))  # one pass, exact case: "CRS", "CAR-T"
    pairs = []
    for short, long in best.items():
        n_short, n_long = tokens[short], lower.count(long)
        common = short if n_short >= MIN_RATIO * n_long else long if n_long >= MIN_RATIO * n_short else None
        pairs.append((short, long, common))
    return pairs


@lru_cache(maxsize=1)
def vocabulary() -> Counter:
    return Counter(w.lower() for text in _texts() for w in WORD.findall(text))


def _one_edit(word: str) -> set[str]:
    splits = [(word[:i], word[i:]) for i in range(len(word) + 1)]
    return ({a + b[1:] for a, b in splits if b}                                  # a letter dropped
            | {a + b[1] + b[0] + b[2:] for a, b in splits if len(b) > 1}         # two letters swapped
            | {a + c + b[1:] for a, b in splits if b for c in LETTERS}           # a letter changed
            | {a + c + b for a, b in splits for c in LETTERS})                   # a letter added


def correct(word: str) -> str:
    """The word, or its correction if it is a clear typo of a word in the papers."""
    if len(word) < MIN_TYPO_LENGTH or sum(ch.isupper() for ch in word) > 1 or "'" in word:
        return word  # short words, abbreviations (CRS, ICANS) and contractions are left alone
    vocab = vocabulary()
    if vocab[word.lower()]:
        return word
    options = sorted(((vocab[c], c) for c in _one_edit(word.lower()) if vocab[c] >= MIN_WORD_COUNT), reverse=True)
    if not options or (len(options) > 1 and options[0][0] == options[1][0]):
        return word  # nothing close, or two equally likely words
    return options[0][1]


def rewrite(question: str, parts: set[str] | None = None) -> str:
    """The question as the search should see it (unchanged when rewriting is off)."""
    parts = REWRITE if parts is None else parts
    if "spelling" in parts:
        question = QUESTION_WORD.sub(lambda m: correct(m.group(0)), question)
    if "names" in parts:
        added = []
        for short, long, common in name_pairs():
            has_short = re.search(rf"(?<![\w-]){re.escape(short)}(?![a-z0-9])", question)
            has_long = re.search(rf"\b{re.escape(long)}\b", question, re.I)
            if has_long and not has_short and common == short:
                added.append(short)
            elif has_short and not has_long and common == long:
                added.append(long)
        if added:
            question = f"{question} {' '.join(added)}"
    return question


if __name__ == "__main__":
    q = " ".join(sys.argv[1:]) or "Do sterids given for side effects reduce how well chimeric antigen receptor T cells work?"
    print(f"names found in the papers: {len(name_pairs())}")
    print(f"  as typed:  {q}\n  rewritten: {rewrite(q, {'names', 'spelling'})}")
