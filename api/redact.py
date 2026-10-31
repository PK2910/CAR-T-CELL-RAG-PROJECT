"""
Remove personal details from text before it is saved (README, Personal details): the logbook keeps what was asked,
not who it was about. Claude still gets the question as typed; only the saved copy is cleaned.

  "My son John Smith, MRN 4829301, born 03/12/2018, email j.smith@gmail.com"
  -> "My son [NAME], [ID], born [DATE], email [EMAIL]"

Uses Microsoft Presidio (rules + spaCy's small English model, run locally) for names, emails, phone numbers and
other ID formats, plus our own rules for medical record numbers, long ID numbers and full dates. Tuned on the 191
test questions, where the untuned version marked medical terms as names, organizations, places and dates
("Kymriah" a person, "CRS" an organization, "Philadelphia" chromosome a place):
  - organizations, places, ages and partial dates ("after 2 years") are not removed: on their own they do not say
    who someone is, and removing them would make the logbook useless;
  - a name is kept if all its words appear at least VOCAB_MIN_COUNT times in the papers (drug names, gene names,
    frequent author surnames such as "Wang"). So "John Wang" is removed, but "Wang" alone is not.
If the check fails, the text is not saved at all (FAILED is saved instead): privacy comes before debugging.
"""

import json
import re
import sys
from collections import Counter
from functools import cache
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
VOCAB_MIN_COUNT = 10
FAILED = "[not saved: the personal-details check failed]"

PLACEHOLDERS = {"PERSON": "[NAME]", "EMAIL_ADDRESS": "[EMAIL]", "PHONE_NUMBER": "[PHONE]", "ID": "[ID]",
                "DATE": "[DATE]", "US_SSN": "[ID]", "CREDIT_CARD": "[ID]", "IP_ADDRESS": "[ID]"}
ID_PATTERNS = [
    # "MRN 4829301", "medical record number: 12345", "patient ID #A1234"
    r"\b(MRN|medical record( number)?|record (number|no\.?)|patient id|hospital (number|id))\s*[:#]?\s*[A-Z]?\d{3,}\b",
    r"\b\d{6,}\b",  # any number of 6+ digits (NCT02435849 is not matched: no word boundary after "NCT")
]
# "my daughter Emily", "our patient, Jordan": the small model misses some first names; after these words a
# capitalized word is a name
NAME_AFTER = re.compile(r"\b(?i:my|our|the)\s+(?i:son|daughter|child|kid|baby|boy|girl|husband|wife|partner|mother|"
                        r"mom|mum|father|dad|brother|sister|grandson|granddaughter|patient|friend)\s*,?\s+"
                        r"([A-Z][a-z'-]+)")
_MONTH = r"(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)[a-z]*\.?"
DATE_PATTERNS = [  # full dates only (a birth date identifies someone; "in 2018" or "day 28" does not)
    r"\b\d{1,2}[/.-]\d{1,2}[/.-](19|20)\d{2}\b",
    r"\b(19|20)\d{2}-\d{1,2}-\d{1,2}\b",
    rf"\b{_MONTH}\s+\d{{1,2}}(st|nd|rd|th)?,?\s+(19|20)\d{{2}}\b",
    rf"\b\d{{1,2}}(st|nd|rd|th)?\s+(of\s+)?{_MONTH},?\s+(19|20)\d{{2}}\b",
]


@cache
def _engines():
    from presidio_analyzer import AnalyzerEngine, Pattern, PatternRecognizer
    from presidio_analyzer.nlp_engine import NlpEngineProvider
    from presidio_anonymizer import AnonymizerEngine
    nlp = NlpEngineProvider(nlp_configuration={
        "nlp_engine_name": "spacy", "models": [{"lang_code": "en", "model_name": "en_core_web_sm"}]}).create_engine()
    analyzer = AnalyzerEngine(nlp_engine=nlp, supported_languages=["en"])
    for entity, patterns in (("ID", ID_PATTERNS), ("DATE", DATE_PATTERNS)):
        analyzer.registry.add_recognizer(PatternRecognizer(
            supported_entity=entity, patterns=[Pattern(f"{entity}{i}", p, 0.9) for i, p in enumerate(patterns)]))
    return analyzer, AnonymizerEngine()


@cache
def _paper_vocabulary() -> frozenset:
    """Words that appear at least VOCAB_MIN_COUNT times in the papers: medical terms, not patients' names."""
    sys.path.insert(0, str(ROOT / "src"))
    from index_config import CHUNKS_FILE
    counts = Counter(w for c in json.loads(CHUNKS_FILE.read_text())
                     for w in re.findall(r"[A-Za-z][A-Za-z0-9+'-]*", c["text"]))
    return frozenset(w for w, n in counts.items() if n >= VOCAB_MIN_COUNT)


def _is_real_name(span: str) -> bool:
    words = re.findall(r"[A-Za-z][A-Za-z0-9+'-]*", span)
    if not words or any(not re.fullmatch(r"[A-Z][a-z'-]+", w) for w in words):
        return False  # names are capitalized words without digits: not "MUC1", "Ph+ B-cell", "multiplex"
    return not all(w in _paper_vocabulary() for w in words)


def redact(text: str | None) -> str | None:
    """The text with personal details replaced by [NAME], [ID], [DATE], [EMAIL], [PHONE]."""
    if not text:
        return text
    try:
        from presidio_analyzer import RecognizerResult
        from presidio_anonymizer.entities import OperatorConfig
        analyzer, anonymizer = _engines()
        found = [r for r in analyzer.analyze(text, language="en", entities=list(PLACEHOLDERS), score_threshold=0.4)
                 if r.entity_type != "PERSON" or _is_real_name(text[r.start:r.end])]
        for m in NAME_AFTER.finditer(text):
            start, end = m.span(1)
            if _is_real_name(m.group(1)) and not any(r.start < end and start < r.end for r in found):
                found.append(RecognizerResult("PERSON", start, end, 0.9))
        return anonymizer.anonymize(text=text, analyzer_results=found, operators={
            e: OperatorConfig("replace", {"new_value": v}) for e, v in PLACEHOLDERS.items()}).text
    except Exception as e:  # any failure (missing package or model, a bug): save nothing rather than the raw text
        print(f"WARNING (personal-details check) failed, text not saved: {type(e).__name__}: {e}")
        return FAILED


def warm_up() -> None:
    """Load the model and the paper vocabulary once at startup, so the first question is not slowed down."""
    redact("warm-up")
