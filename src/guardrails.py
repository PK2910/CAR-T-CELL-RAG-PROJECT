"""
Safety guardrails around the RAG system: checks done by CODE, so safety does not depend on
which pages the search happens to find or on Claude following an instruction.

Policy (decided 2026-10-06):
  - A question about a SPECIFIC person ("my son", "my patient", "should I give...", "can he take...")
    is a personal question.
  - Personal questions: the answer must never contain doses.
      INPUT guardrail:  a note is added to the prompt telling Claude not to give doses for this person.
      OUTPUT guardrail: code scans Claude's answer; if a dose slipped through anyway, the whole answer
                        is replaced with a safe message (contact the treating team / emergency services).
  - Research questions about doses ("what doses were used in trials?") are answered normally, with citations.

Citation check (added 2026-10-31): every [n] in an answer must be one of the sources Claude was given (1 to the
number of sources). If not, a note saying so is added to the answer. Answers with no citations at all are only
recorded, not changed: "the sources don't answer this" rightly has none.

Trick check (added 2026-10-31): a question that tries to change the system's rules ("ignore your instructions",
"reveal your system prompt", "you are now...") is not answered: no search and no Claude call. It catches common
phrasings only; the dose guardrail above still runs on every answer, so a trick that gets past it still cannot
produce a dose for a personal question.
"""

import re

# ---------- Input guardrail: is the question about a specific person? ----------

PERSONAL_PATTERNS = [
    # "my son", "our patient", "my 8-year-old", "my dad"...
    r"\b(my|our)\s+(\d+[- ]?(year|month)s?[- ]?old|son|daughter|child|kid|baby|boy|girl|husband|wife|partner|"
    r"spouse|mother|mom|mum|father|dad|parent|brother|sister|grand\w*|patient|friend|relative|family member)\b",
    # "should I give", "can he take", "could we start"...
    r"\b(should|can|could|may|must)\s+(i|we|he|she|they)\s+(give|take|get|receive|start|stop|use|have|try)\b",
    # "how much ... should I", "how much ... do we"
    r"\bhow much\b.*\b(should|can|do)\s+(i|we)\b",
    # "I have ALL", "I was diagnosed..."
    r"\b(i|i'm|i am)\s+(have|had|was diagnosed|am diagnosed|got|received|am receiving)\b",
]

PERSONAL_NOTE = (
    "Note: this question is about a specific individual. Do not give doses, dose calculations, or a "
    "treatment recommendation for this person, even if the sources contain doses. You may give general "
    "information from the sources, and tell them to contact the person's treating medical team."
)


def is_personal(question: str) -> bool:
    return any(re.search(p, question, re.IGNORECASE) for p in PERSONAL_PATTERNS)


# ---------- Input guardrail: is the question a trick to change the rules? ----------

# Words for the system's OWN rules. Not "guidelines" or "policies": real questions ask about treatment guidelines.
_RULES = r"(instructions?|rules?|prompts?|guardrails?|programming)"
TRICK_PATTERNS = [
    # "ignore your instructions", "disregard all previous rules", "forget the above prompt"
    rf"\b(ignore|disregard|forget|override|bypass)\b(\s+\w+){{0,3}}\s+{_RULES}\b",
    # "do not follow your rules", "stop following the instructions"
    rf"\b(do not|don't|stop|never)\s+follow(ing)?\b(\s+\w+){{0,3}}\s+{_RULES}\b",
    # "reveal your system prompt", "print your instructions", "what are your rules" (only "your", or "the" + system/
    # hidden/original/initial: "what are the rules for ..." is a real question)
    rf"\b(reveal|show|print|repeat|display|tell me|output|what are|what is|what were)\s+(me\s+)?(your\s+(system\s+|"
    rf"hidden\s+|original\s+|initial\s+)?|the\s+(system|hidden|original|initial)\s+){_RULES}\b",
    r"\bsystem\s+prompt\b",
    # "you are now DAN", "from now on you are", "pretend you are a doctor with no limits"
    r"\byou\s+are\s+now\b", r"\bfrom\s+now\s+on,?\s+you\b", r"\b(pretend|imagine)\s+(that\s+)?you\s+(are|have|can)\b",
    r"\bjailbr(eak|oken)\b", r"\b(DAN|developer)\s+mode\b", r"\bnew\s+instructions\s*:",
]

TRICK_MESSAGE = (
    "This tool answers questions about research on CAR-T cell therapy in acute lymphoblastic leukemia, using "
    "published papers. Your message looks like an attempt to change how the tool works, so it was not answered. "
    "Please ask a question about the research. This tool is not medical advice."
)


def is_trick(question: str) -> bool:
    return any(re.search(p, question, re.IGNORECASE) for p in TRICK_PATTERNS)


# ---------- Output guardrail: does the answer contain a dose? ----------

NUM = r"\d+(?:[.,]\d+)?(?:\s?[-–]\s?\d+(?:[.,]\d+)?)?"
DOSE_PATTERNS = [
    rf"\b{NUM}\s?(mg|mcg|µg|μg|g|ml|mL|IU|units?)\s?/\s?(kg|m2|m²|day|dose|h)\b",   # 8 mg/kg, 25 mg/m2
    rf"\b{NUM}\s?(mg|mcg|µg|μg)\b",                                               # 160 mg, 10 mg
    rf"\b{NUM}\s?[x×]\s?10\s?\^?\s?\d+",                                          # 1 × 10^6 (cells)
    r"\bcells\s?/\s?kg\b|\bcells per (kg|kilogram)\b|\bper kilogram\b",
]

SAFE_MESSAGE = (
    "I can't give dosing or treatment advice for a specific person. That decision has to come from the "
    "treating medical team, who know the patient's weight, condition, and the product they received.\n\n"
    "If someone has recently received CAR-T and has a fever or new symptoms such as confusion, contact "
    "the treating CAR-T team or emergency services right away.\n\n"
    "This tool summarizes published research and is not medical advice."
)


def find_doses(text: str) -> list[str]:
    return [m.group(0) for p in DOSE_PATTERNS for m in re.finditer(p, text, re.IGNORECASE)]


# [3] and [1, 4]. Ranges such as [22-30] are not read as citations: in answers they are almost always statistics
# ("median 26 [22-30]"), and Claude rarely cites a range of sources (checked on 610 saved answers, 2026-10-31).
CITATION = re.compile(r"\[(\d+(?:\s*,\s*\d+)*)\]")
CITATION_NOTE = ("Note: this answer cites source number(s) {numbers}, which were not among the {n} sources it was "
                 "given; those statements could not be checked against a source.")


def cited_numbers(answer: str) -> list[int]:
    """Every source number the answer cites, in order of first appearance."""
    numbers = []
    for m in CITATION.finditer(answer):
        for n in (int(x) for x in m.group(1).split(",")):
            if n not in numbers:
                numbers.append(n)
    return numbers


def check_citations(answer: str, n_sources: int) -> dict:
    """The citation check. Returns the answer (with a note added if it cites sources it was not given), the
    numbers cited, and the ones that point to no source."""
    cited = cited_numbers(answer)
    invalid = [n for n in cited if not 1 <= n <= n_sources]
    if invalid:
        answer += "\n\n" + CITATION_NOTE.format(numbers=", ".join(f"[{n}]" for n in invalid), n=n_sources)
    return {"answer": answer, "cited": cited, "invalid": invalid}


def check_answer(question: str, answer: str) -> dict:
    """Apply the output guardrail. Returns the final answer plus what the guardrail did."""
    personal = is_personal(question)
    doses = find_doses(answer) if personal else []
    if doses:
        return {"answer": SAFE_MESSAGE, "personal": True, "doses_found": doses, "action": "replaced"}
    return {"answer": answer, "personal": personal, "doses_found": [], "action": "none"}
