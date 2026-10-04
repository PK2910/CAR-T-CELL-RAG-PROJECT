"""
Box 2: PARSE
Turn each raw PMC XML file into clean text organized by section,
plus the metadata we need for citations.

Input:  data/raw/PMC*.xml
Output: data/processed/papers.json

Run:  python src/parse.py
"""

import json
import re
import statistics
import xml.etree.ElementTree as ET
from pathlib import Path

RAW_DIR = Path("data/raw")
OUT_FILE = Path("data/processed/papers.json")

# Elements whose text we don't want inside paragraphs (baseline skips tables, figures, math)
SKIP_TAGS = {"table-wrap", "fig", "disp-formula", "inline-formula", "math", "supplementary-material"}

# Sections that are administrative boilerplate, not scientific content
BOILERPLATE = re.compile(
    r"acknowledg|funding|conflict|competing interest|disclosure|author contribution|"
    r"data availability|data sharing|ethic|abbreviation|supplementary|consent|publisher",
    re.IGNORECASE,
)


def local(tag: str) -> str:
    """Strip an XML namespace: '{http://...}math' -> 'math'."""
    return tag.rsplit("}", 1)[-1]


# Elements that start a new line/block: they need a space around them,
# otherwise "apheresis<break/>Genetic" becomes "apheresisGenetic"
BLOCK_TAGS = {"break", "p", "list-item", "title", "label", "td", "th", "tr", "caption"}


def text_of(el: ET.Element) -> str:
    """All text inside an element, skipping tables/figures/math, with tidy whitespace."""
    def walk(e):
        parts = [e.text or ""]
        for child in e:
            name = local(child.tag)
            if name in BLOCK_TAGS:
                parts.append(" ")
            if name not in SKIP_TAGS:
                parts.append(walk(child))
            if name in BLOCK_TAGS:
                parts.append(" ")
            parts.append(child.tail or "")  # text after a child belongs to the parent
        return "".join(parts)
    return re.sub(r"\s+", " ", walk(el)).strip()


def paragraphs_in(el: ET.Element) -> list[str]:
    """Collect <p> paragraphs under an element, without entering tables/figures."""
    found = []
    for child in el:
        name = local(child.tag)
        if name in SKIP_TAGS:
            continue
        if name == "p":
            text = text_of(child)
            if text:
                found.append(text)
        else:
            found.extend(paragraphs_in(child))  # e.g. subsections, lists
    return found


def extract_table(wrap: ET.Element) -> dict:
    """Turn one <table-wrap> into {label, caption, markdown, footnote}.
    Tables published only as images have markdown=None. Merged cells (rowspan/colspan)
    are not expanded, so complex tables may come out slightly misaligned."""
    def field(path):
        el = wrap.find(path)
        return text_of(el) if el is not None else ""

    def cell(c):
        return text_of(c).replace("|", "/")  # '|' would break the markdown columns

    rows = [[cell(c) for c in tr if local(c.tag) in ("td", "th")]
            for tr in wrap.iter() if local(tr.tag) == "tr"]
    rows = [r for r in rows if any(r)]

    markdown = None
    if rows:
        width = max(len(r) for r in rows)
        rows = [r + [""] * (width - len(r)) for r in rows]
        lines = ["| " + " | ".join(rows[0]) + " |", "|" + " --- |" * width]
        lines += ["| " + " | ".join(r) + " |" for r in rows[1:]]
        markdown = "\n".join(lines)

    return {
        "label": field("label"),
        "caption": field("caption"),
        "markdown": markdown,
        "footnote": field("table-wrap-foot"),
    }


def clean_title(title: str) -> str:
    """'3. Sequential Use of CAR T' -> 'Sequential Use of CAR T', 'II. Methods' -> 'Methods'"""
    # Numbering must be followed by whitespace, so the "I" in "Introduction" is kept
    return re.sub(r"^(?:\d+(?:\.\d+)*|[IVX]+)\.?\s+", "", title).strip() or title


def parse_paper(path: Path) -> dict | None:
    """Parse one PMC XML file. Returns None for non-English papers."""
    article = ET.parse(path).getroot().find("article")
    if article.get("{http://www.w3.org/XML/1998/namespace}lang", "en") != "en":
        return None

    meta = article.find("front/article-meta")
    ids = {a.get("pub-id-type"): a.text for a in meta.findall("article-id")}

    years = [int(y.text) for y in meta.findall("pub-date/year") if y.text and y.text.isdigit()]
    surname = meta.find("contrib-group/contrib[@contrib-type='author']/name/surname")
    journal = article.find("front/journal-meta/journal-title-group/journal-title")
    title = meta.find("title-group/article-title")

    # Prefer the main abstract over plain-language/graphical summaries
    abstracts = meta.findall("abstract")
    main_abstract = next((a for a in abstracts if a.get("abstract-type") is None), abstracts[0] if abstracts else None)

    sections, dropped = [], []
    body = article.find("body")
    if body is not None:
        loose = [text_of(p) for p in body.findall("p")]  # paragraphs not inside any section
        if any(loose):
            sections.append({"title": "Main text", "paragraphs": [p for p in loose if p]})
        for sec in body.findall("sec"):
            sec_title = clean_title(text_of(sec.find("title"))) if sec.find("title") is not None else "Untitled"
            if BOILERPLATE.search(sec_title) or BOILERPLATE.search(sec.get("sec-type", "")):
                dropped.append(sec_title)
                continue
            paras = paragraphs_in(sec)
            # Untitled sections are often disclosures; judge them by their opening words
            if paras and sec_title == "Untitled" and BOILERPLATE.search(paras[0][:60]):
                dropped.append(sec_title)
                continue
            if paras:
                sections.append({"title": sec_title, "paragraphs": paras})

    return {
        "pmcid": ids.get("pmcid") or f"PMC{ids.get('pmcaid')}",
        "pmid": ids.get("pmid"),
        "doi": ids.get("doi"),
        "title": text_of(title) if title is not None else "",
        "journal": journal.text if journal is not None else None,
        "year": min(years) if years else None,
        "first_author": surname.text if surname is not None else None,
        "article_type": article.get("article-type"),
        # Join the abstract's paragraphs, leaving out headings like "Abstract" / "Background"
        "abstract": " ".join(paragraphs_in(main_abstract)) if main_abstract is not None else "",
        "sections": sections,
        # Extracted now, but the baseline doesn't chunk them (they're a planned add-on)
        "tables": [extract_table(w) for w in article.iter("table-wrap")],
        "dropped_sections": dropped,
    }


def main():
    papers, non_english, failed = [], [], []
    for path in sorted(RAW_DIR.glob("PMC*.xml")):
        try:
            paper = parse_paper(path)
        except ET.ParseError as e:
            failed.append((path.name, str(e)))
            continue
        if paper is None:
            non_english.append(path.stem)
        else:
            papers.append(paper)

    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    OUT_FILE.write_text(json.dumps(papers, indent=2, ensure_ascii=False), encoding="utf-8")

    # ---------- Report: always look at what the parser produced ----------
    words = [len(p["abstract"].split()) + sum(len(par.split()) for s in p["sections"] for par in s["paragraphs"])
             for p in papers]
    print(f"Parsed:            {len(papers)} papers -> {OUT_FILE}")
    print(f"Skipped (non-EN):  {len(non_english)} {non_english}")
    print(f"Failed to parse:   {len(failed)} {failed}")
    print(f"Words per paper:   median {statistics.median(words):,.0f}, min {min(words):,}, max {max(words):,}")
    print(f"Total words:       {sum(words):,}")
    print(f"No body sections:  {sum(1 for p in papers if not p['sections'])} papers (abstract only)")
    print(f"Missing year/PMID: {sum(1 for p in papers if not p['year'])} / {sum(1 for p in papers if not p['pmid'])}")
    tables = [t for p in papers for t in p["tables"]]
    print(f"Tables extracted:  {sum(1 for t in tables if t['markdown'])} as text, "
          f"{sum(1 for t in tables if not t['markdown'])} image-only (not usable) "
          f"[saved for the tables add-on, not used by the baseline]")
    print(f"Boilerplate sections dropped: {sum(len(p['dropped_sections']) for p in papers)}")


if __name__ == "__main__":
    main()
