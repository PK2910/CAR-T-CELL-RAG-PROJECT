"""
Box 1: COLLECT
Search PubMed Central (PMC) for open-access papers on CAR-T in ALL
and download each paper's full-text XML into data/raw/.

Run:  python src/collect.py
"""

import json
import os
import time
from datetime import date
from pathlib import Path

import requests
from dotenv import load_dotenv
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_random_exponential

# ---------- Settings ----------
TARGET_PAPERS = 200
RAW_DIR = Path("data/raw")

# Papers must mention CAR-T AND ALL in the title or abstract, and be open access
QUERY = (
    '("chimeric antigen receptor"[Title] OR "chimeric antigen receptor"[Abstract] '
    'OR "CAR-T"[Title] OR "CAR-T"[Abstract] OR "CAR T"[Title] OR "CAR T"[Abstract]) '
    'AND ("acute lymphoblastic leukemia"[Title] OR "acute lymphoblastic leukemia"[Abstract] '
    'OR "B-ALL"[Title] OR "B-ALL"[Abstract]) '
    'AND "open access"[filter]'
)

EUTILS = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"

# Optional: an NCBI API key raises the limit from 3 to 10 requests/second.
# Put NCBI_API_KEY=... in a .env file (never in code).
load_dotenv()
API_KEY = os.getenv("NCBI_API_KEY")
SECONDS_BETWEEN_REQUESTS = 0.11 if API_KEY else 0.34


class TransientError(Exception):
    """A temporary failure (rate limit, server error) that is worth retrying."""


@retry(
    retry=retry_if_exception_type((TransientError, requests.ConnectionError, requests.Timeout)),
    wait=wait_random_exponential(min=1, max=30),  # exponential backoff + jitter
    stop=stop_after_attempt(5),
)
def eutils_get(endpoint: str, params: dict) -> requests.Response:
    """Call an NCBI E-utilities endpoint, retrying on temporary failures."""
    if API_KEY:
        params = {**params, "api_key": API_KEY}
    time.sleep(SECONDS_BETWEEN_REQUESTS)  # respect NCBI's rate limit
    resp = requests.get(f"{EUTILS}/{endpoint}", params=params, timeout=30)
    if resp.status_code == 429 or resp.status_code >= 500:
        raise TransientError(f"HTTP {resp.status_code}")
    resp.raise_for_status()  # other errors (e.g. 400) are permanent: fail loudly
    return resp


def search_pmc(query: str, max_results: int) -> list[str]:
    """Return PMC IDs matching the query, most relevant first."""
    resp = eutils_get("esearch.fcgi", {
        "db": "pmc",
        "term": query,
        "retmax": max_results,
        "retmode": "json",
        "sort": "relevance",
    })
    result = resp.json()["esearchresult"]
    if result.get("errorlist") or result.get("warninglist"):
        print("NCBI notes:", result.get("errorlist"), result.get("warninglist"))
    print(f"Search matched {result['count']} papers in total")
    return result["idlist"]


def has_full_text(xml: str) -> bool:
    """Some records only contain metadata; real full text has a <body> section."""
    return "<body" in xml


def main():
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    # Ask for extra IDs, because a few records won't contain full text
    ids = search_pmc(QUERY, max_results=int(TARGET_PAPERS * 1.5))

    saved, skipped_existing, no_full_text = [], 0, []
    for pmc_id in ids:
        if len(saved) == TARGET_PAPERS:
            break

        path = RAW_DIR / f"PMC{pmc_id}.xml"
        if path.exists():  # idempotent: re-running never re-downloads
            saved.append(pmc_id)
            skipped_existing += 1
            continue

        xml = eutils_get("efetch.fcgi", {"db": "pmc", "id": pmc_id, "retmode": "xml"}).text
        if not has_full_text(xml):
            no_full_text.append(pmc_id)
            continue

        path.write_text(xml, encoding="utf-8")
        saved.append(pmc_id)
        print(f"[{len(saved):>3}/{TARGET_PAPERS}] saved PMC{pmc_id}")

    # A manifest records exactly how this corpus was built (reproducibility)
    manifest = {
        "query": QUERY,
        "date_collected": date.today().isoformat(),
        "target": TARGET_PAPERS,
        "pmc_ids": [f"PMC{i}" for i in saved],
        "skipped_no_full_text": [f"PMC{i}" for i in no_full_text],
    }
    (RAW_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2))

    print(f"\nDone: {len(saved)} papers in {RAW_DIR}/ "
          f"({skipped_existing} already existed, {len(no_full_text)} had no full text)")


if __name__ == "__main__":
    main()
