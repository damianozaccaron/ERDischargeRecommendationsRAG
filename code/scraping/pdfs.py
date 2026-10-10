"""Step 5: open-access PDFs, checked against their record and stamped with its PMID."""
import re, unicodedata
from pathlib import Path

import pymupdf
import requests

from scraping.common import http
from scraping.pubmed import eutils, summaries

PMC_S3 = "https://pmc-oa-opendata.s3.amazonaws.com/"
VERSIONS = {"publishedVersion": 0, "acceptedVersion": 1}  # OpenAlex's labels, tried in this order, then the rest
ENGLISH = set("the and of to in for with is are on by that this be as from or".split())  # a quarter of English prose


def pdf_urls(rec: dict) -> list[tuple[str, str]]:
    """(url, version) of the record's PDF copies: PMC's first, then OpenAlex's, published versions first. OpenAlex's
    labels are unreliable (repositories often call the publisher's PDF submitted): they order the copies, never drop
    one."""
    urls = []
    try:
        if rec["pmcid"]:  # PMC's open-access subset, served from S3 without the bot check pmc.ncbi.nlm.nih.gov has
            listing = http.get(PMC_S3, params={"list-type": 2, "prefix": rec["pmcid"] + "."}, timeout=30).text
            keys = re.findall(r"<Key>(PMC\d+\.(\d+)/PMC\d+\.\2\.pdf)</Key>", listing)
            urls += [(PMC_S3 + k, "PMC") for k, v in sorted(keys, key=lambda k: -int(k[1]))]
        if rec["doi"]:  # publisher and repository copies OpenAlex knows of
            work = http.get(f"https://api.openalex.org/works/doi:{rec['doi']}", timeout=30)
            if work.ok:
                locs = [work.json().get("best_oa_location") or {}] + work.json().get("locations", [])
                locs = sorted((l for l in locs if l.get("pdf_url")), key=lambda l: VERSIONS.get(l.get("version"), 2))
                urls += [(l["pdf_url"], l.get("version") or "") for l in locs]
    except requests.RequestException:
        pass
    return list(dict.fromkeys(urls))


def twins(rec: dict) -> list[dict]:
    """Other PubMed records with the record's exact title: the guideline co-published in another journal, whose copies
    are the same text. Not errata ("Correction: ..."), whose titles differ."""
    words = [w.lower() for w in re.findall(r"[A-Za-z0-9]{4,}", rec["title"])]
    same = lambda title: re.findall(r"[a-z0-9]+", title.lower())
    try:
        ids = eutils("esearch.fcgi", term=" AND ".join(f"{w}[ti]" for w in words), retmax=20)["esearchresult"]["idlist"]
        recs = summaries([i for i in ids if i != rec["pmid"]])
        return [r for r in recs.values() if same(r["title"]) == same(rec["title"])]
    except Exception:  # a backup source: a failed lookup only means no twins
        return []


def mismatch(path: Path, rec: dict) -> str:
    """What is wrong with a guideline's PDF, '' if nothing: not in English (under 8% of the body's words are ENGLISH
    ones, 15-35% in English guidelines, 1-4% in Italian translations; judged on body pages, since the first ones can be
    all names), the record's DOI and most (80%) of its title's words of 4+ letters missing from the first 3 pages,
    whatever the line breaks, or 2 pages or fewer (an erratum, a preview)."""
    norm = lambda s: unicodedata.normalize("NFKC", s).lower()
    with pymupdf.open(path) as doc:
        pages, text = doc.page_count, norm(" ".join(doc[i].get_text() for i in range(min(3, doc.page_count))))
        body = [doc[i].get_text() for i in {pages // 3, pages // 2, 2 * pages // 3}]
    body = re.findall(r"[^\W\d_]+", norm(" ".join(body)))
    if sum(w in ENGLISH for w in body) < 0.08 * len(body):
        return "not in English"
    text = text.replace("-\n", "")  # words hyphenated across lines
    words = set(re.findall(r"[a-z0-9]{4,}", norm(rec["title"])))
    found = words & set(re.findall(r"[a-z0-9]{4,}", text))
    if not (rec["doi"] and rec["doi"].lower() in text) and len(found) < 0.8 * len(words):
        return "title not found in the PDF"
    return f"{pages} pages" if pages <= 2 else ""


def stamp(path: Path, pmid: str) -> bool:
    """Writes the PMID into the PDF's metadata (key /PMID of its Info dictionary), so the file says which guideline it
    is wherever it is moved or renamed. Saved incrementally, the original bytes stay. False if it won't open as a
    PDF."""
    try:
        with pymupdf.open(path) as doc:
            if doc.xref_get_key(-1, "Info")[0] != "xref":  # no Info dictionary yet
                xref = doc.get_new_xref()
                doc.update_object(xref, "<<>>")
                doc.xref_set_key(-1, "Info", f"{xref} 0 R")
            doc.xref_set_key(int(doc.xref_get_key(-1, "Info")[1].split()[0]), "PMID", pymupdf.get_pdf_str(pmid))
            incremental = doc.can_save_incrementally()
            if incremental:
                doc.saveIncr()
            else:  # a file MuPDF had to repair: saved whole
                doc.save(path.with_suffix(".tmp"))
        if not incremental:
            path.with_suffix(".tmp").replace(path)
        return True
    except Exception:  # not a PDF MuPDF can open or save
        return False


def pmid_of(path: Path) -> str:
    """The PMID stamp() wrote into a PDF's metadata; '' if none."""
    try:
        with pymupdf.open(path) as doc:
            kind, value = doc.xref_get_key(-1, "Info/PMID")
        return value if kind == "string" else ""
    except Exception:
        return ""


def file_names(picks: list[dict], here: dict[str, str]) -> dict[str, str]:
    """PMID -> PDF file name, from the LLM's names. here maps the PDFs already in OUT to the PMID in their metadata:
    a guideline already there keeps its file, a name another file has gets _PMID, chapters picking one guideline
    share it."""
    found = {pmid: f for f, pmid in here.items() if pmid}
    names = {}
    for pick in picks:
        pmid, name = pick["pmid"], f"{pick['name'] or 'PMID' + pick['pmid']}.pdf"
        if pmid not in names:
            taken = name in here or name in names.values()
            names[pmid] = found.get(pmid) or (f"{name[:-4]}_{pmid}.pdf" if taken else name)
    return names


def download(rec: dict, dest: Path) -> tuple[str, list[dict]]:
    """Saves the first copy that passes mismatch(), with the record's PMID stamped in: its own copies, then its twins'
    (looked up only if needed). A PDF that fails the check is kept only if nothing better turns up, and flagged in
    manifest.csv; never one that is broken or not in English. Returns the host it came from ('' if none) and every
    attempt, for downloads.csv."""
    def copies():
        yield from ((rec, *c) for c in pdf_urls(rec))
        for twin in twins(rec):
            yield from ((twin, *c) for c in pdf_urls(twin))

    log, kept = [], None
    for src, url, version in copies():
        row = dict(pmid=rec["pmid"], record=src["pmid"], version=version, url=url)
        try:
            r = http.get(url, timeout=60)
        except requests.RequestException as e:
            log.append(row | dict(result=type(e).__name__))
            continue
        if not r.ok or r.content[:5] != b"%PDF-":
            log.append(row | dict(result=f"HTTP {r.status_code}" if not r.ok else "not a PDF"))
            continue
        dest.write_bytes(r.content)
        problem = mismatch(dest, rec) if stamp(dest, rec["pmid"]) else "broken PDF"
        log.append(row | dict(result=problem or "saved"))
        if not problem:
            return url.split("/")[2], log
        if problem not in ("broken PDF", "not in English"):
            kept = kept or (r.content, url.split("/")[2])
    if kept:  # only PDFs that looked wrong: the first
        dest.write_bytes(kept[0])
        stamp(dest, rec["pmid"])
        return kept[1], log
    dest.unlink(missing_ok=True)
    return "", log or [dict(pmid=rec["pmid"], record=rec["pmid"], version="", url="", result="no copy known")]
