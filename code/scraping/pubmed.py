"""NCBI's E-utilities: searches, records, MeSH headings and their tree numbers."""
import threading, time
import xml.etree.ElementTree as ET

import requests

from scraping.common import http, retry

EUTILS = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"

NCBI = threading.Lock()


def eutils(tool: str, db: str = "pubmed", retmode: str = "json", **params):
    """An E-utilities reply: parsed JSON, or an ElementTree with retmode="xml"."""
    def call():
        with NCBI:  # NCBI allows 3 requests/s without an API key, whatever the threads
            time.sleep(0.34)
        r = http.post(EUTILS + tool, data=dict(db=db, retmode=retmode, tool="er-guideline-scraper", **params),
                      timeout=120)
        r.raise_for_status()
        if retmode == "json":
            data = r.json()  # a cut-off reply raises here, and is retried
            # a search that failed on PubMed's side, under HTTP 200: an error instead of a count (1 of the 277
            # searches of a dry run on 6 October 2026; the same query answered at once a minute later)
            if "count" not in data.get("esearchresult", {"count": 0}):
                raise requests.RequestException(f"ESearch without a count: {str(data['esearchresult'])[:300]}")
            return data
        try:
            return ET.fromstring(r.content)
        except ET.ParseError as e:  # a cut-off XML reply, retried like a JSON one
            raise requests.RequestException(e)
    return retry(call)


def headings(article) -> list[dict]:
    """A PubMed article's MeSH headings: descriptor UI and name, qualifiers, whether it is a main topic (the descriptor
    or one of its qualifiers starred)."""
    out = []
    for h in article.iter("MeshHeading"):
        d, qs = h.find("DescriptorName"), h.findall("QualifierName")
        out.append(dict(ui=d.get("UI"), name=d.text, qualifiers=[q.text for q in qs],
                        major=d.get("MajorTopicYN") == "Y" or any(q.get("MajorTopicYN") == "Y" for q in qs)))
    return out


def mesh(pmids: list[str]) -> dict[str, list[dict]]:
    """PMID -> its MeSH headings; [] for a record not indexed yet (recent ones)."""
    out = {}
    for i in range(0, len(pmids), 200):
        for art in eutils("efetch.fcgi", retmode="xml", id=",".join(pmids[i:i + 200])).iter("PubmedArticle"):
            out[art.findtext("MedlineCitation/PMID")] = headings(art)
    return out


def trees(uis) -> dict[str, list[str]]:
    """MeSH descriptor UI -> its tree numbers (D009157, myasthenia gravis -> C10.114.656 ...): where it sits."""
    out, uis = {}, sorted(uis)
    for i in range(0, len(uis), 100):  # the MeSH database's IDs, found by UI: newer 9-digit UIs don't convert to them
        ids = eutils("esearch.fcgi", db="mesh", term=" OR ".join(f"{u}[MHUI]" for u in uis[i:i + 100]),
                     retmax=200)["esearchresult"]["idlist"]
        res = eutils("esummary.fcgi", db="mesh", id=",".join(ids))["result"] if ids else {"uids": []}
        for d in (res[uid] for uid in res["uids"] if res[uid].get("ds_meshui")):
            out[d["ds_meshui"]] = [x["treenum"] for x in d.get("ds_idxlinks", []) if x.get("treenum")]
    return out


def heading_trees(names) -> dict[str, list[str]]:
    """{preferred name: tree numbers} of the MeSH headings named as PubMed's query translations write them ("community
    acquired pneumonia" finds "Community-Acquired Pneumonia")."""
    out, names = {}, sorted(names)
    for i in range(0, len(names), 100):
        ids = eutils("esearch.fcgi", db="mesh", term=" OR ".join(f'"{n}"[MH]' for n in names[i:i + 100]),
                     retmax=1000)["esearchresult"]["idlist"]
        res = eutils("esummary.fcgi", db="mesh", id=",".join(ids))["result"] if ids else {"uids": []}
        for d in (res[uid] for uid in res["uids"] if res[uid].get("ds_meshterms")):
            out[d["ds_meshterms"][0]] = [x["treenum"] for x in d.get("ds_idxlinks", []) if x.get("treenum")]
    return out


def summaries(pmids: list[str]) -> dict[str, dict]:
    """PMID -> its record. Books (NCBI Bookshelf: the WHO's) have a booktitle and a publisher instead of a title and a
    journal."""
    out = {}
    for i in range(0, len(pmids), 200):
        res = eutils("esummary.fcgi", id=",".join(pmids[i:i + 200]))["result"]
        for p in res["uids"]:
            d, ids = res[p], {a["idtype"]: a["value"] for a in res[p]["articleids"]}
            out[p] = dict(pmid=p, date=d["sortpubdate"][:10], journal=d["source"] or d.get("publishername", ""),
                          title=d["title"] or d.get("booktitle", ""), types=", ".join(d["pubtype"]),
                          doi=ids.get("doi", ""), pmcid=ids.get("pmc", ""))
    return out
