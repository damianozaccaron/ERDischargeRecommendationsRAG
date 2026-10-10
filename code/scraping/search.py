"""Step 3: the PubMed search, its tiers, European filter and fallback, frozen in pubmed.json; the query log's flags."""
import csv, functools, hashlib, json, re, time

from scraping.common import OUT
from scraping.pubmed import eutils, summaries

SNAPSHOT = OUT / "pubmed.json"
FALLBACK_SNAPSHOT = OUT / "pubmed_fallback.json"

# Tier 1: PubMed's Guideline article types, plus "guideline" in the title: some ESC guidelines (2018 syncope) are
# indexed only as Journal Article; and Consensus Statements, the type of SHARE's Kawasaki recommendations and the
# ESNM/ANMS reflux consensus. Tier 2, for chapters with fewer than FEW_HITS: consensus and position papers by title too.
# Consensus Development Conference, PubMed's older type, matches nothing now.
TIER1 = "(guideline[pt] OR practice guideline[pt] OR guideline*[ti] OR consensus statement[pt])"
TIER2 = ('(guideline[pt] OR practice guideline[pt] OR guideline*[ti] OR consensus[ti] OR "position paper"[ti]'
         ' OR "position statement"[ti] OR consensus statement[pt])')
# European: the title, corporate author, affiliation or journal name (its start, "The European ..." too) says so, or
# the title has a Europe-wide society's acronym (ESC, ERS, ESICM...). Not eur*: PubMed ignores a * after fewer than four
# letters, and eur*[ta] matched nothing. The acronyms come from PubMed itself: every acronym-like token in the titles of
# guideline-type records the other clauses miss, labelled by an LLM and reviewed; acronyms.csv has the counts and why
# any Europe-wide one is left out. Acronyms that are also English words (ERA, COST) are checked in capitals by
# filters(). From 2010 on: the older picks were outdated or off-topic.
ACRONYMS = {r["token"]: r["use"] for r in csv.DictReader(open(OUT / "Sources" / "acronyms.csv", encoding="utf-8"))}
EUROPE = " OR ".join(['europe*[ti]', 'europe*[cn]', 'europe*[ad]', 'europe*[ta]', '"the europe*"[ta]']
                     + [f"{a}[ti]" for a, use in ACRONYMS.items() if use == "yes"])
REST = "english[la] AND 2010:3000[dp] NOT (comment[pt] OR editorial[pt] OR letter[pt] OR published erratum[pt])"
# Fallback, for chapters left without a pick: national societies of European Union countries (a country or nationality
# in the title or corporate author, the corporate author also in the society's own language; joint Nordic ones too,
# not the UK's BTS or SIGN); the UK's NICE, the one body from outside the EU, whose guidelines PubMed lists as NCBI
# Bookshelf books, and their summaries in journals (NICE[ti]); the WHO, also Bookshelf books; the Clinical Toxicology
# Recommendations Collaborative, led by the European EAPCCT with AACT and APAMT, whose recommendations PubMed does not
# always type as guidelines. Every hit is a candidate: up to 300 per chapter, median 7 (29 September 2026, before NICE).
EU = """Austria Austrian Belgium Belgian Bulgaria Bulgarian Croatia Croatian Cyprus Cypriot Czech Czechia Denmark Danish
Estonia Estonian Finland Finnish France French Germany German Greece Greek Hellenic Hungary Hungarian Ireland Irish Italy
Italian Latvia Latvian Lithuania Lithuanian Luxembourg Malta Maltese Netherlands Dutch Poland Polish Portugal Portuguese
Romania Romanian Slovakia Slovak Slovenia Slovenian Spain Spanish Sweden Swedish Nordic Scandinavian AWMF""".split()
OWN_LANGUAGE = "italian* deutsch* francais* espanol* nederland* svensk* dansk* suomen polsk* portugues* osterreich*".split()
NATIONAL = " OR ".join([f"{w}[ti]" for w in EU] + [f"{w}[cn]" for w in EU + OWN_LANGUAGE])
WHO = '"World Health Organization"[pubn]'
NICE = '"National Institute for Health and Care Excellence"[pubn] OR NICE[ti]'
CTRC = '"Clinical Toxicology Recommendations Collaborative"[tiab]'
FALLBACK = f"((({NATIONAL} OR {NICE} OR {WHO}) AND {TIER2}) OR {CTRC})"

FEW_HITS = 3  # fewer tier-1 hits are often off-topic: tier 2 adds candidates
MANY_HITS = 300  # flagged in queries.csv: a long selection prompt
MAX_HITS = 10000  # ESearch's most per call; the largest chapter had ~1,450 (29 September 2026)


@functools.cache
def filters() -> str:
    """The European filter, English, from 2010 on. PubMed's [ti] is case-blind (ERA[ti] also finds "era"), so the
    acronyms marked "case" are searched once per run and kept, as PMIDs, only where the title has them in capitals."""
    case = [a for a, use in ACRONYMS.items() if use == "case"]
    ids = eutils("esearch.fcgi", term=f"({' OR '.join(f'{a}[ti]' for a in case)}) AND {TIER2} AND {REST}",
                 retmax=10000)["esearchresult"]["idlist"] if case else []
    capitals = re.compile(rf"\b({'|'.join(case)})\b")
    keep = [p for p, r in summaries(ids).items() if r["title"] != r["title"].upper() and capitals.search(r["title"])]
    return f"({' OR '.join([EUROPE] + [f'{p}[uid]' for p in keep])}) AND {REST}"


def search(query: str, fallback: bool = False) -> dict:
    """ESearch's result and tier (2 when tier 1 finds fewer than FEW_HITS): idlist (every hit: PubMed's relevance
    order changes from one search to the next, so any cut by it is a lottery), count, querytranslation,
    translationset (what each term mapped to). With fallback, tier 3: every FALLBACK hit."""
    if fallback:
        return eutils("esearch.fcgi", term=f"({query}) AND {FALLBACK} AND {REST}",
                      retmax=MAX_HITS)["esearchresult"] | {"tier": 3}
    for tier, types in (1, TIER1), (2, TIER2):
        res = eutils("esearch.fcgi", term=f"({query}) AND {types} AND {filters()}", retmax=MAX_HITS)["esearchresult"]
        if int(res["count"]) >= FEW_HITS:
            break
    return res | {"tier": tier}


def flags(res: dict) -> str:
    """0 or too many hits, and words PubMed found nowhere, which it drops from the query."""
    n = int(res["count"])
    out = ["0 hits"] if n == 0 else [f"{n} hits"] if n > MANY_HITS else []
    return "; ".join(out + [f"not found: {w}" for w in res.get("errorlist", {}).get("phrasesnotfound", [])])


def split_terms(terms: list[str], res: dict) -> str:
    """Phrases with no MeSH match, which PubMed searches as their single words ANDed."""
    mapped = {t["from"].lower() for t in res.get("translationset", [])}
    return "; ".join(t for t in terms if " " in t and t.lower() not in mapped)


def search_defs(tier: int) -> str:
    """What a chapter's search result depends on besides its query: a tier-1 result does not use TIER2, so changing
    TIER2 searches only the tier-2 chapters again (each new search reshuffles PubMed's order, see snapshot()); tier 3
    is the fallback's."""
    if tier == 3:
        return hashlib.sha1(f"{FALLBACK} {REST}".encode()).hexdigest()
    return hashlib.sha1(f"{TIER1} {TIER2 if tier == 2 else ''} {EUROPE} {REST} {MAX_HITS} {FEW_HITS}".encode()).hexdigest()


def snapshot(queries: dict[str, str], refresh: bool, fallback: bool = False) -> tuple[dict[str, dict], dict[str, dict]]:
    """Each chapter's ESearch result and every candidate's ESummary record, kept in pubmed.json (pubmed_fallback.json
    for the fallback's) and reused unless refresh, or the chapter's query or the search definitions its result used
    changed (search_defs()). The hit sets grow as PubMed indexes new records: a frozen snapshot keeps runs
    reproducible and the selections cached."""
    path = FALLBACK_SNAPSHOT if fallback else SNAPSHOT
    snap = json.loads(path.read_text()) if path.exists() and not refresh else {}
    today, chapters, records = time.strftime("%Y-%m-%d"), dict(snap.get("chapters", {})), snap.get("records", {})
    current = lambda c, q: c.get("query") == q and c.get("defs") == search_defs(c["result"]["tier"])
    try:
        for num, q in queries.items():
            if not current(chapters.get(num, {}), q):
                res = search(q, fallback)
                chapters[num] = dict(query=q, date=today, defs=search_defs(res["tier"]), result=res)
    finally:  # an interrupted run keeps the searches it did
        path.write_text(json.dumps(dict(chapters=chapters, records=records)))
    chapters = {num: chapters[num] for num in queries}  # chapters no longer searched (SKIPPED ones) drop out
    needed = list(dict.fromkeys(p for c in chapters.values() for p in c["result"]["idlist"]))
    records |= summaries([p for p in needed if p not in records])
    records = {p: records[p] for p in needed if p in records}
    path.write_text(json.dumps(dict(chapters=chapters, records=records)))
    return {num: c["result"] for num, c in chapters.items()}, records
