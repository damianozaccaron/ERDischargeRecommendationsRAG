"""Step 4: the LLM's picks among each chapter's candidates, and one consolidation pass over all chapters' picks."""
import functools, hashlib, itertools, json, re

from scraping.common import ask_all
from scraping.search import ACRONYMS

MAX_PICKS = 3  # more guidelines for one chapter are flagged in manifest.csv, not dropped
# Title words too common to tell two guidelines apart, left out when looking for picks that may repeat
COMMON = set("""guideline guidelines management with from that this their clinical practice european society
recommendations the and for update diagnosis treatment""".split())
# A title that names a European body: Europe, European, Euro... or a Europe-wide society's acronym, in capitals
NAMED_EUROPEAN = re.compile(r"(?i:\beuro)|\b(?:" + "|".join(
    map(re.escape, sorted(a for a, use in ACRONYMS.items() if use in ("yes", "case")))) + r")\b")

SELECT_PROMPT = """Chapter of Tintinalli's Emergency Medicine: "{title}" (section: {section}).
Topics of this chapter: {terms}
PubMed records, newest first, as "PMID | date | journal | publication types | title":
{records}

Select guidelines for a corpus of European clinical guidelines, following these rules.

Purpose: the corpus helps emergency physicians write discharge letters, so a pick must guide how an emergency \
physician assesses, treats or discharges a patient with the topic's condition.

{qualifies}

Reject: {outsiders}; guidelines on a different condition; guidelines for a different population than the \
chapter (e.g. pediatric-only for an adult chapter); studies of guideline adherence or implementation; reviews and \
commentaries on a guideline; versions republished in another journal or language when the original is listed; pocket \
or summary versions when the full guideline is listed; older versions when a newer version of the same guideline is \
listed; documents limited to imaging, laboratory methods, classification criteria, training or surgical technique; \
systematic reviews that issue no recommendations.

How many: the smallest set of guidelines that together cover the topics.
- Each topic is covered by at most one guideline: the one that covers it best; among equals, the most recent.
- For the same topic, prefer a clinical practice guideline to a consensus statement or position paper.
- Prefer one guideline that covers several topics over several narrower ones.
- Add a guideline only if it covers a topic that no other selected guideline covers.
- A topic may stay uncovered. Do not cover it with a guideline on a neighbouring condition.
- If no record qualifies, return an empty list.

Answer with only JSON:
{{"picks": [{{"pmid": "<PMID>", "covers": ["<topic>", ...], \
"name": "<short CamelCase name of the guideline's topic, e.g. AcutePulmonaryEmbolism>"}}]}}"""
# The EM Model's topics (em.py) are sections of a curriculum, not chapters of the book; the chapters' prompt is left as
# it was, its answers cached by it
EM_SELECT_PROMPT = SELECT_PROMPT.replace(
    "Chapter of Tintinalli's Emergency Medicine",
    "Section of the Model of the Clinical Practice of Emergency Medicine (ABEM, 2022)").replace(
    "Topics of this chapter", "Topics of this section").replace(
    "than the chapter (e.g. pediatric-only for an adult chapter)", "than the topics (e.g. pediatric-only for a topic of all "
    "ages)")
# Who qualifies, in the European tiers and in the fallback
EUROPEAN_RULES = dict(
    qualifies="Qualifies: a clinical practice guideline, consensus statement or position paper issued by a European medical "
              "society (ESC, ERC, ESO, EAACI, ESCMID, ESAIC, ERS, EAU, EASL, UEG, ESPGHAN, ...) whose subject is one of the "
              "topics above. A national society of a European country qualifies only for a topic that no Europe-wide "
              "guideline in the list covers.",
    outsiders="non-European bodies")
FALLBACK_RULES = dict(
    qualifies="Qualifies: a clinical practice guideline, consensus statement or position paper whose subject is one of the "
              "topics above, issued by a European medical society, by medical societies of European Union countries (one "
              "or several), by the United Kingdom's NICE, by the World Health Organization, or by the Clinical Toxicology "
              "Recommendations Collaborative (led by the European EAPCCT with AACT and APAMT). For the same topic, prefer "
              "a Europe-wide guideline to a national one (NICE's included), and an Italian society's to another "
              "country's.",
    outsiders="bodies of countries outside the European Union other than NICE (the United Kingdom's BTS or SIGN "
              "included) and other international bodies")

CONSOLIDATE_PROMPT = """Below are numbered pairs of records with alike titles: a guideline picked for a corpus and \
either another pick or a newer record, each as "PMID | date | journal | title". Give a verdict for every pair:
- "same": the same document in two journals or translated, or a summary, pocket or practical-instruction version of \
the other; keep the full original, in the society's own journal;
- "newer": two versions of one guideline (same society, same topic; an "update of" a guideline is a newer version); \
keep the newest, if it is the full guideline (not a summary of it or a commentary on it). A guideline from another \
society (a national one included), or a practical, pocket or summary version, is never a newer version;
- "different": different documents: other topics or populations, separate parts of one guideline, a focused update \
(titled so) and the guideline it updates, companion documents; keep both.
Answer with only JSON: {{"verdicts": [{{"pair": <number>, "verdict": "same|newer|different", \
"keep": "<PMID to keep, for same and newer>"}}, ...]}}, one per pair.

{pairs}"""


def select_prompt(chapter, terms: list[str], cands: list[dict], rules: dict = EUROPEAN_RULES) -> str:
    """The selection prompt, candidates newest first (PMID among the same date): the same candidates make the same
    prompt whatever PubMed's order, so a rerun reuses the cached answer (key())."""
    num, title, section = chapter
    cands = sorted(cands, key=lambda c: (c["date"], c["pmid"]), reverse=True)
    records = "\n".join(f"{c['pmid']} | {c['date']} | {c['journal']} | {c['types']} | {c['title']}" for c in cands)
    prompt = EM_SELECT_PROMPT if num.startswith("EM") else SELECT_PROMPT
    return prompt.format(title=title, section=section, terms="; ".join(terms), records=records, **rules)


def key(prompt: str) -> str:
    """A prompt's cache key in selections.json."""
    return hashlib.sha1(prompt.encode()).hexdigest()


def last_json(reply: str, key: str) -> dict:
    """The last {"key": ...} object in an LLM reply: on hard cases the model reasons out loud before answering."""
    starts = [m.start() for m in re.finditer(r'\{\s*"%s"' % key, reply)]
    if not starts:
        raise ValueError(f'no {{"{key}": ...}} in the reply, cut off at max_tokens? It ends: ...{reply[-80:]!r}')
    return json.JSONDecoder().raw_decode(reply, starts[-1])[0]


def read_picks(reply: str, terms: list[str], cands: list[dict]) -> dict:
    """The LLM's picks among the candidates, with flags to review by hand: more than MAX_PICKS picks (kept), PMIDs not
    among the candidates (made up: dropped), covered topics not in the chapter's list. Raises on an unreadable reply."""
    picks = last_json(reply, "picks")["picks"]
    known, listed = {c["pmid"] for c in cands}, {t.lower() for t in terms}
    flags = [f"made-up PMID: {p.get('pmid')}" for p in picks if str(p.get("pmid")) not in known]
    picks = [dict(pmid=str(p["pmid"]), covers=p.get("covers") or [], name=re.sub(r"\W", "", p.get("name") or ""))
             for p in picks if str(p.get("pmid")) in known]
    flags += [f"topic not listed: {t}" for p in picks for t in p["covers"] if t.lower() not in listed]
    if len(picks) > MAX_PICKS:
        flags.insert(0, f"{len(picks)} picks")
    return dict(picks=picks, flags="; ".join(flags))


def alike(picks: list[dict], pool: list[dict]) -> list[tuple[dict, dict]]:
    """Pairs that may be one guideline twice: two picks, or a pick and a newer record of the pool (every chapter's
    candidates), whose titles share 60% of the shorter title's words of 4+ letters (years and COMMON words left out).
    Measured on the shorter title, since some list every section (ERC 2010: "... drowning, accidental hypothermia,
    ..."). Deliberately broad; consolidate()'s LLM calls keep different documents apart. Titles with different part
    numbers (or one only in one) are parts of one guideline, never paired: the LLM took a "Part II" for a newer
    version of the unnumbered first part."""
    @functools.cache  # each title once: the pool has every chapter's hits
    def title(text):
        return (set(re.findall(r"[a-z]{4,}", text.lower())) - COMMON,
                re.findall(r"\bpart\s*([0-9]+|[ivx]+)\b", text.lower()))
    def close(a, b):
        (wa, pa), (wb, pb) = title(a["title"]), title(b["title"])
        return wa and wb and len(wa & wb) >= 0.6 * min(len(wa), len(wb)) and pa == pb
    ids = {r["pmid"] for r in picks}
    newer = [r for r in pool if r["pmid"] not in ids]
    return ([(a, b) for a, b in itertools.combinations(picks, 2) if close(a, b)]
            + [(a, b) for a in picks for b in newer if b["date"] > a["date"] and close(a, b)])


def read_verdicts(reply: str, chunk: list[tuple[dict, dict]]) -> list[list[str]]:
    """[PMID, PMID, verdict, PMID to keep] per pair of a consolidation reply, as the model gave it; replacements()
    decides what to apply. Raises on an unreadable reply."""
    out = []
    for v in last_json(reply, "verdicts")["verdicts"]:
        if 0 < int(v.get("pair", 0)) <= len(chunk):
            a, b = chunk[int(v["pair"]) - 1]
            out.append([a["pmid"], b["pmid"], str(v.get("verdict")), str(v.get("keep"))])
    return out


def replacements(verdicts: list[list[str]], records: dict[str, dict]) -> dict[str, str]:
    """PMID -> PMID to use instead, from the verdicts: a "same" or "newer" pair replaces the record not kept. A record
    replaced in several pairs (2010 by 2015 and by 2025) takes the newest; chains (2010 -> 2021 -> 2025) are followed.
    A "newer" verdict never replaces a guideline whose title names a European body (NAMED_EUROPEAN) by one whose title
    names none: the model took the 2019 French meningitis guideline, titled "Management of acute community-acquired
    bacterial meningitis", for a newer version of ESCMID's 2016 one. Only the titles count, not the journals: picks are
    often reprints or summaries in other journals (the 2018 ESC pregnancy guideline in Kardiol Pol, a BMJ summary of
    NICE's head injury guideline), whose newer versions a same-journal rule blocked on 6 October 2026."""
    european = lambda p: bool(NAMED_EUROPEAN.search(records.get(p, {}).get("title", "")))
    date = lambda p: records.get(p, {}).get("date", "")
    replace = {}
    for a, b, verdict, keep in verdicts:
        if verdict not in ("same", "newer") or keep not in (a, b):
            continue
        old = b if keep == a else a
        if verdict == "newer" and european(old) and not european(keep):
            print(f"  consolidation: {old} kept, not replaced by {keep}: a European guideline, by a newer version "
                  f"that names no European body")
        elif date(keep) > date(replace.get(old, "")):
            replace[old] = keep
    for a in replace:
        for _ in range(len(replace)):
            replace[a] = replace.get(replace[a], replace[a])
    return {a: b for a, b in replace.items() if a != b}


def consolidate(picks: list[dict], pool: list[dict], cache: dict) -> dict[str, str]:
    """PMID -> PMID to use instead, for all chapters' picks: the same guideline in two journals or translated, a summary
    of a full guideline, an older version of a newer one, picked elsewhere or only found by some chapter's search.
    Each chapter's call sees only its own candidates, so these repeat across chapters. The alike() pairs go to LLM
    calls of 60 pairs, a verdict per pair (asked only for the replacements, the model skipped some), each call's
    verdicts cached in selections.json by its prompt; a failed one leaves its picks as they are until the next run."""
    pairs = alike(picks, pool)
    line = lambda r: f"{r['pmid']} | {r['date']} | {r['journal']} | {r['title']}"
    chunks = [pairs[i:i + 60] for i in range(0, len(pairs), 60)]
    prompts = {key(p): (p, chunk) for chunk in chunks for p in [CONSOLIDATE_PROMPT.format(
        pairs="\n\n".join(f"{i}.\n{line(a)}\n{line(b)}" for i, (a, b) in enumerate(chunk, 1)))]}
    done = {k: v for k, v in cache.get("verdicts", {}).items() if k in prompts}
    for k, reply in ask_all({k: p for k, (p, _) in prompts.items() if k not in done}, "consolidation").items():
        try:
            done[k] = read_verdicts(reply, prompts[k][1])
        except Exception as e:
            print(f"  consolidation: {type(e).__name__}: {e}")
    cache["verdicts"] = done
    cache.pop("consolidation", None)  # replacements cached before 6 October 2026, without the verdicts
    return replacements([v for vs in done.values() for v in vs], {r["pmid"]: r for r in pool + picks})
