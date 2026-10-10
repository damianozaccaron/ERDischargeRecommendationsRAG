"""Steps 1-2: the book's chapters, and each chapter's search terms, from its title and from its references, cached
in topics.json."""
import collections, json, re
from concurrent.futures import ThreadPoolExecutor

import pymupdf
from bs4 import BeautifulSoup

from scraping.common import OUT, ask_all, http, retry
from scraping.pubmed import eutils, mesh, trees

TOC = OUT / "Sources" / "TintinalliChapters.html"
REFERENCES = OUT / "Sources" / "references.json"

# Prehospital care, disasters, resuscitative procedures, the neurologic examination, death notification, military
# medicine, legal issues; child, partner and elder abuse (screening, forensic examination, reporting: 150 has only
# imaging guidance). Sexual assault (293) stays: its post-exposure prophylaxis is a discharge prescription.
SKIPPED = {*range(1, 11), *range(22, 34), 150, 164, 294, 295, *range(301, 304)}
# Topics the 2019 book predates, searched like its chapters: (number, title, section)
EXTRA = [("305", "COVID-19", "Infectious Diseases")]
# A term made only of these (stopwords included) pulls in unrelated guidelines; inside a longer term they stay.
GENERIC = set("""guideline guidelines management treatment diagnosis recommendation recommendations emergency
emergencies evaluation assessment disposition issue issues problem problems approach initial general lesion lesions
of the and in for to a an with on or""".split())

POPULATION = {"pediatric": "(child OR infant OR pediatric OR adolescent)", "pregnancy": "(pregnancy OR pregnant OR obstetric)",
              "elderly": "(elderly OR geriatric OR older adults)"}

TOPIC_PROMPT = """Below are chapters of Tintinalli's Emergency Medicine, as "number | section | title".
For each chapter, give:
- terms: 1-6 phrases that the title of a clinical practice guideline on the chapter's core topic would use.
- population: "pediatric", "pregnancy" or "elderly" if the chapter (its title or section) is about that patient group \
and the terms don't already name it; otherwise null.

Rules:
- Only the chapter's core topic. No broader categories and no neighbouring conditions (e.g. "Accidental Hypothermia" \
must not include therapeutic hypothermia).
- No generic words: guideline, management, treatment, diagnosis, recommendation.
- Plain phrases only: no quotes, operators, parentheses or wildcards.

Return only JSON: {"<number>": {"terms": [...], "population": ...}}
Examples:
{"273": {"terms": ["hip fracture", "femoral fracture", "proximal femur fracture"], "population": null},
 "121": {"terms": ["otitis media", "otitis externa", "mastoiditis"], "population": "pediatric"},
 "118": {"terms": ["sudden infant death syndrome", "sudden unexpected infant death"], "population": null}}

"""

# Qualifiers that make a substance a toxicology topic (Acetaminophen/poisoning); see topic()
POISON = ("poisoning", "toxicity", "adverse effects")
# Reference topics: cited this often in the chapter to be a candidate, candidates shown per chapter, picks per chapter,
# chapters per call
REF_MIN, REF_SHOWN, REF_MAX, REF_CHAPTERS = 2, 8, 3, 15

REF_PROMPT = """Below are chapters of Tintinalli's Emergency Medicine, each with the guideline search terms already \
taken from its title and the MeSH topics its own references cite most, as "topic (references in this chapter, \
chapters citing it)".
For each chapter, pick up to 3 reference topics that are core subjects of the chapter the title terms miss, so the \
guideline search should also cover them. Skip topics a title term already covers (same condition, synonym, subtype or \
parent), topics that are only complications, comorbidities, differential diagnoses or treatment side effects, and \
generic topics cited across many chapters. Fewer than 3, or none, is fine.
Return only JSON: {"<number>": ["<topic exactly as listed>", ...]}

"""


def chapters() -> list[tuple[str, str, str]]:
    """(number, title, section) for every chapter; 29 is split into 29A and 29B."""
    text = BeautifulSoup(TOC.read_text(encoding="utf-8"), "lxml").get_text("\n", strip=True)
    out, section = [], ""
    for line in text.splitlines():
        if m := re.match(r"SECTION \d+\. (.+)", line):
            section = m[1]
        elif m := re.match(r"Chapter (\d+[AB]?): (.+)", line):
            out.append((m[1], m[2], section))
    return out


def corpus_chapters() -> list[tuple[str, str, str]]:
    return [c for c in chapters() + EXTRA if int(c[0].rstrip("AB")) not in SKIPPED]


def save_topics(done: dict):
    """topics.json, one chapter per line, to edit by hand."""
    (OUT / "topics.json").write_text("{\n" + ",\n".join(f" {json.dumps(k)}: {json.dumps(v)}" for k, v in done.items())
                                     + "\n}\n")


def topics(chs) -> dict[str, dict]:
    """{number: {"terms": [...], "population": ...}}; chapters missing from the LLM's answer are asked again next run."""
    path = OUT / "topics.json"
    done = json.loads(path.read_text()) if path.exists() else {}
    todo = [c for c in chs if c[0] not in done]
    prompts = {f"terms{i // 40}": TOPIC_PROMPT + "\n".join(f"{n} | {s} | {t}" for n, t, s in todo[i:i + 40])
               for i in range(0, len(todo), 40)}
    for reply in ask_all(prompts, "title terms").values():
        try:
            got = json.loads(reply[reply.find("{"):reply.rfind("}") + 1])
        except json.JSONDecodeError as e:
            print(f"  unreadable batch, asked again next run: {e}")
            continue
        for num, v in got.items():
            terms = [re.sub(r'["()\[\]*]', "", t).strip() for t in v.get("terms") or []]  # PubMed syntax
            terms = [t for t in terms if not set(re.findall(r"[a-z]+", t.lower())) <= GENERIC]
            if terms:
                done[num] = dict(terms=terms, population=v.get("population"))
    save_topics(done)
    return done


def references() -> dict:
    """The book's own references: each chapter's PMIDs (from the companion site's reference PDFs, which the saved TOC
    links), each reference's MeSH headings, each heading's tree numbers. What a chapter covers, in its authors'
    sources. The 9e's references don't change: fetched once into Sources/references.json."""
    if REFERENCES.exists():
        return json.loads(REFERENCES.read_text())
    soup = BeautifulSoup(TOC.read_text(encoding="utf-8"), "lxml")
    urls = {m[1]: a["href"] for a in soup.find_all("a", href=True)
            if (m := re.match(r"Chapter (\d+[AB]?):", a.get_text(strip=True)))}
    def pmids(url):
        with pymupdf.open(stream=retry(lambda: http.get(url, timeout=60)).content, filetype="pdf") as doc:
            return list(dict.fromkeys(re.findall(r"PMID:\s*(\d+)", " ".join(p.get_text() for p in doc))))
    with ThreadPoolExecutor(4) as pool:
        chapters_ = dict(zip(urls, pool.map(pmids, urls.values())))
    heads = {p: [h for h in hs if h["major"]] for p, hs in mesh(sorted({p for v in chapters_.values() for p in v})).items()}
    refs = dict(chapters=chapters_, mesh=heads, trees=trees({h["ui"] for hs in heads.values() for h in hs}))
    REFERENCES.write_text(json.dumps(refs))
    return refs


def topic(h: dict, tree: dict) -> tuple[str, str] | None:
    """(key, name) of the chapter topic a reference's MeSH heading names, None if none: a main-topic disease (tree C)
    or mental disorder (F03), or a main-topic substance (D) indexed as poisoning, toxicity or adverse effects, since
    poisonings are indexed on the substance (Acetaminophen/poisoning)."""
    t = tree.get(h["ui"], [])
    if not h["major"]:
        return None
    if any(x.startswith(("C", "F03")) for x in t):
        return h["ui"], h["name"]
    q = next((q for q in POISON if q in h["qualifiers"]), None)
    if q and any(x.startswith("D") for x in t):
        return f"{h['ui']}/{q}", f"{h['name']} {q}"
    return None


def chapter_topics(refs: dict) -> tuple[dict, dict, collections.Counter]:
    """Per chapter, how many of its references have each topic(); each topic's name; how many chapters cite it (a
    comorbidity like hypertension is cited in many, a chapter's own subject in one or two)."""
    per, names = {}, {}
    for num, pmids in refs["chapters"].items():
        counts = collections.Counter()
        for p in pmids:
            found = dict(t for h in refs["mesh"].get(p, []) if (t := topic(h, refs["trees"])))
            names |= found
            counts.update(found.keys())
        per[num] = counts
    return per, names, collections.Counter(k for c in per.values() for k in c)


def ref_candidates(counts: collections.Counter, names: dict, translation: str) -> list[str]:
    """A chapter's reference topics cited REF_MIN+ times, most cited first, but those its title query already searches:
    PubMed's translation of the query has their MeSH heading ("esophageal perforation"[MeSH Terms]; for a substance,
    the substance's)."""
    translation = translation.lower()
    heading = lambda k: names[k][:-len(k.split("/")[1]) - 1] if "/" in k else names[k]
    return [k for k, n in counts.most_common() if n >= REF_MIN
            and f'"{heading(k).lower()}"[mesh terms]' not in translation][:REF_SHOWN]


def read_refs(reply: str, chapters: list[str], cands: dict[str, list[str]], names: dict) -> dict[str, list[str]]:
    """The topics picked for each chapter of a reference-topic prompt, by MeSH name: only topics listed for the chapter,
    exactly, REF_MAX at most. A chapter the reply leaves out gets []: the model omits the chapters it adds nothing to,
    and an unmarked chapter would be screened again, and its search and selection redrawn, on every run. Raises on
    an unreadable reply."""
    got = json.loads(reply[reply.find("{"):reply.rfind("}") + 1])
    out = {}
    for num in chapters:
        listed, picked = {names[k] for k in cands[num]}, got.get(num)
        out[num] = list(dict.fromkeys(t for t in picked if t in listed))[:REF_MAX] if isinstance(picked, list) else []
    return out


def reference_topics(chs, tops: dict[str, dict]) -> dict[str, dict]:
    """Adds to each chapter's terms up to REF_MAX topics of its references that its title terms miss, picked by the LLM
    among ref_candidates() (GERD in Esophageal Emergencies, Kawasaki disease in Rashes in Infants and Children), by
    their MeSH names, which PubMed maps exactly. Each chapter is screened once: topics.json lists what was added under
    "refs", [] for nothing; a chapter whose prompt got no readable reply is asked again next run."""
    todo = [c for c in chs if c[0] in tops and "refs" not in tops[c[0]]]
    if not todo:
        return tops
    per, names, citing = chapter_topics(references())
    translate = lambda num: eutils("esearch.fcgi", term=build_query(tops[num]), retmax=0)["esearchresult"].get(
        "querytranslation", "")
    with ThreadPoolExecutor(3) as pool:  # NCBI's limit
        cands = {num: ref_candidates(per.get(num, collections.Counter()), names, tr)
                 for num, tr in zip([c[0] for c in todo], pool.map(translate, [c[0] for c in todo]))}
    for num, _, _ in todo:
        if not cands[num]:  # nothing to screen: no references (chapter 32, EXTRA), or the title terms cover them
            tops[num]["refs"] = []
    asked = [c for c in todo if cands[c[0]]]
    line = lambda num, title: (f"{num} | {title} | terms: {'; '.join(tops[num]['terms'])} | topics: " + "; ".join(
        f"{names[k]} ({per[num][k]}, {citing[k]})" for k in cands[num]))
    groups = {f"refs{i // REF_CHAPTERS}": asked[i:i + REF_CHAPTERS] for i in range(0, len(asked), REF_CHAPTERS)}
    prompts = {k: REF_PROMPT + "\n".join(line(n, t) for n, t, _ in g) for k, g in groups.items()}
    for k, reply in ask_all(prompts, "reference topics").items():
        try:
            picked = read_refs(reply, [c[0] for c in groups[k]], cands, names)
        except json.JSONDecodeError as e:
            print(f"  reference topics: unreadable reply, asked again next run: {e}")
            continue
        for num, refs in picked.items():
            lower = {t.lower() for t in tops[num]["terms"]}
            tops[num]["terms"] += [t for t in refs if t.lower() not in lower]
            tops[num]["refs"] = refs
    save_topics(tops)
    added = {n: t["refs"] for n, t in tops.items() if t.get("refs")}
    print(f"Reference topics: {sum(map(len, added.values()))} added to {len(added)} chapters")
    return tops


def build_query(topic: dict) -> str:
    """((term) OR (term) ...), AND the population's clause if any."""
    query = "(" + " OR ".join(f"({t})" for t in topic["terms"]) + ")"
    if pop := topic.get("population"):
        query += " AND " + POPULATION.get(pop, f"({pop})")
    return query
