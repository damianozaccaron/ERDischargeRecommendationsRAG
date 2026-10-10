"""Step 2b: the topics of the Model of the Clinical Practice of Emergency Medicine (EM Model; ABEM, 2022) that no
chapter searches. Its Table 4 is read from ABEM's PDF; each clinical item is named by the LLM under its headings; an item
is covered when a chapter term reads, in PubMed's own term mapping, as the same concepts; the uncovered items, tumours
left out, are pooled under their EM heading into topics searched and selected like the chapters."""
import collections, csv, functools, html, json, re, subprocess
from concurrent.futures import ThreadPoolExecutor

from scraping.common import OUT, ask_all, http, retry
from scraping.pubmed import eutils, heading_trees
from scraping.topics import GENERIC

URL = "https://www.abem.org/wp-content/uploads/2024/07/2022-EM-Model-for-Website.pdf"
PDF = OUT / "Sources" / "2022-EM-Model-for-Website.pdf"
TABLE = OUT / "Sources" / "em_model_2022.csv"
NAMES = OUT / "em_names.json"
MAPPINGS = OUT / "mappings.json"
# Categories 2-18. 1 (signs and symptoms) is left out: a generic symptom has no guideline of its own, and the book's
# presentation chapters (syncope, chest pain, headache) already search the ones that do. 19-20 (procedures, core
# competencies) are left out for the reasons the book's SKIPPED chapters are.
CATEGORIES = range(2, 19)
# Items left out with their branches, as the book's abuse chapters are (topics.SKIPPED): tobacco use disorder, and
# violence, abuse and neglect, whose sexual assault prophylaxis chapter 293 searches.
SKIPPED = ("14.1.4", "14.6")
# The EM Model's branches on pregnancy, labour, delivery and the puerperium, the one population it sets apart: a chapter
# term searched for pregnancy covers only their items, and one searched for children or the elderly covers none.
PREGNANCY = ("13.2", "13.3", "13.4", "13.5", "13.6", "13.7", "13.8", "18.2")
# Tumours, left out: MeSH's neoplasms by histologic type and by site. Not all of C04, which also files cysts (pilonidal
# sinus, ovarian cysts). Cancer care guidelines do not guide an emergency discharge; oncologic emergencies are chapters.
TUMOUR = ("C04.557", "C04.588")

NAME_PROMPT = """Below are items of the Model of the Clinical Practice of Emergency Medicine (ABEM, 2022), as \
"number | section | headings › item".
For each item, give the condition or problem it stands for, named as the title of a clinical practice guideline on it \
would name it:
- Read the item under its headings: "Acid" under "Toxic effects of caustic agents" is "acid ingestion", "Tension" under \
"Pneumothorax" is "tension pneumothorax".
- One name per condition: an item joining conditions with "/" gets one for each ("Epididymitis/Orchitis": \
"epididymitis", "orchitis"). No synonyms, no broader or narrower conditions.
- Plain phrases only: no quotes, operators, parentheses or wildcards.

Return only JSON: {"<number>": ["<name>", ...]}
Examples:
{"2.2.2.3.1": ["acid ingestion"], "16.2.6.2": ["tension pneumothorax"], "15.5.3.2": ["epididymitis", "orchitis"],
 "17.1.11.3": ["beta-blocker poisoning"]}

"""


def parse(pdf) -> list[dict]:
    """Table 4's category headings (num "2" for "2.0") and items, each with its title and marks: the columns (Critical,
    Emergent, Lower) of its Xs. From the words pdftotext -bbox places: a line is the words at one height, an item a line
    that starts with its number, and a line without one continues the item, or the category heading, above it."""
    xml = subprocess.run(["pdftotext", "-bbox", str(pdf), "-"], capture_output=True, text=True, check=True).stdout
    rows, cols, heading = [], {}, False
    for page in xml.split("<page ")[1:]:
        lines = []  # [height, [(x, word), ...]], words 3 points apart in height at most
        for x0, y0, x1, y1, word in sorted(re.findall(r'<word xMin="([\d.]+)" yMin="([\d.]+)" xMax="([\d.]+)" '
                                                     r'yMax="([\d.]+)">([^<]*)</word>', page),
                                          key=lambda w: float(w[1]) + float(w[3])):
            x, y = (float(x0) + float(x1)) / 2, (float(y0) + float(y1)) / 2
            if lines and abs(lines[-1][0] - y) <= 3:
                lines[-1][1].append((x, html.unescape(word)))
            else:
                lines.append([y, [(x, html.unescape(word))]])
        for _, words in lines:
            words.sort()
            tokens = [w for _, w in words]
            text = " ".join(tokens)
            if re.fullmatch(r"\d+\.0 [^a-z]+", text):  # "2.0 ABDOMINAL AND GASTROINTESTINAL DISORDERS"
                rows.append(dict(num=tokens[0][:-2], title=text.split(" ", 1)[1], marks=set()))
                heading = True
                continue
            if not rows or text.startswith(("Model of the Clinical Practice", "Page ")):
                continue  # pages before Table 4 (Figure 1 lists changes by number too), running heads
            if {"Critical", "Emergent", "Lower"} <= set(tokens):
                cols = {w: x for x, w in words if w in ("Critical", "Emergent", "Lower")}
                continue
            if re.fullmatch(r"\d+(\.\d+)+", tokens[0]):
                rows.append(dict(num=tokens[0], title="", marks=set()))
                heading = False
                words = words[1:]
            elif heading:  # "19.0 PROCEDURES AND ... OF EMERGENCY", then "MEDICINE"
                rows[-1]["title"] += " " + text
                continue
            row = rows[-1]
            row["title"] = " ".join([row["title"]] + [w for _, w in words if w != "X"]).strip()
            row["marks"] |= {min(cols, key=lambda c: abs(cols[c] - x)) for x, w in words if w == "X"}
    return rows


def table() -> list[dict]:
    """Table 4 of the EM Model: num, title, and "1" in critical, emergent and lower for its acuity marks. Read once into
    Sources/em_model_2022.csv from ABEM's PDF (downloaded if missing) with poppler's pdftotext: PyMuPDF's get_text drops
    ten of its rows, diabetic ketoacidosis among them (PyMuPDF 1.28.2, 7 October 2026)."""
    if not TABLE.exists():
        if not PDF.exists():
            PDF.write_bytes(retry(lambda: http.get(URL, timeout=120)).content)
        with open(TABLE, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["num", "title", "critical", "emergent", "lower"])
            w.writerows([r["num"], r["title"], *("1" if c in r["marks"] else "" for c in ("Critical", "Emergent", "Lower"))]
                        for r in parse(PDF))
    return list(csv.DictReader(open(TABLE, encoding="utf-8")))


@functools.cache
def titles() -> dict[str, str]:
    """Every heading and item of Table 4 by number, cross-references ("(See 12.10)") cut."""
    return {r["num"]: re.sub(r"\s*\(see [^)]*\)", "", r["title"], flags=re.I).strip() for r in table()}


def path(num: str) -> str:
    """An item or heading under its headings: "Esophagus › Inflammatory disorders › Toxic effects of caustic agents ›
    Acid"."""
    parts = num.split(".")
    return " › ".join(titles()[".".join(parts[:k])] for k in range(2, len(parts) + 1))


def section(num: str) -> str:
    return titles()[num.split(".")[0]].capitalize()


def items() -> list[dict]:
    """The clinical items (CATEGORIES): the leaves of Table 4, its most specific topics. A heading is left to its
    leaves even when it has acuity marks: some are conditions the chapters search anyway (diabetic ketoacidosis,
    pharyngitis), others classes too broad for one search ("Viral", "Other bacterial diseases")."""
    parents = {n.rsplit(".", 1)[0] for n in titles()}
    return [dict(num=n, path=path(n), section=section(n)) for n in titles()
            if "." in n and int(n.split(".")[0]) in CATEGORIES and n not in parents
            and not any(n == s or n.startswith(s + ".") for s in SKIPPED)]


def names(its: list[dict]) -> dict[str, list[str]]:
    """{number: [name, ...]}, the LLM's names for the items, cached in em_names.json, one item per line, to edit by hand;
    names made only of GENERIC words are dropped, and items left without one are asked again next run."""
    done = json.loads(NAMES.read_text()) if NAMES.exists() else {}
    todo = [i for i in its if i["num"] not in done]
    asked = {i["num"] for i in todo}
    prompts = {f"names{k // 40}": NAME_PROMPT + "\n".join(f"{i['num']} | {i['section']} | {i['path']}"
                                                          for i in todo[k:k + 40]) for k in range(0, len(todo), 40)}
    for reply in ask_all(prompts, "EM names").values():
        try:
            got = json.loads(reply[reply.find("{"):reply.rfind("}") + 1])
        except json.JSONDecodeError as e:
            print(f"  unreadable batch, asked again next run: {e}")
            continue
        for num, ns in got.items():
            ns = [re.sub(r'["()\[\]*]', "", n).strip() for n in ns if isinstance(n, str)]  # PubMed syntax
            ns = [n for n in ns if n and not set(re.findall(r"[a-z]+", n.lower())) <= GENERIC]
            if num in asked and ns:
                done[num] = ns
    NAMES.write_text("{\n" + ",\n".join(f" {json.dumps(k)}: {json.dumps(v)}" for k, v in done.items()) + "\n}\n")
    return done


def mappings(terms, refresh=False) -> dict[str, str]:
    """{term: PubMed's translation of it}, frozen in mappings.json like the searches: MeSH and PubMed's term mapping
    change from year to year."""
    done = json.loads(MAPPINGS.read_text()) if MAPPINGS.exists() and not refresh else {}
    todo = sorted(set(terms) - set(done))
    translate = lambda t: eutils("esearch.fcgi", term=t, retmax=0)["esearchresult"].get("querytranslation", "")
    try:
        with ThreadPoolExecutor(3) as pool:  # NCBI's limit
            for t, tr in zip(todo, pool.map(translate, todo)):
                done[t] = tr
    finally:  # an interrupted run keeps the translations it got
        MAPPINGS.write_text(json.dumps(done))
    return done


def norm(name: str) -> str:
    """A MeSH heading as PubMed's translations write it: lowercase, without hyphens or apostrophes."""
    return re.sub(r"\s+", " ", re.sub(r"[-'’]", " ", name.lower())).strip()


def split(s: str, op: str) -> list[str]:
    """s split at op (" AND ", " OR ") outside parentheses and quotes."""
    parts, depth, quoted, start = [], 0, False, 0
    for i, c in enumerate(s):
        if c == '"':
            quoted = not quoted
        elif not quoted and c in "()":
            depth += 1 if c == "(" else -1
        elif not quoted and not depth and i >= start and s.startswith(op, i):
            parts.append(s[start:i])
            start = i + len(op)
    return [p.strip() for p in parts + [s[start:]] if p.strip()]


def unwrap(s: str) -> str:
    """s without the parentheses that enclose all of it."""
    while s.startswith("(") and s.endswith(")"):
        depth, quoted = 0, False
        for i, c in enumerate(s):
            if c == '"':
                quoted = not quoted
            elif not quoted and c in "()":
                depth += 1 if c == "(" else -1
                if not depth and i < len(s) - 1:
                    return s
        s = s[1:-1].strip()
    return s


HARM = {"poisoning", "poisonings", "poisons", "toxicity", "toxic", "adverse effects", "intoxication", "overdose",
        "drug overdose"}


def concepts(term: str, translation: str) -> list[dict]:
    """How PubMed reads a term: the concepts it ANDs, each as the MeSH headings, supplementary concepts, subheadings and
    words it searches it by (nested ANDs, its word-by-word fallbacks, left out). Toxicity, poisoning, intoxication and
    overdose are one concept: "digoxin toxicity" is "digoxin poisoning"."""
    out = []
    for part in split(unwrap(translation), " AND ") if translation else []:
        c = dict(mesh=set(), sc=set(), sh=set(), text=set())
        for alt in split(unwrap(part), " OR "):
            if m := re.fullmatch(r'"([^"]*)"\[([^\]]+)\]', alt):
                field = m[2]
                kind = ("mesh" if field.startswith("MeSH Terms") else "sc" if field == "Supplementary Concept"
                        else "sh" if "Subheading" in field else "text" if field == "All Fields" else "")
                if kind:
                    c[kind].add(norm(m[1]) if kind == "mesh" else m[1].lower())
        if c["mesh"] <= HARM and (c["mesh"] | c["sh"] | c["text"]) & HARM:
            c = dict(mesh={"harm"}, sc=set(), sh=set(), text=set())
        if any(c.values()):
            out.append(c)
    return out or [dict(mesh=set(), sc=set(), sh=set(), text={term.lower()})]


def same(a: list[dict], b: list[dict]) -> bool:
    """Whether two terms read as the same concepts: each concept of one shares a MeSH heading, supplementary concept or
    subheading with one of the other, or, when neither is mapped to MeSH, a word or phrase."""
    eq = lambda x, y: bool(x["mesh"] & y["mesh"] or x["sc"] & y["sc"] or x["sh"] & y["sh"]
                           or x["text"] & y["text"] and not (x["mesh"] and y["mesh"]))
    return all(any(eq(x, y) for y in b) for x in a) and all(any(eq(x, y) for x in a) for y in b)


def counts(num: str, population) -> bool:
    """Whether a chapter term searched for population covers item num (see PREGNANCY)."""
    return population is None or population == "pregnancy" and any(num == p or num.startswith(p + ".") for p in PREGNANCY)


def tumour(cs: list[dict], trees: dict[str, list[str]]) -> bool:
    """Whether a name is a tumour: one of its concepts maps only to MeSH headings under TUMOUR. Only: PubMed reads the
    word "tumor" (and "cyst") as neoplasms, cysts and neurofibroma at once, so "tumor hemorrhage" is not one."""
    return any(c["mesh"] and all(any(t.startswith(TUMOUR) for t in trees.get(m, [])) for m in c["mesh"]) for c in cs)


def pool(num: str) -> str:
    """The EM heading an uncovered item is pooled under: its parent, or itself when its parent is a category (pooling
    "Tumors" with "Pulmonary Hypertension" would mix unrelated conditions)."""
    parent = num.rsplit(".", 1)[0]
    return parent if "." in parent else num


def topics(tops: dict[str, dict], refresh=False) -> tuple[list[tuple[str, str, str]], dict[str, dict]]:
    """The EM topics, as (number, title, section) like the chapters ("EM4.6", "Papular/Nodular Lesions", ...), and their
    terms: the names of the pooled items no chapter term covers (see counts()), tumours left out. em_coverage.csv has
    every item's verdict, to review."""
    its = items()
    named = names(its)
    its = [i for i in its if i["num"] in named]  # the LLM left it out: named next run
    terms = [(ch, t, v.get("population")) for ch, v in tops.items() for t in v["terms"]]
    read = {t: concepts(t, tr) for t, tr in mappings([n for ns in named.values() for n in ns]
                                                     + [t for _, t, _ in terms], refresh).items()}
    for i in its:
        cover = {n: [f"{ch}: {t}" for ch, t, pop in terms if counts(i["num"], pop) and same(read[n], read[t])]
                 for n in named[i["num"]]}
        i |= dict(names="; ".join(named[i["num"]]), new=[n for n, c in cover.items() if not c],
                  covered_by="; ".join(c for cs in cover.values() for c in cs))
    trees = {norm(h): t for h, t in heading_trees({m for i in its for n in i["new"] for c in read[n]
                                                    for m in c["mesh"]} - {"harm"}).items()}
    pooled = collections.defaultdict(dict)
    for i in its:
        i["verdict"] = ("covered" if not i["new"] else "tumour" if any(tumour(read[n], trees) for n in i["new"])
                        else "EM" + pool(i["num"]))
        if i["verdict"].startswith("EM"):
            for n in i["new"]:
                pooled[i["verdict"]].setdefault(n.lower(), n)
    with open(OUT / "em_coverage.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, ["num", "section", "path", "names", "verdict", "new", "covered_by"])
        w.writeheader()
        w.writerows(dict(i, new="; ".join(i["new"])) for i in its)
    c = collections.Counter(i["verdict"] if i["verdict"] in ("covered", "tumour") else "pooled" for i in its)
    print(f"EM Model: {len(its)} items named ({len(items()) - len(its)} not yet), {c['covered']} covered by a chapter "
          f"term, {c['tumour']} tumours left out, {c['pooled']} pooled into {len(pooled)} topics, in "
          f"{OUT / 'em_coverage.csv'}")
    return ([(k, path(k[2:]), section(k[2:])) for k in pooled],
            {k: dict(terms=list(v.values())) for k, v in pooled.items()})
