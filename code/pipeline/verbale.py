"""
The Verbale di Pronto Soccorso, the PDF the ER software prints for a visit, read into an anonymised Clinical Note in
the format of Examples/ITA, with its diagnosis, its prognosis and outcome, and the physician's discharge advice when
it has them (the provisional print has none). It reads the known layouts: the provisional print, the same print with
a Dimissione section (the synthetic notes of Data/), and the final one (SIO).
"""
import re
import sys
import unicodedata
from typing import NamedTuple

import pymupdf

# section title prefix -> label in the Clinical Note, in the order the note needs them. Procedure and Attività
# Infermieristiche hold what the emergency department did and ordered (a splint, sutures, the drugs of the stay), which
# the discharge advice follows from: ordered, not given ("Olmesartan 20 mg" is there, which the diary says was not given)
KEPT = {"anamnesi": "Anamnesi:", "esame obiettivo": "Esame obiettivo:", "diario": "Diario clinico:",
        "procedure": "Procedure:", "attività infermieristiche": "Terapie disposte in PS:"}
# the rows every visit has in Procedure, from the software's catalogue: the visit, the blood tests, the ECG, the
# monitoring, the billing line for injections, the tests requested
ROUTINE_PROCEDURE = re.compile(r"^(?:prestazioni eseguite|visita di pronto soccorso|prelievo|ecg\b|elettrocardiogramma|"
                               r"ega\b|emogas|monitoraggio|iniezione di altre sostanze|osservazione breve|trasferimento|"
                               r"richiest[io] accertamenti|note:)", re.IGNORECASE)
# a Verbale has these; the objective examination may be missing (4 of the 25 synthetic notes)
REQUIRED = ["Accettazione", "Anamnesi", "Diario"]
# the header's patient line: "COGNOME E NOME" in the provisional print, "Cognome, Nome" in the final one
PATIENT = re.compile(r"\b(?:cognome\W+(?:e\W+)?nome|nome\W+(?:e\W+)?cognome)\b", re.IGNORECASE)
SEX_AGE = re.compile(r"\b(?:sesso|genere)\b.*\bet[àa]\b", re.IGNORECASE)  # the same line when its label is missing
NEXT_FIELD = re.compile(r"\b(?:sesso|genere|et[àa]|nat[oa]|data di nascita|nr|nosologico)\b", re.IGNORECASE)
FOOTER = re.compile(r"^(?:Data ed ora della stampa|Sede legale|Pagina \d)", re.IGNORECASE)
GAP = 10  # a wider gap between two pieces of a line separates table cells
TIME = re.compile(r"^(?=.*\d)\s*(?:\d{1,2}/\d{1,2}/\d{4}\s*)?(?:\d{1,2}:\d{2})?\s*$")  # a date or a time alone
LOG = re.compile(r"^Inserit[ao] il |ha preso in carico il paziente\.?$")
# name particles: too common as words to remove on their own, and no use without the rest of the name
PARTICLES = {"de", "di", "da", "del", "dei", "dal", "dell", "dall", "della", "delle", "dello", "degli", "dalla", "dalle",
             "dallo", "la", "le", "lo"}
NAME = r"(?:(?:d[aei]|de[il]|della|la|lo)\s+|d['’])?[A-ZÀ-Þ][\w'’-]*"
# titles that are never anything else (not "prof.", which is also "profonda")
TITLED = re.compile(r"(\b(?i:dott\.?ssa|dr\.?ssa|prof\.?ssa|dott(?:or(?:e|essa)?)?|dr|sig\.?(?:ra|na)|"
                    r"signor(?:[ae]|ina)?|professor(?:e|essa)?)\b\.?)\s*" + NAME + r"(?:\s+" + NAME + r"){0,3}")
PII = re.compile(r"\b[A-Z]{6}\d{2}[A-Z]\d{2}[A-Z]\d{3}[A-Z]\b|\S+@\S+\.\w+|(?<![\d+])\+?\d(?:[\s.-]?\d){8,}(?!\d)",
                 re.IGNORECASE)
OMISSIS = "[omissis]"

# Dimissione fields holding the physician's advice, by label prefix
ADVICE_FIELDS = ("prescrizion", "si consiglia", "consigli")
# Condizioni alla dimissione: a piece made only of these words is the patient's state, any other piece is advice
STATE = re.compile(r"(?:buon|stabil|stazionari|migliorat|invariat|discret|asintomatic|paucisintomatic|eupnoic|"
                   r"apiretic|vigil|lucid|orientat|emodinamicamente|condizion|general)\w*|in|e|ed|ma", re.IGNORECASE)
POINTER = re.compile(r"^vedi\b", re.IGNORECASE)  # "Vedi diario": the advice is in the diary
# advice in a kept section: a sentence opening with a heading or an impersonal advice verb, a referral back or a call for
# work-up anywhere in it, the return precautions. A consultant's advice ("il cardiologo consiglia...") is the visit's own.
ADVICE = re.compile(r"^(?:\d+\.\s*)?(?:indicazioni\s+(?:terapeutiche|alla\s+dimissione|domiciliari)|consigli\b|"
                    r"si\s+(?:consiglia|raccomanda|prescrive|suggerisce)\b)|\bsi\s+riaffida\b|\bmeritevol\w*\s+di\b|"
                    r"\b(?:pronto\s+soccorso|p\.?s\.?)\s.*\b(?:in\s+caso\s+d|qualora)", re.IGNORECASE)
# The examination checklist prints "Si" after a region with nothing found ("SEGNI DI TVP Si", glued in some notes:
# "CUTE, SOTTOCUTE, MUCOSESi"), its box for "No reperti patologici : Si", which a translation reads as "SIGNS OF DVT Yes"
CHECKED = re.compile(r"(?m)^([A-ZÀ-Þ][A-ZÀ-Þ ,'/-]*[A-ZÀ-Þ]{2})\s*Si\s*$")
LIST_ITEM = re.compile(r"^\s*[-•*]")
# A line the layout wrapped carries on the one before: its first word did not fit there (unless it opens with a capital
# after a word in lower case, which in these notes is a typed line: "gomito sinistro" / "Nega traumi", "Colazione e
# cena" / "TOUJEO", while "virus SARS" / "COV-2" and "L'ANDAMENTO" / "CLINICO" are wrapped), the one
# before stops on a word no sentence ends on ("perforazione diverticolare del" / "sigma distale"), it opens with a word
# that carries a sentence on ("e Bisoprololo 2,5mg"), or a page break cut a sentence ("adeguamento" / "terapeutico.")
OPEN_END = re.compile(r"(?:[,\-–+]|\b(?:di|del|dello|della|dei|degli|delle|da|dal|dall|dalla|dai|in|nel|nella|nei|con|"
                      r"per|su|sul|sulla|a|al|allo|alla|ai|e|ed|o|il|lo|la|gli|le|un|uno|una|che|tra|fra|presso))$",
                      re.IGNORECASE)
CARRIED = re.compile(r"(?:e|ed|o|oppure|ma|con|per|di|del|della|dei|delle|da|dal|dall|dalla|nel|nella|che)\b")  # not
# "in", "a", "su": "in anamnesi pregressa...", "in terapia con...", "a domicilio..." open a statement as often
STOPWORDS = {"per", "con", "del", "della", "dei", "delle", "dello", "degli", "alla", "alle", "allo", "dal", "dalla",
             "nel", "nella", "che", "una", "uno", "gli", "non", "sono", "come", "anche", "piu", "poi", "tra", "fra",
             "sul", "sulla", "ore", "die"}


class Verbale(NamedTuple):
    note: str            # the anonymised Clinical Note, without diagnosis and prognosis
    dropped: list[str]   # the titles of the sections left out of it
    diagnosis: str       # "" when the Verbale has none, as the provisional print
    prognosis: str       # the Prognosi and the Esito ("Ricoverato presso Pneumologia"), "" when it has neither
    advice: list[str]    # the physician's discharge advice, one item per line, never in the note


def fold(text: str) -> str:
    """The text without accents, one character for one, so positions found in it hold in the original."""
    return "".join(unicodedata.normalize("NFD", c)[0] for c in text)


def stems(text: str) -> set[str]:
    """The words of a text that carry its content, cut to 6 letters so a typo or an ending does not hide one."""
    return {w[:6] for w in re.findall(r"[^\W\d_]{3,}", fold(text).lower()) if w not in STOPWORDS}


def is_name(text: str) -> bool:
    """A cell as the registry prints a name: capitals, no digits ("Inf." is a role, "20:05" a time)."""
    return bool(re.search(r"[^\W\d_]{2}", text)) and not re.search(r"\d", text) and text == text.upper()


def is_stamp(span: dict) -> bool:
    """Who wrote a record: the layouts print it in italics, on the line of the record's time."""
    return span["italic"] and not span["bold"]


def writer(text: str) -> str:
    """The name in a stamp, without the role or title before it ("Inf.", "Dr.", "Dott.ssa") and the colon after."""
    return re.sub(r"^(?:[^\W\d_]{1,4}\.(?:ssa|ra|na)?\s*)+", "", text.strip()).strip(" :")


def scrub(text: str, names: set[str]) -> str:
    """The text without the words of the names, whole words in any case or accent, and without the PII patterns."""
    words = {w for n in names for w in re.findall(r"[^\W\d_]+", fold(n).lower()) if len(w) > 1 and w not in PARTICLES}
    if words:
        pattern = re.compile(r"\b(?:" + "|".join(map(re.escape, words)) + r")\b", re.IGNORECASE)
        for a, b in reversed([m.span() for m in pattern.finditer(fold(text))]):
            text = text[:a] + OMISSIS + text[b:]
    text = PII.sub(OMISSIS, TITLED.sub(lambda m: m.group(1) + " " + OMISSIS, text))
    return re.sub(re.escape(OMISSIS) + r"(?:[\s,]+" + re.escape(OMISSIS) + ")+", OMISSIS, text)


def read_lines(page) -> list[list[dict]]:
    """The page's text as lines top to bottom, each a list of spans left to right."""
    spans = []
    for block in page.get_text("rawdict")["blocks"]:
        for line in block.get("lines", []):
            if abs(line["dir"][1]) > 0.01:  # rotated text is a watermark, never the note
                continue
            for s in line["spans"]:
                text = "".join(c["c"] for c in s["chars"])
                if not text.strip():
                    continue
                word = text.split()[0]
                i = text.index(word)
                spans.append({"x0": s["bbox"][0], "x1": s["bbox"][2], "y": (s["bbox"][1] + s["bbox"][3]) / 2,
                              "text": text, "bold": bool(s["flags"] & pymupdf.TEXT_FONT_BOLD),
                              "italic": bool(s["flags"] & pymupdf.TEXT_FONT_ITALIC),
                              "width": s["chars"][i + len(word) - 1]["bbox"][2] - s["chars"][i]["bbox"][0]})
    lines = []
    for s in sorted(spans, key=lambda s: s["y"]):
        if lines and s["y"] - lines[-1][0]["y"] < 3:
            lines[-1].append(s)
        else:
            lines.append([s])
    return [sorted(line, key=lambda s: s["x0"]) for line in lines]


def text_of(line: list[dict]) -> str:
    out = line[0]["text"]
    for a, b in zip(line, line[1:]):
        # the software glues a label to the next ("QUIK LOOKQuick Look"), a table row leaves a gap
        out += (" " if b["x0"] - a["x1"] > 1 or (a["text"][-1].isalpha() and b["text"][0].isalpha()) else "") + b["text"]
    return re.sub(r" {2,}", " ", out.strip())


def cells(line: list[dict]) -> list[list[dict]]:
    """The line cut where a gap separates table cells."""
    out = [[line[0]]]
    for a, b in zip(line, line[1:]):
        if b["x0"] - a["x1"] > GAP:
            out.append([b])
        else:
            out[-1].append(b)
    return out


def opens_paragraph(span: dict) -> bool:
    """A label ("Anamnesi patologica remota:", "ALLERGIE:", bold or in italics) or a bold heading ("TORACE")."""
    t = span["text"].strip()
    return bool((span["bold"] or span["italic"]) and t.endswith((":", "?")) or
                span["bold"] and t.isupper() and len(re.findall(r"[^\W\d_]", t)) >= 4)


def label_of(title: str) -> str | None:
    return next((label for prefix, label in KEPT.items() if title.lower().startswith(prefix)), None)


def fields(lines: list[list[dict]]) -> dict[str, list[list[dict]]]:
    """
    A section of labelled fields (the Dimissione) as {label: value lines}. A label is a bold line start at the
    section's left edge ("Diagnosi", "Prognosi:", "Prescrizione e Consigli"), its value the rest of that line and the
    lines below that start where the value does: not the signature further right, nor "Prognosi di 5 gg" mistaken for
    a label. The first field of a label wins.
    """
    left = min((line[0]["x0"] for line in lines if line[0]["bold"]), default=0)
    out = []
    for line in lines:
        if line[0]["bold"] and line[0]["x0"] < left + 3:
            out.append((line[0]["text"].strip(" :").lower(), [line[1:]] if line[1:] else []))
        elif out and (not out[-1][1] or line[0]["x0"] < out[-1][1][0][0]["x0"] + 3):
            out[-1][1].append(line)
    return dict(reversed(out))


def take_advice(lines: list[str], advice: list[str]) -> list[str]:
    """
    The lines of a record without the advice in them, which goes to `advice`: each sentence
    ADVICE marks, the list under one that ends in a colon or is a heading ("Accesso immediato al più vicino Pronto
    Soccorso in caso di:" and the symptoms below it), and the lower-case line carrying on one left unfinished at a page
    break ("Meritevole di approfondimento" / "specialistico.").
    """
    out, taking = [], None
    for line in lines:
        item = LIST_ITEM.sub("", line).strip()
        if taking == "items" and LIST_ITEM.match(line):  # each a piece of advice, under a heading that is none
            advice.append(item)
            continue
        if taking == "join" and LIST_ITEM.match(line) or taking == "sentence" and line[:1].islower():
            advice[-1] += " " + item  # the symptoms the return precautions list are one piece of advice
            taking = None if taking == "sentence" and re.search(r"[.:;!?]$", line) else taking
            continue
        sentences = re.split(r"(?<=[^\W\d_][.!?])\s+(?=[A-ZÀ-Þ])", line)  # not after "7." of a heading
        rest = [s for s in sentences if not ADVICE.search(s)]
        taking = None
        if line.isupper() and len(rest) < len(sentences):
            taking = "items"
        elif len(rest) < len(sentences):
            advice.append(LIST_ITEM.sub("", " ".join(s for s in sentences if ADVICE.search(s))).strip())
            if line.rstrip().endswith(":"):
                taking = "join"
            elif ADVICE.search(sentences[-1]) and not re.search(r"[.:;!?]$", line):
                taking = "sentence"
        if rest:
            out.append(" ".join(rest))
    return out


def parse_verbale(pdf: bytes) -> Verbale:
    """
    The Verbale read into its anonymised Clinical Note, the titles of the sections left out of it, its diagnosis,
    prognosis and outcome, and the physician's discharge advice. A file that is not a Verbale raises ValueError,
    with a message for the physician.
    """
    try:
        doc = pymupdf.open(stream=pdf, filetype="pdf")
    except Exception:
        raise ValueError("Il file non è un PDF leggibile.")
    names, body_lines, missing, header, bare, labelled = set(), [], [], "", [], False
    for n, page in enumerate(doc):
        lines = read_lines(page)
        texts = [text_of(line) for line in lines]
        # the first patient line is the header's: one further down (a signature block) must not move the cut
        cut = next((i for i, t in enumerate(texts) if PATIENT.search(t)), None)
        if cut is not None:
            labelled = True
            name = NEXT_FIELD.split(texts[cut][PATIENT.search(texts[cut]).end():], 1)[0]
            if re.search(r"[^\W\d_]{2}", name):
                names.add(name)
        else:
            cut = next((i for i, t in enumerate(texts) if SEX_AGE.search(t)), None)
        if cut is None:
            bare.append(str(n + 1))
            continue
        if n == 0:
            verbale = next((i for i, t in enumerate(texts) if "verbale di pronto soccorso" in t.lower()), None)
            if verbale is None:
                missing.append("VERBALE DI PRONTO SOCCORSO")
            cut = max(cut, verbale or 0)
            header = "\n".join(texts[:cut + 1])
        end = next((i for i in range(cut + 1, len(lines)) if FOOTER.match(texts[i])), len(lines))
        if end > cut + 1:
            lines[cut + 1][0]["top"] = True  # a sentence may go on from the page before
        body_lines += lines[cut + 1:end]
    if bare:  # nothing else can be judged: a page without its header is never read
        raise ValueError("Il file non sembra un verbale di Pronto Soccorso: intestazione con cognome e nome del "
                         f"paziente non trovata{f' a pagina ' + ', '.join(bare) if len(bare) < doc.page_count else ''}.")
    # section titles: the bold lines furthest left (x=20 in the provisional print, 31 in the final one)
    left = min((line[0]["x0"] for line in body_lines if line[0]["bold"]), default=0)
    sections = []
    for line in body_lines:
        if line[0]["bold"] and line[0]["x0"] < left + 3:
            sections.append((text_of(line), []))
        elif sections:
            sections[-1][1].append(line)
    missing += [r for r in REQUIRED if not any(t.lower().startswith(r.lower()) for t, _ in sections)]
    if missing:
        raise ValueError(f"Il file non sembra un verbale di Pronto Soccorso: non trovati {', '.join(missing)}.")
    if not labelled:  # with no patient line found, a name printed in some other place would go through
        raise ValueError("Nome del paziente non trovato nell'intestazione: impossibile anonimizzare il verbale.")

    def rows_of(lines):
        """Each line as (opens an entry, its cells but the times, the line), the writers' names taken."""
        rows = []
        for line in lines:
            if TIME.match(line[0]["text"]) and len(line) > 1:  # "18:29 Dr. <Surname, Name>:" set close as one cell
                line = line[1:]
                dated = True
            else:
                dated = False
            every = cells(line)
            rest = [c for c in every if not TIME.match(text_of(c))]
            dated = dated or len(rest) < len(every)  # the time before an entry opens it, and its writer sits by it
            for s in line:
                w = writer(s["text"]) if dated and is_stamp(s) else ""
                if "," in w or is_name(w):  # "Fauri, Ivo" in the final print, "SIMONE BERLANDA" in the provisional
                    names.add(w)
            rows.append((dated, rest, line))
        return rows

    unstamped = lambda dated, c: [s for s in c if not (dated and is_stamp(s))]

    def blocks_of(rows, right, title=""):
        """
        The running text of the rows as records (a diary entry, an anamnesis), each a list of paragraphs, the lines
        the layout wrapped (before `right`) joined. A row without running text closes the record.
        """
        blocks = [[]]
        for dated, spans, line in rows:
            t = text_of(spans) if spans else ""
            if not t or LOG.search(t):
                names.update(s["text"] for s in line if is_name(s["text"]))  # a Validatore cell, a signature
                blocks.append([])
                continue
            if dated:  # the time before an entry opens it
                blocks.append([])
            block = blocks[-1]
            if t.lower() == title.lower() or title.lower().startswith(t.lower() + " "):  # the record's own title
                continue
            # DeepL translates each line on its own ("memoria anterograda: non" / "ricorda tre oggetti")
            prev, end = block[-1] if block else ("", 0)
            if block and not opens_paragraph(spans[0]) and not LIST_ITEM.match(t) and (
                    OPEN_END.search(prev) or CARRIED.match(t)
                    or end + 3 + spans[0]["width"] > right and (not t[:1].isupper() or prev.endswith(":") or
                                                                re.search(r"\b[A-ZÀ-Þ][\w'’-]*$", prev))
                    or line[0].get("top") and t[:1].islower() and not re.search(r"[.:;!?]$", prev)):
                block[-1] = [f"{prev} {t}", spans[-1]["x1"]]
            else:
                block.append([t, spans[-1]["x1"]])
        return [[t for t, _ in block] for block in blocks if block]

    # in a kept section, running text sits in one column, the leftmost; a cell or a signature on its own sits right
    # of it. Each line as (opens an entry, its running text or [], the line)
    kept = []
    for title, lines in sections:
        if label_of(title):
            rows = [(dated, cs[0] if len(cs) == 1 else None, line) for dated, cs, line in rows_of(lines)]
            column = min((c[0]["x0"] for dated, c, _ in rows if c and unstamped(dated, c)), default=0)
            kept.append((label_of(title), title, [(dated, unstamped(dated, c) if c and c[0]["x0"] < column + 3 else [],
                                                   line) for dated, c, line in rows]))
    # where running text wraps: the widest line seen, which errs towards joining two lines rather than splitting one
    right = max((spans[-1]["x1"] for _, _, rows in kept for _, spans, _ in rows if spans), default=0)

    # the Dimissione: every cell of a field's value is its text (a Prognosi sits in two)
    found = {}
    for title, lines in sections:
        if title.lower().startswith("dimissione"):
            found = {label: (value, blocks_of([(dated, unstamped(dated, [s for c in cs for s in c]), line)
                                               for dated, cs, line in rows_of(value)], right))
                     for label, value in fields(lines).items()}
    flat = lambda label: [p for b in found.get(label, ([], []))[1] for p in b]
    advice = [p for label in found if label.startswith(ADVICE_FIELDS) for p in flat(label)
              if not POINTER.match(p) and re.search(r"[^\W\d_]", p)]
    for p in flat("condizioni alla dimissione"):  # "Buone, asintomatica. Da rivalutare a breve dal CSM", a list
        advice += [x for x in re.split(r"\s*(?:[,;]|\.(?=\s|$))\s*", p) if x and not POINTER.match(x)
                   and not all(STATE.fullmatch(w) for w in re.findall(r"[^\W\d_]+", x))]
    # the physician's diagnosis is bold, the coded one below it plain ("Riacutizzazione in BPCO" / "Bronchite acuta"),
    # and the software may append the code's wording to it ("frattura gomito sx, Frattura dell' estremità prossimale
    # dell' omero, chiusa"), which goes unless it is all there is
    value = found.get("diagnosi", ([], []))[0]
    diagnosis = " ".join(text_of(b) for b in ([s for s in line if s["bold"]] for line in value) if b)
    diagnosis = diagnosis or (text_of(value[0]) if value else "")
    for line in value:
        coded = text_of(line) if not any(s["bold"] for s in line) else ""
        if coded and diagnosis.endswith(coded) and diagnosis[:-len(coded)].rstrip(" ,"):
            diagnosis = diagnosis[:-len(coded)].rstrip(" ,")
    prognosis = [p for p in flat("prognosi") if not re.match(r"prognosi riservata\W*no\b", p, re.IGNORECASE)]
    prognosis = ". ".join(re.sub(r"^prognosi\s+(?:di\s+)?", "", p, flags=re.IGNORECASE).strip(" .")
                          for p in prognosis + flat("esito")[:1])  # the final print goes on below with its footer

    # advice written in the kept sections leaves the note, and joins the advice unless it repeats the Dimissione's ("Si
    # consiglia esecuzione ECG Holter e visita aritmologica" / "ECG Holter in elezione", "Visita Aritmologica"). A diary
    # line restating the Dimissione's advice leaves the note too: every word of it in the advice, as "si invia per
    # valutazione ortopedica", while the home therapy "Trimbow 1 puff 3 volte al giorno con distanziatore" is not
    # "Prosegua con Trimbow 2 spray con distanziatore" (4 of its 5)
    given = set().union(*map(stems, advice))
    restates = lambda text: len(s := stems(text)) >= 3 and s <= given
    repeats = lambda text: len(s := stems(text)) and len(s & given) >= 0.6 * len(s)
    body = {label: [] for label in KEPT.values()}
    for label, title, rows in kept:
        for block in blocks_of(rows, right, title):
            more = []
            block = take_advice(block, more)
            if label == KEPT["diario"]:
                block = [p for p in block if not restates(p)]
            advice += [p for p in more if not repeats(p)]
            if label == KEPT["procedure"]:
                block = [p for p in block if not ROUTINE_PROCEDURE.match(p)]
            if block:
                body[label].append("• " + "\n".join(block) if label == KEPT["diario"] else "\n".join(block))

    sex = re.search(r"\b(?:sesso|genere)\W*([MF])", header, re.IGNORECASE)
    sex = {"F": "Femmina", "M": "Maschio"}.get(sex[1].upper() if sex else "", "Altro")
    age = re.search(r"\bet[àa]\W*(\d+)\s*(mes[ie])?", header, re.IGNORECASE)
    age = f"{age[1]} {'mesi' if age[2] else 'anni'}" if age else ""
    note = f"{sex}, {age}\n" + "\n\n".join(f"{label}\n" + "\n".join(body[label]) for label in KEPT.values()
                                            if body[label])  # a Verbale may have no objective examination
    note = re.sub(r"(?im)^anamnesi (patologica (?:prossima|remota))", lambda m: m.group(1).capitalize(), note)
    note = re.sub(r"(?m)^([^:\n]{3,60}:)\s*\1", r"\1", note)  # "Anamnesi farmacologica: Anamnesi farmacologica:"
    note = CHECKED.sub(r"\1: nessun reperto patologico", note)
    return Verbale(scrub(note, names), [t for t, _ in sections if not label_of(t)], scrub(diagnosis, names),
                   scrub(prognosis, names), [scrub(a, names) for a in advice])


if __name__ == "__main__":
    out = scrub("Visto dal Dr. Mario Rossi e dalla dott.ssa de Luca; BUCCELLETTI, tel. +39 333 1234567, "
                "CF ZRZMLL57A46G347I; chiamato Nicolò", {"FRANCESCO BUCCELLETTI", "NICOLO'"})
    assert out == "Visto dal Dr. [omissis] e dalla dott.ssa [omissis]; [omissis], tel. [omissis], CF [omissis]; " \
                  "chiamato [omissis]", out
    advice = []
    out = take_advice(["Diuresi valida.", "7. INDICAZIONI TERAPEUTICHE", "- Controllo pressorio domiciliare.",
                       "- Statina.", "Noduli tiroidei. Meritevoli di approfondimento", "specialistico.", "- non a domicilio",
                       "Il cardiologo consiglia NAO.", "Accesso in PS in caso di:", "- cefalea;", "- vertigine."], advice)
    assert out == ["Diuresi valida.", "Noduli tiroidei.", "- non a domicilio", "Il cardiologo consiglia NAO."], out
    assert advice == ["Controllo pressorio domiciliare.", "Statina.", "Meritevoli di approfondimento specialistico.",
                      "Accesso in PS in caso di: cefalea; vertigine."], advice

    path, words = sys.argv[1], set(sys.argv[2:])
    v = parse_verbale(open(path, "rb").read())
    print(f"{v.note}\n\n[Diagnosi] {v.diagnosis}\n[Prognosi] {v.prognosis}\n[Sezioni escluse] {', '.join(v.dropped)}")
    print("[Consigli]\n" + "\n".join(f"- {a}" for a in v.advice))
    labels = [i for i in (v.note.find(label) for label in KEPT.values()) if i != -1]
    assert labels and labels == sorted(labels), labels
    assert not [a for a in v.advice if a in v.note], "advice left in the note"
    # every word more than half under a black box, and every word given, must be gone
    for page in pymupdf.open(path):
        boxes = [d["rect"] for d in page.get_drawings() if d.get("fill") == (0.0, 0.0, 0.0)]
        for *bbox, word, _, _, _ in page.get_text("words"):
            r = pymupdf.Rect(bbox)
            if any((r & b).get_area() > r.get_area() / 2 for b in boxes):
                words |= set(re.findall(r"\w{3,}", word))
    if not words:
        sys.exit("\nNo black boxes and no words given, so nothing to check the note against: read it above.")
    text = "\n".join([v.note, v.diagnosis, v.prognosis, *v.advice])
    leaks = sorted(w for w in words if re.search(rf"\b{re.escape(w)}\b", text, re.IGNORECASE))
    assert not leaks, f"leaked: {leaks}"
    print(f"\nOK: none of the {len(words)} words to hide is in the output")
