import re

# a run of digits (with commas and dashes) glued to the end of a word, bracket or stop is a reference
REF_MARKER = re.compile(r'(?<=[a-zA-Z)][.,])\d+(?:[,\-–]\d+)*')


def detect_biblio(text: str, threshold: float = 0.1) -> bool:
    """
    A page of bibliography: many lines with a Vancouver "2014;93", a DOI or a link. Reference lists in the other
    styles, and those sharing a page with body text, are left to drop_references.
    """
    lines = [l.strip() for l in text.split('\n') if l.strip()]
    if not lines:
        return False

    journal_citation = re.compile(r'\d{4}\s*;\s*\d+\s*[:(/]*') # "2021;28:" or "2014;93(" or "1993; 36:" (journal volume citations)
    ref_lines = 0
    for line in lines:
        has_citation = journal_citation.search(line) or re.search(r'\bdoi\b|http', line, re.IGNORECASE)  # not "doing", "Doizi"
        if has_citation:
            ref_lines += 1

    return ref_lines/len(lines) >= threshold


# A journal locator, the volume and pages every reference style gives a cited article
LOCATOR = re.compile(
    r'\b(?:19|20)\d\d[a-z]?\s*;\s*\d'
    r'|\b\d{1,4}\s*\([^()\n]{1,15}\)\s*:\s*[eS]?\d'
    r'|\b\d{1,4}\s*:\s*[eS]?\d{1,6}\s*[–—\-e]\s*[eS]?\d'
    r'|\b\d{1,4}\s*:\s*e\d{3,}'
    r'|(?<=[A-Za-z.)]),\s*\d{1,4}(?:\s*\([^()\n]{1,15}\))?,\s*(?:\d{1,6}\s*[–—-]\s*\d|e?\d{5,}\b)'
    r'|\b(?:19|20)\d\d\s+\d{1,4}\s+[eS]?\d{1,6}\s*[–—-]\s*\d'
    r'|\b\d{1,4}\s*\((?:\d{1,4}|Suppl[^()\n]{0,10})\),\s*(?:[eS]?\d{1,6}\s*[–—-]\s*[eS]?\d|e\d{3,})'
    r'|\b\d{1,4},\s*[eS]?\d{1,6}\s*[–—-]\s*[eS]?\d{1,6}\s*\((?:19|20)\d\d\)'
)
# three authors in a row as Vancouver sets them
AUTHORS = re.compile(r"(?:\b[A-Z][a-z'’\-]+(?: [a-z]{1,3})? [A-Z]{1,3},\s*){3,}")


def is_reference(text: str, density: float = 3.0) -> bool:
    """
    A block of a reference list: at least one journal locator per 1000/density characters, or a locator after an
    author list (an entry with twenty authors runs past that).
    """
    flat = " ".join(text.replace("\u00ad", "").split()) 
    hits = len(LOCATOR.findall(flat))
    return hits > 0 and (hits * 1000 >= density * len(flat) or bool(AUTHORS.search(flat)))


def drop_references(blocks: list[str], max_gap: int = 2) -> list[str]:
    """
    The blocks of a page that are not part of a reference list, whether the list fills the page or shares it with body
    text (the conclusion above the first references stays). Up to max_gap blocks between two reference blocks belong
    to the list too: some layouts cut an entry over several blocks and leave its locator in the last one.
    """
    ref = [is_reference(b) for b in blocks]
    marks = [i for i, r in enumerate(ref) if r]
    for i, j in zip(marks, marks[1:]):
        if j - i <= max_gap + 1:
            ref[i + 1:j] = [True] * (j - i - 1)
    return [b for b, r in zip(blocks, ref) if not r]


def detect_index(text: str, threshold: float = 0.4) -> bool:
    """
    Detects if a page is likely to be part of the index section.
    Uses a combination of keyword detection, line patterns, and leader patterns to score the likelihood of being an index page.
    """

    if not text or len(text.strip()) < 20:
        return False
 
    lines = [l.strip() for l in text.strip().splitlines() if l.strip()]
 
    score = 0.0
 
    # Keyword detection (weight: 0.30)
    # TOC/index headers
    keyword_pattern = re.compile(
        r'\b(table of contents|index|'
        r'|list of figures|list of tables)\b',
        re.IGNORECASE
    )
    header_text = " ".join(lines[:20])
    if keyword_pattern.search(header_text):
        score += 0.30
 
    # 2. Lines ending with page numbers (weight: 0.40)
    line_with_number = re.compile(
        r'[a-zA-Z]'              # at least one letter somewhere
        r'.*'                     # any content
        r'[\s.\-·…\t]{2,}'       # leader: dots, spaces, dashes, etc.
        r'\d{1,4}\s*$'           # ends with a page number
    )
    number_ending_count = sum(1 for line in lines if line_with_number.match(line))
    number_ratio = number_ending_count / len(lines)
    score += 0.40 * min(number_ratio / 0.5, 1.0)  # full score at 50%+ lines matching
 
    # 3. Dot/dash leaders (weight: 0.20)
    leader_pattern = re.compile(r'[.\-·…]{5,}')
    leader_count = sum(1 for line in lines if leader_pattern.search(line))
    leader_ratio = leader_count / len(lines)
    score += 0.2 * min(leader_ratio / 0.3, 1.0)  # full score at 30%+ lines
 
    return score >= threshold


# an acronym alone on its line, with two capitals at least: "R24" and "R-12" are the numbers of recommendations
ACRONYM_LINE = re.compile(r'^(?=(?:[^A-Z]*[A-Z]){2})[A-Z][A-Z0-9\-]{1,9}\s*$')


def detect_abbreviations(text: str, threshold: int = 12, order: float = 0.8) -> bool:
    """
    A page holding an abbreviation list, usually at the start of the guideline
    """
    acronyms = list(dict.fromkeys(l.strip() for l in text.split("\n") if ACRONYM_LINE.match(l.strip())))
    ordered = sum(a.lower() <= b.lower() for a, b in zip(acronyms, acronyms[1:]))
    return len(acronyms) >= threshold and ordered >= order * (len(acronyms) - 1)


def drop_abbreviations(blocks: list[str], page_text: str) -> list[str]:
    """
    The blocks of a page that are not part of its abbreviation list, judged on the whole page 
    """
    if not detect_abbreviations(page_text):
        return blocks
    def in_list(block):
        lines = [l.strip() for l in block.split("\n") if l.strip()]
        return sum(bool(ACRONYM_LINE.match(l)) for l in lines) * 4 >= len(lines)
    return [b for b in blocks if not in_list(b)]


def is_page_bad(text: str) -> bool:
    return detect_biblio(text) or detect_index(text)


def remove_reference_numbers(lines: list[str]) -> list[str]:

    cleaned = []
    for line in lines:
        # remove a number glued behind a period or comma without a space (a reference marker)
        line = REF_MARKER.sub('', line)
        line = line.strip()
        if line:
            cleaned.append(line)
    return cleaned


def remove_captions(sentences: list[str]) -> list[str]:
    return [s for s in sentences
        if not re.match(r'^\s*(Figure|Fig\.|Table|Supplementary|Recommendation Table)\s*+\d+', s, re.IGNORECASE)]

def remove_figure_nonsense(lines: list[str]) -> list[str]:
    """
    Remove lines that are mostly garbled/unrenderable characters
    from broken font mappings in PDF extraction.
    """
    cleaned = []
    for line in lines:

        bad_chars = sum(1 for c in line if ord(c) > 127 or not c.isprintable())
        ratio = bad_chars / max(len(line), 1)

        if ratio > 0.8 or not line.strip():
            continue

        cleaned.append(line)

    return cleaned

def remove_summaries(lines: list[str]) -> list[str]:

    cleaned = []
    for line in lines:
        line = line.strip()

        if "©" in line:
            continue
        if line.count(";") >= 3:
            continue
        # if len(line) >= 300 and line.count(",") / len(line) > 0.05:
            # continue

        cleaned.append(line)

    return cleaned


def clean_sentences(lines: list[str]) -> list[str]:
    
    lines = remove_summaries(lines)
    lines = remove_captions(lines)

    return lines


def clean_lines(lines: list[str]) -> list[str]:

    lines = remove_reference_numbers(lines)
    lines = remove_figure_nonsense(lines)

    return lines


if __name__ == "__main__":
    # one reference entry per citation style, page dashes as the publishers set them
    for entry in ["Rosenfeld BA (1998) Radial artery pressure monitoring underestimates central arterial pressure during vasopressor therapy in critically ill surgical patients. Crit Care Med 26:1646–1649",
                  "Chuang MJ, Chang WN, Chang HW, et al. Predictors and long-term outcome of seizures after bacterial brain abscess. J Neurol Neurosurg Psychiatry 2010;81:913e7.",
                  "Shahravan, A., Haghdoost, A.A. & Rad, M. (2019) Effect of different concentrations of sodium hypochlorite on outcome of primary root canal treatment. Journal of Endodontics, 45, 357–\u00ad363.",
                  "van der Zweerde, T., Bisdounis, L., Kyle, S. D., Lancee, J., & van Straten, A. (2019). Cognitive behavioral therapy for insomnia: A meta-analysis. Sleep Medicine Reviews, 48, 101208.",
                  "Hoorn EJ & Zietse R. Hyponatremia and mortality: how innocent is the bystander? Clinical Journal of the American Society of Nephrology 2011 6 951–953.",
                  "Khan BA, Perkins AJ, Khan SH et al (2024) Mobile critical care recovery program for survivors of acute respiratory failure: a randomized clinical trial. JAMA Netw Open 7:e2353158",
                  "Batlle, D.C., von Riotte, A.B., Gaviria, M., Grupp, M., 1985. Amelioration of polyuria by amiloride in patients receiving long-term lithium therapy. N. Engl. J. Med. 312 (7), 408–414.",
                  "Wu, Z. & McGoogan, J. M. Characteristics of and important lessons from the coronavirus disease 2019 (COVID-19) outbreak in China. JAMA 323, 1239–1242 (2020).",
                  "Kirkpatrick AW, Pereira BM, Sugrue M, De Keulenaer B, Bodnar Z, Acosta S, Ejike J, Tayebi S, Stiens J, Cordemans C, Van Regenmortel N, Elbers PWG, Monnet X, Wise R, Malbrain M (2023) Intra-abdominal hypertension and abdominal compartment syndrome in critically ill patients: an updated consensus of the World Society of the Abdominal Compartment Syndrome. Anaesthesiol Intensive Ther 55:1–14"]:
        assert is_reference(entry), entry
    # in-text citations, doses and ratios, and one source cited inline in a paragraph are running text
    for text in ["In a study by Heim et al. (2015), syndesmotic screws placed tricortically showed loosening in 91 % of the patients [12, 27, 29–31].",
                 "Early administration of 70 mg/kg (5 g) hydroxocobalamin is advocated for all intermediate or severely poisoned patients; adrenaline 1:1000 is given IM.",
                 "Of the 120 patients, 45 (12%), 23 (8%) and 12 (4%) developed AKI within 48–72 h; median age 67 (58–74) years.",
                 "Extubation is always an elective manoeuvre, and the plan should be developed and discussed beforehand with the whole team. Based on the extubation strategy published by Weatherall and colleagues (Pediatr Anesth 2022; 32: 1234), a structured approach including an assessment of readiness, a clear plan for the procedure itself and a backup plan for reintubation is recommended for every child with a difficult airway."]:
        assert not is_reference(text), text
    # an entry cut over blocks, its locator in the next one, belongs to the list; the body text around it stays
    body = "The management of OME depends on the clinical history, otoscopy findings, and hearing testing."
    ref = "Rosenfeld RM, Shin JJ, Schwartz SR et al (2016) Clinical practice guideline: otitis media with effusion. Otolaryngol Head Neck Surg 154:S1–S41"
    cut = "12. Vlastarakos PV, Nikolopoulos TP, Korres S, Tavoulari E (2007) Grommets in otitis media with effusion:"
    assert drop_references([body, ref, cut, ref, body]) == [body, body]
    assert drop_references([ref, body, body, body, ref]) == [body, body, body]
    assert not detect_biblio("\n".join(["doing well on treatment", "Doizi S, Traxer O"] + ["plain text"] * 8))
    # an abbreviation list names each acronym once, in alphabetical order; running text extracted one word per line
    # repeats one ("OME"), a table of genes lists them in any order
    glossary = "\n".join(f"{a}\n{e}" for a, e in [("AAD", "Antiarrhythmic drugs"), ("ACE", "Angiotensin-converting enzyme"), ("AF", "Atrial fibrillation"),
        ("BP", "Blood pressure"), ("CI", "Confidence interval"), ("CKD", "Chronic kidney disease"), ("ECG", "Electrocardiogram"), ("HF", "Heart failure"),
        ("HR", "Hazard ratio"), ("LV", "Left ventricle"), ("OAC", "Oral anticoagulant"), ("RCT", "Randomized controlled trial"), ("VKA", "Vitamin K antagonist")])
    assert detect_abbreviations(glossary)
    assert not detect_abbreviations("\n".join(("The management of OME depends on the clinical history and hearing testing in OME").split() * 12))
    assert not detect_abbreviations("\n".join(f"{g}\nAD\nArrest of neutrophil maturation" for g in ["ELANE", "HAX1", "G6PC3", "CSF3R", "WAS", "CXCR4", "SBDS", "VPS45", "JAGN1", "GFI1", "TAZ", "CLPB", "SRP54"]))
    # beside the list, the recommendations on the same page stay
    rec = "8 ESGE recommends, for patients with an ulcer with a nonbleeding visible vessel, endoscopic hemostasis\nwith clips or thermal therapy."
    assert drop_abbreviations([glossary, rec], glossary + "\n" + rec) == [rec]
    assert drop_abbreviations([rec], rec) == [rec]
    # numbered recommendations ("R17" ... "R30") are in order but no acronyms
    numbered = [f"In patients on high doses of anticoagulants no routine testing is suggested.\n2C\nR{n}" for n in range(17, 31)]
    assert drop_abbreviations(numbered, "\n".join(numbered)) == numbered
    print("cleaning self-check passed")
