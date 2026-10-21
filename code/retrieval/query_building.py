import math
import re
from collections import Counter
from difflib import get_close_matches
from transformers import pipeline
from sklearn.feature_extraction.text import TfidfVectorizer

from config import LLM_QUERIES, QUERY_MODEL, API, MODEL_NAME, OPENROUTER_KEY, FINAL_N
from pipeline.user_input import extract_patient_fields
from generation.prompting import is_hospitalised


# NER categories returned by extract_entities by default
KEPT_TYPES = {"DISEASE_DISORDER", "MEDICATION", "DIAGNOSTIC_PROCEDURE", "SIGN_SYMPTOM"}

# Query building without an LLM.
FACET = "Recommendations for {item} in patients with {dx}."

SECTIONS_HOME = [
    "Recommended treatment for patients with {dx}.",
    "Recommendations for drug therapy and dosage in patients with {dx}.",
    "Recommendations for follow-up of patients with {dx}.",
    "Recommendations for lifestyle and risk factor management in patients with {dx}.",
    "Recommendations for complications and warning signs in patients with {dx}.",
]
SECTIONS_ADMITTED = [
    "Recommended treatment for patients with {dx}.",
    "Recommendations for drug therapy and dosage in patients with {dx}.",
    "Recommendations for monitoring of patients with {dx}.",
    "Recommendations for diagnosis and assessment of {dx}.",
    "Recommendations for complications in patients with {dx}.",
]
MAX_FINDINGS = 3        # findings are the least specific items: the few the diagnosis's guideline talks about most
MIN_ON_TOPIC = 0.3      # ...and only if at least this share of their top chunks comes from that guideline
MAX_ITEM_WORDS = 12     # a longer clause is narrative reasoning, not one item
MAX_LISTED = 3          # pieces of a listed history, those the diagnosis's guideline names most (a drug list runs
                        # to a dozen)
TOPIC_SHARE = 0.05      # a word in more of the corpus's chunks than this names no topic ("type", "surgery")
MIN_NAME_SIMILARITY = 0.33
# what a piece of a listed history names: a condition (NER tags a past one HISTORY), a drug or a procedure
PIECE_TYPES = {"DISEASE_DISORDER", "SIGN_SYMPTOM", "HISTORY", "MEDICATION", "THERAPEUTIC_PROCEDURE"}
# a past history that denies it or says it is over ("No known dyslipidaemia", "Previous appendectomy"), or has nothing
# to tell ("Remote pathology: None", "NDR", the Italian "Muta" left as it is)
NEGATED_START = re.compile(r"(?:no|not|without|denies|never|non|former|previous|none|nil|nothing|ndr|unremarkable|"
                           r"negative|silent|muta)\b", re.IGNORECASE)
# a history or diary line about a relative ("mother, aged 82, hypertensive and on treatment" under "Family history:")
RELATIVE = re.compile(r"^(?:family history|familiarity|mother|father|brother|sister|son|daughter|parents?|siblings?|"
                      r"grand\w+|relatives?)\b", re.IGNORECASE)
# a home drug the history says is off ("Tirzepatide ... currently suspended due to gastrointestinal intolerance"), a
# diary drug that was not given ("Olmesartan 20 mg not administered (prescribed in error)")
STOPPED = re.compile(r"\b(?:suspended|discontinued|stopped|withdrawn|no longer|not tolerated|intoleran\w*)\b", re.IGNORECASE)
NOT_GIVEN = re.compile(r"\bnot (?:administered|given|taken)\b|\bin error\b", re.IGNORECASE)
# what the emergency department applies that the patient goes home with, whatever NER makes of it
ER_PROCEDURE = re.compile(r"\b(?:sutur\w*|stitch\w*|dressing|plaster|bandag\w*|splint\w*|cast|immobili[sz]\w*|"
                          r"drainage|catheteri[sz]\w*|local an(?:a)?esthe\w*)\b", re.IGNORECASE)
# a guideline for children, which an adult's diagnosis may land in ("Urinary tract infection" -> PediatricUrinaryTract...)
PAEDIATRIC = re.compile(r"p(?:a)?ediatric|neonat|child|infant|newborn", re.IGNORECASE)

HEDGES = re.compile(r"\b(?:suspected(?: of)?|probabl[ey]|possibl[ey]|presum(?:ed|ably)|likely(?: to be)?|"
                    r"appears? to be)\b(?:\s+(?:an?|the)\b)?", re.IGNORECASE)  # "Likely to be a stomach bug"
SIDES = re.compile(r"\b(?:left|right)\b", re.IGNORECASE)
# NER types that say nothing clinical on their own
NON_CLINICAL = {"AGE", "SEX", "DATE", "DURATION", "TIME", "FREQUENCY", "COREFERENCE", "NONBIOLOGICAL_LOCATION",
                "OCCUPATION", "SUBJECT", "QUALITATIVE_CONCEPT"}
# Diary clauses that are logistics or routine rather than a result, a therapy or a decision
TEST_ORDER = re.compile(r"\b(?:performed|measured|carried out|taken|requested|established|referred|scheduled|"
                        r"contacted|examined|examination|assessed|monitoring|consultation|transfer\w*)\b", re.IGNORECASE)
VITALS = re.compile(r"^(?:blood pressure|arterial pressure|bp|pao|heart rate|hr|pulse|spo2|(?:oxygen |o2 )?saturation|"
                    r"fio2|body temperature|temperature|weight|respiratory rate|acvpu|avpu|gcs|nrs|news2?)\b", re.IGNORECASE)
NORMAL = re.compile(r"\b(?:normal(?:ly)?|normo\w+|negative|preserved|stable|unremarkable|physiological|no abnormalities|"
                    r"not visualisable|ruling out)\b", re.IGNORECASE)
# a diary's patient doing well: no result to search for (a history's "heel ulcer improved but very deep" is current)
WELL = re.compile(r"\b(?:asymptomatic|calm|comfortable|intact|eupn\w*|afebrile|apyretic|alert|oriented|improv\w*|"
                  r"subsided|resolved|regress\w*|(?:feel\w*|generally|reports? feeling) well|"
                  r"well[- ](?:perfused|controlled|hydrated))\b", re.IGNORECASE)
# the ward's routine and the patient's life, in a nurse's diary ("goes to the toilet", "her daughter lives in ...")
ROUTINE = re.compile(r"\b(?:toilet|bedpan|urine output|diuresis|bowel|meals?|ate|eats|eating|breakfast|lunch|dinner|"
                     r"slept|sleeps?|sleeping|asleep|telephone|phone|moves? independently|able to move|lives with|"
                     r"lives alone|daughter|son|wife|husband|relatives?|family members?|independently|"
                     r"home (?:treatment|therapy|medications?))\b", re.IGNORECASE)
# (an administration is one when it names its drug: the nurse's "the following is administered:", "the prescribed
# treatment is being administered" name none)
DECISION = re.compile(r"\b(?:need for|indication for|decided to|initiated|started|applied)\b", re.IGNORECASE)
FLUIDS = re.compile(r"\b(?:(?:in|of|with)\s+)?(?:normal saline|saline solution|saline|rehydration solution)\b",
                    re.IGNORECASE)
# A value with its unit and the preposition tying it to the next word ("1 g of", "30 mg in"), not a number naming a
# category ("type 2 diabetes", "grade 3 sprain"), nor the drug a count goes with ("Administered 1 amlodipine 5 mg tablet")
UNIT = (r"(?:mg|mcg|µg|μg|ug|g|kg|ml|dl|l|lt|litres?|liters?|cc|ui|iu|u|units?|mmhg|bpm|beats|breaths|meq|mmol|°c|c|cm|mm|"
        r"m|h|hrs?|hours?|min|minutes?|days?|weeks?|months?|years?|gg|aa|times|x|cp|cpr|cps|tablets?|capsules?|caps|puffs?|"
        r"drops?|gtt|fl|vials?|ampoules?|sachets?|bags?|points?|am|pm)")
MEASURE = re.compile(r"(?<![\w-])(?<!type )(?<!grade )(?<!stage )(?<!class )[<>≤≥~+]?\d[\d.,]*(?:[/x]\d[\d.,]*)*\s*"
                     r"(?:(?:" + UNIT + r"|[a-zA-Zµμ]+(?=/))(?:/[a-zA-Zµμ\d]+)?\b|%)?(?:\s+(?:of|in|to)\b)?", re.IGNORECASE)
# Narrative around the item: a passive verb of the ward routine, or a lead-in that names who did what
FILLER = re.compile(r"\b(?:is|was|were|are|has been|have been)\s+(?:calculated|performed|administered|initiated|"
                    r"applied|given|established|started|carried out|reviewed|observed|detected|visible|measured)\b|"
                    r"^(?:and|but|with|therefore|thus|then|given\b[^,]*,?|it (?:is|was) decided to|"
                    r"the (?:patient|family) (?:is|was|are|were) informed of the diagnosis(?:,| and)?|"
                    r"(?:the patient |he |she )?reports|currently(?: taking)?|who (?:advised|recommended|suggested))\s*|"
                    r"\s*,?\s*\b(?:as per (?:the )?(?:medical )?(?:instructions?|orders?|prescription)|as prescribed)\b",
                    re.IGNORECASE)


def load_ner_model(model_name = "Clinical-AI-Apollo/Medical-NER", device="cuda"):
    ner_model = pipeline("ner", model=model_name, aggregation_strategy="simple", device=device)
    return ner_model


NEGATION_PATTERNS = re.compile(
    r'\b('
    r'no|not|none|neither|never|nor|'
    r'without|absent|absence of|'
    r'denies|denied|deny|negative|ruling out|'
    r'rule[sd]? out|exclude[sd]?'
    r')\b',
    re.IGNORECASE
)
# Resolved or historical: a former smoker is not a smoker
HISTORICAL = re.compile(r"\b(?:former|ex|previous(?:ly)?|stopped|quit|ceased|no longer|eradicated|underwent|undergone|"
                        r"operated)\b", re.IGNORECASE)
SCOPE_RESET = re.compile(r',\s*(?:but|and|reports?|laments?|complains?|refers?|presents?|notes?|has|shows?|with)\b', re.IGNORECASE)
# A remote date says an event is over 
MONTH = (r"(?-i:January|February|March|April|May|June|July|August|September|October|November|December|Jan|Feb|Mar|Apr|"
         r"Jun|Jul|Aug|Sept?|Oct|Nov|Dec)")  # capitalised: "may" is a verb
DATE = (r"(?:\b\d{1,2}[/.-]\d{1,2}[/.-](?:(?:19|20)\d{2}|\d{2})\b|\b(?:\d{1,2}(?:st|nd|rd|th)?\s+)?" + MONTH +
        r"\s+(?:19|20)\d{2}\b|\b(?:19|20)\d{2}\b(?!\s*(?:[a-zµ]{0,4}/|mg|mcg|µg|g|ml|l|ui|iu|u|"
        r"units?|mmhg|mmol|meq|bpm|mm|cm|kg|cells)\b|\s*%)|\b\d+\s+(?:years?|yrs?)\s+ago\b)")
DATED = re.compile(r"(?<!since )(?<!from )" + DATE, re.IGNORECASE)

DATE_HEADING = re.compile(r"^\W*(?:(?:on|in)\s+)?" + DATED.pattern + r"|" + DATED.pattern + r"\s*:", re.IGNORECASE)
DATE_AFTER = re.compile(r"\s*(?:\(|,?\s*(?:in|on)\s+|-\s*)?" + DATE, re.IGNORECASE)  # "hernia repair (2013)"
ONSET = re.compile(r"\bdiagnos", re.IGNORECASE)


def dated_past(text: str, start: int, end: int) -> bool:
    """
    Whether a remote date says the entity at text[start:end] is over: one before it in its comma piece, one right after
    it ("umbilical hernia repair (2013)", "acute diverticulitis in 2009", "polyp removed 10 years ago"), or a date
    heading its sentence, which covers the list after it like a denial does, but not a parenthesis opened after it
    ("2006: road polytrauma with humerus fracture, left ankle (residual paresis of the left arm)").
    """
    sentence = re.split(r"[.;](?:\s|$)|\n", text[:start])[-1]
    piece = re.split(r"[,:]", sentence)[-1]
    if DATED.search(piece) and not ONSET.search(piece):
        return True
    headings = list(DATE_HEADING.finditer(sentence))
    if headings:
        after = sentence[headings[-1].end():]
        if after.count("(") <= after.count(")") and not ONSET.search(after):
            return True
    return bool(DATE_AFTER.match(text, end))
def is_negated(text: str, entity_start: int, entity_end: int, window: int = 80) -> bool:
    """
    Detects if an entity is negated by looking for negation cues in a window of text before and after the entity.
    """
    raw_start = max(0, entity_start - window)
    preceding_raw = text[raw_start:entity_start]

    # Trim to the last clause boundary. A comma counts only when a re-asserting verb follows it: "denies chest pain,
    # laments migraines" reopens the scope, "denies chest pain, syncope or presyncope" is one negated list.
    last_boundary = max(preceding_raw.rfind('.'), preceding_raw.rfind(':'), preceding_raw.rfind(';'))
    for m in SCOPE_RESET.finditer(preceding_raw):
        last_boundary = max(last_boundary, m.start())
    # an exception leaves the denial ("not currently on any treatment, apart from valsartan")
    for m in re.finditer(r"\b(?:apart from|except(?: for)?|other than|besides)\b", preceding_raw, re.IGNORECASE):
        last_boundary = max(last_boundary, m.end() - 1)
    if last_boundary != -1:
        preceding = preceding_raw[last_boundary + 1:]
    else:
        preceding = preceding_raw

    if NEGATION_PATTERNS.search(preceding) or HISTORICAL.search(preceding.rsplit(",", 1)[-1]) or \
            dated_past(text, entity_start, entity_end):
        return True

    # Look-ahead: "negative(s)" after the entity
    post_start = entity_end
    post_end = min(len(text), entity_end + 15)
    following_raw = text[post_start:post_end]

    # Trim to the next clause boundary (period or comma)
    first_boundary = len(following_raw)
    for sep in ['.', ',', ':', ';']:
        pos = following_raw.find(sep)
        if pos != -1:
            first_boundary = min(first_boundary, pos)
    following = following_raw[:first_boundary]

    if re.search(r'\bnegatives?\b', following, re.IGNORECASE):
        # print(f"Negated {text[entity_start:entity_end]} because of: {following} (Following)")
        return True

    return False


def extract_entities(text: str, ner_model, filter_negated = True, types = KEPT_TYPES) -> dict[str, list[str]]:
    """
    Run NER on text and return entities grouped by type.
    Only keeps entity types listed in types (KEPT_TYPES by default). Line by line: the model reads 512 tokens, and
    on a long note the diary's drugs never reached it.
    """
    lines = list(re.finditer(r"[^\n]+", text))
    results = [dict(e, start=e["start"] + m.start(), end=e["end"] + m.start())
               for m, found in zip(lines, ner_model([m.group() for m in lines]) if lines else []) for e in found]
 
    entity_group = set([ent["entity_group"] for ent in results if ent["entity_group"] in types])
    entities = {}
 
    for entity in list(entity_group):
        seen = set()
        deduped = []
        for ent in results:
            if ent["entity_group"] != entity:
                continue
 
            # Skip negated entities
            if filter_negated and is_negated(text, ent["start"], ent["end"]):
                continue
            
            # Eliminate duplicates
            word = ent["word"].strip()
            word_key = word.lower()
            if word_key not in seen:
                deduped.append(word)
                seen.add(word_key)
        entities[entity] = deduped
    return entities


# A line written in capitals (three words or more, most letters capital), which the NER cuts into fragments
# ("PRESENTS WITH BILIARY VOMITING" -> 'MIT', "DIAGNOSTIC" -> 'NOSTIC', OLMESARTAN not a drug): sentence case, but for
# a short token that is not a common word (COPD, CPAP, IV, TIA)
CAPS_LINE = re.compile(r"^(?=(?:[^a-z\n]*\b[A-Z]{2,}\b){3})[^a-z\n]*(?:[a-z][^a-z\n]*){0,3}$", re.MULTILINE)
SHORT_WORDS = {"A", "AN", "THE", "AND", "OR", "OF", "IN", "ON", "AT", "TO", "FOR", "WITH", "BY", "IS", "ARE", "WAS", "WERE",
               "BE", "BEEN", "IF", "NO", "NOT", "AS", "PER", "FROM", "INTO", "HAS", "HAD", "HAVE", "ITS", "HIS", "HER",
               "BUT", "ALL", "ANY", "DAY", "DAYS", "THEN", "THAN", "THIS", "THAT", "ONE", "TWO", "UP", "OUT", "OFF", "ONLY"}


def normalize(text: str) -> str:
    SUBSCRIPT_MAP = str.maketrans('₀₁₂₃₄₅₆₇₈₉⁰¹²³⁴⁵⁶⁷⁸⁹', '01234567890123456789')
    return CAPS_LINE.sub(lambda m: re.sub(r"\b[A-Z][A-Z'’]*\b", lambda w: w.group().lower() if len(w.group()) > 4 or
                                          w.group() in SHORT_WORDS else w.group(), m.group()), text.translate(SUBSCRIPT_MAP))

# A form line of a real Verbale, which says nothing to search for: a label with nothing after it ("ALLERGIES:",
# "TREATMENT:") or a checklist answer ("Drug allergies: no", "Paracetamol no", "Allergy status checked? Yes")
FORM_LINE = re.compile(r"(?:[:?]|\b(?:yes|no))\s*\.?$", re.IGNORECASE)

def split_sentences(text: str) -> list[str]:
    """
    Sentences on period or newline (each line of a Verbale stands alone, bullet or not), or on a list dash typed after
    a period or between two items ("Hypothyroidism.- Hiatal hernia", "1 tablet in the morning - Furosemide"), form
    lines left out.
    """
    return [s for s in (s.strip().lstrip('-–• ') for s in
                         re.split(r'(?<=[.;])\s*(?=[-–•]\s*[^\W\d_])|(?<!\d)\s[-–]\s(?=[^\W\d_])|(?<=[.])\s+|\n', text))
            if s and not FORM_LINE.search(s)]


def parse_diagnosis(diagnosis: str, ner_model) -> tuple[list[str], list[str]]:
    """
    The conditions on the diagnosis line and the qualifiers that come with them. Parentheticals, hedges
    (suspected, likely to be a) and sides (left) go: no guideline says which leg. A part after a dash, a semicolon or
    a sentence end ("Acute bronchitis. Cough.") is a second condition when NER finds a disorder in it
    (erysipelas/cellulitis — infective endocarditis), a qualifier otherwise (Kawasaki disease — classic form); what
    follows "with" is a qualifier too (atrial fibrillation with rapid ventricular response). Qualifiers become facets
    of their own.
    """
    conditions, qualifiers = [], []
    for i, part in enumerate(re.split(r"\s+[—–-]\s+|;|(?<=[^\W\d_]{2})\.\s+", diagnosis)):  # "Acute bronchitis. Cough."
        part = SIDES.sub(" ", HEDGES.sub(" ", re.sub(r"\([^)]*\)", " ", part)))
        part, _, extra = re.sub(r"\s+", " ", part).strip(" .,:-").partition(" with ")
        if not part:
            continue
        if i and not any(e["entity_group"] == "DISEASE_DISORDER" for e in ner_model(part)):
            qualifiers.append(part)
        else:
            conditions.append(part)
        if extra:
            qualifiers.append(extra)
    return conditions, qualifiers


def head(condition: str, ner_model) -> str:
    """
    The condition without the place or population after "of"/"in" when a multi-word name stands on its own
    (deep vein thrombosis | of the gastrocnemius veins), the whole of it otherwise: a one-word name (fracture of the
    malleolus), a name that is an episode of the disease after it (acute exacerbation of chronic bronchitis) or a
    pair of lesions of the site after it (occlusion and stenosis of the carotid artery).
    Facets name the head: the cross-encoder marks a passage down for every query element it lacks, and the one on
    oral contraceptives and DVT scored 0.54 with the gastrocnemius veins in the query, 0.71 without.
    """
    m = re.search(r"\s+(?:of|in)\s+", condition)
    name, rest = (condition[:m.start()], condition[m.end():]) if m else (condition, "")
    if len(name.split()) < 2 or not rest or " and " in name or \
            any(e["entity_group"] == "DISEASE_DISORDER" for e in ner_model(rest)):
        return condition
    return name


def clauses(text: str) -> list[str]:
    """
    A note section cut at colons, semicolons, dashes and commas, one finding or action per piece. Commas and
    colons inside parentheses are no cut (a lab range belongs to its value), nor is a colon before a number
    (the CHA2DS2-VASc score is calculated: 3 points). A form's denial becomes the clause's own, since the cut would
    part it from its item ("Respiratory distress: Absent" -> "no Respiratory distress").
    """
    out = []
    for s in split_sentences(text):
        s = re.sub(r"\(([^)]*)\)", lambda m: "(" + re.sub(r"[,:]", " ", m.group(1)) + ")", s)
        s = re.sub(r"(^|[,;]\s*)([^,;:]+?)\s*:\s*(?:absent|none|no|negative)\b", r"\1no \2", s, flags=re.IGNORECASE)
        s = re.sub(r":\s+(?=\d)", " ", s)
        out += [c.strip() for c in re.split(r"[:;]\s*|\s+[—–]\s*|,\s+", s)
                if len(c.split()) >= 2 or ER_PROCEDURE.search(c)]
    return out


def clean_item(clause: str) -> str:
    """
    A clause as a query item. A parenthesis stays when it interprets ("(slightly elevated)", "(heroin)") and goes
    when it measures; values, infusion fluids, sides, hedges and the narrative around the item go too
    ("5 mg metoprolol tartrate is administered intravenously" -> "metoprolol tartrate intravenously").
    """
    s = re.sub(r"\(([^)]*)\)", lambda m: " " if re.search(r"\d", m.group(1)) else f" {m.group(1)} ", clause)
    s = MEASURE.sub(" ", FLUIDS.sub(" ", s))
    for _ in range(2):  # a lead-in can hide another ("Therefore, it is decided to")
        s = FILLER.sub(" ", s.strip()).strip(" ,")
    s = re.sub(r"\b(in|of)\s+(?:in|of)\b", r"\1", SIDES.sub(" ", HEDGES.sub(" ", s)))
    s = re.sub(r"\s+([,.])", r"\1", re.sub(r"\s+", " ", s)).strip(" .,;:-—+\"“”'‘’/")
    s = re.sub(r"\s+(?:in|of|with|to|for)$", "", s)
    first = s.split()[0] if s else ""
    # lower-case to sit inside the frame, unless it is an acronym or a code (MRSA, CHA2DS2-VASc)
    return s if len(first) > 1 and (first.isupper() or any(ch.isdigit() for ch in first)) else s[:1].lower() + s[1:]


def pieces(text: str, found: list[dict]) -> list[str]:
    """
    The items of a sentence that lists them ("medical history of thyroiditis, gouty arthropathy, arterial
    hypertension", "on therapy with enoxaparin, amiodarone"): each part between commas, colons or semicolons that
    names a condition, a drug or a procedure the sentence does not negate, from the first word naming one to the
    last, so the lead-in goes ("on therapy with"). Whole words of the note, which NER cuts ("thyroid" + "itis").
    """
    out = []
    for m in re.finditer(r"[^,:;]+", text):
        spans = [(e["start"], min(e["end"], m.end())) for e in found if e["entity_group"] in PIECE_TYPES
                 and m.start() <= e["start"] < m.end() and not is_negated(text, e["start"], e["end"])]
        if spans:
            a, b = min(a for a, _ in spans), max(b for _, b in spans)
            while text[a].isspace():
                a += 1
            while a > m.start() and text[a - 1].isalnum():
                a -= 1
            while b < m.end() and text[b].isalnum():
                b += 1
            out.append(text[a:b])
    return out


def note_items(fields: dict, known: set[str], ner_model) -> tuple[list[str], list[str], list[str]]:
    """
    What the note says about this patient beyond the diagnosis, as (items, findings, listed):
      - past history: a sentence naming one thing (a comorbidity, a home drug, a habit) is an item, whole, since
        its negation spans commas ("Former smoker, approximately 1 packet per day"). A sentence listing several,
        or too long to be one item, gives its pieces as listed instead ("Drug history: on therapy with enoxaparin,
        amiodarone, furosemide" -> enoxaparin, amiodarone, furosemide), and so does every sentence of an anamnesis
        without the Remote pathology label, where the history and the presenting complaint mix;
      - diary: a clause stating a therapy or a decision is an item, one stating an abnormal result a finding.
        Test orders, vitals, normal or negated results and scores drop: a score serves the diagnosis, or leads to
        a decision the diary states on its own ("Indication for oral anticoagulant therapy").
    An item made only of `known` words (the diagnosis restated), longer than MAX_ITEM_WORDS, or a finding left
    with nothing but a test name once its value is gone ("Sodium 139 mEq/L") drops too.
    """
    history = [s for s in split_sentences(fields.get("remote_medical_history", ""))
               if not RELATIVE.match(s) and not STOPPED.search(s)]
    # a drug history is a list, one drug per line or not: its drugs are listed pieces, which the diagnosis's
    # guideline must name (a home therapy runs to fifteen drugs)
    drugs = [s for s in split_sentences(fields.get("drug_history", "")) if not STOPPED.search(s)]
    sentences = history + [s for s in split_sentences(fields.get("anamnesis", "")) if not RELATIVE.match(s)] + drugs
    stay = "\n".join(fields.get(k, "") for k in ("clinical_diary", "procedures", "orders"))
    diary = [c for c in clauses(stay)
             if not re.match(r"(?:no|not|without|denies)\b", c, re.I) and not VITALS.match(c) and not RELATIVE.match(c)
             and not re.search(r"\bscore\b", c, re.I) and not NOT_GIVEN.search(c) and not ROUTINE.search(c)]
    items, findings, listed = [], [], []
    for i, (text, found) in enumerate(zip(sentences + diary, ner_model(sentences + diary) if sentences + diary else [])):
        kept = [e for e in found if e["entity_group"] not in NON_CLINICAL and not is_negated(text, e["start"], e["end"])]
        types = {e["entity_group"] for e in kept}
        if i < len(sentences):
            listing = pieces(text, found)
            if i >= len(history) or len(listing) > 1 or len(text.split()) > MAX_ITEM_WORDS:
                listed += [x for x in listing if not NEGATED_START.match(x)]  # NER tags "previous PSVT" whole
            elif kept and not NEGATED_START.match(text) and not NORMAL.search(text):
                items.append(text)
        elif ER_PROCEDURE.search(text) and not NOT_GIVEN.search(text):
            items.append(text)
        elif TEST_ORDER.search(text):
            continue
        elif any(e["entity_group"] == "MEDICATION" and not FLUIDS.search(e["word"]) for e in kept) or (
                not FLUIDS.search(text) and not NORMAL.search(text) and not WELL.search(text) and (
                DECISION.search(text) or ("THERAPEUTIC_PROCEDURE" in types and
                                          not re.search(r"\bvenous (?:access|line|cannula|catheter)|\bperipheral "
                                                        r"(?:line|cannula)|\baccess\b|\b(?:AVP|CVP)\b", text, re.I)))):
            items.append(text)
        elif types & {"SIGN_SYMPTOM", "DISEASE_DISORDER", "LAB_VALUE"} and not NORMAL.search(text) and not WELL.search(text):
            findings.append(text)

    def keep(texts):
        out = {}
        for x in map(clean_item, texts):
            if x and len(x.split()) <= MAX_ITEM_WORDS and not set(re.findall(r"\w+", x.lower())) <= known:
                out.setdefault(x.lower(), x)
        return list(out.values())

    items, findings, listed = keep(items), keep(findings), keep(listed)
    # one item per drug: the diary and the orders name the stay's drugs over and over ("metoprolol bolus", "metoprolol
    # as a bolus", "metoprolol intravenously"), and each copy took a floor
    if items:
        by_drug = {}
        for x, found in zip(items, ner_model(items)):
            drug = next((e["word"].split()[0].strip(" .,").lower() for e in found
                         if e["entity_group"] == "MEDICATION" and e["word"].split()), x.lower())
            by_drug.setdefault(drug, x)
        items = list(by_drug.values())
    if findings:
        findings = [x for x, found in zip(findings, ner_model(findings))
                    if any(e["entity_group"] in {"SIGN_SYMPTOM", "DISEASE_DISORDER", "LAB_VALUE"} for e in found)]
    return items, findings, listed


def primary_guidelines(conditions: list[str], embedder, index, metadata, k: int = 20, share: float = 0.2,
                       child: bool = True, min_similarity: float = MIN_NAME_SIMILARITY) -> set[str]:
    """
    The guideline(s) of the diagnosis: those holding at least `share` of the top-k chunks of a condition's name
    alone. The bare name finds its guideline where a framed query drifts; a name the corpus words otherwise
    ("basal pneumonia", "peroneal malleolus") splits across guidelines instead, and may name none. A guideline for
    children is not an adult's (urinary tract infection at 26, second-degree AV block at 61), and its chunks do not count.
    A diagnosis the corpus has no guideline for still gathers its nearest chunks in some guideline, by a word they
    share (a wrist contusion in haemodialysis access, at the wrist): a guideline also needs a chunk at min_similarity
    or closer to the name, or the queries would be searched in it alone.
    """
    sims, ids = index.search(embedder.encode_query(conditions, convert_to_numpy=True, normalize_embeddings=True), k)
    docs = set()
    for srow, row in zip(sims, ids):
        counts, best = Counter(), {}
        for s, i in zip(srow, row):
            if i != -1 and (child or not PAEDIATRIC.search(metadata[i]["doc_id"])):
                counts[metadata[i]["doc_id"]] += 1
                best.setdefault(metadata[i]["doc_id"], s)  # the ranking is by similarity: the first is the closest
        docs |= {d for d, n in counts.items() if n >= share * k and best[d] >= min_similarity}
    return docs


def on_topic(queries: list[str], guidelines: set[str], embedder, index, metadata, k: int = 30) -> list[float]:
    """
    Share of each query's top-k chunks that come from the given guidelines, i.e. how much the diagnosis's
    guideline talks about it: troponin for reflux 0.07, sinus tachycardia for Kawasaki 0.0, a tricuspid
    vegetation for endocarditis 0.73, free fluid in the pouch of Douglas for ectopic pregnancy 0.93.
    """
    _, ids = index.search(embedder.encode_query(queries, convert_to_numpy=True, normalize_embeddings=True), k)
    return [sum(metadata[i]["doc_id"] in guidelines for i in row) / k for row in ids]


def patient_qualifiers(fields: dict) -> str:
    """Age bracket and gender appended to broad queries, e.g. "elderly male"."""
    age = fields.get("age")
    bracket = ""
    if age is not None:
        if age <= 1 or fields.get("neonatal"):
            bracket = "neonatal"
        elif age <= 14:
            bracket = "pediatric"
        elif age >= 65:
            bracket = "elderly"
    return " ".join(filter(None, [bracket, fields.get("gender", "").lower()]))


def fix_oov(text: str, vectorizer: TfidfVectorizer, tokenizer) -> str:
    """
    Query-side spelling repair for translation leftovers and typos: a word of 5+ letters missing from
    the corpus vocabulary becomes its closest vocabulary word with the same initial when the two are
    near-identical (metfromin -> metformin, ventricolar -> ventricular). A word the NER tokenizer knows
    whole is real English the corpus merely never uses (election, proximate) and is left alone; the
    same initial and a high cutoff keep unrelated neighbours out (mesocardium -> myocardium). Query
    building only: the note the LLM and the dose check see stays untouched, since a wrong repair on a
    drug name there could change a dose.
    """
    vocab = vectorizer.vocabulary_
    english = tokenizer.get_vocab()
    by_initial = {}
    for w in vocab:
        by_initial.setdefault(w[0], []).append(w)

    def repair(m):
        w = m.group().lower()
        if w in vocab or "▁" + w in english:
            return m.group()
        close = get_close_matches(w, by_initial.get(w[0], []), n=1, cutoff=0.88)
        return close[0] if close else m.group()

    return re.sub(r"[A-Za-z]{6,}", repair, text)




def build_queries_facets(translated_note: str, ner_model, vectorizer: TfidfVectorizer, embedder, index, metadata):
    """
    One query per discharge section and one per item of the note, each naming the diagnosis:
      - sections (SECTIONS_HOME or SECTIONS_ADMITTED by prognosis) name all conditions and are searched within
        the diagnosis's guideline(s) only;
      - facets (FACET) are the diagnosis's qualifiers, the past history, the therapies and decisions of the stay,
        the MAX_FINDINGS abnormal findings the diagnosis's guideline talks about most and the MAX_LISTED pieces
        of a listed history it names most. Each names the head of the condition it sits closest to (a second
        condition keeps its own facets). All but the findings and the listed pieces hold a floor of one slot: the
        section frames outscore a comorbidity everywhere, yet in the example notes three quarters of the chunks a
        facet brought in were relevant, against a third of those the sections brought in. The listed pieces are
        searched within the diagnosis's guideline(s), like the sections.
    The presenting complaint is left out on purpose: it describes how the diagnosis was reached, which is
    settled by discharge, and its narrative pulls case descriptions from unrelated guidelines.
    Returns (queries, {query index: floor}, {query index: guidelines it is searched in}).
    """
    translated_note = fix_oov(normalize(translated_note), vectorizer, ner_model.tokenizer)
    fields = extract_patient_fields(translated_note)
    conditions, qualifiers = parse_diagnosis(fields.get("diagnosis", ""), ner_model)
    # an unknown age keeps the guidelines for children
    child = fields.get("age") is None or fields["age"] <= 14 or fields.get("neonatal", False)
    guidelines = primary_guidelines(conditions, embedder, index, metadata, child=child)
    sections = SECTIONS_ADMITTED if is_hospitalised(fields.get("prognosis", "")) else SECTIONS_HOME
    queries = [s.format(dx=" and ".join(conditions)) for s in sections]

    known = set(re.findall(r"\w+", " ".join(conditions + qualifiers).lower())) | {"of", "the", "with", "in", "a", "an",
                                                                                  "and", "is", "was"}
    items, findings, listed = note_items(fields, known, ner_model)
    items = qualifiers + items
    heads = [head(c, ner_model) for c in conditions]

    if findings:
        shares = on_topic([FACET.format(item=x, dx=heads[0]) for x in findings], guidelines, embedder, index, metadata)
        findings = [x for s, x in sorted(zip(shares, findings), key=lambda t: -t[0]) if s >= MIN_ON_TOPIC][:MAX_FINDINGS]
    # A listed piece is kept when the diagnosis's guideline names it, not by on_topic: in a query about the
    # diagnosis, a word the embedder knows little of leaves the diagnosis to decide ("cardioasa" 0.80 for AF), while
    # a real comorbidity pulls in its own guideline (amiodarone 0.60, hypertension 0.47, named in 37 and 32 chunks of
    # the AF guideline). Named means one chunk holds every word of it that could name a topic, as a word start
    # ("hypertens" for hypertensive), and whole when the corpus lacks it: a typo or a brand is never named.
    chunks = [m["text"].lower() for m in metadata if m["doc_id"] in guidelines]
    stop, vocab = vectorizer.get_stop_words() or set(), vectorizer.vocabulary_
    common = 1 + math.log((1 + len(metadata)) / (1 + TOPIC_SHARE * len(metadata)))  # the idf of such a word

    def named(text):
        words = [w for w in re.findall(r"[a-z]{4,}", text.lower())
                 if w not in stop and (w not in vocab or vectorizer.idf_[vocab[w]] >= common)]
        stems = [re.compile(r"\b" + re.escape(w[:max(5, len(w) - 3)] if w in vocab else w) + ("" if w in vocab else r"\b"))
                 for w in words]
        return sum(all(s.search(c) for s in stems) for c in chunks) if stems else 0

    listed = [x for n, x in sorted(((named(x), x) for x in listed), key=lambda t: -t[0]) if n][:MAX_LISTED]
    others = findings + listed
    closest = [0] * (len(items) + len(others))
    if len(heads) > 1 and closest:
        sims = (embedder.encode_query(items + others, convert_to_numpy=True, normalize_embeddings=True)
                @ embedder.encode_query(heads, convert_to_numpy=True, normalize_embeddings=True).T)
        closest = sims.argmax(axis=1)
    facets = [FACET.format(item=x, dx=heads[j]) for x, j in zip(items + others, closest)]
    queries = list(dict.fromkeys(queries + facets))
    # a floor for every item but the findings, which compete on score: they are the least specific
    floors = {i: 1 for i, q in enumerate(queries) if q in facets[:len(items)]}
    # a listed piece was chosen because the diagnosis's guideline names it, so it is searched there: colchicine
    # from the drug list of a patient whose diagnosis has no guideline of its own brought in pericarditis
    listed = set(facets[len(items) + len(findings):])
    scopes = {i: guidelines for i, q in enumerate(queries) if i < len(sections) or q in listed} if guidelines else {}
    return queries, floors, scopes


# The LLM path. Same lessons as the frames, but the LLM also knows the guidelines' name for the condition and the
# therapy the note never mentions. The count is left to the LLM within a ceiling: every query adds retrieval time,
# and only FINAL_N passages reach the letter. The letter the queries serve, and so the topics worth searching,
# depends on whether the patient goes home or is admitted.
MAX_LLM_QUERIES = 10
QUERY_PURPOSE = {
    False: "The patient is being discharged home. The recommendations tell the patient and their GP what to take (drug, "
           "dose, duration), how to manage the patient's other conditions and current medications alongside this "
           "diagnosis, what to change in lifestyle, when to be seen again, and which symptoms mean coming back.",
    True: "The patient is being admitted to hospital. The recommendations hand the patient over to the admitting team: "
          "the definitive treatment to start, the monitoring and investigations still needed, the complications to "
          "anticipate, and how the patient's other conditions and current medications affect that management.",
}
QUERY_TOPICS = {
    False: """- the definitive treatment of the condition and its drug therapy: agent, dose, duration
- the patient's other conditions and current medications, where they change that treatment
- follow-up, monitoring and reassessment after discharge
- lifestyle and self-care, only for a risk factor the patient actually has
- complications to watch for""",
    True: """- the definitive treatment (drug protocol or procedure) and its timing
- drug therapy: agent, dose, duration, and adjustment for this patient
- the patient's other conditions and current medications, where they change that treatment
- the monitoring and investigations still needed, and the criteria for escalation
- the complications to anticipate""",
}

QUERY_PROMPT = """You write search queries for a retrieval system over a corpus of official clinical guidelines. The passages it retrieves are the only source for this patient's recommendations. {purpose}

How the search works: each query is embedded and compared with guideline passages, then a cross-encoder scores the passages it found against that query. The passages that score best are the ones that read like the query. So:
- Write each query as the guideline recommendation you expect to find, in the guideline's own wording: a declarative sentence such as "<intervention> is recommended in patients with <condition> [and <patient factor>] to <aim>". Not a question, not a heading, not a list of keywords.
- One topic per query. A query that lists several topics matches none of them well; if two things matter, write two queries.
- Name the condition as the guidelines name it (e.g. the formal disease name, its type or stage), not with the ward shorthand of the note. Leave out the side of the body and words like "suspected"; mention sex or age only when they change management (pregnancy, children, frailty).
- Be specific where this patient is specific: the drug classes and agents, doses or thresholds, scores, comorbidities, current medications and contraindications that change what the guideline says for this patient. Use only what the note states as present: skip conditions that are denied, past or resolved.
- Stay within what a guideline on this condition would say. Generic ward care (antiemetics, routine analgesia, standard perioperative thromboprophylaxis) is rarely in it: include such a topic only when the note gives a specific reason.
- Do not search for the diagnostic work-up the note has already completed; search for what comes next.
- Keep each query under 35 words.

Topics to consider, skipping those that do not apply to this patient and splitting one when two distinct things matter:
{topics}

How many: one query per distinct topic, usually 5 to 8 and never more than {max_queries}. Every query adds retrieval time and only {final_n} passages reach the recommendations, so do not pad or split a topic into near-duplicates.

Output only the queries, one per line, in English, with no numbering, headings or other text.

Clinical note:
{note}"""


def build_queries_llm(translated_note: str, llm_model=None, tokenizer=None, model: str = None, meta: dict = None,
                      provider: dict = None):
    """
    Queries written by an LLM, one per topic it judges worth retrieving for this patient, at most MAX_LLM_QUERIES.
    model (an OpenRouter id), else QUERY_MODEL, picks the writer -- an OpenRouter id always goes through the API, None
    reuses the output model (the API one, or the local model already loaded). Temperature 0, so a note gets nearly the same
    queries; reasoning at low effort, since it adds seconds to a call that is already most of the retrieval time
    and the current Claude models refuse it off. Its hidden tokens count against max_tokens (at 800 they truncated
    the queries, and emptied them for one note; Nemotron 3 Super at low effort passed 4000 on one note), hence the margin.
    Floors of 1, the rest of the selection by score. meta and provider go to the OpenRouter call (generate_letter_openrouter).
    """
    from generation.output_prod import generate_letter

    admitted = is_hospitalised(extract_patient_fields(translated_note).get("prognosis", ""))
    prompt = QUERY_PROMPT.format(purpose=QUERY_PURPOSE[admitted], topics=QUERY_TOPICS[admitted],
                                 max_queries=MAX_LLM_QUERIES, final_n=FINAL_N, note=translated_note)
    model = model or QUERY_MODEL or (MODEL_NAME if API else None)
    if model:
        output = generate_letter(prompt, model=model, tokenizer=None, backend="openrouter",
                                 temperature=0, max_tokens=16000, api_key=OPENROUTER_KEY, reasoning_effort="low",
                                 meta=meta, provider=provider)
    elif llm_model is None:
        raise RuntimeError("LLM queries without QUERY_MODEL or the API need a local model, but none was loaded")
    else:
        output = generate_letter(prompt, model=llm_model, tokenizer=tokenizer, backend="local",
                                 temperature=0, max_tokens=800)

    # one query per line, whatever the model wraps them in: numbering and bullets go, and a line that is a
    # heading ("Here are the queries:") or too short to be a guideline sentence was never one of them
    lines = (re.sub(r"^\s*(?:[-*\u2022]|\d+[.)])\s*", "", line).strip() for line in (output or "").splitlines())
    queries = [line for line in lines if len(line.split()) >= 8 and not line.endswith(":")][:MAX_LLM_QUERIES]
    if not queries:
        raise ValueError(f"No queries in the LLM output: {(output or '')[:200]!r}")
    return queries, {i: 1 for i in range(len(queries))}


def build_queries(translated_note: str, ner_model, vectorizer: TfidfVectorizer, embedder, index, metadata,
                  llm_model=None, tokenizer=None, llm_queries=LLM_QUERIES, on_fallback=None):
    """
    The queries for one note, as (queries, {query index: floor}, {query index: guidelines it is searched in}).
    Each query both retrieves and reranks its own candidates. llm_queries (LLM_QUERIES unless the UI says
    otherwise) picks the writer, and a failing LLM falls back to the frames rather than to no queries, passing
    its error to on_fallback when given (the UI shows it); the LLM's queries search the whole corpus.
    """
    if llm_queries:
        try:
            queries, floors = build_queries_llm(translated_note, llm_model, tokenizer)
            return queries, floors, {}
        except Exception as error:
            print(f"  [queries] LLM query building failed ({error}), falling back to the facet frames")
            if on_fallback:
                on_fallback(error)

    return build_queries_facets(translated_note, ner_model, vectorizer, embedder, index, metadata)


if __name__ == "__main__":
    out = split_sentences("• EARLY PAIN TREATMENT\nDrug allergies: no\nParacetamol no\nTREATMENT:\nParacetamol 1 g orally. "
                          "Codeine 30 mg\nHas the patient's allergy status been checked? Yes\n• Enoxaparin s.c. administered")
    assert out == ["EARLY PAIN TREATMENT", "Paracetamol 1 g orally.", "Codeine 30 mg", "Enoxaparin s.c.", "administered"], out

    # a historical cue stays in its item, a denial runs over the list
    text = "thyroiditis, previous PSVT, arterial hypertension. Denies chest pain, syncope"
    at = lambda w: (text.index(w), text.index(w) + len(w))
    assert [is_negated(text, *at(w)) for w in ("PSVT", "hypertension", "syncope")] == [True, False, True]

    # a remote date says an event is over, a date of onset does not, nor a value that looks like a year
    text = ("Admitted to medicine in 2022 for respiratory failure. Gallstones. 2006: road polytrauma with brachial plexus "
            "injury, humerus fracture, left ankle (residual paresis of the left arm). Diverticular disease (acute "
            "diverticulitis in 2009). Previous operations: hernia repair (2013). Hypertension since 2010, diabetes "
            "diagnosed in 2015. Pro-BNP 2000 pg/mL with heart failure. Removal of sinus polyp 10 years ago. On 12/05/2022 "
            "the patient underwent a procedure consisting of: iliac stenting")
    past = ["respiratory failure", "polytrauma", "humerus fracture", "acute diverticulitis", "hernia repair",
            "sinus polyp", "iliac stenting"]
    now = ["Gallstones", "residual paresis", "Diverticular disease", "Hypertension", "diabetes", "heart failure"]
    assert [w for w in past if not is_negated(text, *at(w))] == [], "missed past"
    assert [w for w in now if is_negated(text, *at(w))] == [], "current marked past"
    # as DeepL writes dates; an exception leaves the denial
    text = ("On 12 May 2022, the patient underwent a procedure consisting of: iliac stenting. In May 2019 hepatitis. "
            "He may have asthma. Not currently on any treatment, apart from valsartan")
    assert [is_negated(text, *at(w)) for w in ("iliac stenting", "hepatitis", "asthma", "valsartan")] == \
           [True, True, False, False]
    assert normalize("PRESENTS WITH BILIARY VOMITING IN COPD. On CPAP") == "presents with biliary vomiting in COPD. On CPAP"

    # NER cuts words and tags a lead-in: the pieces are whole words, from the first named thing to the last
    text = "Drug history: on therapy with enoxaparin, type 2 diabetes on metformin, hypothyroidism"
    ents = [("MEDICATION", " enoxaparin"), ("DISEASE_DISORDER", " type 2 diabetes"), ("MEDICATION", " metformin"),
            ("DISEASE_DISORDER", " hypo"), ("HISTORY", "thyroid")]
    found, pos = [], 0
    for group, word in ents:
        pos = text.index(word, pos)
        found.append({"entity_group": group, "start": pos, "end": pos + len(word)})
        pos += len(word)
    out = pieces(text, found)
    assert out == ["enoxaparin", "type 2 diabetes on metformin", "hypothyroidism"], out
    assert clean_item("type 2 diabetes on metformin 500 mg") == "type 2 diabetes on metformin"

    # a guideline holding enough of the name's chunks is the diagnosis's only if one of them is close to the name
    class Index:  # FAISS's answer for one name: similarities and chunk ids, closest first
        def search(self, query, k):
            return [[0.40, 0.38, 0.31, 0.30, 0.29, 0.28]], [[0, 1, 2, 3, 4, 5]]
    class Embedder:
        def encode_query(self, texts, **kw):
            return texts
    meta = [{"doc_id": d} for d in ("Right", "Right", "Paediatric", "Wrong", "Wrong", "Wrong")]
    assert primary_guidelines(["name"], Embedder(), Index(), meta, k=6, share=0.3) == {"Right"}
    assert primary_guidelines(["name"], Embedder(), Index(), meta, k=6, share=0.3, min_similarity=0.25) == {"Right", "Wrong"}
