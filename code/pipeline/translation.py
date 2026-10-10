from config import DEEPL_AUTH_KEY, DEEPL_GLOSSARY_ID, API, MODEL_NAME, LLM_QUERIES, QUERY_MODEL, OPENROUTER_KEY
from generation.output_prod import generate_letter
from generation.prompting import OUTPUT_TRANSLATION_PROMPT
from pathlib import Path
import deepl
import json
import re

# The note's section labels and their English, as extract_patient_fields reads them (user_input.SECTION_LABELS). DeepL
# words a label as it likes whatever the glossary says ("Patologica prossima" came back as "Reason for presentation",
# "Current medical condition" and "Pathological proximate symptoms" in three notes), so labels never go through it.
LABELS = {"anamnesi farmacologica": "Drug history", "anamnesi": "Medical history",
          "patologica prossima": "Pathological proximate", "patologica remota": "Remote pathology",
          "allergie": "Allergies", "esame obiettivo": "Objective examination", "diario clinico": "Clinical diary",
          "procedure": "Procedures", "terapie disposte in ps": "Treatments ordered in the ED", "diagnosi": "Diagnosis",
          "prognosi": "Prognosis"}
SECTION = re.compile(r"(?im)^([^\w\n]*)(" + "|".join(LABELS) + r")([ \t]*:)")
SEX = {"maschio": "Male", "uomo": "Male", "femmina": "Female", "donna": "Female", "altro": "Other"}
DEMOGRAPHICS = re.compile(r"\s*(" + "|".join(SEX) + r")\s*,\s*(?:(\d+)\s*(?:anni|aa))?\s*(?:e\s*)?(?:(\d+)\s*mesi)?\s*",
                          re.IGNORECASE)


def demographics(line: str) -> str | None:
    """The note's first line in English ("Femmina, 89 anni" -> "Female, 89 years old"), None when it is not one."""
    m = DEMOGRAPHICS.fullmatch(line)
    return m and SEX[m[1].lower()] + ", " + " and ".join(filter(None, [m[2] and f"{m[2]} years old",
                                                                     m[3] and f"{m[3]} months"]))


def deepl_translation(text, auth_key = DEEPL_AUTH_KEY, glossary = DEEPL_GLOSSARY_ID):
    """
    The note in English by DeepL, section by section, each with its label for context ("Patologica remota: Muta" is
    "unremarkable", "Muta" alone a wetsuit). The label DeepL wrote, up to its colon, is then replaced by the fixed one.
    A drug history goes line by line: sent whole, two of its lines came back merged into one wrong drug. Line by line
    everywhere lost the meaning of every other abbreviation ("BAV I grado" became second-degree, "in AA" on oxygen), and
    DeepL's `context` (the whole note) made it guess ("ACVPU: Alert" became "GCS", "Caduta accidentale" a burn), so
    neither is used. The first line (sex and age, which DeepL writes "aged 89") is put as extract_patient_fields reads it.
    """
    # DeepL leaves all-caps words untranslated or half-translated (VENTRICOLAR, APPENDICITE, CELLULITE);
    # the diagnosis line is the only part of the template written in caps, and nothing downstream needs its case.
    # Words under 5 letters stay as they are: those are acronyms (IVDU, TEP, FA) that DeepL mangles once lowercased.
    # ponytail: a 5+ letter acronym (STEMI, NSTEMI) gets lowercased too, add an allowlist if one shows up
    text = re.sub(r'(Diagnosi\s*:)(.*)', lambda m: m.group(1) + re.sub(r'[A-ZÀ-Ý]{5,}', lambda w: w.group().lower(), m.group(2)), text)

    first, _, rest = text.partition("\n")
    head = demographics(first)
    units, drugs = [], False  # a section's lines, or one line of a drug history
    for line in (rest if head else text).split("\n"):
        m = SECTION.match(line)
        drugs = m[2].lower() == "anamnesi farmacologica" if m else drugs
        if m or not units or drugs and line.strip():
            units.append([line])
        else:
            units[-1].append(line)
    sources = []  # (label match, what DeepL gets, the blank lines after it)
    for unit in units:
        m = SECTION.match(unit[0])
        lines = [unit[0][m.end():].strip() if m else unit[0]] + unit[1:]
        while len(lines) > 1 and not lines[-1].strip():
            lines.pop()
        body = "\n".join(lines)
        source = (f"{m[2]}:" + ("\n" if not lines[0] else " ") + body.lstrip("\n") if m else body) if body.strip() else None
        sources.append((m, source, len(unit) - len(lines)))
    texts = [t for _, t, _ in sources if t]
    client, done = deepl.DeepLClient(auth_key), []
    for i in range(0, len(texts), 50):  # DeepL takes 50 texts a request
        done += [r.text for r in client.translate_text(texts[i:i + 50], source_lang="IT", target_lang="EN-GB",
                                                       glossary=glossary)]
    done = iter(done)
    out = []
    for m, source, blanks in sources:
        english = next(done) if source else ""
        if m:
            english = english.split(":", 1)[1] if ":" in english else english
            english = english if english.startswith("\n") else " " + english.strip() if english.strip() else ""
            english = m[1] + LABELS[m[2].lower()] + m[3] + english.rstrip()
        out.append(english + "\n" * blanks)
    return (f"{head}\n" if head else "") + "\n".join(out)

def deepl_translation_en_it(text, auth_key = DEEPL_AUTH_KEY):

    deepl_client = deepl.DeepLClient(auth_key)
    result = deepl_client.translate_text(text, source_lang="EN", target_lang="IT", formality="prefer_more")

    return result.text

def llm_translation_en_it(text, model, tokenizer, backend):
    """
    The output translated by the model that wrote it (OUTPUT_LLM_TRANSLATION), in a second call at temperature 0.
    Reasoning stays at the model's default, as in the generation call: some models (Gemini 3.x Flash) refuse to turn
    it off, and their hidden tokens count against max_tokens, hence the margin.
    """
    return generate_letter(OUTPUT_TRANSLATION_PROMPT + text, model=model, tokenizer=tokenizer, backend=backend,
                           temperature=0, max_tokens=8192, api_key=OPENROUTER_KEY).strip()

# The LLM gets the DeepL glossary's entries, read from the file that created it so the two never drift apart: the
# section headers in it are what extract_patient_fields looks for in the translated note.
_glossary_file = (Path(__file__).parent / "CreateGlossaryDeepL.txt").read_text(encoding="utf-8")
# its entries run over several lines, each closing and reopening the shell quote ('...'\ newline '...'), which the shell
# joins back into one argument
GLOSSARY = json.loads(_glossary_file.split("--data '", 1)[1].rsplit("'", 1)[0].replace("'\\\n'", ""))["dictionaries"][0]["entries"]

TRANSLATION_PROMPT = """Translate the Italian clinical note below into English.

- Translate the whole note into English, literally and completely: every sentence of the note appears once in the output, in English, and the Italian text is never copied. Do not add, remove, summarise, reorder, rephrase, 
interpret or explain anything, and do not improve the style: a telegraphic or clumsy sentence stays telegraphic or clumsy.
- Do not correct anything, even what looks like an error or a typo, and do not complete what is left unfinished.
- Keep the layout exactly: line breaks, blank lines, bullets, punctuation, capitalisation, and each section header followed by its colon.
- Numbers, values, units, doses, dates and scores stay exactly as written. An Italian abbreviation becomes its standard English equivalent (e.g. FA -> AF, e.v. -> IV) or, where there is none, stays as written.
- Use standard English medical terminology. A term in the glossary below always takes the glossary translation, and no other.
- A drug brand name in the glossary is always written as the generic name the glossary gives, with its dose unchanged (e.g. Clexane 4000 UI -> enoxaparin 4000 IU). This is part of the translation, not a correction or an interpretation.

Glossary (Italian, tab, English):
{glossary}

Output only the translation, with no preamble, notes or comments.

Clinical note:
{note}"""

def glossary_for(text):
    """
    The glossary entries whose Italian term the note contains, in any case. Given all ~500, the LLM left most brand
    names as they were and once dropped the note's first line; a note holds a few dozen.
    """
    return "\n".join(e for e in GLOSSARY.split("\n")
                     if re.search(r"(?<![\w-])" + re.escape(e.split("\t")[0]) + r"(?![\w-])", text, re.IGNORECASE))

def llm_translation(text, llm_model=None, tokenizer=None):
    """
    The note translated by an LLM the pipeline already calls: the query model when the queries go through the API,
    otherwise the output model (the API one, or the local model already loaded). Temperature 0 and reasoning off,
    as for the queries: a note gets nearly the same translation every time (the API is not fully deterministic even
    at temperature 0), and no hidden tokens eat into max_tokens.
    """
    prompt = TRANSLATION_PROMPT.format(glossary=glossary_for(text), note=text)
    api_model = QUERY_MODEL if LLM_QUERIES and QUERY_MODEL else MODEL_NAME if API else None
    if api_model:
        output = generate_letter(prompt, model=api_model, tokenizer=None, backend="openrouter", temperature=0,
                                 max_tokens=4096, api_key=OPENROUTER_KEY, use_reasoning=False)
    elif llm_model is None:
        raise RuntimeError("LLM translation without the API needs a local model, but none was loaded")
    else:
        output = generate_letter(prompt, model=llm_model, tokenizer=tokenizer, backend="local", temperature=0,
                                 max_tokens=4096)
    return output.strip()

# Instruction for glossary creation can be found in this folder in the file CreatGlossaryDeepL.txt (to use from terminal)


if __name__ == "__main__":
    class Fake:  # DeepL, offline: each text marked as translated
        def __init__(self, key): pass
        def translate_text(self, texts, **kw):  # the label reworded, the text marked, one result per text
            return [type("R", (), {"text": re.sub(r"^[^:\n]*:(\s*)", r"Whatever:\1<", t, count=1) + ">" if ":" in t
                                   else f"<{t}>"}) for t in texts]
    deepl.DeepLClient, real = Fake, deepl.DeepLClient
    out = deepl_translation("Femmina, 89 anni\nAnamnesi:\nANAMNESI GENERALE\n- Patologica prossima: tosse\n"
                            "Anamnesi farmacologica: Eutirox\n\nDiario clinico:\n• vomito\n\nDiagnosi: BRONCHITE\n\nPrognosi: ")
    assert out == ("Female, 89 years old\nMedical history:\n<ANAMNESI GENERALE>\n- Pathological proximate: <tosse>\n"
                   "Drug history: <Eutirox>\n\nClinical diary:\n<• vomito>\n\nDiagnosis: <bronchite>\n\nPrognosis:"), out
    assert demographics("MASCHIO, 2 AA e 8 MESI") == "Male, 2 years old and 8 months" and demographics("Tosse") is None
    deepl.DeepLClient = real
    sent = glossary_for("Cefalea frontale, al momento. Assume CLEXANE 4000 UI e Frontal. Patologica prossima: dolore")
    assert sent.split("\n") == ["Patologica Prossima\tPathological proximate", "clexane\tenoxaparin", "frontal\talprazolam"], sent
