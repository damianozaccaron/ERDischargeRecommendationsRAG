from config import DEEPL_AUTH_KEY, DEEPL_GLOSSARY_ID, API, MODEL_NAME, LLM_QUERIES, QUERY_MODEL, OPENROUTER_KEY
from generation.output_prod import generate_letter
from generation.prompting import OUTPUT_TRANSLATION_PROMPT, CITATION
from pathlib import Path
from xml.sax.saxutils import escape, unescape
import deepl
import json
import re

# The note's section labels and their English, as extract_patient_fields reads them (user_input.SECTION_LABELS).
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
    The note in English by DeepL, section by section, each with its label for context. The label DeepL wrote, up to its colon, is then replaced by the fixed one.
    """
    # DeepL leaves all-caps words untranslated or half-translated
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
    """
    The output in Italian by DeepL. Its citations go in a tag DeepL is told to leave as written
    """
    marked = CITATION.sub(lambda m: f"<x>{m[0]}</x>", escape(text))
    result = deepl.DeepLClient(auth_key).translate_text(marked, source_lang="EN", target_lang="IT", formality="prefer_more",
                                                        tag_handling="xml", ignore_tags=["x"])
    return unescape(re.sub(r"</?x>", "", result.text))

def llm_translation_en_it(text, model, tokenizer, backend):
    """
    The output translated by the model that wrote it (OUTPUT_LLM_TRANSLATION), in a second call at temperature 0.
    Reasoning stays at the model's default, as in the generation call.
    """
    return generate_letter(OUTPUT_TRANSLATION_PROMPT + text, model=model, tokenizer=tokenizer, backend=backend,
                           temperature=0, max_tokens=8192, api_key=OPENROUTER_KEY).strip()

# The LLM gets the DeepL glossary's entries, read from the file that created it
_glossary_file = (Path(__file__).parent / "CreateGlossaryDeepL.txt").read_text(encoding="utf-8")

GLOSSARY = json.loads(_glossary_file.split("--data '", 1)[1].rsplit("'", 1)[0].replace("'\\\n'", ""))["dictionaries"][0]["entries"]

TRANSLATION_PROMPT = """Translate the Italian clinical note below into English.

- Translate the whole note into English, literally and completely: every sentence of the note appears once in the output, in English, and the Italian text is never copied. Do not add, remove, summarise, reorder, rephrase, 
interpret or explain anything, and do not improve the style: a telegraphic or clumsy sentence stays telegraphic or clumsy.
- Do not correct anything, even what looks like an error or a typo, and do not complete what is left unfinished.
- Keep the layout exactly: line breaks, blank lines, bullets, punctuation and capitalisation. Headings and labels are translated like the rest of the text (ANAMNESI GENERALE -> GENERAL MEDICAL HISTORY, Descrizione: -> Description:); the section labels already in English stay exactly as they are, each followed by its colon.
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

def llm_translation(text, llm_model=None, tokenizer=None, api_model=None, provider=None, meta=None, max_tokens=16000,
                    temperature=0):
    """
    The note translated by an LLM the pipeline already calls: the query model when the queries go through the API,
    otherwise the output model (the API one, or the local model already loaded). Temperature 0 and reasoning at low
    effort.
    """
    first, _, rest = text.partition("\n")
    head = demographics(first)
    note = SECTION.sub(lambda m: m[1] + LABELS[m[2].lower()] + m[3], rest if head else text)
    prompt = TRANSLATION_PROMPT.format(glossary=glossary_for(text), note=note)
    api_model = api_model or (QUERY_MODEL if LLM_QUERIES and QUERY_MODEL else MODEL_NAME if API else None)
    if api_model:
        output = generate_letter(prompt, model=api_model, tokenizer=None, backend="openrouter", temperature=temperature,
                                 max_tokens=max_tokens, api_key=OPENROUTER_KEY, reasoning_effort="low", meta=meta,
                                 provider=provider)
    elif llm_model is None:
        raise RuntimeError("LLM translation without the API needs a local model, but none was loaded")
    else:
        output = generate_letter(prompt, model=llm_model, tokenizer=tokenizer, backend="local", temperature=0,
                                 max_tokens=4096)
    return (f"{head}\n" if head else "") + output.strip()

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
    class FakeEnIt:  # DeepL translating "p." everywhere but in the ignored tags
        def __init__(self, key): pass
        def translate_text(self, text, **kw):
            return type("R", (), {"text": re.sub(r"(<x>.*?</x>)|p\.", lambda m: m[1] or "pag.", text)})
    deepl.DeepLClient = FakeEnIt
    out = deepl_translation_en_it("1 - (AtrialFibrillation, p.2-3)(Clinical note) - eGFR < 30 & p.")
    assert out == "1 - (AtrialFibrillation, p.2-3)(Clinical note) - eGFR < 30 & pag.", out
    deepl.DeepLClient = real
    # the LLM gets the labels in English and the note without its first line, which comes back written in code
    generate_letter, real = lambda prompt, **kw: prompt.rsplit("Clinical note:\n", 1)[1], generate_letter
    out = llm_translation("Femmina, 31 anni\nAnamnesi:\n- Patologica prossima: tosse\nALLERGIE:\n\nTerapie disposte in PS:\n"
                          "Toradol 1 fl\n\nDiagnosi: contusione\n\nPrognosi: 5 gg")
    assert out == ("Female, 31 years old\nMedical history:\n- Pathological proximate: tosse\nAllergies:\n\n"
                   "Treatments ordered in the ED:\nToradol 1 fl\n\nDiagnosis: contusione\n\nPrognosis: 5 gg"), out
    generate_letter = real
    sent = glossary_for("Cefalea frontale, al momento. Assume CLEXANE 4000 UI e Frontal. Patologica prossima: dolore")
    assert sent.split("\n") == ["Patologica Prossima\tPathological proximate", "clexane\tenoxaparin", "frontal\talprazolam"], sent
