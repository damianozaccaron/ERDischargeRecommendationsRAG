import re

from config import USE_TABLES

# The prognosis line carries the outcome: "ricovero urgente" in a typed note, the Esito of a Verbale ("Ricoverato presso
# Pneumologia", "Rinviato a domicilio", "Rifiuta ricovero"), in Italian or as any translation words it. A refused
# admission sends the patient home.
ADMISSION = re.compile(r"\b(?:hospitali[sz]|admi(?:t|ssion)|inpatient|transferred|ricover|trasferit)", re.IGNORECASE)
REFUSAL = re.compile(r"\b(?:refus|declin|rifiut|against medical advice)", re.IGNORECASE)

# The rules of any translation of the output, whether in the generation call or in a second one.
TRANSLATION_RULES = (
    "- Translate faithfully and completely: the same items in the same order and numbering, every sentence in "
    "full. Do not add, omit, merge, split, summarise, reword or reinterpret anything, and do not correct or "
    "complete the content.\n"
    "- Drugs take their Italian name (enoxaparin -> enoxaparina). Numbers, doses and units keep their exact value, "
    "written the Italian way (8,000 IU -> 8.000 UI; 2.5 mg -> 2,5 mg). The source citations stay exactly as "
    "written.\n"
    "- Use standard Italian medical terminology.\n"
)

# Appended to either prompt on request (the UI's "Con traduzione in italiano", OUTPUT_TRANSLATION_IN_PROMPT in the
# backend): the output stays in English, next to the English excerpts it cites, and gains a faithful Italian copy of
# all of it for the patient.
ITALIAN_TRANSLATION = (
    "\nITALIAN TRANSLATION:\n"
    "After Section 2, write \"TRADUZIONE IN ITALIANO\" on a line of its own, then translate into Italian "
    "everything you wrote above: both sections, with their titles, every numbered item and every reference line. "
    "This is the only addition allowed after the two sections.\n"
    + TRANSLATION_RULES
)

# The same translation as a second call, on the finished output.
OUTPUT_TRANSLATION_PROMPT = (
    "Translate into Italian the document below: both sections, with their titles, every numbered item and every "
    "reference line.\n"
    + TRANSLATION_RULES +
    "Output only the translation, with no preamble, notes or comments.\n\n"
    "Document:\n"
)

# Section 2 names the source itself, not the [E#]/[T#] label, so the reader can find it without the prompt.
CITATION_RULE = (
    "Cite each source by copying the (document, pages) shown next to its [E#] or [T#] label, "
    "never the label itself, and each distinct source once. Cite only a source that states what the item says.\n"
)


def is_hospitalised(prognosis: str) -> bool:
    return bool(ADMISSION.search(prognosis)) and not REFUSAL.search(prognosis)


def build_preamble(patient_data: str, retrieved_chunks: list[dict], role: str, tables: list[dict] = ()) -> list[str]:
    """Shared preamble: ROLE + CLINICAL NOTE + GUIDELINE EXCERPTS (+ GUIDELINE TABLES when rendered)."""

    prompt = []

    prompt.append(f"ROLE:\n{role}\n")

    prompt.append("CLINICAL NOTE:\n")
    prompt.append(patient_data)
    prompt.append("\n")

    prompt.append("GUIDELINE EXCERPTS:\n")
    for i, chunk in enumerate(retrieved_chunks, 1):
        prompt.append(
            f"[E{i}] ({chunk['doc_id']}, "
            f"p.{chunk['page_start']}-{chunk['page_end']})\n"
        )
        prompt.append(chunk["text"])
        prompt.append("\n")

    if USE_TABLES and tables:
        prompt.append("GUIDELINE TABLES:\n")
        for i, t in enumerate(tables, 1):
            prompt.append(f"[T{i}] ({t['doc_id']}, p.{t['page_start']}-{t['page_end']}) {t['caption_raw']}\n")
            prompt.append(t["markdown"])
            prompt.append("\n")

    return prompt


def build_discharge_prompt(patient_data: str, retrieved_chunks: list[dict], tables: list[dict] = ()) -> str:
    """Prompt for patients discharged home."""

    role = (
        "You are an emergency department physician writing discharge "
        "recommendations for a patient. The document will be read by "
        "the patient and their general practitioner."
    )

    prompt = build_preamble(patient_data, retrieved_chunks, role, tables)
    with_tables = USE_TABLES and bool(tables)
    table_rules = (
        "- A dose may come from a cited table: copy the value or range exactly as written, never compute or "
        "interpolate. Where a table distinguishes routes, use the oral maintenance column and ignore intravenous "
        "ones. Before starting a drug from a table, check its contraindication or criteria columns against the "
        "clinical note; if one applies, do not start it.\n"
    ) if with_tables else ""

    # ── TASK ──
    prompt.append(
        "TASK:\n"
        "Write the discharge recommendations: exactly the two sections below, in plain text, with nothing "
        "before, between or after them.\n"
        "Base the recommendations on the excerpts" + (" and tables" if with_tables else "") + " that concern "
        "this patient's diagnosis or a condition treated in this visit (the treatment, measures and follow-up "
        "they recommend after discharge, and any home medication they say to stop), together with what the "
        "clinical note records (therapy begun in the emergency department, specialist advice). The excerpts were "
        "retrieved automatically and some may concern other conditions: ignore those.\n\n"

        "SECTION 1 — DISCHARGE RECOMMENDATIONS\n"
        "A numbered list of instructions for the patient. Go through these groups in this order, one "
        "instruction per item, and skip every group where nothing applies:\n"
        "a. Medications to start, including a therapy begun in the emergency department that goes on at home: "
        "\"Take [drug] [dose] [route] [frequency].\" or \"Continue [drug] [dose] [route] [frequency].\"\n"
        "b. Medications to stop or change: \"Stop [drug].\" or \"Stop [drug] and take [drug] [dose] [route] "
        "[frequency] instead.\"\n"
        "c. Other treatments and self-care measures, as direct instructions, including the care of anything applied "
        "in the emergency department (cast, splint, dressing).\n"
        "d. Lifestyle changes (smoking, alcohol, diet, exercise): \"[Change] is strongly recommended.\" Only "
        "what needs to change or start; say nothing about what the patient already does right.\n"
        "e. Follow-up visits, tests and specialist referrals: \"A prescription is issued for [specialist type] "
        "evaluation for [clinical purpose].\" or \"A prescription is issued for [test] for [clinical purpose].\" "
        "Take the specialist type from the clinical note when it names one.\n"
        "f. Always: \"Go to the nearest emergency department immediately if you experience [warning symptoms "
        "for this diagnosis].\"\n"
        "g. Always, last: \"You are referred back to your general practitioner for any further diagnostic or "
        "therapeutic needs and ongoing care.\"\n\n"

        "Medications:\n"
        "- Never mention a medication the patient already takes and keeps taking unchanged: the patient goes on "
        "with it without being told. Mention a home medication only to stop or change it, and only when the "
        "clinical note or an excerpt that applies to this visit calls for it.\n"
        "- Give every medication you mention its dose, route and frequency. Take them from the excerpts"
        + (" or tables" if with_tables else "") + " when they state them, otherwise copy them exactly from the "
        "clinical note; where neither states them, write \"as directed by the specialist\" in their place. "
        "Intravenous therapy does not go on at home: give the oral form when the excerpts"
        + (" or tables" if with_tables else "") + " state it.\n"
        + table_rules +
        "- Stopping a medication and its replacement is one item. Never put a medication and a non-medication "
        "instruction in the same item.\n\n"

        "Every item:\n"
        "- One or two sentences. No reasons or explanations (no \"because\", no \"to reduce the risk of\"), no "
        "speculation about future treatment (duration of therapy, drug switches), no citations, no bold, headers "
        "or sub-bullets.\n"
        "- If the patient needs a decision that the excerpts do not cover, such as the further work-up of the "
        "problem of this visit, do not make it: issue the referral of group e instead. Refer only for the problem "
        "of this visit, never for chronic conditions it did not concern.\n"
        "- Never write the same instruction twice in different words.\n\n"

        "SECTION 2 — CLINICAL REFERENCES (for the reviewing physician)\n"
        "One line for every item of Section 1, with its number, in this format:\n"
        "[number] - [sources] - [one-sentence clinical justification]\n"
        "[sources] is always one of:\n"
        "- the excerpts" + (" or tables" if with_tables else "") + " that support the item. " + CITATION_RULE +
        "- (Clinical note) when the item rests on the clinical note and no excerpt supports it. When both do, "
        "write the excerpts and then (Clinical note).\n"
        "- (No guideline source) when neither an excerpt nor the clinical note supports it.\n\n"

        "Example of the format only, for a different patient who takes ramipril and ibuprofen at home (ramipril "
        "goes on unchanged, so it is not mentioned):\n"
        "SECTION 1 — DISCHARGE RECOMMENDATIONS\n"
        "1. Stop ibuprofen and take diclofenac 75 mg orally every 12 hours instead.\n"
        "2. Drinking at least 2 litres of water a day is strongly recommended.\n"
        "3. A prescription is issued for urologist evaluation for management of the ureteral stone.\n"
        "4. Go to the nearest emergency department immediately if you experience fever, persistent vomiting or "
        "pain not controlled by the prescribed therapy.\n"
        "5. You are referred back to your general practitioner for any further diagnostic or therapeutic needs "
        "and ongoing care.\n\n"
        "SECTION 2 — CLINICAL REFERENCES (for the reviewing physician)\n"
        "1 - (RenalColic, p.6-6)(RenalColic, p.8-9) - Diclofenac is first-line analgesia for renal colic; "
        "ibuprofen stopped to avoid two NSAIDs.\n"
        "2 - (RenalColic, p.12-12) - High fluid intake reduces stone recurrence.\n"
        "3 - (Clinical note) - Urology follow-up requested by the on-call urologist.\n"
        "4 - (RenalColic, p.4-4) - Fever with an obstructed kidney needs urgent decompression.\n"
        "5 - (No guideline source) - Standard closing item.\n\n"
    )

    # ── RULES ──
    prompt.append(
        "RULES:\n"
        "- Use ONLY the clinical note and the excerpts that apply to it. Do not rely on prior medical knowledge.\n"
        "- Do NOT invent drug names, doses, frequencies or clinical facts.\n"
        "- When several excerpts support the same instruction, write one item and cite them all.\n"
        "- Section 1 must be usable as a discharge document with no editing.\n"
        "- Before answering, check that no unchanged home medication appears in Section 1 and that every line of "
        "Section 2 has the number of its item and sources in one of the three forms above.\n"
    )

    return "\n".join(prompt)


def build_hospitalisation_prompt(patient_data: str, retrieved_chunks: list[dict]) -> str:
    """Prompt for patients admitted to hospital from the ER."""

    role = (
        "You are an emergency department physician writing recommendations "
        "for a patient being admitted to hospital. The document "
        "will be read by the admitting team, the patient, and their family."
    )

    prompt = build_preamble(patient_data, retrieved_chunks, role)

    # ── TASK ──
    prompt.append(
        "TASK:\n"
        "The patient is being admitted to hospital. The ER does not "
        "prescribe ongoing therapy or follow-up — that is the admitting "
        "team's responsibility. Produce a brief handover note in two "
        "sections.\n\n"

        "SECTION 1 — ADMISSION NOTE\n"
        "In a numbered list: \n"
        "- State that the patient is admitted to [department] "
        "for [key planned interventions], extracting the department "
        "and planned interventions from the clinical note "
        "(e.g. \"Patient is admitted urgently to the Paediatrics "
        "department for high-dose intravenous immunoglobulin therapy, "
        "dedicated transthoracic echocardiography with coronary "
        "evaluation, and clinical-laboratory monitoring.\").\n"
        "- State that the patient and/or family have been informed "
        "of the diagnosis, the expected clinical course based on the "
        "guideline excerpts (e.g. treatment protocol, possible "
        "complications, need for monitoring), and the planned "
        "treatment pathway. Do not write examples.\n"
        "- Do NOT give examples of complications or risks in parentheses. "
        "State the information directly without \"e.g.\", \"such as\", "
        "or \"for example\".\n"
        "- If the patient is being hospitalised to perform surgery,"
        "state that the patient has been prepped. \n"
        "- State that the patient is handed over to the care of "
        "the [department] team.\n\n"
        "- Do NOT prescribe medications, follow-up visits, or "
        "lifestyle changes.\n"
        "- Do NOT cite sources in this section.\n"
        "- Keep each item to one or two sentences.\n\n"

        "SECTION 2 — CLINICAL REFERENCES (for the reviewing physician)\n"
        "For each item in Section 1 that draws on guideline content, "
        "write:\n"
        "[number] - [sources] - brief clinical justification\n"
        + CITATION_RULE +
        "Example:\n"
        "2 - (KawasakiDisease, p.4-5)(KawasakiDisease, p.9-9) - IVIG 2g/kg and aspirin 30-50mg/kg/day are "
        "first-line treatment for Kawasaki disease to prevent coronary "
        "artery aneurysms.\n\n"
    )

    # ── RULES ──
    prompt.append(
        "RULES:\n"
        "- Use ONLY the provided guideline excerpts. "
        "Do not rely on prior medical knowledge.\n"
        "- Do NOT invent clinical facts not in the clinical note "
        "or excerpts.\n"
        "- Section 1 must be directly usable as a handover note "
        "with no editing needed.\n"
    )

    return "\n".join(prompt)


def build_prompt(
    patient_data: str,
    retrieved_chunks: list[dict],
    prognosis: str = "",
    tables: list[dict] = (),
) -> str:
    """
    Selects the appropriate prompt template based on the prognosis of hospitalisation.
    If the patient is hospitalised, the prompt gives more generic instructions since patient is handed over to a different team.
    """

    if is_hospitalised(prognosis):
        return build_hospitalisation_prompt(patient_data, retrieved_chunks)
    else:
        return build_discharge_prompt(patient_data, retrieved_chunks, tables)


def fit_prompt(max_chars: int, patient_data: str, retrieved_chunks: list[dict], prognosis: str = "",
               tables: list[dict] = ()) -> tuple[str, int, int]:
    """
    build_prompt, dropping the lowest-ranked excerpts and tables until it fits max_chars, so a chat app with an
    input cap does not cut the instructions at the end. Each drop takes the tail of whichever list has kept the
    larger share of its items; one excerpt always stays. Returns the prompt and how many excerpts and tables went.
    """
    chunks, kept_tables = list(retrieved_chunks), list(tables)
    prompt = build_prompt(patient_data, chunks, prognosis, kept_tables)
    while len(prompt) > max_chars and len(chunks) > 1:
        if kept_tables and len(kept_tables) / len(tables) > len(chunks) / len(retrieved_chunks):
            kept_tables.pop()
        else:
            chunks.pop()
        prompt = build_prompt(patient_data, chunks, prognosis, kept_tables)
    return prompt, len(retrieved_chunks) - len(chunks), len(tables) - len(kept_tables)