import re

# the Clinical Note's section labels as the translation words them (pipeline/CreateGlossaryDeepL.txt)
SECTION_LABELS = {
    "anamnesis": r"^Medical history\s*:",
    "medical_history": r"Pathological proximate[^:\n]*:",
    "remote_medical_history": r"Remote pathology[^:\n]*:",
    "drug_history": r"Drug history\s*:",
    "allergies": r"^\W*Allergies\s*:",
    "exams": r"Objective examination\s*:",
    "clinical_diary": r"Clinical (?:diary|record)\s*:",
    "procedures": r"^\W*Procedures\s*:",
    "orders": r"^\W*Treatments ordered in the ED\s*:",
    "diagnosis": r"Diagnosis\s*:",
    "prognosis": r"Prognosis\s*:",
}

def collect_patient_input() -> str:
    """
    Mock function to collect patient data. In a real implementation, this would be replaced by a form or an interface to input the data.
    Only gives the idea that the user follows a pre-determined structure to input the data.
    """

    def prompt_required(prompt_text: str) -> str:
        value = ""
        while not value.strip():
            print(prompt_text)
            value = input()
            if not value.strip():
                print("Questo campo non può essere vuoto.")
        return value.strip()

    def prompt_optional(prompt_text: str) -> str:
        print(prompt_text)
        return input().strip()
    
    clinical_note = []

    print("Genere (M/F/A): ")
    gender = ""
    while gender.lower() not in ('m', 'f', 'a'):
        gender = input()
        if gender.lower() not in ('m', 'f', 'a'):
            print("Genere deve essere M (maschio), F (femmina) o A (altro): ")
    
    gender_exp = "Maschio"
    if gender.lower() == 'f':
        gender_exp = "Femmina"
    elif gender.lower() == 'a':
        gender_exp = "Altro"

    age = prompt_required("\n\nEtà: ")
    clinical_note.append(gender_exp+", "+age)

    clinical_note.append("\nAnamnesi: ")

    pat_prox = prompt_required("\n\nPatologica Prossima: ")
    clinical_note.append("- Patologia Prossima:\n" + pat_prox)

    pat_rem = prompt_optional("\n\nPatologica Remota (lasciare vuoto se non presente): ")
    if pat_rem:
        clinical_note.append("- Patologia Remota:\n" + pat_rem)

    exam = prompt_required("\n\nEsame Obiettivo: ")
    clinical_note.append("\nEsame Obiettivo:\n" + exam)
    
    diary = prompt_required("\n\nDiario Clinico: ")
    clinical_note.append("\nDiario Clinico: \n" + diary)

    diagnosis = prompt_required("\n\nDiagnosi: ")
    clinical_note.append("\nDiagnosi: "+diagnosis)

    prognosis = prompt_required("\n\nPrognosi: ")
    clinical_note.append("\nPrognosi: "+prognosis)


    clinical_note_str = "\n".join(clinical_note)

    return clinical_note_str


def extract_patient_fields(translated_note: str) -> dict:
    """
    Takes as input the translated note as raw text and returns a dictionary with the relevant fields extracted.
    """
    fields = {}

    # demographics (everything before comma is gender, the first digits after the comma are the age: "71 years old"
    # from DeepL, "71 years", "aged 71" or "71 yo" from an LLM)
    demo_match = re.match(r'^[^,]+,\s*(?:(\d+)\s*(?:years?(?:\s*old)?|y\.?o\.?|aa)|aged\s+(\d+)(?!\s*months?))?\s*(?:(?:aged\s+)?(\d+)\s*months?)?',
                          translated_note, re.IGNORECASE)

    gender = translated_note.split(",", 1)[0].strip()
    if gender.lower() == "other":
        gender = ""
    fields["gender"] = gender

    # age
    years = int(demo_match.group(1) or demo_match.group(2)) if demo_match.group(1) or demo_match.group(2) else None
    months = int(demo_match.group(3)) if demo_match.group(3) else None

    if years is not None:
        fields["age"] = years
    elif months is not None:
        fields["age"] = 0
    else:
        fields["age"] = None

    # neonatal
    if years is None and months is not None:
        fields["neonatal"] = True
    else: 
        fields["neonatal"] = False

    # each section runs from its label to the next label the note has: a Verbale may lack the remote history, the drug
    # history or the objective examination. Diagnosis and prognosis close the note, so their last label is theirs
    # ("Previous diagnosis: ..." in the history is not)
    found = []
    for key, pattern in SECTION_LABELS.items():
        matches = list(re.finditer(pattern, translated_note, re.IGNORECASE | re.MULTILINE))
        if matches:
            m = matches[-1] if key in ("diagnosis", "prognosis") else matches[0]
            found.append((m.start(), m.end(), key))
    found.sort()
    sections = {key: translated_note[end:after].strip()
                for (_, end, key), (after, _, _) in zip(found, found[1:] + [(len(translated_note), 0, "")])}
    for key in SECTION_LABELS:
        fields[key] = sections.get(key, "")
    # an anamnesis written as free text, without the history labels (the final print of the Verbale): the whole of it
    if fields["medical_history"] or fields["remote_medical_history"]:
        fields["anamnesis"] = ""
    return fields
