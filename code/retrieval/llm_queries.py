import re

from config import QUERY_MODEL, API, MODEL_NAME, OPENROUTER_KEY, FINAL_N
from pipeline.user_input import extract_patient_fields
from generation.prompting import is_hospitalised


# The LLM path. 
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
