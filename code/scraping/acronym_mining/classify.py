# One-off: the LLM labels each acronym-like token (with enough titles) as a Europe-wide, national, other or no body.
import csv, json, sys
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from scraping.common import ask

PROMPT = """Below are tokens from the titles of medical guidelines, consensus statements and position papers in \
PubMed, as "token | example title (journal) | example title (journal)".
For each token, say whether it is the acronym of an organisation that issues such documents, and its scope:
- "europe": a Europe-wide medical society, council, network or agency, e.g. ESC (European Society of Cardiology), \
ERC (European Resuscitation Council), UEG (United European Gastroenterology), EORTC, ECDC.
- "national": a society, college or agency of one European country, e.g. NICE, AWMF, SPILF, BTS.
- "other": an organisation outside Europe or worldwide, e.g. AHA, IDSA, WHO, KDIGO.
- "none": not an organisation: a disease, test, drug, technique, trial, place or anything else.
Judge by the examples when a token has several meanings.

Return only JSON: {"<token>": {"org": "<the organisation's full name, or null>", "scope": "europe|national|other|none"}}

"""

src, dst, k = sys.argv[1], sys.argv[2], int(sys.argv[3])
rows = [r for r in csv.DictReader(open(src, encoding="utf-8")) if int(r["titles"]) >= k]
cache = Path(dst + ".json")  # labels so far: a rerun (after an API error) asks only for the rest
got = json.loads(cache.read_text()) if cache.exists() else {}
todo = [r for r in rows if r["token"] not in got]
batches = [todo[i:i + 150] for i in range(0, len(todo), 150)]


def label(batch):
    lines = "\n".join(f"{r['token']} | {r['example1']} | {r['example2']}" for r in batch)
    reply = ask(PROMPT + lines)
    return json.loads(reply[reply.find("{"):reply.rfind("}") + 1])


with ThreadPoolExecutor(1) as pool:  # every batch that succeeds is kept, whatever fails around it
    for fut in as_completed([pool.submit(label, b) for b in batches]):
        try:
            got |= fut.result()
            cache.write_text(json.dumps(got))
        except Exception as e:
            print("batch failed, a rerun asks for it again:", type(e).__name__, str(e)[:120])
with open(dst, "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["token", "titles", "case_blind_titles", "scope", "org", "example1", "example2"])
    for r in rows:
        g = got.get(r["token"]) or {}
        w.writerow([r["token"], r["titles"], r["case_blind_titles"], g.get("scope", "?"), g.get("org") or "",
                    r["example1"], r["example2"]])
print(len(rows), "tokens labelled;", sum(r["token"] not in got for r in rows), "missing from the answers")
