"""What every step shares: the output folder, the HTTP session and its retries, the LLM calls, the manifest."""
import csv, hashlib, json, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests
from requests.adapters import HTTPAdapter, Retry

from config import GUIDELINES_BATCH, GUIDELINES_MODEL, OPENROUTER_KEY
from generation.output_prod import generate_letter_openrouter

OUT = Path("Guidelines")
FIELDS = ["chapter", "section", "chapter_title", "query", "candidates", "flags", "covers", "pmid", "date", "journal", "title", "types",
          "doi", "pmcid", "file", "source"]

MAX_TOKENS = 16000
BATCHES = "https://openrouter.ai/api/v1/batches"
PENDING = OUT / "batches.json"  # batches submitted and not yet collected, picked up again after an interrupted run

http = requests.Session()
http.headers["User-Agent"] = "Mozilla/5.0 (X11; Linux x86_64; rv:140.0) Gecko/20100101 Firefox/140.0"
http.mount("https://", HTTPAdapter(max_retries=Retry(total=4, backoff_factor=1, allowed_methods=None,
                                                     status_forcelist=[429, 500, 502, 503, 504])))


def retry(call, waits=(5, 15, 45)):
    """call(), tried again after each wait when the connection drops (also halfway through a reply), the reply is not
    whole or the server is busy (429, 5xx): NCBI and OpenRouter both have their moments, and one failed request would
    end a run. A client error (a malformed query, 4xx) is raised at once."""
    for wait in (*waits, None):
        try:
            return call()
        except requests.RequestException as e:
            status = getattr(e.response, "status_code", None)
            if wait is None or status is not None and status != 429 and status < 500:
                raise
            time.sleep(wait)


def ask(prompt: str) -> str:
    """One standard call to GUIDELINES_MODEL, reasoning left to the model's default (on for Opus 5.5)."""
    return retry(lambda: generate_letter_openrouter(prompt, model=GUIDELINES_MODEL, api_key=OPENROUTER_KEY,
                                                    temperature=0, max_tokens=MAX_TOKENS))


def ask_all(prompts: dict[str, str], step: str) -> dict[str, str]:
    """The replies to a step's prompts, by key, as one OpenRouter batch (GUIDELINES_BATCH) or as standard calls. A
    prompt that failed is printed and left out: its step asks it again on the next run."""
    if not prompts:
        return {}
    if GUIDELINES_BATCH:
        return batch(prompts, step)
    def one(key):
        try:
            return ask(prompts[key])
        except Exception as e:  # API failure: asked again next run
            print(f"  {step} {key}: {type(e).__name__}: {e}")
    with ThreadPoolExecutor(8) as pool:
        return {k: r for k, r in zip(prompts, pool.map(one, prompts)) if r}


def batch(prompts: dict[str, str], step: str) -> dict[str, str]:
    """One batch of the prompts on the model's :batch variant, polled every minute until it ends (24 hours at most).
    Its id is kept in PENDING until the replies are in, so a run stopped meanwhile collects the same batch, unbilled
    twice, if it asks the same prompts again."""
    head = {"Authorization": f"Bearer {OPENROUTER_KEY}"}
    key = hashlib.sha1(json.dumps(prompts, sort_keys=True).encode()).hexdigest()
    pending = json.loads(PENDING.read_text()) if PENDING.exists() else {}
    if key not in pending:
        reqs = [dict(custom_id=k, body=dict(messages=[dict(role="user", content=p)], temperature=0,
                                            max_tokens=MAX_TOKENS)) for k, p in prompts.items()]
        # requests last: OpenRouter reads the body as a stream
        body = dict(endpoint="/v1/chat/completions", model=f"{GUIDELINES_MODEL}:batch", completion_window="24h",
                    requests=reqs)
        r = http.post(BATCHES, headers=head, json=body, timeout=300)
        if not r.ok:
            raise SystemExit(f"{step}: batch refused, HTTP {r.status_code} {r.text[:300]}\n"
                             "Set GUIDELINES_BATCH = False in config.py for standard calls.")
        pending[key] = r.json().get("id") or r.json()["data"]["id"]
        PENDING.write_text(json.dumps(pending, indent=1))
    print(f"  {step}: batch {pending[key]}, {len(prompts)} requests, {time.strftime('%H:%M')}; polling every minute")
    def poll():
        r = http.get(f"{BATCHES}/{pending[key]}", headers=head, timeout=120)
        r.raise_for_status()
        return r.json()
    t, status = time.time(), ""
    while status not in ("completed", "failed", "expired", "cancelled"):
        time.sleep(60)
        b = retry(poll)
        b = b.get("data", b)
        if b["status"] != status:
            status = b["status"]
            print(f"  {step}: {status} after {(time.time() - t) / 60:.0f} min")
    out = {}
    for item in b.get("results") or []:
        try:
            out[item["custom_id"]] = item["response"]["body"]["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError):  # asked again next run
            print(f"  {step} {item.get('custom_id')}: {str(item.get('error') or item.get('response'))[:200]}")
    del pending[key]
    PENDING.write_text(json.dumps(pending, indent=1))
    return {k: r for k, r in out.items() if r}


def write_manifest(rows):
    with open(OUT / "manifest.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, FIELDS, restval="", extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
