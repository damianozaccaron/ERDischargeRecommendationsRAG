# One-off: titles of every guideline-type record (tier-2 types, English, 2010+) that the europe* filter misses
# ("missed") or catches ("caught"), one row per PMID and year as PMID, year, journal, title.
import csv, sys, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from scraping.pubmed import eutils
from scraping.search import TIER2

E = "(europe*[ti] OR europe*[cn] OR europe*[ad] OR europe*[ta])"
base = f"{TIER2} AND english[la] NOT (comment[pt] OR editorial[pt] OR letter[pt] OR published erratum[pt])"
out, op = sys.argv[1], {"missed": "NOT", "caught": "AND"}[sys.argv[2]]
YEARS = range(2010, 2027)  # one search per year: ESearch returns at most 10,000 PMIDs


def call(tool, **p):
    for i in range(5):
        try:
            return eutils(tool, **p)
        except Exception as e:
            print("retry", tool, type(e).__name__, flush=True)
            time.sleep(5 * (i + 1))
    raise RuntimeError(tool)


def year(y):
    ids = call("esearch.fcgi", term=f"({base} AND {y}[dp]) {op} {E}", retmax=10000)["esearchresult"]["idlist"]
    with open(f"{out}.{y}", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, delimiter="\t")
        for i in range(0, len(ids), 500):
            res = call("esummary.fcgi", id=",".join(ids[i:i + 500]))["result"]
            for p in res["uids"]:
                w.writerow([p, y, res[p].get("source", ""), res[p].get("title", "")])
    print(y, len(ids), flush=True)


with ThreadPoolExecutor(3) as pool:  # NCBI: at most 3 requests/s without an API key; each request takes seconds
    list(pool.map(year, YEARS))
Path(out).write_text("".join(Path(f"{out}.{y}").read_text(encoding="utf-8") for y in YEARS), encoding="utf-8")
