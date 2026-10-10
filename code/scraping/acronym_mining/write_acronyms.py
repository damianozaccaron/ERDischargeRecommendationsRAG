# One-off: Guidelines/Sources/acronyms.csv, the reviewed list scraping/search.py reads: the LLM's labels and
# the manual review. use: "yes" adds ACRONYM[ti] to the European filter, "case" keeps only titles with it in capitals.
import collections, csv, re, sys

S, dst = sys.argv[1], sys.argv[2]
dedupe = lambda p: list({r[0]: r for r in csv.reader(open(p, encoding="utf-8"), delimiter="\t")}.values())
caught = collections.Counter(t for r in dedupe(f"{S}/caught_titles.tsv")
                             for t in set(re.findall(r"[A-Za-z][A-Za-z0-9]*", r[3])))
labels = list(csv.DictReader(open(f"{S}/labels.csv", encoding="utf-8")))
CASE = ["ERA", "COST", "SHARE", "NATA"]
why = {**dict.fromkeys(CASE, "collides with an English word: searched case-blind, kept only where the title has it in "
                             "capitals"),
       "GA": "in capitals too it is also general anaesthesia, gestational age, geographic atrophy; GA(2)LEN documents "
             "are caught through LEN",
       **dict.fromkeys(["SIOPe", "EuroGuiderm"], "case variant of a listed acronym: [ti] is case-blind"),
       "AFAS": "mislabelled: French national society", "GEIL": "mislabelled: French national group",
       "FMF": "mislabelled: UK-based international foundation", "PIC": "mislabelled: international regulatory scheme",
       **dict.fromkeys(["EMA", "EFSA"], "regulatory agency, not clinical practice guidelines"),
       **dict.fromkeys(["EuroFIR", "EEMCO", "FELASA", "ECVDI", "ECEIM"], "outside human clinical medicine")}
org = {"SHARE": "Single Hub and Access point for paediatric Rheumatology in Europe"}  # the LLM guessed a survey
use = lambda r: "case" if r["token"] in CASE else "yes" if r["scope"] == "europe" and r["token"] not in why else "no"
rows = [dict(token=r["token"], use=use(r), missed_records=r["titles"], case_blind_records=r["case_blind_titles"],
             caught_titles=caught[r["token"]], scope=r["scope"], org=org.get(r["token"], r["org"]),
             note=why.get(r["token"], ""), example1=r["example1"], example2=r["example2"]) for r in labels]
rows.sort(key=lambda r: (r["use"] == "no", r["scope"] != "europe", -int(r["missed_records"]), r["use"] == "case"))
with open(dst, "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, list(rows[0])); w.writeheader(); w.writerows(rows)
print({u: sum(r["use"] == u for r in rows) for u in ("yes", "case", "no")})
