# One-off: what the labelled acronyms would add, which collide with English words, which labels look wrong.
import collections, csv, re, sys

S = sys.argv[1]
dedupe = lambda path: list({r[0]: r for r in csv.reader(open(path, encoding="utf-8"), delimiter="\t")}.values())
missed, caught = dedupe(f"{S}/missed_titles.tsv"), dedupe(f"{S}/caught_titles.tsv")
labels = list(csv.DictReader(open(f"{S}/labels.csv", encoding="utf-8")))
words = lambda title: set(re.findall(r"[a-z0-9]+", title.lower()))  # a [ti] search is case-blind
caught_upper = collections.Counter(t for r in caught for t in set(re.findall(r"[A-Za-z][A-Za-z0-9]*", r[3])))

print(collections.Counter(r["scope"] for r in labels))
europe = [r for r in labels if r["scope"] == "europe"]
print(f"\n{len(europe)} Europe-wide tokens (token, records as acronym, case-blind records, already-caught titles, org):")
for r in europe:
    ratio = int(r["case_blind_titles"]) / int(r["titles"])
    print(f"  {r['token']:10} {r['titles']:>4} {r['case_blind_titles']:>5}{' COLLIDES' if ratio > 1.5 else '         '}"
          f" {caught_upper[r['token']]:>4}  {r['org'][:70]}")

keys = {r["token"].lower() for r in europe}
added = [r for r in missed if words(r[3]) & keys]
print(f"\nThe whole list adds {len(added)} of {len(missed)} missed records (the filter catches {len(caught)} today)")

print("\nNot labelled Europe-wide but often next to explicit European markers (token, scope, missed, caught, org):")
for r in labels:
    n_in, n_out = caught_upper[r["token"]], int(r["titles"])
    if r["scope"] != "europe" and n_in >= 3 and n_in / (n_in + n_out) >= 0.3:
        print(f"  {r['token']:10} {r['scope']:8} {n_out:>4} {n_in:>4}  {r['org'][:60]} | {r['example1'][:70]}")

national = sorted((r for r in labels if r["scope"] == "national"), key=lambda r: -int(r["titles"]))
print(f"\n{len(national)} national tokens, top 40:", ", ".join(f"{r['token']} {r['titles']}" for r in national[:40]))
