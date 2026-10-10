# One-off: acronym-like tokens in the titles the European filter misses, ranked by how many records each would add.
import collections, csv, re, sys

rows = list({r[0]: r for r in csv.reader(open(sys.argv[1], encoding="utf-8"), delimiter="\t")}.values())  # one per PMID: a record dated in two years came twice
upper, words, examples = collections.Counter(), collections.Counter(), collections.defaultdict(list)
for pmid, year, journal, title in rows:
    words.update(set(re.findall(r"[a-z0-9]+", title.lower())))  # what a case-blind [ti] search matches
    if title.upper() == title:  # all-caps title: every word looks like an acronym
        continue
    for t in set(re.findall(r"[A-Za-z][A-Za-z0-9]*", title)):
        if 2 <= len(t) <= 12 and sum(c.isupper() for c in t) >= 2:
            upper[t] += 1
            if len(examples[t]) < 2:
                examples[t].append(f"{title[:120]} ({journal})")

print(len(rows), "titles;", len(upper), "tokens;", sum(n >= 3 for n in upper.values()), "in 3+ titles;",
      sum(n >= 5 for n in upper.values()), "in 5+")
with open(sys.argv[2], "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["token", "titles", "case_blind_titles", "example1", "example2"])
    for t, n in upper.most_common():
        w.writerow([t, n, words[t.lower()], *examples[t]])
