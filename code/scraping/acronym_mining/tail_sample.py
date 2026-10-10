# One-off: a fixed random sample of 300 single-record tokens, to estimate what the unlabelled tail could add.
import csv, random, sys

rows = list(csv.DictReader(open(sys.argv[1], encoding="utf-8")))
single = [r for r in rows if r["titles"] == "1"]
random.seed(0)
sample = random.sample(single, 300)
with open(sys.argv[2], "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, list(rows[0]))
    w.writeheader(); w.writerows(sample)
print(len(single), "single-title tokens; sampled", len(sample))
