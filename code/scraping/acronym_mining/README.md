# Acronym mining

How `Guidelines/Sources/acronyms.csv` was made (September 2026). That file lists the Europe-wide society acronyms
that the scraper (`scraping/search.py`) adds to its European filter. Kept as a record of the method: the pipeline
only reads the CSV and never runs these scripts.

**Why.** The European filter (`europe*` in title, corporate author, affiliation or journal name) missed guidelines
that name their society only by acronym, such as the 2023 ERS/ESICM/ESCMID/ALAT guideline on severe
community-acquired pneumonia. Of 97,935 guideline-type records (tier-2 types, English, 2010 onwards), it caught 9,601.

**Steps**, run from the project root with `PYTHONPATH=code`, `W` being a scratch folder:

1. `mine_titles.py W/missed_titles.tsv missed` and `mine_titles.py W/caught_titles.tsv caught`: the titles of the
   88,334 records the filter misses and the 9,601 it catches (ESearch by year, then ESummary).
2. `acronyms.py W/missed_titles.tsv W/tokens.csv`: every token with at least two capitals in the missed titles
   (titles written entirely in capitals skipped), with the number of records where it appears in capitals and in any
   case. 8,830 tokens.
3. `classify.py W/tokens.csv W/labels.csv 2`: an LLM labels the 3,571 tokens found in 2 or more records as
   Europe-wide, national, other organisation or no organisation, from two example titles each.
4. `tail_sample.py W/tokens.csv W/tail_sample.csv` and `classify.py W/tail_sample.csv W/tail_labels.csv 1`: 300 of
   the 5,259 single-record tokens, sampled with seed 0 (`tail_labels.csv` here). 5 were Europe-wide, so the
   unlabelled tail could add about 90 records (95% interval about 35–200), on niche topics.
5. `analyze.py W`: the Europe-wide list, acronyms that collide with English words (far more matches in any case
   than in capitals, since PubMed's `[ti]` ignores case), and non-Europe labels that often sit next to explicit
   European markers (none turned out to be a missed European society).
6. `write_acronyms.py W Guidelines/Sources/acronyms.csv`: the labels plus the manual review, 156 acronyms
   searched as `ACRONYM[ti]` and 4 (ERA, COST, SHARE, NATA) that `scraping.search.filters()` checks in capitals.
   The `note` column says why each excluded Europe-wide acronym is left out.

**Models**, temperature 0, through OpenRouter: the main labels (step 3) came from Sonnet 5
(`anthropic/claude-sonnet-5`), and the tail sample (step 4) from Opus 5 (`anthropic/claude-opus-5`), which
`classify.py` used before the account ran out of credit.

**Effect** on the European filter: tier-1 records 6,282 → 7,913, tier-2 records 9,601 → 11,889.
