"""
Scrapes European clinical guidelines, one per chapter of Tintinalli's Emergency Medicine 9e (303 chapters, less the
SKIPPED ones, which have nothing for a discharge letter), plus EXTRA topics the 2019 book predates (COVID-19), and
the topics of the Model of the Clinical Practice of Emergency Medicine (EM Model; ABEM, 2022) that no chapter searches.

1. Chapter titles come from a saved copy of the book's public companion site (tintinalliem.com/references_9e.php):
   the AccessEmergencyMedicine table of contents sits behind a Cloudflare browser check, and the 9e page may not
   outlive the next edition.
2. An LLM gives each chapter's core terms and patient group, cached in topics.json: edit a line there to steer a
   chapter. Terms made only of GENERIC words are dropped. Then up to 3 topics of the chapter's own references that its
   title terms miss (GERD in Esophageal Emergencies): its references' main MeSH topics cited twice or more and not
   already searched by the title terms, screened by the LLM, added by their MeSH names (under "refs" in topics.json).
   The code builds the PubMed query from the terms, unquoted so PubMed maps each term to MeSH.
2b. The EM Model's clinical items, the leaves of categories 2-18 of its Table 4 (read once from ABEM's PDF with
   poppler's pdftotext into Sources/em_model_2022.csv), get a name each from the LLM, read under their headings
   (em_names.json, editable like topics.json). An item is covered when a chapter term reads, in PubMed's own term
   mapping (frozen in mappings.json), as the same concepts, a term searched for a population covering only items of
   that population. The uncovered items, tumours left out, are pooled under their EM heading into EM topics
   ("EM4.6"), searched and selected like the chapters; em_coverage.csv has every item's verdict.
3. PubMed search for European guidelines and consensus statements in English from 2010 on, every hit (PubMed's
   relevance order changes from one search to the next: any cut by it is a lottery); chapters with fewer than FEW_HITS
   get a second tier that also takes consensus and position papers by title. The results are frozen in pubmed.json,
   each chapter with its search date, and reused until its query changes or --refresh. queries.csv logs how PubMed
   read each query, flags 0 or too many hits and words not found, and lists the terms it split into single words.
4. The LLM picks, among the candidates newest first, the smallest set of European society guidelines that covers the
   chapter's topics (its terms), the most recent among equals, or none. A chapter left with none, or without
   candidates, is searched again among the guidelines of national societies of European Union countries, the UK's
   NICE, the WHO and the Clinical Toxicology Recommendations Collaborative (FALLBACK: every hit, frozen in
   pubmed_fallback.json), and the LLM picks among those, an Italian society's before another country's. Then one
   consolidation pass over all chapters' picks, since each chapter's call sees only its own candidates: picks with
   alike titles go to LLM calls, which replace duplicates, translations, summaries and superseded versions with the one
   to keep (a newer version never replaces a guideline titled as European by one that is not). manifest.csv has one
   row per chapter or EM topic and pick, and flags what to review; the run counts chapters and EM topics apart.
   Every LLM step uses GUIDELINES_MODEL, as one OpenRouter batch per step with GUIDELINES_BATCH (config.py): half
   price, but each step waits minutes to hours.
5. PDFs from open-access copies: PMC open data, then the copies OpenAlex lists, published versions first, then those
   of the guideline's twins (the same title in another journal). A copy counts when it is in English (OpenAlex files
   some translations under the original's DOI), its first pages have most of the title's words and it has more than 2
   pages; downloads.csv logs every attempt. Publisher sites block scripts
   (Cloudflare) and many need a subscription, so --assist opens the remaining guidelines in your own browser, one
   at a time, where your institutional login applies (or --proxy, your library's EZproxy prefix), and files the PDF
   you save into your downloads folder (--downloads), whatever its name, under the open guideline. Every
   PDF gets its PMID written into its metadata (/PMID): reruns recognise a guideline by it, whatever its file is called.

The steps are in code/scraping/: 1-2 in topics.py, 2b in em.py, 3 in search.py, 4 in selection.py, 5 in pdfs.py,
and scrape.py runs them. Run from the project root; the output also goes to Guidelines/run.log, each run under its
date and options:

  python code/scrape_guidelines.py              steps 1-5, writes Guidelines/manifest.csv
  python code/scrape_guidelines.py --queries    steps 1-3 only, writes queries.csv and em_coverage.csv for review
  python code/scrape_guidelines.py --refresh    searches PubMed again instead of reusing pubmed.json (with any of these)
  python code/scrape_guidelines.py --assist     the PDFs step 5 could not get, through your browser (saved in ~/Downloads)
  python code/scrape_guidelines.py --check      smoke test (network, one LLM call)
"""
import argparse, sys, time
from pathlib import Path

from scraping.assist import assist
from scraping.check import check
from scraping.common import OUT
from scraping.scrape import run


class Tee:
    """sys.stdout that also appends to a file: the run log, where the thesis's counts and times come from."""
    def __init__(self, *streams):
        self.streams = streams

    def write(self, text):
        for s in self.streams:
            s.write(text)
            s.flush()

    def flush(self):
        for s in self.streams:
            s.flush()


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--assist", action="store_true", help="fetch the missing PDFs through your browser")
    ap.add_argument("--downloads", type=Path, default=Path.home() / "Downloads", help="where your browser saves files")
    ap.add_argument("--proxy", default="", help='EZproxy prefix, e.g. "https://ezproxy.example.it/login?url="')
    ap.add_argument("--check", action="store_true", help="smoke test")
    ap.add_argument("--queries", action="store_true", help="stop after the PubMed search, to review queries.csv")
    ap.add_argument("--refresh", action="store_true", help="search PubMed again instead of reusing pubmed.json")
    args = ap.parse_args()
    OUT.mkdir(exist_ok=True)
    sys.stdout = Tee(sys.stdout, open(OUT / "run.log", "a", encoding="utf-8"))
    print(f"=== {time.strftime('%Y-%m-%d %H:%M')} {' '.join(sys.argv[1:]) or '(full run)'}")
    check() if args.check else assist(args.downloads, args.proxy) if args.assist else run(args.queries, args.refresh)
