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
