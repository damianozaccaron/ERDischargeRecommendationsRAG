"""--assist: the PDFs step 5 could not get, fetched through the user's own browser."""
import csv, select, shutil, sys, time, webbrowser
from pathlib import Path

from scraping.common import OUT, write_manifest
from scraping.pdfs import mismatch, stamp


def new_pdfs(folder: Path, since: float, seen: set) -> list[Path]:
    """Finished PDFs that appeared in folder after since (moved there too) and were not looked at yet."""
    out = []
    for p in folder.glob("*.pdf"):
        try:
            if p in seen or max(p.stat().st_mtime, p.stat().st_ctime) <= since or Path(f"{p}.part").exists():
                continue
            with p.open("rb") as f:
                if f.read(5) == b"%PDF-":
                    out.append(p)
        except OSError:  # renamed or removed by the browser meanwhile
            pass
    return out


def next_pdf(folder: Path, since: float, seen: set):
    """The next finished PDF saved into folder; "" when the user presses Enter (skip), None on q (stop)."""
    quiet = time.time()
    while True:
        if new := new_pdfs(folder, since, seen):
            seen.add(new[0])
            return new[0]
        if select.select([sys.stdin], [], [], 1)[0]:
            line = sys.stdin.readline()
            return None if not line or line.strip().lower() == "q" else ""
        if quiet and time.time() - quiet > 60:
            print(f"  (nothing new in {folder} for a minute: the browser must save the PDFs there, or rerun with "
                  f"--downloads DIR)")
            quiet = 0


def assist(downloads: Path, proxy: str):
    """Opens in the browser, one at a time, the guidelines without a PDF and those whose PDF fails mismatch() (kept by
    step 5 for want of a better copy), and files the next PDF saved into downloads under the open one, whatever its
    name, replacing any PDF it had."""
    rows = list(csv.DictReader(open(OUT / "manifest.csv", encoding="utf-8")))
    problem = lambda r: mismatch(OUT / r["file"], r) if (OUT / r["file"]).exists() else "no PDF yet"
    # a PDF kept by hand although not in English (source browser) is not offered again
    todo = [r | dict(problem=p) for r in {r["pmid"]: r for r in rows if r["pmid"]}.values() if (p := problem(r))
            and not (p == "not in English" and r["source"] == "browser")]
    print(f"{len(todo)} guidelines to fetch. Log in to your institution in your browser first (or pass --proxy).\n"
          f"Save each PDF into {downloads}, under any name (from the PDF viewer too: its download button).\n"
          f"Enter skips a guideline, q and Enter stops; a PDF taken for not English can be kept anyway.")
    seen, since = set(), time.time()
    for i, r in enumerate(todo, 1):
        url = proxy + (f"https://doi.org/{r['doi']}" if r["doi"] else f"https://pubmed.ncbi.nlm.nih.gov/{r['pmid']}/")
        print(f"\n[{i}/{len(todo)}] {r['title'][:100]}\n  {url}" +
              ("" if r["problem"] == "no PDF yet" else f"\n  replaces {r['file']}: {r['problem']}"))
        webbrowser.open_new_tab(url)
        while pdf := next_pdf(downloads, since, seen):
            try:
                problem = mismatch(pdf, r)
            except Exception:  # a PDF MuPDF cannot read
                problem = "does not open as a PDF"
            # the word count misjudges a guideline whose body pages are in another language (a French abstract): the
            # user's word stands, and source browser keeps the next run from fetching it again
            if problem == "not in English" and input(f"  {pdf.name}: not in English? y and Enter keeps it anyway, "
                                                     f"Enter alone refuses it: ").strip().lower() == "y":
                problem = ""
            # checked before stamping, a refused file stays untouched
            if problem in ("not in English", "does not open as a PDF") or not stamp(pdf, r["pmid"]):
                print(f"  {pdf.name}: {problem or 'cannot be stamped'}, left in {downloads}. Save another copy, or "
                      f"Enter to skip")
                continue
            shutil.move(pdf, OUT / r["file"])
            if problem:
                print(f"  check it's the right PDF: {problem}")
            for row in rows:
                if row["pmid"] == r["pmid"]:
                    row["source"] = "browser"
            write_manifest(rows)  # after every file, so an interrupted session keeps what it got
            print(f"  saved {r['file']}")
            break
        if pdf is None:
            print(f"Stopped: {len(todo) - i + 1} guidelines left for the next session")
            return
