"""
Score table reading (caption and header) against tables_gold.json, hand-labelled from rendered pages.

The gold holds 72 regions the table finder returned, drawn at random from the Guidelines in three
batches of 24: 62 real tables and 10 regions that are not tables (reference lists, tables of contents,
affiliations...). Batch 3 was labelled after the table reading was tuned on batches 1 and 2 and is the
fairest estimate. Keys are "Guideline|page|index of the table in find_tables() on that page". A label
of "?" means the answer is ambiguous and is not scored; rotated tables are not scored.

  python -m evaluation.eval_tables        scores per batch
  python -m evaluation.eval_tables -v     and every miss, gold beside prediction

Run from code/. Needs the layout model (a couple of minutes).
"""
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import pymupdf

from ingestion.pdf_parsing import page_model, read_found_table, trim_description, upright

GOLD = Path(__file__).with_name("tables_gold.json")
GUIDELINES = Path(__file__).resolve().parents[2] / "Guidelines"


def tokens(s: str) -> set:
    """Words of a caption or header, citation markers dropped and a spaced-out "T A B L E" closed up."""
    s = re.sub(r"\[[\d\s,–-]+\]", " ", s.lower().replace("ﬁ", "fi").replace("ﬂ", "fl"))
    s = re.sub(r"\bt ?a ?b ?l ?e\b", "table", s)
    return set(re.findall(r"[a-z0-9]+", s))


def caption_ok(gold: str, pred: str) -> bool:
    """Most words of the gold caption found, and not buried in text that is not the caption."""
    g, p = tokens(trim_description(gold)), tokens(pred)
    if not g:
        return not p
    return len(g & p) >= 0.8 * len(g) and len(p) <= 2 * len(g) + 3


def header_ok(gold: list, pred: list) -> bool:
    """Most header words found, and most predicted header words real header words (no body row in it)."""
    g, p = tokens(" ".join(gold)), tokens(" ".join(pred))
    if not g:
        return not p
    return bool(p) and len(g & p) >= 0.8 * len(g) and len(g & p) >= 0.7 * len(p)


def main(verbose: bool = False) -> dict:
    gold = json.load(open(GOLD))
    by_page = defaultdict(list)
    for key, g in gold.items():
        doc, page, ti = key.split("|")
        if g["table"] and not g.get("rotated"):
            by_page[(doc, int(page))].append((int(ti), key, g))
    score = defaultdict(lambda: [0, 0, 0, 0])  # batch -> captions right, scored, headers right, scored
    for (doc, pno), items in sorted(by_page.items()):
        page = upright(pymupdf.open(GUIDELINES / f"{doc}.pdf")[pno - 1])
        if page.rotation:
            page.set_rotation(0)  # as extract_pdf_text_layout_aware does
        found = page.find_tables().tables
        model = page_model(page, [t.bbox for t in found])
        for ti, key, g in items:
            t = read_found_table(page, found[ti], model)
            s = score[g["batch"]]
            ok_c = ok_h = None
            if g["caption"] != "?":
                ok_c = caption_ok(g["caption"], t["description"])
                s[0] += ok_c
                s[1] += 1
            if g["header"] != "?":
                ok_h = header_ok(g["header"], t["header"])
                s[2] += ok_h
                s[3] += 1
            if verbose and False in (ok_c, ok_h):
                print(f"\n{key}\n  caption {'ok' if ok_c is not False else 'MISS'}  gold {g['caption']!r}\n"
                      f"                pred {t['description']!r}\n  header  {'ok' if ok_h is not False else 'MISS'}  gold {g['header']}\n"
                      f"                pred {t['header']}")
    for batch, (c, nc, h, nh) in sorted(score.items()):
        print(f"batch {batch}: captions {c}/{nc}, headers {h}/{nh}")
    return dict(score)


if __name__ == "__main__":
    main("-v" in sys.argv)
