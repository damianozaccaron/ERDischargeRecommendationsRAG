from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool
import os
import re
import pymupdf
import time
from collections import Counter

from config import USE_LAYOUT, INGESTION_WORKERS
from ingestion.cleaning import is_page_bad, drop_references, drop_abbreviations, REF_MARKER
from ingestion.tables import MAX_HEADER_CELL, DROP_HEADER, overlong_cell
from ingestion.table_layout import read_table, label, column_of

if USE_LAYOUT:
    import pymupdf.layout  # registers the ML page-segmentation hook that find_tables() picks up
else:
    pymupdf.no_recommend_layout()  # silence the suggestion printed by find_tables()

# Evidence Tables are the one kind of table we are sure is useless
EVIDENCE = re.compile(r'^\s*(Supplementary\s+)?Evidence\s+Table', re.IGNORECASE)
SENTENCE_END = re.compile(r'[.!?](?=\s+[A-Z]|\s*$)|;')  # not a comma (captions hold lists), a bracket, "2.5" or "pet. and"
MAX_DESCRIPTION = 200
SMALL_FONT = 0.75          # a span below this share of the cell's median font size is a footnote marker
MAX_BODY_SHARE = 0.5       # share of a table's text that the layout model labels body text


def trim_description(text: str) -> str:
    """
    Keep a caption down to its first sentence: some run on with notes on sources and abbreviations.
    The search starts after the label ("Table 12", "TABLE 4.2") so a caption like "Table 13: List of
    dosages" is not cut at the number, and stops at the first semicolon or sentence end ("(SEMS)",
    "2.5 mg" and "pet. and" are none).
    """
    m = label(text)
    start = m.end() if m else 0
    sep = re.match(r'[.:;\-\u2013\u2014]\s*', text[start:])  # "Table 5." separates, it does not end a sentence
    if sep:
        start += sep.end()
    end = SENTENCE_END.search(text, start)
    if end:
        text = text[: end.start()]
    return text[:MAX_DESCRIPTION].strip()


def empty_page_warning(doc_id: str, empty_page_counter: int, total_pages: int, threshold: float = 0.3):
    """
    Simple function to print a warning if too many pages are empty in a document.
    """
    if empty_page_counter / total_pages > threshold:
        print(
            f"[WARNING] {doc_id}: "
            f"{empty_page_counter*100/total_pages:.1f}% pages are empty. "
            "Possible scanned PDF or extraction failure."
        )


RUNNING_HEAD = ("page-header", "page-footer")
RUNNING_HEAD_BAND = 0.08  # wider lets in the body pages the model mislabels; the 5% margins alone miss 43% of running heads


def running_head(rect: pymupdf.Rect, layout: list, bounds: pymupdf.Rect) -> bool:
    """
    A running head or foot the 5% margins miss because it straddles their line ("Journal of Hepatology, August 2025.
    vol. 83 502–583"): the layout box covering most of the block is labelled page header or footer, and the block's
    centre lies within RUNNING_HEAD_BAND of a page edge. The label alone is not trusted: the model labels whole body
    pages page-header (recommendations in ANCAAssociatedVasculitis).
    """
    best = max(layout, key=lambda b: (rect & b[0]).get_area(), default=None)
    if best is None or best[1] not in RUNNING_HEAD or (rect & best[0]).is_empty:
        return False
    c = (rect.tl + rect.br) / 2
    return (min(c.y - bounds.y0, bounds.y1 - c.y) < RUNNING_HEAD_BAND * bounds.height
            or min(c.x - bounds.x0, bounds.x1 - c.x) < RUNNING_HEAD_BAND * bounds.width)


def is_superscript(span: dict, prev: dict | None) -> bool:
    """A span flagged superscript, or one smaller than its predecessor and raised above its baseline
    (author affiliations, some citation fonts). A subscript is smaller too but sits below, and stays.
    Text after a glyph ("▶ Table 2") is smaller than the glyph without being a superscript."""
    if span["flags"] & 1:
        return True
    return (prev is not None and any(ch.isalnum() for ch in prev["text"])
            and span["size"] < prev["size"] * 0.85 and span["origin"][1] < prev["origin"][1] - 0.5)


def prose_blocks(page) -> list[tuple]:
    """
    page.get_text("blocks") without the superscript spans, which are the citation markers ("included46",
    "glucocorticoids64"). Dropping them by typography keeps the letter-adjacent digits that belong to a
    name, CHA2DS2-VA, SGLT2, P2Y12, HbA1c, which a regex on the plain text cannot tell from a citation.
    Same tuple shape as get_text("blocks"): (x0, y0, x1, y1, text, block_no, block_type).
    """
    blocks = []
    for b in page.get_text("dict")["blocks"]:
        if b["type"] != 0:
            continue
        lines = []
        for l in b["lines"]:
            kept, prev = [], None
            for s in l["spans"]:
                if not is_superscript(s, prev):
                    kept.append(s["text"])
                prev = s
            lines.append("".join(kept))
        blocks.append((*b["bbox"], "\n".join(lines), b["number"], 0))
    return blocks


def cell_text(page, cell) -> str:
    """
    Text of one table cell, read straight from the page instead of through Table.extract(): with the
    layout hook active find_tables() disables quad corrections, which corrupts spacing inside cells
    ("2.5-5 mg" -> "2 .5 -5 mg", "maximal cumulative dose" -> "maximalcumulativedose"). find_tables()
    restores the setting before returning, so extracting here is safe.
    Footnote markers are dropped by font: spans flagged superscript, or of one or two characters well
    below the cell's median size ("Agent^a", "Beta-blockers^b"; the length guard keeps small caps,
    whose lowercase letters are just as small). Reference numbers glued to a word (".458-460") go
    through REF_MARKER.
    """
    if not cell:
        return ""
    lines = [[s for i, s in enumerate(l["spans"]) if not is_superscript(s, l["spans"][i - 1] if i else None)]
             for b in page.get_text("dict", clip=pymupdf.Rect(cell))["blocks"] for l in b.get("lines", [])]
    sizes = sorted(s["size"] for l in lines for s in l if s["text"].strip())
    if not sizes:
        return ""
    median = sizes[len(sizes) // 2]
    # whitespace-only spans are the word separators between font runs: always kept
    text = " ".join("".join(s["text"] for s in l if s["size"] >= SMALL_FONT * median or len(s["text"].strip()) not in (1, 2)) for l in lines)
    return " ".join(REF_MARKER.sub("", text).split())


def row_cells(page, row, span: dict) -> list[str]:
    """
    One row's cell texts. A cell spanning several rows (one contraindication for a whole drug group)
    is reported once, with a rect reaching down past its row, and as None on the rows below: those
    rows inherit its text. span keeps the last real cell per column across the table's rows.
    """
    bottom = pymupdf.Rect(row.bbox).y1
    cells = []
    for j, c in enumerate(row.cells):
        if c is None:
            prev = span.get(j)
            cells.append(prev[1] if prev and prev[0].y1 >= bottom - 1 else "")
        else:
            span[j] = (pymupdf.Rect(c), cell_text(page, c))
            cells.append(span[j][1])
    return cells


def body_text_share(page, rect: pymupdf.Rect, table_chars: int) -> float:
    """
    Share of a table's text that the layout model labelled running text rather than table. Only
    available with the layout hook: find_tables() leaves its page segmentation on the page.
    """
    boxes = getattr(page, "layout_information", None) or []
    body = sum(len(page.get_text("text", clip=pymupdf.Rect(b[:4]) & rect)) for b in boxes if b[-1] == "text")
    return body / table_chars if table_chars else 0.0


def text_style(spans: list[dict]) -> tuple | None:
    """(font, colour, size, bold) most of the spans' characters share, or None when there is no text."""
    c = Counter()
    for s in spans:
        if n := len(s["text"].strip()):
            c[(s["font"], s["color"], round(s["size"], 1), bool(s["flags"] & 16))] += n
    return c.most_common(1)[0][0] if c else None


def fill_under(rect: pymupdf.Rect, fills: list) -> str | None:
    """Colour of the smallest filled area covering most of rect, None on the page background."""
    best = None
    for f, color in fills:
        i = rect & f
        if not i.is_empty and i.get_area() > 0.6 * rect.get_area() and (best is None or f.get_area() < best[0].get_area()):
            best = (f, color)
    return "%02x%02x%02x" % tuple(round(x * 255) for x in best[1][:3]) if best else None


def horizontal_rules(drawings: list) -> list[tuple]:
    """Horizontal rules as (x0, y, x1): stroked lines and hairline rectangles in a visible colour. A rule
    some typesetters draw column by column is joined back into one."""
    segs = []
    for d in drawings:
        color = d.get("color") or d.get("fill")
        if not color or all(x > 0.97 for x in color[:3]):
            continue
        for it in d["items"]:
            if it[0] == "l" and abs(it[1].y - it[2].y) < 1:
                segs.append((min(it[1].x, it[2].x), (it[1].y + it[2].y) / 2, max(it[1].x, it[2].x)))
            elif it[0] == "re" and it[1].height <= 2:
                segs.append((it[1].x0, (it[1].y0 + it[1].y1) / 2, it[1].x1))
    rules, group = [], []
    for seg in sorted(segs, key=lambda s: s[1]) + [(0, float("inf"), 0)]:
        if group and seg[1] - group[0][1] >= 1:
            for x0, y, x1 in sorted(group):
                if rules and rules[-1][1] == group[0][1] and x0 <= rules[-1][2] + 3:
                    rules[-1] = (rules[-1][0], rules[-1][1], max(rules[-1][2], x1))
                else:
                    rules.append((x0, group[0][1], x1))
            group = []
        group.append(seg)
    return rules


def page_model(page, boxes: list) -> dict:
    """What table_layout reads a table from: the page's text lines with their style (superscripts dropped,
    rotated lines skipped), filled areas, horizontal rules, the layout model's labels, the body style,
    and the boxes of all the tables the finder found."""
    lines = []
    for b in page.get_text("dict")["blocks"]:
        for l in b.get("lines", []):
            spans = [s for i, s in enumerate(l["spans"]) if not is_superscript(s, l["spans"][i - 1] if i else None)]
            text = "".join(s["text"] for s in spans).strip()
            if text and abs(l["dir"][1]) < 0.1:
                # height from baseline and size: some fonts report boxes several lines tall
                ink = [s for s in spans if s["text"].strip()]
                rect = pymupdf.Rect(min(s["bbox"][0] for s in ink), min(s["origin"][1] - 0.8 * s["size"] for s in ink),
                                    max(s["bbox"][2] for s in ink), max(s["origin"][1] + 0.25 * s["size"] for s in ink))
                lines.append({"rect": rect, "text": text, "spans": spans})
    drawings = page.get_drawings()
    fills = [(pymupdf.Rect(d["rect"]), d["fill"]) for d in drawings if d.get("fill") and not all(x > 0.97 for x in d["fill"][:3])]
    for l in lines:
        style = text_style(l["spans"])
        l["style"] = style + (fill_under(l["rect"], fills),) if style else None
    body = text_style([s for l in lines for s in l["spans"]])
    return {"lines": lines, "fills": fills, "rules": horizontal_rules(drawings), "body": body + (None,) if body else None,
            "layout": [(pymupdf.Rect(b[:4]), b[-1]) for b in (getattr(page, "layout_information", None) or [])],
            "boxes": [pymupdf.Rect(b) for b in boxes]}


def cell_style(model: dict, cell) -> tuple | None:
    """The style of a cell's text and the fill under it, None for an empty or spanned cell."""
    if not cell:
        return None
    r = pymupdf.Rect(cell)
    spans = [s for l in model["lines"] if l["rect"].intersects(r) for s in l["spans"]
             if r.contains(pymupdf.Point((s["bbox"][0] + s["bbox"][2]) / 2, (s["bbox"][1] + s["bbox"][3]) / 2))]
    style = text_style(spans)
    return style + (fill_under(r, model["fills"]),) if style else None


LEADER = re.compile(r"(?:\.\s?){4,}|…{2,}")                                   # "Introduction . . . . 53"
CITATION = re.compile(r"\b(?:19|20)\d\d\s*;\s*\d+|\bdoi\b|https?://|^\s*\[?\d+[\].]\s+[A-Z]", re.IGNORECASE)  # "2014;93", "[8] Spielmann"
ACRONYM = re.compile(r"[A-Z0-9][\w\-/+]{0,11}")


def junk(rows: list[list[str]]) -> str:
    """What a region laid out like a table but holding no guidance is, or "" for a real table: a table
    of contents (dot leaders), a reference list (citations on many rows and in more than one column:
    references set in page columns; a table cites in one column), an abbreviation list (acronym and expansion pairs, in two or four columns). By shape: the
    page-level detectors in cleaning take a score column for page numbers, grade codes (AII, BIII) or
    gene names for acronyms, and a table's column of citations for a bibliography."""
    texts = [" ".join(c for c in r if c.strip()) for r in rows]
    if sum(bool(LEADER.search(t)) for t in texts) >= 0.4 * len(texts):
        return "a table of contents"
    cited_cols = {j for r in rows for j, c in enumerate(r) if CITATION.search(c)}
    if len(cited_cols) >= 2 and sum(any(CITATION.search(c) for c in r) for r in rows) >= 0.3 * len(rows):
        return "a reference list"

    def pairs(f):
        return len(f) in (2, 4) and all(all(ACRONYM.fullmatch(w) and sum(ch.isupper() for ch in w) >= 2 for w in a.split())
                                        and len(b) > len(a) + 3 and any(ch.islower() for ch in b) for a, b in zip(f[::2], f[1::2]))
    entries = [f for f in ([c.strip() for c in r if c.strip()] for r in rows) if len(f) >= 2]  # one-cell rows: wrapped expansions
    if len(entries) >= 8 and sum(map(pairs, entries)) >= 0.7 * len(entries):
        return "an abbreviation list"
    return ""


def read_found_table(page, tab, model: dict) -> dict:
    """One table the finder found, read by table_layout: description, header, body rows, and the regions
    (box, caption and header rows outside it) the table covers on the page. A row the finder cut into
    fewer cells than the table has columns (a header row read as one cell) is read again line by line
    onto the columns."""
    # each column's x-range: its narrowest cell (a cell spanning columns is wider, a spanned one is None)
    width = max(len(r.cells) for r in tab.rows)
    cols = [min(((r.cells[j][0], r.cells[j][2]) for r in tab.rows if len(r.cells) == width and r.cells[j]),
                key=lambda c: c[1] - c[0], default=None) for j in range(width)]
    rows, span = [], {}
    for r in tab.rows:
        cells = row_cells(page, r, span)
        styles = [cell_style(model, c) for c in r.cells]
        if len(r.cells) < len(cols):
            inside = [l for l in model["lines"] if pymupdf.Rect(r.bbox).contains(l["rect"].tl + (1, 1))]
            cells, styles = [""] * len(cols), [None] * len(cols)
            for l in sorted(inside, key=lambda l: (l["rect"].y0, l["rect"].x0)):
                if (j := column_of(l, cols)) is not None:
                    cells[j], styles[j] = (cells[j] + " " + l["text"]).strip(), styles[j] or l["style"]
        if any(cells):
            rows.append({"cells": cells, "styles": styles, "rect": pymupdf.Rect(r.bbox)})
    rect = pymupdf.Rect(tab.bbox)
    t = read_table(rect, rows, cols, model)
    return {"description": trim_description(t["caption"]), "header": t["header"], "rows": t["rows"],
            "rect": rect, "regions": [rect] + t["regions"], "evident": t["evident"], "ncols": len(cols)}


def upright(page):
    """A page whose text runs vertically (a landscape table printed on a portrait page) as an upright
    landscape copy the rest of the pipeline reads like any page; any other page as it is. Directions
    are read, and the copy made, in unrotated space, where the rest of the pipeline works."""
    rotation = page.rotation
    if rotation:
        page.set_rotation(0)
    ink = Counter()
    for b in page.get_text("dict")["blocks"]:
        for l in b.get("lines", []):
            ink["up" if l["dir"][1] < -0.9 else "down" if l["dir"][1] > 0.9 else "flat"] += sum(len(s["text"].strip()) for s in l["spans"])
    turn = max(("up", "down"), key=ink.__getitem__)
    if ink[turn] > ink["flat"]:
        copy = pymupdf.open().new_page(width=page.rect.height, height=page.rect.width)
        copy.show_pdf_page(copy.rect, page.parent, page.number, rotate=-90 if turn == "up" else 90)
    else:
        copy = page
    if rotation:
        page.set_rotation(rotation)
    return copy


def extract_tables(page, prev_cols: int | None = None, name: str = "") -> tuple[list[dict], list[pymupdf.Rect]]:
    """
    Run PyMuPDF's table finder on a page. Returns the tables worth keeping and the rectangles (tables,
    and the captions and header rows found outside the finder's box) that must be subtracted from the
    prose stream. Caption and header are read by table_layout. A region holding no guidance (a table of
    contents, an abbreviation list) is dropped with its text. A single row or column (a box) and a reference
    list are left to the prose pipeline, where drop_references removes the references block by block and keeps
    the body text the finder can sweep in beside them (a page set in two columns). A region that is not evidently
    a table (no labelled caption, no header set apart, not the continuation of the table that ended the
    previous page, whose column count is prev_cols) and looks like running text the finder over-reached
    into (a header cell of paragraph length, a paragraph in a leading column, or mostly text the layout
    model labelled as body text) is rejected and left to the prose pipeline. Cells never go through
    line-level cleaning, so dosages inside them stay intact.
    """
    tables, exclude = [], []
    found = page.find_tables().tables
    model = page_model(page, [t.bbox for t in found]) if found else None
    for i, tab in enumerate(found):
        t = read_found_table(page, tab, model)
        description, header, body = t["description"], t["header"], t["rows"]
        where = f"{name} {description or '(no description)'}"
        # a single row or column is a box, not a table: its text stays in the prose stream
        if len(body) < 2 or tab.col_count < 2:
            continue
        # Evidence Tables and study listings are trial data, not guidance, and go with their text
        if EVIDENCE.match(description) or DROP_HEADER.search(" ".join(header).lower()):
            exclude += t["regions"]
            continue
        if kind := junk(body):
            if kind == "a reference list":
                print(f"[rejected table] {where}: {kind}", flush=True)
                continue  # the region stays in the prose stream, for drop_references
            print(f"[dropped table] {where}: {kind}", flush=True)
            exclude += t["regions"]
            continue
        continues = i == 0 and not description and t["ncols"] == prev_cols
        reason = "" if t["evident"] or continues else (
            "header cell over %d chars" % MAX_HEADER_CELL if any(len(h) > MAX_HEADER_CELL for h in header)
            else "paragraph in a leading column" if overlong_cell(body)
            else "mostly body text" if body_text_share(page, t["rect"], sum(len(c) for r in body for c in r)) > MAX_BODY_SHARE
            else ""
        )
        if reason:
            print(f"[rejected table] {where}: {reason}", flush=True)
            continue  # the region stays in the prose stream
        exclude += t["regions"]
        tables.append({"description": description, "header": header, "rows": body, "ncols": t["ncols"]})
    return tables, exclude


def extract_pdf_text_layout_aware(
    pdf_path: Path,
    min_chars_per_block: int = 10,
    header_ratio: float = 0.05,
    footer_ratio: float = 0.05,
    side_margin_ratio: float = 0.05,
) -> list[dict]:
    """
    Extract text from a single PDF using PyMuPDF. Excludes text written in page margins (page numbers, download references, footnotes, ...),
    running heads straddling them (running_head), and reference lists, whole pages of them by is_page_bad and the rest block by block by drop_references. An abbreviation
    list goes block by block too (drop_abbreviations): its page keeps the text and tables beside it.
    Tables are pulled out first and their regions removed from the prose, so they are never shredded into it.
    Returns a list of dicts with keys "doc_id", "page", "text" and "tables".
    """

    doc = pymupdf.open(pdf_path)
    results = []
    doc_id = pdf_path.stem

    empty_page_counter = 0
    prev_cols = None  # column count of the table that ended the previous page, which the next may continue

    for page_idx, page in enumerate(doc):
        page = upright(page)

        # find_tables works in unrotated space and restores page.rotation before returning, so cell
        # rects would not match a rotated page. Normalise first: blocks, tables and clips then agree.
        page_rotation = page.rotation
        if page_rotation:
            page.set_rotation(0)
        bounds = pymupdf.Rect(page.rect)  # in the blocks' space, so margins are measured on the right sides
        page_height = bounds.height
        page_width = bounds.width

        blocks = prose_blocks(page)
        tables, exclude = extract_tables(page, prev_cols, f"{doc_id} p.{page_idx + 1}")
        prev_cols = tables[-1]["ncols"] if tables else None
        # find_tables() ran the layout model on the page; None when it is off
        layout = [(pymupdf.Rect(b[:4]), b[-1]) for b in (getattr(page, "layout_information", None) or [])]

        if page_rotation:
            page.set_rotation(page_rotation)

        textpage = []      # prose only
        textpage_all = []  # prose + table text, used only to judge whether the page is noise

        empty_block_counter = 0
        for block in blocks: 
            # PyMuPDF divides chunk of text inside a pages in blocks and gives their position relative to the page. If it's a useful block, we'll keep it in the page.
            x0, y0, x1, y1, text, *_ = block
            text = text.strip()

            # if the block does not meet minimum characters, skip it
            if len(text) < min_chars_per_block:
                empty_block_counter += 1
                continue

            # if the block is in the header, footer, or side margins, skip it
            if y1 < page_height * header_ratio:
                empty_block_counter += 1
                continue
            if y0 > page_height * (1 - footer_ratio):
                empty_block_counter += 1
                continue
            if x1 < page_width * side_margin_ratio or x0 > page_width * (1 - side_margin_ratio):
                empty_block_counter += 1
                continue
            rect = pymupdf.Rect(x0, y0, x1, y1)
            if running_head(rect, layout, bounds):
                empty_block_counter += 1
                continue

            textpage_all.append(text)
            if any(rect.intersects(r) for r in exclude):
                continue
            textpage.append(text)

        # At this point, we have a page in text form and with block-level checks completed
        if empty_block_counter == len(blocks):
            empty_page_counter += 1

        textpage = drop_abbreviations(drop_references(textpage), "\n".join(textpage_all))
        if textpage or tables:
            # If the page is identified as bibliography or index, ignore it (tables included)
            if textpage_all and is_page_bad("\n".join(textpage_all)):
                continue
            
            results.append({
                "doc_id": doc_id,
                "page": page_idx + 1,
                "text": "\n".join(textpage),
                "tables": tables,
            })

    # At this point, we have all the (hopefully) useful pages for the file and can check how many we have. 
    # If it's too little, something probably went wrong (threshold for warning set at 30% arbitrarily).
    empty_page_warning(doc_id, empty_page_counter, len(doc))

    return results


WORKER_GB = 1.0  # measured peak RSS ~0.63 GB on a 66-page guideline, rounded up for the longest ones


def available_gb() -> float | None:
    """Free memory in GB, or None when it cannot be determined (non-Linux)."""
    try:
        kb = re.search(r"MemAvailable:\s+(\d+)", Path("/proc/meminfo").read_text()).group(1)
        return int(kb) / 1e6
    except Exception:
        return None


def worker_count(n_docs: int, configured: int = INGESTION_WORKERS) -> int:
    """
    How many extraction processes to actually start: never more than the documents to parse, the
    cores present (oversubscribing CPU-bound work only loses time), or what free RAM affords.
    configured=0 means auto, 1 means serial.
    """
    if configured == 1 or n_docs <= 1:
        return 1
    cores = os.cpu_count() or 1
    limit = configured if configured > 0 else cores
    free = available_gb()
    if free is not None:
        limit = min(limit, int(free * 0.8 / WORKER_GB))  # 80%: leave the OS some room
    return max(1, min(limit, cores, n_docs))


def report(pdf_file: Path, pages: list[dict]) -> None:
    n_tables = sum(len(p["tables"]) for p in pages)
    n_described = sum(1 for p in pages for t in p["tables"] if t["description"])
    print(f"Extracted {len(pages)} pages, {n_tables} tables ({n_described} with description) from {pdf_file.name}", flush=True)


def extract_isolated(pdf_path: Path) -> list[dict]:
    """
    One PDF in its own process, so a native crash inside MuPDF loses that process, not the run: the layout
    model segfaults MuPDF's text extraction on some pages (page 2 of OpioidOverdoseFirstAid.pdf, pymupdf and
    pymupdf-layout 1.28.2). A PDF that crashes is parsed again here without the layout model, which only
    costs it the borderless tables that vector-line detection misses.
    """
    try:
        with ProcessPoolExecutor(max_workers=1) as pool:
            return pool.submit(extract_pdf_text_layout_aware, pdf_path).result()
    except BrokenProcessPool:
        print(f"[WARNING] {pdf_path.name} crashed the parser; parsing it again without the layout model", flush=True)
        layout, pymupdf._get_layout = pymupdf._get_layout, None  # the hook find_tables() calls; None switches it off
        try:
            return extract_pdf_text_layout_aware(pdf_path)
        finally:
            pymupdf._get_layout = layout


def extract_folder(
    folder_path: Path
) -> list[dict]:
    """
    Extract text from all PDFs in a folder, one process per PDF. If the pool cannot be started or a worker
    dies (out of memory, or a MuPDF crash on one PDF), the PDFs not yet parsed go one at a time, each in
    its own process, so the one that crashes is found and retried without the layout model.
    """
    pdf_files = sorted(folder_path.glob("*.pdf"))
    workers = worker_count(len(pdf_files))

    start_time = time.time()
    all_pages = []
    done = 0

    if workers > 1:
        print(f"Parsing {len(pdf_files)} PDFs across {workers} processes")
        try:
            with ProcessPoolExecutor(max_workers=workers) as pool:
                # map keeps results in submission order, so output stays identical to the serial path
                for pdf_file, pages in zip(pdf_files, pool.map(extract_pdf_text_layout_aware, pdf_files)):
                    report(pdf_file, pages)
                    all_pages.extend(pages)
                    done += 1
        except (BrokenProcessPool, OSError) as e:
            print(f"[WARNING] parallel extraction failed ({type(e).__name__}: {e}); "
                  f"parsing the remaining {len(pdf_files) - done} PDFs one at a time")
            workers = 1

    if workers == 1:
        for pdf_file in pdf_files[done:]:  # the ones the pool finished are kept, in order
            print(f"Parsing {pdf_file.name}")
            pages = extract_isolated(pdf_file)
            report(pdf_file, pages)
            all_pages.extend(pages)

    end_time = time.time()
    print(f"Extraction took {end_time - start_time:.3f} seconds")
    return all_pages


if __name__ == "__main__":
    import sys
    # a configured count is clamped by docs, cores and RAM; 1 and single-document folders stay serial
    assert worker_count(10, configured=1) == 1 and worker_count(1, configured=8) == 1
    assert worker_count(3, configured=8) == 3, "never more workers than documents"
    assert worker_count(500, configured=10_000) <= (os.cpu_count() or 1), "never more workers than cores"
    assert worker_count(500, configured=0) >= 1
    # running heads: a page-header straddling the 5% line goes, the same label on body text mid-page or a text block
    # at the edge stays
    R, page = pymupdf.Rect, pymupdf.Rect(0, 0, 600, 800)
    head = R(100, 32, 500, 48)
    assert running_head(head, [(R(90, 30, 510, 50), "page-header")], page)
    assert running_head(R(100, 760, 500, 775), [(R(90, 755, 510, 780), "page-footer"), (R(0, 0, 600, 750), "text")], page)
    assert not running_head(R(100, 400, 500, 450), [(R(0, 0, 600, 800), "page-header")], page)
    assert not running_head(head, [(R(90, 30, 510, 50), "text")], page)
    assert not running_head(head, [(R(90, 300, 510, 350), "page-header")], page) and not running_head(head, [], page)
    # captions: sectioned numbers survive, a separator after the number is not a sentence end,
    # a closing bracket is kept, and body text that detection swept in gets cut at the first stop
    assert trim_description("TABLE 4.2 Dosing regimens") == "TABLE 4.2 Dosing regimens"
    assert trim_description("Table 13: List of dosages") == "Table 13: List of dosages"
    assert trim_description("Table 5. Preparations (ie, oral) may vary. Ignore this") == "Table 5. Preparations (ie, oral) may vary"
    assert trim_description("▶ Meta-analyses of stent (SEMS) placement in patients with 2.5 mg doses.") == "▶ Meta-analyses of stent (SEMS) placement in patients with 2.5 mg doses"
    assert trim_description("TABLE 2 | The optimal doses of pet. and liquid preparations. Doses vary") == "TABLE 2 | The optimal doses of pet. and liquid preparations"
    assert trim_description("Table 9 Recommendations. Body text follows here") == "Table 9 Recommendations"
    assert len(trim_description("Table 1 " + "x " * 300)) <= MAX_DESCRIPTION
    # junk regions by shape; score, grade, gene and citation columns do not make one
    assert junk([["Introduction . . . . . . .", "53"], ["Methods . . . . . .", "54"], ["Imaging", "55"]]) == "a table of contents"
    assert junk([[f"{i}. Smith AB, Jones C. A trial. Lancet", f"{i + 9}. Brown CG. Resuscitation 2019;{i}:1-9"] for i in range(5)]
                + [["2019;201:110266. https://doi.org/10.1016/j", "Crit Care 2023;27(1):496."]] * 3) == "a reference list"
    assert junk([[a, e] for a, e in [("AAD", "Antiarrhythmic drugs"), ("ACS", "Acute coronary syndrome"), ("AF", "Atrial fibrillation"),
                ("BP", "Blood pressure"), ("CI", "confidence interval"), ("ECG", "Electrocardiogram"), ("HR", "Hazard ratio"), ("OR", "odds ratio")]]) == "an abbreviation list"
    assert junk([["Congestive heart failure", "1"], ["Hypertension", "1"], ["Age 75 years or above", "2"], ["Diabetes mellitus", "1"]]) == ""
    assert junk([[d, g, "Consider in patients at risk"] for d, g in [("Posaconazole", "AII"), ("Voriconazole", "BIII"), ("Micafungin", "CII")] * 3]) == ""
    assert junk([[g, "Sarcomere", "30%", "Smith et al. 2019;12:1-9"] for g in ["MYH7", "MYBPC3", "TNNT2", "TTN", "LMNA", "FLNC", "DSP", "PKP2"]]) == ""
    pdf = Path(sys.argv[1] if len(sys.argv) > 1 else "Guidelines/AtrialFibrillation.pdf")
    pages = extract_pdf_text_layout_aware(pdf)
    tables = [(p["page"], t) for p in pages for t in p["tables"]]
    print(f"{pdf.name}: {len(pages)} pages, {len(tables)} tables")
    for page_no, t in tables[:3]:
        print(f"\n--- p.{page_no} | {t['description'] or '(no description)'} ---")
        print(t["header"])
        for r in t["rows"][:3]:
            print(r)
