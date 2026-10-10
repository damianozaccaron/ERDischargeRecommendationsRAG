"""
A table's caption and header, read from the page around and inside the box the table finder returns.

The finder's box rarely matches the table: the caption may be its first row or sit above, below or
beside it, and a header row may be inside it or on the line above it. A table is read as a stack, top
to bottom: caption, title rows, header rows, body. Nothing here knows a publisher. A caption is found
by its label ("Table 3", "TA BL E 3", "Tableau 7", "Box 1", "Recommendation 12"); a header by being set
in a style no body row uses, by a rule drawn across the table under it, or failing both by the shape
of the first row.

Input is plain data so the logic can be checked without a PDF (pdf_parsing builds it from the page):
  rows  [{"cells": [str], "styles": [style | None], "rect": Rect}]  the finder's rows, empty ones dropped
  cols  [(x0, x1) | None]                                           column x-ranges
  page  {"lines": [{"rect", "text", "style"}], "rules": [(x0, y, x1)], "layout": [(Rect, cls)], "body": style}
A style is (font, colour, size, bold, fill): the text style most of a cell's characters share and the
fill drawn under the cell.
"""
import re
from collections import Counter

import pymupdf

# "Table 3", "TA BL E 3", "Tableau 7", "Tabela III", "Supplementary Table S2", "▶ Table 2", "Box 1",
# "Recommendation Table 19", "Recommendation 12"
LABEL = re.compile(r"^\s*[^\w\s]?\s*(?:(?:supplementa\w*|online|recommendations?|evidence|summary)\s+)?"
                   r"(?:t\s?a\s?b\s?l\s?e\s?(?:a\s?u)?|tabel+[ae]?|tabla|tab\.|box|panel|recommendation)\s*"
                   r"[A-Z]?(?:\d+(?:[.\-]\d+)*|[IVXL]+\b)", re.IGNORECASE)
BARE_LABEL = re.compile(r"^\s*(?:t\s?a\s?b\s?l\s?e|tableau|tabela|box|panel)\s*$", re.IGNORECASE)  # the title on the next line
LETTER_SPACED = re.compile(r"^(\s*[^\w\s]?\s*)(t\s?a\s?b\s?l\s?e)\b", re.IGNORECASE)
GROUP_MAX = 60        # a single cell longer than this at the top is a title, shorter a group line
LINE_GAP = 4          # most space between two lines of one caption
ABOVE_GAP, BELOW_GAP, SIDE_GAP, HEADING_GAP = 60, 40, 60, 15  # how far from the table a caption may sit
ROWS_ABOVE_MAX = 200  # how far above its box the finder may have left rows of a table


def label(text: str):
    """The caption label opening text, or None. Prose citing a table ("Table 2 presents ...") is not a
    label: a caption goes on with a separator, a capital, a quote or an opening bracket, never a lowercase
    word, nor a closing bracket or comma (a line wrapped inside "(see also Evidence Table 4)")."""
    m = LABEL.match(text) or BARE_LABEL.match(text)
    return m if m and not re.match(r"\s*[a-z),;\]]", text[m.end():]) else None


def filled(cell: str) -> bool:
    return any(ch.isalnum() for ch in cell)


def n_filled(cells: list[str]) -> int:
    return sum(filled(c) for c in cells)


def single(cells: list[str]) -> str | None:
    f = [c for c in cells if filled(c)]
    return f[0] if len(f) == 1 else None


def join_lines(texts: list[str]) -> str:
    """Caption lines as one text: a word hyphenated across lines is rejoined, a spaced-out label closed up."""
    out = ""
    for t in texts:
        t = " ".join(t.replace("\xad", "").split())
        out = out[:-1] + t if out.endswith("-") and t[:1].islower() else (out + " " + t).strip()
    return LETTER_SPACED.sub(lambda m: m.group(1) + m.group(2).replace(" ", ""), out)


# ── the lines around the table ───────────────────────────────────────────────────────────────────

def bands(lines: list[dict]) -> list[dict]:
    """Lines sharing a baseline, top to bottom: a label and its title set apart, the cells of a header row."""
    out = []
    for l in sorted(lines, key=lambda l: (l["rect"].y0, l["rect"].x0)):
        mid = (l["rect"].y0 + l["rect"].y1) / 2
        if out and out[-1]["rect"].y0 <= mid <= out[-1]["rect"].y1:
            out[-1]["lines"].append(l)
            out[-1]["rect"] = out[-1]["rect"] | l["rect"]
        else:
            out.append({"rect": pymupdf.Rect(l["rect"]), "lines": [l]})
    for b in out:
        b["lines"].sort(key=lambda l: l["rect"].x0)
        b["text"] = " ".join(l["text"] for l in b["lines"])
    return out


def band_cells(band: dict, cols: list) -> list[str]:
    """A band as a table row: each line to the column it overlaps most."""
    cells = [""] * len(cols)
    for l in band["lines"]:
        if (j := column_of(l, cols)) is not None:
            cells[j] = (cells[j] + " " + l["text"]).strip()
    return cells


def paragraphs(bs: list[dict], cols: list | None) -> list[list[dict]]:
    """Consecutive bands a line apart, split before every labelled band. A band spread over two or more
    columns of the table is a row, never a caption line."""
    out = []
    for b in bs:
        gap = b["rect"].y0 - out[-1][-1]["rect"].y1 if out else None
        if (out and gap <= max(LINE_GAP, 0.5 * b["rect"].height) and not label(b["text"])
                and (cols is None or n_filled(band_cells(b, cols)) < 2)):
            out[-1].append(b)
        else:
            out.append([b])
    return out


def captions_near(box, page: dict, cols: list) -> list[tuple]:
    """Labelled caption paragraphs above or below the box as (gap, text, rect, side), nearest first: the
    one below must start within BELOW_GAP of the box; the one above may end further off than ABOVE_GAP,
    since rows the finder left out can lie between (read_table checks it against the whole table)."""
    over = [l for l in page["lines"] if l["rect"].x1 > box.x0 and l["rect"].x0 < box.x1]
    found = []
    above = paragraphs(bands([l for l in over if l["rect"].y1 <= box.y0 + 2]), cols)
    below = paragraphs(bands([l for l in over if l["rect"].y0 >= box.y1 - 2]), cols)
    for side, paras, gap_of in (("above", above, lambda p: box.y0 - p[-1]["rect"].y1),
                                ("below", below, lambda p: p[0]["rect"].y0 - box.y1)):
        for p in paras:
            gap = gap_of(p)
            if label(p[0]["text"]) and -2 <= gap <= (ROWS_ABOVE_MAX if side == "above" else BELOW_GAP):
                found.append((gap, join_lines([b["text"] for b in p]), pymupdf.Rect(p[0]["rect"]) | p[-1]["rect"], side))
    return sorted(found, key=lambda f: f[0])


def caption_beside(box, page: dict) -> tuple | None:
    """A labelled caption paragraph in the margin beside the table (text, rect): its near edge within
    SIDE_GAP of the table, opening level with the table's top or closing level with its bottom."""
    for left in (True, False):
        lines = [l for l in page["lines"] if box.y0 - 10 <= l["rect"].y0 <= box.y1
                 and (l["rect"].x1 <= box.x0 + 2 if left else l["rect"].x0 >= box.x1 - 2)]
        for p in paragraphs(bands(lines), None):
            rect = pymupdf.Rect(p[0]["rect"])
            for b in p:
                rect |= b["rect"]
            gap = box.x0 - rect.x1 if left else rect.x0 - box.x1
            if label(p[0]["text"]) and gap <= SIDE_GAP and (abs(rect.y0 - box.y0) <= 10 or abs(rect.y1 - box.y1) <= 10):
                return join_lines([b["text"] for b in p]), rect
    return None


def column_of(line: dict, cols: list) -> int | None:
    ov = [min(line["rect"].x1, c[1]) - max(line["rect"].x0, c[0]) if c else -1 for c in cols]
    j = max(range(len(cols)), key=ov.__getitem__)
    return j if ov[j] > 0 else None


def rows_above(box, top_limit: float, page: dict, cols: list, sizes: set) -> list[dict]:
    """Rows the finder left above its box (a header row, or the first rows of a table it cut short):
    bands walked upward from the box, each a line apart from the one below, none labelled, none running
    across the columns like prose (a label spanning a few columns is short), none in a type size the
    table does not use, none inside another table. A band with an empty first column continues the row
    above it (cells set on two lines). Rows as read_table takes them."""
    def across(l):
        j = column_of(l, cols)
        return j is None or (l["rect"].x0 < cols[j][0] - 3 or l["rect"].x1 > cols[j][1] + 3) and l["rect"].width > 0.4 * box.width

    lines = [l for l in page["lines"] if l["rect"].x1 > box.x0 and l["rect"].x0 < box.x1
             and top_limit - 1 <= l["rect"].y0 and l["rect"].y1 <= box.y0 + 2
             and not any(b != box and b.contains(l["rect"].tl + (1, 1)) for b in page.get("boxes", []))]
    taken, top = [], box.y0
    for b in reversed(bands(lines)):
        if (top - b["rect"].y1 > max(8, 1.5 * b["rect"].height) or label(b["text"])
                or any(map(across, b["lines"]))
                or not all(l["style"] and any(abs(l["style"][2] - z) <= 1 for z in sizes) for l in b["lines"])):
            break
        taken.insert(0, b)
        top = b["rect"].y0
    rows = []
    for b in taken:
        cells, styles = [""] * len(cols), [None] * len(cols)
        for l in b["lines"]:
            if (j := column_of(l, cols)) is not None:
                cells[j], styles[j] = (cells[j] + " " + l["text"]).strip(), styles[j] or l["style"]
        if rows and not filled(cells[0]) and len(cols) > 1:
            last = rows[-1]
            last["cells"] = [(a + " " + c).strip() for a, c in zip(last["cells"], cells)]
            last["styles"] = [a or c for a, c in zip(last["styles"], styles)]
            last["rect"] |= b["rect"]
        elif any(map(filled, cells)):
            rows.append({"cells": cells, "styles": styles, "rect": pymupdf.Rect(b["rect"])})
    return rows


def heading_above(box, top: float, page: dict) -> tuple | None:
    """An unlabelled title right above the table that the layout model calls a caption or a heading
    ("Recommendations for the diagnosis of AF" set as a heading over an ESC table)."""
    bs = bands([l for l in page["lines"] if l["rect"].x1 > box.x0 and l["rect"].x0 < box.x1
                and 0 <= top - l["rect"].y1 <= HEADING_GAP])
    if not bs:
        return None
    b = bs[-1]
    for rect, cls in page["layout"]:
        if cls in ("caption", "section-header", "title") and rect.contains(b["rect"].tl + (1, 1)):
            return b["text"], b["rect"]
    return None


# ── the rows inside the box ──────────────────────────────────────────────────────────────────────

def row_style(row: dict):
    """The style most of a row's cells share, or None when the row mixes styles (cells coloured by value)."""
    styles = [s for c, s in zip(row["cells"], row["styles"]) if c.strip() and s]
    if not styles:
        return None
    common, n = Counter(styles).most_common(1)[0]
    return common if len(styles) == 1 or n >= max(2, 2 * len(styles) / 3) else None


def header_run(rows: list[dict], k: int) -> int:
    """How many rows from k form a header by style: a run set in one style that no row after it is set
    in (row banding repeats its shades) and that is not the prevailing style of the body cells in most of
    its columns (a bold stub column may share the header's style; a continuation row shares the style of
    the body cells under it). One-cell rows ending the run are group lines in the header's style (ESC),
    not header; inside it they are header cells set on several lines."""
    style = row_style(rows[k])
    if style is None:
        return 0
    m = k + 1
    while m < len(rows) - 1 and row_style(rows[m]) == style:
        m += 1
    while m - 1 > k and n_filled(rows[m - 1]["cells"]) < 2:
        m -= 1
    if not any(n_filled(r["cells"]) >= 2 for r in rows[k:m]):
        return 0
    after = [r for r in rows[m:] if n_filled(r["cells"]) >= 2]
    if not after or style in {row_style(r) for r in after}:
        return 0
    apart = []
    for j in {j for r in rows[k:m] for j, c in enumerate(r["cells"]) if c.strip()}:
        col = [r["styles"][j] for r in after if j < len(r["cells"]) and r["cells"][j].strip() and r["styles"][j]]
        if col:
            apart.append(col.count(style) < len(col) / 2)
    return m - k if apart and sum(apart) > len(apart) / 2 else 0


def lower_level(rows: list[dict], k: int, run: int) -> bool:
    """A second header level under the first (a lighter fill under a darker one) reads as labels, and is set
    in a font or on a fill of its own: a plain row the finder left above its box is data."""
    style = row_style(rows[k])
    body = [st for r in rows[k + run:] for c, st in zip(r["cells"], r["styles"]) if c.strip() and st]
    fonts, fills = Counter(st[0] for st in body), {st[4] for st in body}
    return (all(labels(r["cells"]) for r in rows[k:k + run])
            and (not fonts or style[0] != fonts.most_common(1)[0][0] or (style[4] is not None and style[4] not in fills)))


def ruled_header(rows: list[dict], box, rules: list[tuple]) -> int:
    """Rows above the first rule drawn across the table, when rules do not simply separate every row
    (a ruled grid says nothing about the header). 0 when there is no such rule."""
    inner = sorted({round(y) for x0, y, x1 in rules
                    if rows[0]["rect"].y0 + 2 < y < box.y1 - 3 and x1 - x0 >= 0.85 * box.width
                    and min(x1, box.x1) - max(x0, box.x0) >= 0.85 * box.width})
    if not inner or len(inner) >= 0.6 * (len(rows) - 1):
        return 0
    k = sum(r["rect"].y1 <= inner[0] + 2 for r in rows)
    return k if 1 <= k <= 3 and k < len(rows) and any(n_filled(r["cells"]) >= 2 for r in rows[:k]) else 0


NUMERIC = re.compile(r"^[\s\d.,:;%±+\-–<>≥≤=/()\[\]]+$")


def labels(cells: list[str]) -> bool:
    """Every filled cell short, and none a sentence or a number."""
    return all(len(c) <= GROUP_MAX and not c.rstrip().endswith((".", ";")) and not NUMERIC.match(c) for c in cells if filled(c))


def header_like(cells: list[str], below: list[list[str]]) -> bool:
    """With no style or rule to go by, the first row is the header unless it cannot be one: a single cell,
    a sentence or paragraph, a number, or a value its column repeats below (a group label)."""
    if n_filled(cells) < 2 or not labels(cells):
        return False
    return bool(below) and not any(filled(h) and any(j < len(r) and r[j] == h for r in below) for j, h in enumerate(cells))


def read_table(box, rows: list[dict], cols: list, page: dict) -> dict:
    """
    The stack of one table: caption (as written), header (one text per column, empty when the table has
    none, as on a continuation page), body rows, the regions outside the box that belong to the table
    (its caption and rows above it), which the prose must not repeat, and whether it is evidently a
    table (a labelled caption, or a header set apart by style or a rule).
    """
    # 1. title rows: a caption the finder swept in as its first row (labelled, or a single cell too long for
    # a group line), with the rows it wraps onto
    title = []
    while rows:
        txt = " ".join(c for c in rows[0]["cells"] if filled(c))
        if title and single(rows[0]["cells"]) and re.match(r"\s*[a-z]", txt):
            title.append(txt)
        elif not title and (label(txt) or len(single(rows[0]["cells"]) or "") > GROUP_MAX):
            title.append(txt)
        else:
            break
        rows = rows[1:]

    # 2. rows the finder left above its box, below any caption paragraph there (found first, so its
    # lines are never read as rows); a caption above counts only if it sits near the whole table. A box
    # cutting through the caption leaves its last line as a one-cell first row right under it.
    near = captions_near(box, page, cols)
    above = next((c for c in near if c[3] == "above"), None)
    if above and not title and rows and single(rows[0]["cells"]) and rows[0]["rect"].y0 - above[2].y1 <= LINE_GAP + 2:
        above = (above[0], join_lines([above[1], single(rows[0]["cells"])]), above[2] | rows[0]["rect"], "above")
        rows = rows[1:]
    sizes = {st[2] for r in rows for st in r["styles"] if st}
    out = [] if title else rows_above(box, above[2].y1 if above else box.y0 - ROWS_ABOVE_MAX, page, cols, sizes)
    top = out[0]["rect"].y0 if out else box.y0
    if above and top - above[2].y1 > ABOVE_GAP:
        above = None
    below = next((c for c in near if c[3] == "below"), None)
    regions = [r["rect"] for r in out]
    rows = out + rows
    # a one-cell line in the first column set over a header ("Overview of treatments by acne type",
    # "▶ (Continuation)") is a title, not a group line
    while not title and len(rows) > 2 and single(rows[0]["cells"]) and filled(rows[0]["cells"][0]) and header_run(rows, 1):
        title.append(single(rows[0]["cells"]))
        rows = rows[1:]

    # 3. header: a run of rows in a style of their own, else the rows above the first rule across the
    # table, else the first row if nothing rules it out
    k = 0
    while k < len(rows) - 1 and (run := header_run(rows, k)) and (k == 0 or lower_level(rows, k, run)):
        k += run
    if not k and rows:
        k = ruled_header(rows, box, page["rules"])
    set_apart = k > 0  # by style or by a rule, not merely by the shape of the first row
    if not k and rows:
        k = int(header_like(rows[0]["cells"], [r["cells"] for r in rows[1:]]))
    # a header set on two lines comes out as two rows, the second opening lowercase ("Evidence, / consensus")
    if k and k < len(rows) - 1 and n_filled(rows[k]["cells"]) < n_filled(rows[k - 1]["cells"]) \
            and re.match(r"\s*[a-z]", next((c for c in rows[k]["cells"] if filled(c)), "")):
        k += 1
    header = [" ".join(r["cells"][j] for r in rows[:k] if j < len(r["cells"]) and filled(r["cells"][j])) for j in range(len(cols))] if k else []

    # 4. caption: the labelled paragraph nearest the table above or below, else the title rows, else one
    # in the margin beside the table, else a heading right above
    caption, rect, labelled = "", None, True
    if above or below:
        gap_above = top - above[2].y1 if above else float("inf")
        _, caption, rect, _ = above if gap_above <= (below[0] if below else float("inf")) else below
    elif title:
        caption, labelled = join_lines(title), bool(label(title[0]))
    elif h := caption_beside(pymupdf.Rect(box.x0, top, box.x1, box.y1), page):
        caption, rect = h
    elif h := heading_above(box, top, page):
        (caption, rect), labelled = h, False
    else:
        labelled = False
    if rect is not None:
        regions.append(rect)
    # a labelled caption or a header set apart is evidence of a real table, which the checks for a
    # finder that over-reached into running text must not reject
    return {"caption": caption, "header": header, "rows": [r["cells"] for r in rows[k:]], "regions": regions,
            "evident": labelled or set_apart}


if __name__ == "__main__":
    R = pymupdf.Rect
    body, bold, blue = ("Reg", "000000", 8.0, False, None), ("Bold", "000000", 8.0, True, None), ("Bold", "000000", 8.0, True, "b9daf3")

    def row(y, cells, styles):
        return {"cells": cells, "styles": styles, "rect": R(100, y, 500, y + 10)}

    def line(x0, y, text, style=body):
        return {"rect": R(x0, y, x0 + 6 * len(text), y + 8), "text": text, "style": style}

    cols = [(100, 250), (250, 400), (400, 500)]
    empty = {"lines": [], "rules": [], "layout": [], "body": body}
    # labels: spaced out, other languages, a marker glyph; prose citing a table is not one
    assert all(label(t) for t in ["Table 3 Doses", "TA BL E 2 Summary", "Tableau 7 Répartition", "Tabela III. Postępowanie",
                                  "▶ Table 2 Definitions", "Recommendation Table 19 — Recs", "Box", "TABLE 2 | Doses", "Recommendation 12"])
    assert not any(label(t) for t in ["Table 2 presents the summary", "Tables are shown", "Boxes of", "Evidence Table 4)", "Table 3, and"])
    assert join_lines(["TA BL E 2 Summary of elec-", "tive surgery"]) == "TABLE 2 Summary of elective surgery"
    # a caption the finder swept in as the first row (ESGE "▶ ..." with no "Table N"), wrapped over two rows
    rows = [row(10, ["▶ Oncological outcome of self-expandable metal stent placement versus emergency surgery in patients with malignant colonic", "", ""], [body, None, None]),
            row(20, ["obstruction.", "", ""], [body, None, None]),
            row(30, ["First author, year", "Study population", "Results"], [blue] * 3),
            row(40, ["Foo, 2019", "Patients with obstruction", "Higher recurrence"], [body] * 3),
            row(50, ["Yang, 2018", "Patients with obstruction", "No difference"], [body] * 3)]
    t = read_table(R(100, 10, 500, 60), rows, cols, empty)
    assert t["caption"].endswith("malignant colonic obstruction.") and t["header"] == ["First author, year", "Study population", "Results"], t
    assert t["rows"][0][0] == "Foo, 2019" and len(t["rows"]) == 2
    # a header row the finder left above its box (ERC), the caption above that; banded body rows are not a header
    band = ("Reg", "000000", 8.0, False, "ceebe8")
    page = {"lines": [line(100, -30, "Table 8 – Major causes of cardiac arrest.", ("Cap", "000000", 8.0, False, "74ccd4")),
                      line(100, -12, "Cause", bold), line(250, -12, "Mechanism", bold), line(400, -12, "Impact", bold)],
            "rules": [], "layout": [], "body": body}
    rows = [row(0, ["Hypovolaemic shock", "Blood loss", "Low preload"], [band] * 3), row(10, ["Tension pneumothorax", "Pressure", "Low return"], [body] * 3),
            row(20, ["Cardiac tamponade", "Blood", "Low filling"], [band] * 3), row(30, ["Hypoxia", "Airway", "Arrest"], [body] * 3)]
    t = read_table(R(100, 0, 500, 40), rows, cols, page)
    assert t["header"] == ["Cause", "Mechanism", "Impact"] and t["caption"] == "Table 8 – Major causes of cardiac arrest." and len(t["rows"]) == 4, t
    # a box cutting through the caption: its last line is the first row
    page = {**empty, "lines": [line(100, -9, "Table 1", bold)]}
    rows = [row(0, ["Summary of recommendations", "", ""], [body, None, None]), row(10, ["Topic", "Recommendation", "Strength"], [blue] * 3),
            row(20, ["Allergy", "All patients labelled penicillin allergic should be assessed.", "Strong"], [body] * 3),
            row(30, ["Testing", "Nonallergists should perform drug challenge tests.", "Strong"], [body] * 3)]
    t = read_table(R(100, 0, 500, 40), rows, cols, page)
    assert t["caption"] == "Table 1 Summary of recommendations" and t["header"] == ["Topic", "Recommendation", "Strength"], t
    # a row of symbols only under the header is no second header line (and no crash)
    rows = [row(0, ["Agent", "Dose", "Note"], [blue] * 3), row(10, ["•", "", ""], [body, None, None]),
            row(20, ["Apixaban", "5 mg", "twice daily"], [body] * 3), row(30, ["Edoxaban", "60 mg", "once daily"], [body] * 3)]
    assert read_table(R(100, 0, 500, 40), rows, cols, empty)["header"] == ["Agent", "Dose", "Note"]
    # a table the finder cut short: caption, header and first row lie above its box; the caption is too far
    # from the box but near the whole table
    rows = [row(0, ["2", "A weaker or conditional recommendation, where the risks and benefits are balanced"], [band, band]),
            row(10, ["A", "High-quality evidence from well-performed randomized trials"], [band, band]),
            row(20, ["B", "Moderate-quality evidence from randomized trials with flaws"], [body, body])]
    page = {"lines": [line(100, -78, "TABLE 1 Grading of recommendation and the quality of evidence.", ("Cap", "000000", 9.0, False, None)),
                      line(100, -62, "Grade", bold), line(250, -62, "Recommendation", bold),
                      line(100, -50, "1", band), line(250, -50, "A strong recommendation to do (or not do)", band),
                      line(250, -40, "something, where benefits clearly", band), line(250, -30, "outweigh risks for most patients.", band),
                      line(250, -20, "Clinicians would want to follow", band), line(250, -10, "a strong recommendation.", band)],
            "rules": [], "layout": [], "body": body}
    t = read_table(R(100, 0, 500, 30), rows, [(100, 250), (250, 500)], page)
    assert t["caption"].startswith("TABLE 1 Grading") and t["header"] == ["Grade", "Recommendation"], t
    assert t["rows"][0] == ["1", "A strong recommendation to do (or not do) something, where benefits clearly outweigh risks for most patients. "
                                 "Clinicians would want to follow a strong recommendation."] and len(t["rows"]) == 4, t
    # a continuation page: no caption, no header, the first recommendation is body (ESC, rows coloured by class)
    green, teal, grey = ("Reg", "000000", 8.0, False, "58bf8e"), ("Reg", "000000", 8.0, False, "1a9aa4"), ("Reg", "000000", 8.0, False, "ededee")
    rows = [row(0, ["It is recommended to postpone the switch from heparin back to oral anticoagulants.", "I", "C"], [grey, green, teal]),
            row(10, ["DOACs are not recommended during pregnancy.", "III", "C"], [grey, ("Reg", "000000", 8.0, False, "d9534f"), teal]),
            row(20, ["Clinical surveillance is recommended during pregnancy.", "I", "C"], [grey, green, teal])]
    t = read_table(R(100, 0, 500, 30), rows, cols, empty)
    assert t["caption"] == "" and t["header"] == [] and len(t["rows"]) == 3, t
    # a plain table: the rule under the header row decides; a two-line header is glued back
    rows = [row(0, ["No. GoR Evidence,", "Recommendation", "Status"], [body] * 3), row(10, ["consensus", "", "2022"], [body, None, body]),
            row(20, ["3 Key recommendation", "Laparoscopy can be safely performed.", "New"], [body] * 3),
            row(30, ["1 B", "Midline laparotomy should be preferred.", "Confirmed"], [body] * 3)]
    t = read_table(R(100, 0, 500, 40), rows, cols, {**empty, "rules": [(100, 20.5, 500)]})
    assert t["header"] == ["No. GoR Evidence, consensus", "Recommendation", "Status 2022"] and len(t["rows"]) == 2, t
    # the first row is no header when its value repeats below (a group column)
    rows = [row(0, ["Demographic factors", "Age", ""], [body] * 3), row(10, ["Demographic factors", "Male sex", ""], [body] * 3)]
    assert read_table(R(100, 0, 500, 20), rows, cols, empty)["header"] == []
    # a caption beside the table, level with its top (Karger, Springer, Wiley margin captions)
    page = {**empty, "lines": [line(20, 0, "Table 2. Different HDF regimens")]}
    rows = [row(0, ["", "Convection", "Efficacy"], [None, blue, blue]), row(10, ["Low-volume", "++", "+"], [body] * 3), row(20, ["High-volume", "+++", "++"], [body] * 3)]
    t = read_table(R(220, 0, 500, 30), rows, [(220, 300), (300, 400), (400, 500)], page)
    assert t["caption"] == "Table 2. Different HDF regimens" and t["header"] == ["", "Convection", "Efficacy"], t
    print("ok")
