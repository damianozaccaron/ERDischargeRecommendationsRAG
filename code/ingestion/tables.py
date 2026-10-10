"""
Guideline tables, from the records pdf_parsing extracts to what the index stores.

A table is structured by shape alone (no publisher vocabulary): group headers become a row
attribute, broken rows are merged, and cell shape decides where the table goes. Sentence-type
tables (recommendations) become text chunks for the prose index; grid tables (doses, criteria,
scores) are serialized as markdown plus short surrogates the table retriever matches against.
"""
import re
import unicodedata

from ingestion.cleaning import REF_MARKER

LONG_CELL = 80
GROUP_MAX = 60
MAX_HEADER_CELL = 120     # rejection thresholds, shared with pdf_parsing
MAX_DATA_CELL = 400       # applies to every column but the last (last two from 5 columns), where long text legitimately sits
PART_MAX = 4000           # markdown characters before a grid table is split at group boundaries
SURROGATE_MAX = 400
UNIT = re.compile(r"(?<![a-z])(?:mg|µg|μg|mcg|g|kg|ml|l|iu|u|mmol|meq|daily|hourly|weekly|monthly|once|twice|bid|tid|qid)(?![a-z])"
                  r"|/(?:kg|h|min|day)(?![a-z])", re.IGNORECASE)
STAT = re.compile(r"(?<![A-Za-z])(?:IQR|CI|SD|SE|OR|HR|RR|n|N|p)(?![A-Za-z])")
NUMERIC_NOISE = re.compile(r"[\d.,±%()\[\]\-–≥≤<>=/]")
DROP_HEADER = re.compile(r"(?<![a-z])(?:first author|ref\.|study type|population|rct)(?![a-z])")
DROP_CAPTION = re.compile(r"summary of studies|(?<![a-z])epidemiolog|evidence table")
# a cell that only lists cited studies ("Parkinson et al. (2015), D'Oria et al. (2022)"): evidence, not guidance
CITED = re.compile(r"(?:\s*[A-Z][^()]{0,60}?\((?:19|20)\d\d[a-z]?\)\s*[,;]?)+\s*")


def filled(cell: str) -> bool:
    return any(ch.isalnum() for ch in cell)


def overlong_cell(rows: list[list[str]]) -> bool:
    """A paragraph in a leading column: the finder swallowed running text. The trailing column(s) are exempt."""
    exempt = 2 if rows and len(rows[0]) >= 5 else 1
    return any(len(c) > MAX_DATA_CELL for r in rows for c in r[:-exempt])


def prose_column(cells: list[str]) -> bool:
    """Most filled cells of the column are paragraphs or lines cut mid-sentence (starting lowercase)."""
    cells = [c for c in cells if filled(c)]
    return bool(cells) and sum(len(c) > LONG_CELL or bool(re.match(r"\s*[a-z]", c)) for c in cells) > len(cells) / 2


def is_numeric(cell: str) -> bool:
    """A value with no unit and no word left once numbers, brackets, comparison signs and statistics are removed."""
    if not filled(cell) or UNIT.search(cell):
        return False
    rest = NUMERIC_NOISE.sub(" ", STAT.sub(" ", cell))
    return not re.search(r"[A-Za-z]{3,}", rest)


def is_group_header(cell: str) -> bool:
    c = cell.strip()
    return len(c) <= GROUP_MAX and not re.match(r"[\d(a-z]", c) and not c.endswith((".", ",", ";"))


def normalize(text: str) -> list[str]:
    """
    Tokens for word matching, applied identically to query terms and surrogates: lowercase, no
    diacritics, beta spelled out, hyphenated or Greek-prefixed words glued ("beta-blocker",
    "beta blocker", "betablocker" all give betablocker), plurals dropped on tokens over 4 chars.
    """
    s = text.lower().replace("β", "beta")
    s = "".join(ch for ch in unicodedata.normalize("NFKD", s) if not unicodedata.combining(ch))
    s = re.sub(r"(?<=[a-z])[-‐‑–](?=[a-z])", "", s)
    s = re.sub(r"\b(alpha|beta|gamma|delta) (?=[a-z])", r"\1", s)
    out = []
    for t in re.findall(r"[a-z0-9]+", s):
        if len(t) > 4 and t.endswith("s"):
            t = t[:-2] if re.search(r"(ss|x|z|ch|sh)es$", t) else t[:-1]
        out.append(t)
    return out


# ── Phase 1: from extracted records to structured tables ──────────────────────────────────────────

def merge_continuations(pages: list[dict]) -> list[dict]:
    """
    One record per table across page breaks: a continuation is the first table of the next page,
    uncaptioned (or "continued"), with the parent's column count. Phantom columns are dropped.
    """
    ncols = lambda t: max([len(t["header"])] + [len(r) for r in t["rows"]])
    tables = []
    for page in pages:
        for first_on_page, t in enumerate(page.get("tables", [])):
            prev = tables[-1] if tables else None
            continued = (
                prev is not None
                and prev["doc_id"] == page["doc_id"]
                and page["page"] == prev["page_end"] + 1
                and first_on_page == 0
                and ncols(t) == ncols(prev)
                and (not t["description"] or "continued" in t["description"].lower())
            )
            if continued:
                # a header repeated on the continuation page is dropped; one that differs was a data row
                if t["header"] != prev["header"] and any(t["header"]):
                    prev["rows"].append(t["header"])
                prev["rows"].extend(t["rows"])
                prev["page_end"] = page["page"]
                continue
            tables.append({"doc_id": page["doc_id"], "page_start": page["page"], "page_end": page["page"],
                           "caption_raw": t["description"], "header": t["header"], "rows": list(t["rows"])})
    for t in tables:
        header, rows = t["header"], t["rows"]
        width = max(len(header), *(len(r) for r in rows))
        header = header + [""] * (width - len(header))
        rows = [r + [""] * (width - len(r)) for r in rows]
        keep = [j for j in range(width) if filled(header[j]) or any(filled(r[j]) for r in rows)]
        # a headerless column past the last headed one, holding paragraphs, is running text the finder
        # swept in beside the table (a prose column next to a recommendation table), not a column
        last_headed = max((j for j in keep if filled(header[j])), default=-1)
        keep = [j for j in keep if last_headed < 0 or j <= last_headed or not prose_column([r[j] for r in rows])]
        t["header"] = [header[j] for j in keep]
        t["rows"] = [[r[j] for j in keep] for r in rows]
    return [t for t in tables if t["header"] and t["rows"]]  # no column survives cleaning (marker-only cells): not a table


def key_column(header: list[str], rows: list[list[str]]) -> int:
    """
    The leftmost column with a header cell, unless a column before it is a label column: a headerless
    corner column filled on most rows with short cells ("Minor stress", "Moderate and major stress").
    A sparse column of section lines is not one. If every header cell is empty and no label column
    exists, the most filled column.
    """
    for j, h in enumerate(header):
        cells = [r[j] for r in rows if filled(r[j])]
        if filled(h) or (len(cells) >= len(rows) / 2 and sum(len(c) <= GROUP_MAX for c in cells) > len(cells) / 2):
            return j
    return max(range(len(header)), key=lambda j: sum(filled(r[j]) for r in rows))


def structure_rows(rows: list[list[str]], key: int) -> list[dict]:
    """
    Step 4: group headers become a row attribute, a row with an empty key cell (or one starting
    lowercase: a label the finder cut mid-phrase) is glued to the row above, a trailing single key
    cell (a footnote the finder swept in) is dropped. The first row, when its key cell is empty, has no
    row above: kept when it is one long cell (a statement set above the row of its grades, as in a
    recommendation box), dropped otherwise. Returns [{"cells": [...], "group": str}].
    """
    out, group = [], ""
    first = next((i for i, r in enumerate(rows) if any(filled(c) for c in r)), None)
    for i, r in enumerate(rows):
        cols = [j for j, c in enumerate(r) if filled(c)]
        if not cols:
            continue
        if len(cols) == 1 and is_group_header(r[cols[0]]) and any(filled(x[key]) for x in rows[i + 1:]):
            group = r[cols[0]].strip()
            continue
        if not filled(r[key]) or (out and re.match(r"\s*[a-z]", r[key])):  # ASCII only: "β-blockers" opens a row
            if out:
                for j in cols:
                    out[-1]["cells"][j] = (out[-1]["cells"][j] + " " + r[j]).strip()
                continue
            if i != first or len(cols) > 1 or len(r[cols[0]]) <= LONG_CELL:
                continue
        if len(cols) == 1 and i == len(rows) - 1 and out:
            continue
        out.append({"cells": list(r), "group": group})
    return out


def classify(header: list[str], rows: list[dict], caption: str, key: int) -> str:
    """
    Step 5, first match wins: drop (study listings), sentence (one long cell per row), numeric, grid.
    The long cell of a sentence row is its key cell (or sits under an empty header): a recommendation
    is the row's entity, whereas a dosing table keeps a short drug name in the key column and its
    long criteria cell elsewhere.
    """
    if DROP_HEADER.search(" ".join(header).lower()) or DROP_CAPTION.search(caption.lower()):
        return "drop"
    if not rows or any(len(h) > MAX_HEADER_CELL for h in header):
        return "drop"
    sentence = sum(1 for r in rows
                   if [j for j, c in enumerate(r["cells"]) if len(c) > LONG_CELL] in [[key]] + [[j] for j, h in enumerate(header) if not filled(h)])
    if sentence >= 0.6 * len(rows):
        return "sentence"
    cells = [c for r in rows for c in r["cells"] if filled(c)]
    if sum(is_numeric(c) for c in cells) >= 0.5 * len(cells):
        return "numeric"
    return "grid"


def clean_caption(raw: str, header: list[str]) -> str:
    s = re.sub(r"\s*\(part \d+/\d+\)\s*$", "", raw)
    while True:
        t = re.sub(r"\s*\([^()]*\)\s*$", "", s)
        if t == s:
            break
        s = t
    s = " ".join(REF_MARKER.sub("", s).split())
    return s or " ".join(h for h in header if filled(h))


SAME_AS_ABOVE = 40  # a cell longer than this repeating the one above is a cell spanning rows: shown once


def markdown(header: list[str], rows: list[dict]) -> str:
    grouped = any(r["group"] for r in rows)
    lines = [(["Group"] if grouped else []) + header]
    lines.append(["---"] * len(lines[0]))
    prev = None
    for r in rows:
        cells = [c.replace("|", "/") for c in r["cells"]]
        if prev:
            cells = ["same as above" if len(c) > SAME_AS_ABOVE and c == p else c for c, p in zip(cells, prev)]
        lines.append(([r["group"]] if grouped else []) + cells)
        prev = [c.replace("|", "/") for c in r["cells"]]
    return "\n".join("| " + " | ".join(l) + " |" for l in lines)


def split_parts(header: list[str], rows: list[dict]) -> list[list[dict]]:
    """Step 7: whole if it fits; else parts cut where the group changes, a group too big on its own cut between rows."""
    size = lambda rs: len(markdown(header, rs))
    if size(rows) <= PART_MAX:
        return [rows]
    groups, parts = [], []
    for r in rows:
        if groups and groups[-1][0]["group"] == r["group"]:
            groups[-1].append(r)
        else:
            groups.append([r])
    for g in groups:
        if parts and size(parts[-1] + g) <= PART_MAX:
            parts[-1] += g
        else:
            parts.append(list(g))
    out = []
    for p in parts:
        cur = []
        for r in p:
            if cur and size(cur + [r]) > PART_MAX:
                out.append(cur)
                cur = []
            cur.append(r)
        out.append(cur)
    return out


def serialize(t: dict, rows: list[dict], table_id: str, part: int = 1, parts: int = 1) -> dict:
    """Step 8: markdown, table surrogate, row surrogates and the token sets word matching runs on."""
    header, caption = t["header"], t["caption_clean"]
    key = t["key"]
    groups = list(dict.fromkeys(r["group"] for r in rows if r["group"]))
    surrogate = ". ".join(filter(None, [caption, ", ".join(filter(filled, header)), ", ".join(groups),
                                        ", ".join(r["cells"][key] for r in rows)]))
    out_rows = []
    for r in rows:
        pairs = "; ".join(f"{h}: {c}" for h, c in zip(header, r["cells"]) if filled(c))
        other = [c for j, c in enumerate(r["cells"]) if j != key]
        out_rows.append({
            "cells": r["cells"], "group": r["group"],
            "row_surrogate": f"{caption} | {r['group']} | {pairs}"[:SURROGATE_MAX],
            "key_tokens": set(normalize(r["cells"][key])),
            "group_tokens": set(normalize(r["group"])),
            "other_tokens": set(normalize(" ".join(other))),
        })
    return {
        "table_id": table_id, "part": part, "parts": parts,
        "doc_id": t["doc_id"], "page_start": t["page_start"], "page_end": t["page_end"],
        "caption_raw": t["caption_raw"], "caption_clean": caption, "header": header,
        "header_tokens": set(normalize(" ".join(header))),
        "markdown": markdown(header, rows), "table_surrogate": surrogate, "embed_text": surrogate,
        "rows": out_rows,
    }


def sentence_chunks(t: dict, rows: list[dict], table_id: str, max_chars: int = 1200) -> list[dict]:
    """
    Phase 3 step 1: rows packed into text chunks of about max_chars, a row never cut, the caption once
    per chunk and the group line where it changes. Each row is its long cell with the short cells in
    brackets: "Diuretics are recommended ... (I, C)". A cell that only lists cited studies is left out.
    """
    lines = []
    for r in rows:
        cells = [c for c in r["cells"] if filled(c) and not CITED.fullmatch(c)]
        if not cells:
            continue
        long = max(cells, key=len)
        other = [c for c in cells if c is not long]
        lines.append((r["group"], long + (f" ({', '.join(other)})" if other else "")))
    chunks, cur, start, group = [], [], 0, None
    def flush(end):
        chunks.append({"doc_id": t["doc_id"], "page_start": t["page_start"], "page_end": t["page_end"], "kind": "sentence_row",
                       "table_id": table_id, "row": f"{start}-{end}", "text": (f"{t['caption_clean']}. " if t["caption_clean"] else "") + " ".join(cur)})
    for i, (g, line) in enumerate(lines):
        if cur and sum(map(len, cur)) + len(line) > max_chars:
            flush(i - 1)
            cur, start, group = [], i, None
        if g and g != group:
            cur.append(f"{g}:")
        group = g
        cur.append(line)
    if cur:
        flush(len(lines) - 1)
    return chunks


def structure(pages: list[dict]) -> list[dict]:
    """Phase 1 steps 4-6 on every extracted table: adds key, rows (structured), caption_clean and kind."""
    tables = merge_continuations(pages)
    for t in tables:
        t["key"] = key_column(t["header"], t["rows"])
        t["rows"] = structure_rows(t["rows"], t["key"])
        t["caption_clean"] = clean_caption(t["caption_raw"], t["header"])
        t["kind"] = classify(t["header"], t["rows"], t["caption_clean"], t["key"])  # clean: "(see also Evidence Table 5)" is not an Evidence Table
    return tables


def build_tables(pages: list[dict]) -> tuple[list[dict], list[dict], list[dict]]:
    """
    Returns (sentence-row chunks for the text index, grid table records for the table store, every
    structured table with its kind). Table ids number tables per guideline; the parts of a split grid
    table share its id and carry part/parts.
    """
    chunks, grids, seq = [], [], {}
    tables = structure(pages)
    for t in tables:
        next_id = lambda: f"{t['doc_id']}:{seq.__setitem__(t['doc_id'], seq.get(t['doc_id'], 0) + 1) or seq[t['doc_id']]}"
        if t["kind"] == "sentence":
            chunks += sentence_chunks(t, t["rows"], next_id())
        elif t["kind"] == "grid":
            parts, table_id = split_parts(t["header"], t["rows"]), next_id()
            for k, part in enumerate(parts, 1):
                grids.append(serialize(t, part, table_id, k, len(parts)))
    return chunks, grids, tables


if __name__ == "__main__":
    # normalizer: hyphen and Greek-prefix gluing, plurals, diacritics
    assert normalize("β-blockers") == normalize("Beta blockers") == normalize("betablocker") == ["betablocker"]
    assert normalize("Non-dihydropyridine calcium channel antagonists") == ["nondihydropyridine", "calcium", "channel", "antagonist"]
    assert normalize("Metoprolol XL (succinate)") == ["metoprolol", "xl", "succinate"]
    assert normalize("doses classes boxes") == ["dose", "class", "box"] and normalize("Dose") == ["dose"]
    # cell shapes
    assert all(map(is_numeric, ["62.4±15.6", "19 (IQR 11–35)", "2013–2017", "52/210", "I"]))
    assert not any(map(is_numeric, ["25–100 mg twice daily", "N/A mg", "Nonaggregated data 65.8", "IIa"]))
    assert all(map(is_group_header, ["Beta-blockers", "Non-dihydropyridine calcium channel antagonists", "Specific disease"]))
    assert not any(map(is_group_header, ["(twice daily)", "3 days)", "hospitalisation costs", "Adapted with permission from … [8]."]))
    assert overlong_cell([["x" * 500, "a", "b"]]) and not overlong_cell([["a", "b", "x" * 500]])
    assert overlong_cell([["a", "b", "c", "x" * 500, "e"]]) is False and overlong_cell([["a", "b", "x" * 500, "d", "e"]])
    # a spanned cell repeats down the rows in the store but is shown once in the markdown
    md = markdown(["Agent", "Contraindicated"], [{"cells": ["A", "C" * 50], "group": ""}, {"cells": ["B", "C" * 50], "group": ""}, {"cells": ["D", "N/A"], "group": ""}, {"cells": ["E", "N/A"], "group": ""}])
    assert "| B | same as above |" in md and "| E | N/A |" in md, md
    # a headerless prose column swept in beside a recommendation table is dropped; a headerless leading column is not
    _, _, ts = build_tables([{"doc_id": "D", "page": 1, "tables": [{"description": "Rec Table 1", "header": ["Recommendations", "Class", ""],
                              "rows": [["R" * 150, "I", "p" * 300], ["S" * 150, "IIa", "q" * 300]]}]}])
    assert ts[0]["header"] == ["Recommendations", "Class"] and ts[0]["kind"] == "sentence", ts[0]
    # a continuation page with no header joins its parent; one repeating the parent's header drops it
    cont = build_tables([{"doc_id": "D", "page": 1, "tables": [{"description": "Table 3 Doses", "header": ["Drug", "Dose"], "rows": [["A", "1 mg"], ["B", "2 mg"]]}]},
                         {"doc_id": "D", "page": 2, "tables": [{"description": "", "header": [], "rows": [["C", "3 mg"], ["D", "4 mg"]]}]},
                         {"doc_id": "D", "page": 3, "tables": [{"description": "Table 3 (continued)", "header": ["Drug", "Dose"], "rows": [["E", "5 mg"], ["F", "6 mg"]]}]}])[2]
    assert len(cont) == 1 and [r["cells"][0] for r in cont[0]["rows"]] == list("ABCDEF") and cont[0]["page_end"] == 3, cont
    # a table with no surviving cells is discarded before structuring
    assert build_tables([{"doc_id": "D", "page": 1, "tables": [{"description": "", "header": ["", ""], "rows": [["", "-"], ["", ""]]}]}])[2] == []
    # key column: a headerless label column wins over the first headed column; long headerless text does not
    assert key_column(["", "General considerations", "Examples"], [["Minor stress", "a", "b"], ["", "c", "d"], ["Moderate", "e", "f"]]) == 0
    assert key_column(["", "Class", "Level"], [["r" * 120, "I", "A"], ["s" * 90, "IIa", "B"]]) == 1
    assert key_column(["", "", "Class"], [["r" * 120, "", "I"], ["", "Section 4.", ""], ["s" * 90, "", "IIa"], ["t" * 90, "", "B"]]) == 2
    assert structure_rows([["Moderate", "x", ""], ["and major", "y", ""], ["stress", "", "z"]], 0) == [{"cells": ["Moderate and major stress", "x y", "z"], "group": ""}]
    # a recommendation box: the statement set alone above the row of its grades is kept, the cited studies are no
    # guidance; a short leading cell with no row above is still dropped
    cited = "Parkinson et al. (2015), D’Oria et al. (2022), Lancaster et al. (2022), Lindholt et al. (2000), Scott et al. (2016)"
    box = [["S" * 100, "", ""], ["I", "C", cited]]
    assert [r["cells"][0] for r in structure_rows(box, 1)] == ["S" * 100, "I"]
    assert structure_rows([["short", "", ""], ["I", "C", cited]], 1) == [{"cells": ["I", "C", cited], "group": ""}]
    rec = build_tables([{"doc_id": "D", "page": 1, "tables": [{"description": "Recommendation 6 Changed", "header": [], "rows": box}]}])[0]
    assert [c["text"] for c in rec] == ["Recommendation 6 Changed. " + "S" * 100 + " I (C)"], rec
    # row structure: group header, continuation glued to the row above, trailing footnote dropped
    rows = [["Beta-blockers", "", ""], ["Metoprolol", "2.5 mg", "yes"], ["", "5 mg", ""], ["Other", "", ""],
            ["Amiodarone", "300 mg", ""], ["Adapted from [8].", "", ""]]
    s = structure_rows(rows, 0)
    assert [r["group"] for r in s] == ["Beta-blockers", "Other"], s
    assert s[0]["cells"] == ["Metoprolol", "2.5 mg 5 mg", "yes"] and s[1]["cells"][0] == "Amiodarone"
    # classification and serialization end to end
    pages = [{"doc_id": "D", "page": 3, "tables": [
        {"description": "Table 12 Drugs for rate control (see also Evidence Table 3)", "header": ["Agent", "Oral dose", "Note"], "rows": rows[:5]},
        {"description": "Recommendation Table 1 Recs", "header": ["Recommendations", "Class"], "rows": [["X" * 100, "I"], ["β" + "y" * 90, "IIa"]]},
        {"description": "TABLE 1.1 Summary of epidemiological studies", "header": ["First author [ref.], year", "n"], "rows": [["A", "1"], ["B", "2"]]},
        {"description": "Table 9 Numbers", "header": ["Score", "Rate"], "rows": [["1", "2.5%"], ["2", "3 (1–5)"]]},
    ]}]
    pages[0]["tables"].append({"description": "Table 11 DOAC doses", "header": ["DOAC", "Dose", "Criteria"],
                               "rows": [["Apixaban", "5 mg", "c" * 100], ["Edoxaban", "60 mg", "c" * 90]]})
    # headerless, uncaptioned (a continuation whose parent is not on the previous page): the section line is a group
    pages[0]["tables"].append({"description": "", "header": [],
                               "rows": [["Recommendations for AF patients undergoing PCI", "", ""], ["R" * 150, "I", "A"], ["S" * 120, "IIa", "B"]]})
    chunks, grids, tables = build_tables(pages)
    assert [t["kind"] for t in tables] == ["grid", "sentence", "drop", "numeric", "grid", "sentence"], [t["kind"] for t in tables]
    assert chunks[-1]["text"] == "Recommendations for AF patients undergoing PCI: " + "R" * 150 + " (I, A) " + "S" * 120 + " (IIa, B)", chunks[-1]["text"]
    assert len(chunks) == 2 and chunks[0]["text"] == "Recommendation Table 1 Recs. " + "X" * 100 + " (I) β" + "y" * 90 + " (IIa)" and chunks[0]["kind"] == "sentence_row"
    # packing: rows fill chunks up to max_chars, a row is never cut, the group line opens its run
    packed = sentence_chunks({"doc_id": "D", "page_start": 1, "page_end": 1, "caption_clean": "Cap"},
                             [{"cells": ["A" * 500, "I"], "group": "S1"}, {"cells": ["B" * 500, "I"], "group": "S1"}, {"cells": ["C" * 500, "I"], "group": "S2"}], "D:9")
    assert [c["row"] for c in packed] == ["0-1", "2-2"] and packed[0]["text"].startswith("Cap. S1: AAA") and packed[1]["text"].startswith("Cap. S2: CCC"), packed
    g = grids[0]
    assert g["caption_clean"] == "Table 12 Drugs for rate control" and g["table_id"] == "D:1"
    assert g["markdown"].splitlines()[0] == "| Group | Agent | Oral dose | Note |" and "| Other | Amiodarone | 300 mg |  |" in g["markdown"]
    assert g["table_surrogate"] == "Table 12 Drugs for rate control. Agent, Oral dose, Note. Beta-blockers, Other. Metoprolol, Amiodarone"
    assert g["rows"][0]["row_surrogate"] == "Table 12 Drugs for rate control | Beta-blockers | Agent: Metoprolol; Oral dose: 2.5 mg 5 mg; Note: yes"
    assert "betablocker" in g["rows"][0]["group_tokens"] and "metoprolol" in g["rows"][0]["key_tokens"] and "mg" in g["rows"][0]["other_tokens"]
    # splitting: parts cut at group boundaries, every part keeps caption and header
    big = {"doc_id": "D", "page": 1, "tables": [{"description": "Table 2 Big", "header": ["A", "B", "C"],
           "rows": [["G1", "", ""]] + [[f"A{i}", f"{i}" + "v" * 300, f"{i}" + "w" * 200] for i in range(8)] + [["G2", "", ""]] + [[f"B{i}", f"{i}" + "v" * 300, f"{i}" + "w" * 200] for i in range(8)]}]}
    _, parts, _ = build_tables([big])
    assert len(parts) == 4 and all(len(p["markdown"]) <= PART_MAX for p in parts) and {p["caption_clean"] for p in parts} == {"Table 2 Big"}
    assert {p["table_id"] for p in parts} == {"D:1"} and [p["part"] for p in parts] == [1, 2, 3, 4] and parts[0]["parts"] == 4
    assert parts[0]["rows"][-1]["group"] == "G1" and parts[2]["rows"][0]["group"] == "G2", [p["rows"][0]["group"] for p in parts]
    print("ok")
