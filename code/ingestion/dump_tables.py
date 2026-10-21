"""
Debugging to check what ingestion is extracting from the guidelines.
"""
import sys
import tempfile
from collections import Counter
from pathlib import Path

from ingestion.pdf_parsing import extract_pdf_text_layout_aware
from ingestion.chunking import create_chunks
from ingestion.tables import build_tables, markdown, normalize


def dump(guidelines=Path("Guidelines"), out_dir=Path("Examples/results/tables")):
    out_dir.mkdir(parents=True, exist_ok=True)
    for pdf in sorted(guidelines.glob("*.pdf")):
        _, grids, tables = build_tables(extract_pdf_text_layout_aware(pdf))
        counts = Counter(t["kind"] for t in tables)
        out = out_dir / f"{pdf.stem}_tables.txt"
        with open(out, "w", encoding="utf-8") as f:
            f.write(f"{pdf.name}: {len(tables)} tables {dict(counts)}, {len(grids)} grid parts (rejected ones are logged at extraction)\n")
            for t in tables:
                f.write(f"\n{'=' * 80}\n--- {t['kind'].upper()} | p.{t['page_start']}-{t['page_end']} | {t['caption_raw'] or '(no caption)'}\n")
                f.write(f"caption_clean: {t['caption_clean']}\n{'-' * 80}\n{markdown(t['header'], t['rows'])}\n")
        print(f"{pdf.name}: {dict(counts)} -> {out}")


def dump_chunks(guidelines=Path("Guidelines"), out_dir=Path("Examples/results/text")):
    out_dir.mkdir(parents=True, exist_ok=True)
    punkt = Path(tempfile.gettempdir()) / "punkt_dump.pkl"
    for pdf in sorted(guidelines.glob("*.pdf")):
        pages = extract_pdf_text_layout_aware(pdf)
        chunks = create_chunks(pages, punkt_path=str(punkt)) + build_tables(pages)[0]
        counts = Counter(c.get("kind", "prose") for c in chunks)
        out = out_dir / f"{pdf.stem}_chunks.txt"
        with open(out, "w", encoding="utf-8") as f:
            f.write(f"{pdf.name}: {len(chunks)} chunks {dict(counts)}\n")
            for i, c in enumerate(chunks, 1):
                origin = f" | {c['table_id']} row {c['row']}" if "table_id" in c else ""
                f.write(f"\n--- Chunk {i} | {c.get('kind', 'prose')} | p.{c['page_start']}-{c['page_end']}{origin} | {len(c['text'])} chars\n{c['text']}\n")
        punkt.unlink(missing_ok=True)
        print(f"{pdf.name}: {dict(counts)} -> {out}")


def lookup(token: str):
    from storage.storage import load_table_store
    tables, _ = load_table_store()
    tok = normalize(token)
    tok = tok[0] if tok else token
    for t in tables:
        if tok in t["header_tokens"]:
            print(f"{t['table_id']:28} header      | {t['caption_clean']}")
        for r in t["rows"]:
            for name in ("key_tokens", "group_tokens", "other_tokens"):
                if tok in r[name]:
                    print(f"{t['table_id']:28} {name:12}| {r['row_surrogate'][:110]}")


def show(table_id: str):
    from storage.storage import load_table_store
    tables, _ = load_table_store()
    for t in tables:
        if t["table_id"] == table_id:
            print(f"[{t['table_id']} part {t.get('part', 1)}/{t.get('parts', 1)}] ({t['doc_id']}, p.{t['page_start']}-{t['page_end']}) {t['caption_raw']}\n{t['markdown']}")
            print(f"\nsurrogate: {t['table_surrogate']}")


if __name__ == "__main__":
    if len(sys.argv) == 2 and sys.argv[1] == "chunks":
        dump_chunks()
    elif len(sys.argv) == 3 and sys.argv[1] == "lookup":
        lookup(sys.argv[2])
    elif len(sys.argv) == 3 and sys.argv[1] == "show":
        show(sys.argv[2])
    else:
        dump()
