"""Steps 1-5 in one run, from the chapters to manifest.csv."""
import csv, json, time
from concurrent.futures import ThreadPoolExecutor

from scraping import em
from scraping.common import OUT, ask_all, write_manifest
from scraping.pdfs import download, file_names, mismatch, pmid_of
from scraping.search import flags, snapshot, split_terms
from scraping.selection import EUROPEAN_RULES, FALLBACK_RULES, consolidate, key, read_picks, select_prompt
from scraping.topics import build_query, corpus_chapters, reference_topics, topics


def run(queries_only=False, refresh=False):
    OUT.mkdir(exist_ok=True)
    start = t = time.time()
    chs = corpus_chapters()
    cached = reference_topics(chs, topics(chs))
    tops = {num: cached.get(num) or dict(terms=[title]) for num, title, _ in chs}  # the title if the LLM gave nothing
    print(f"{len(chs)} chapters, {len(cached)} with terms ({time.time() - t:.0f}s)")

    t = time.time()
    em_chs, em_tops = em.topics(tops, refresh)  # the EM Model's topics no chapter searches, after the chapters
    chs, tops = chs + em_chs, tops | em_tops
    queries = {num: build_query(tp) for num, tp in tops.items()}
    print(f"EM topics ready ({time.time() - t:.0f}s)")

    t = time.time()
    results, recs = snapshot(queries, refresh)
    log = [dict(chapter=num, section=section, title=title, hits=results[num]["count"], tier=results[num]["tier"], flags=flags(results[num]),
                split=split_terms(tops[num]["terms"], results[num]), query=queries[num],
                translation=results[num].get("querytranslation", "")) for num, title, section in chs]
    with open(OUT / "queries.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, list(log[0]))
        w.writeheader()
        w.writerows(log)
    print(f"PubMed: {sum(bool(r['flags']) for r in log)} chapters and EM topics flagged, "
          f"{sum(bool(r['split']) for r in log)} with split terms, in {OUT / 'queries.csv'} ({time.time() - t:.0f}s)")
    if queries_only:
        return
    hits = {num: r["idlist"] for num, r in results.items()}
    print(f"PubMed: {sum(map(bool, hits.values()))} chapters and EM topics with candidates, {len(recs)} records "
          f"({time.time() - t:.0f}s)")

    t = time.time()
    # selections are cached per chapter and its prompt's key(): a rerun only asks for failed chapters and changed
    # prompts (edited topics.json lines or SELECT_PROMPT, new candidates after --refresh)
    path = OUT / "selections.json"
    cache = json.loads(path.read_text()) if path.exists() else {}

    def choose(chosen, cands, store, rules=EUROPEAN_RULES, step="selection"):  # store: cache, or its "fallback" entry
        prompts = {c[0]: select_prompt(c, tops[c[0]]["terms"], cands[c[0]], rules) for c in chosen if cands[c[0]]}
        todo = {num: p for num, p in prompts.items() if store.get(num, {}).get("prompt") != key(p)}
        for num, reply in ask_all(todo, step).items():
            try:
                store[num] = read_picks(reply, tops[num]["terms"], cands[num]) | dict(prompt=key(todo[num]))
            except Exception as e:  # an unreadable answer: asked again next run
                print(f"  chapter {num}: {type(e).__name__}: {e}")
        path.write_text(json.dumps(cache, indent=1))
        return {num: store[num] for num, p in prompts.items() if store.get(num, {}).get("prompt") == key(p)}  # current

    cands = {c[0]: [recs[p] for p in hits[c[0]] if p in recs] for c in chs}
    sel = choose(chs, cands, cache)
    empty = [c for c in chs if not cands[c[0]] or c[0] in sel and not sel[c[0]]["picks"]]  # a failed selection is asked again
    results3, recs3 = snapshot({c[0]: queries[c[0]] for c in empty}, refresh, fallback=True)
    cands3 = {num: [recs3[p] for p in r["idlist"] if p in recs3] for num, r in results3.items()}
    sel3 = choose(empty, cands3, cache.setdefault("fallback", {}), FALLBACK_RULES, "fallback selection")
    sel |= {num: dict(s, flags="; ".join(filter(None, [f"fallback, {len(cands3[num])} candidates", s["flags"]])))
            for num, s in sel3.items() if s["picks"]}
    recs |= recs3
    print(f"Fallback: {sum(bool(s['picks']) for s in sel3.values())} of the {len(empty)} chapters without a European "
          f"pick have a national, NICE, WHO or CTRC one, among {len(recs3)} records")
    replace = consolidate(list({p["pmid"]: recs[p["pmid"]] for s in sel.values() for p in s["picks"]}.values()),
                          list(recs.values()), cache)
    path.write_text(json.dumps(cache, indent=1))
    for num, s in sel.items():  # new dicts: the cache keeps each chapter's own answer
        merged = {}
        for p in s["picks"]:
            m = merged.setdefault(replace.get(p["pmid"], p["pmid"]), dict(p, pmid=replace.get(p["pmid"], p["pmid"]),
                                                                          covers=[]))
            m["covers"] = m["covers"] + [t for t in p["covers"] if t not in m["covers"]]
        notes = [f"{p['pmid']} replaced by {replace[p['pmid']]}" for p in s["picks"] if p["pmid"] in replace]
        sel[num] = dict(s, picks=list(merged.values()), flags="; ".join(filter(None, [s["flags"], *notes])))
    here = {p.name: pmid_of(p) for p in OUT.glob("*.pdf")}  # file -> the PMID in its metadata, '' if none
    names = file_names([p for s in sel.values() for p in s["picks"]], here)
    # chapters and EM topics counted apart: each count is its own list's yield (a chapter's subject may also be covered
    # by an EM topic's pick, which neither count credits)
    book, emt = [c[0] for c in chs if not c[0].startswith("EM")], [c[0] for c in chs if c[0].startswith("EM")]
    picked = lambda nums: [n for n in nums if sel.get(n, {}).get("picks")]
    pmids = lambda nums: {p["pmid"] for n in nums for p in sel.get(n, {}).get("picks", [])}
    print(f"Selection: {len(picked(book))}/{len(book)} chapters and {len(picked(emt))}/{len(emt)} EM topics with a "
          f"pick, {len(names)} distinct guidelines ({len(pmids(emt) - pmids(book))} picked for EM topics only), "
          f"{len(replace)} picks replaced by another version or copy, {sum(bool(s['flags']) for s in sel.values())} "
          f"flagged ({time.time() - t:.0f}s)")

    t = time.time()
    # a PDF kept from an earlier run keeps the source manifest.csv gave it then (a host, or browser): the counts of
    # full texts by source stay readable after a rerun
    old = {r["pmid"]: r["source"] for r in csv.DictReader(open(OUT / "manifest.csv", encoding="utf-8"))
           if r["pmid"]} if (OUT / "manifest.csv").exists() else {}
    # a PDF here already is kept, unless it is not in English (a translation OpenAlex filed under the original's DOI)
    # and was not kept by hand in --assist
    todo = [p for p in names if here.get(names[p]) != p or old.get(p) != "browser"
            and mismatch(OUT / names[p], recs[p]) == "not in English"]
    with ThreadPoolExecutor(8) as pool:
        done = dict(zip(todo, pool.map(lambda p: download(recs[p], OUT / names[p]), todo)))
    got = {p: done[p][0] if p in done else old.get(p) or "exists" for p in names}
    with open(OUT / "downloads.csv", "w", newline="", encoding="utf-8") as f:  # this run's attempts
        w = csv.DictWriter(f, ["pmid", "record", "version", "result", "url"])
        w.writeheader()
        w.writerows(row for _, log in done.values() for row in log)
    pdf_flags = {p: mismatch(OUT / names[p], recs[p]) for p in names if got[p]}  # old PDFs too: 12 ms each
    print(f"Open-access PDFs: {sum(map(bool, got.values()))}/{len(names)}, {sum(map(bool, pdf_flags.values()))} "
          f"to check by hand ({time.time() - t:.0f}s)")
    if stale := sorted(f for f, pmid in here.items() if pmid not in names):
        print(f"{len(stale)} PDFs here are no current pick's (or have no PMID in their metadata): {', '.join(stale)}")

    rows = []  # one per chapter and pick; chapters without a pick keep a row
    for num, title, section in chs:
        failed = cands[num] and num not in sel
        row = dict(chapter=num, section=section, chapter_title=title, query=queries[num], candidates=len(hits[num]),
                   flags="selection failed, rerun" if failed else sel.get(num, {}).get("flags", ""))
        picks = sel.get(num, {}).get("picks", [])
        rows += [row | recs[p["pmid"]] | dict(covers="; ".join(p["covers"]), file=names[p["pmid"]],
                                             source=got[p["pmid"]], flags="; ".join(filter(None, [row["flags"],
                                             pdf_flags.get(p["pmid"], "")]))) for p in picks] or [row]
    write_manifest(rows)
    print(f"Wrote {OUT / 'manifest.csv'}; python code/scrape_guidelines.py --assist fetches the {len(names) - sum(map(bool, got.values()))} missing PDFs"
          f" and those to check")
    print(f"Total: {(time.time() - start) / 60:.0f} min")
