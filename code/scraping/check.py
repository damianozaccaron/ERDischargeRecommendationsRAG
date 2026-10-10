"""--check: smoke test of every step, offline checks first, then the network and one LLM call."""
import collections, tempfile, time
from pathlib import Path

import pymupdf
import requests

from scraping.assist import new_pdfs
from scraping.common import ask, retry
from scraping import em
from scraping.pdfs import file_names, mismatch, pmid_of, stamp, twins
from scraping.pubmed import eutils, summaries
from scraping.search import TIER1, TIER2, filters, flags, search, split_terms
from scraping.selection import EM_SELECT_PROMPT, alike, read_picks, read_verdicts, replacements, select_prompt
from scraping.topics import build_query, chapters, read_refs, ref_candidates, topic


def check():
    calls = []
    def flaky():  # the connection reset of 25 September, twice, then a reply
        calls.append(1)
        if len(calls) < 3:
            raise requests.exceptions.ChunkedEncodingError("Connection broken: ConnectionResetError(104)")
        return "reply"
    assert retry(flaky, waits=(0, 0, 0)) == "reply" and len(calls) == 3, calls
    tree = {"D1": ["C14.280"], "D2": ["C14.280.067"], "D3": ["F03.600"], "D4": ["D02.065"], "D5": ["E01.370"]}
    h = lambda ui, q=(), major=True: dict(ui=ui, name=ui, qualifiers=list(q), major=major)
    got = [topic(x, tree) for x in (h("D2"), h("D3"), h("D4", ["poisoning"]), h("D4"), h("D5"), h("D2", major=False))]
    assert got == [("D2", "D2"), ("D3", "D3"), ("D4/poisoning", "D4 poisoning"), None, None, None], got
    names = {"R": "Gastroesophageal Reflux", "P": "Esophageal Perforation", "B": "Barrett Esophagus",
             "A/poisoning": "Acetaminophen poisoning", "S": "Shock"}
    counts = collections.Counter({"R": 5, "P": 4, "A/poisoning": 3, "B": 2, "S": 1})
    translation = '"esophageal perforation"[MeSH Terms] OR "acetaminophen"[MeSH Terms] OR "reflux"[All Fields]'
    assert ref_candidates(counts, names, translation) == ["R", "B"], "searched or once-cited topics not left out"
    reply = 'Thinking... {"77": ["Gastroesophageal Reflux", "Shock", "Gastroesophageal Reflux"]}'
    got = read_refs(reply, ["77", "142"], {"77": ["R", "B"], "142": ["B"]}, names)
    assert got == {"77": ["Gastroesophageal Reflux"], "142": []}, f"a chapter left out stays unscreened: {got}"
    cands = [dict(pmid="1", date="2016/01/01", journal="J A", types="", title="A"),
             dict(pmid="2", date="2019/01/01", journal="J B", types="", title="B"),
             dict(pmid="3", date="2019/01/01", journal="J A", types="", title="C")]
    ch = ("1", "Chapter", "Section")
    assert select_prompt(ch, ["t"], cands) == select_prompt(ch, ["t"], cands[::-1]), "the prompt depends on the order"
    assert select_prompt(ch, ["t"], cands).index("3 | 2019") < select_prompt(ch, ["t"], cands).index("1 | 2016")
    reply = '{"verdicts": [{"pair": 1, "verdict": "newer", "keep": "2"}, {"pair": 9, "verdict": "same", "keep": "3"}]}'
    assert read_verdicts(reply, [(cands[0], cands[1])]) == [["1", "2", "newer", "2"]], "a verdict misread"
    recs = {"m16": dict(date="2016/03", title="ESCMID guideline: diagnosis and treatment of acute bacterial meningitis."),
            "m19": dict(date="2019/08", title="Management of acute community-acquired bacterial meningitis."),
            "p18": dict(date="2019/01", title="2018 ESC Guidelines for the management of cardiovascular diseases during "
                        "pregnancy."),  # the Kardiol Pol reprint
            "p25": dict(date="2025/11", title="2025 ESC Guidelines for the management of cardiovascular disease and "
                        "pregnancy."),
            "h14": dict(date="2014/01", title="Early management of head injury: summary of updated NICE guidance."),
            "h23": dict(date="2023/05", title="Head injury: assessment and early management")}
    verdicts = [["m16", "m19", "newer", "m19"], ["p18", "p25", "newer", "p25"], ["h14", "h23", "newer", "h23"]]
    assert replacements(verdicts, recs) == {"p18": "p25", "h14": "h23"}, "the European guard misapplied"
    with tempfile.TemporaryDirectory() as tmp:
        pdf = Path(tmp) / "blank.pdf"
        doc = pymupdf.open()
        doc.new_page()
        doc.save(pdf)  # no Info dictionary: stamp() has to create one
        assert stamp(pdf, "123") and pmid_of(pdf) == "123", "PMID not written into the PDF's metadata"
        assert mismatch(pdf, dict(title="Anything", doi="")) == "title not found in the PDF"
        doc = pymupdf.open()
        for _ in range(3):
            doc.new_page().insert_text((72, 72), "Guidelines on the manage-\nment of syncope")
        doc.save(pdf)
        assert mismatch(pdf, dict(title="Guidelines for the management of syncope.", doi="")) == ""
        assert mismatch(pdf, dict(title="Guidelines for atrial fibrillation", doi="")) == "title not found in the PDF"
        assert new_pdfs(Path(tmp), time.time() - 60, set()) == [pdf] and not new_pdfs(Path(tmp), time.time() - 60, {pdf})
        doc.delete_page(0)
        doc.save(pdf)
        assert mismatch(pdf, dict(title="Guidelines for the management of syncope.", doi="")) == "2 pages"
        doc = pymupdf.open()  # a translation: the original's English title up front, the body in Italian
        for i in range(3):
            doc.new_page().insert_text((72, 72), "Guidelines for the management of syncope" if i == 0 else
                                       "Linee guida per la gestione della sincope nella pratica clinica italiana")
        doc.save(pdf)
        assert mismatch(pdf, dict(title="Guidelines for the management of syncope.", doi="")) == "not in English"
    erc = [dict(pmid="1", date="2021", title="European Resuscitation Council Guidelines 2021: Cardiac arrest in "
                "special circumstances."),
           dict(pmid="2", date="2010", title="European Resuscitation Council Guidelines for Resuscitation 2010 Section "
                "8. Cardiac arrest in special circumstances: drowning, accidental hypothermia, asthma, anaphylaxis."),
           dict(pmid="3", date="2025", title="European Resuscitation Council Guidelines 2025 Special Circumstances "
                "in Resuscitation."),
           dict(pmid="4", date="2025", title="European Resuscitation Council Guidelines 2025 Special Circumstances. "
                "Part II.")]
    got = {(a["pmid"], b["pmid"]) for a, b in alike(erc[:2], erc[2:])}
    assert got == {("1", "2"), ("1", "3"), ("2", "3")}, got  # a newer pool record pairs; a Part II never does
    here = {"AtrialFibrillation.pdf": "1", "Syncope.pdf": "", "Old.pdf": "2"}  # the last guideline, unstamped, renamed
    picks = [dict(pmid="2", name="New"), dict(pmid="3", name="AtrialFibrillation"), dict(pmid="4", name="Syncope"),
             dict(pmid="5", name="Stroke"), dict(pmid="6", name="Stroke"), dict(pmid="5", name="Other")]
    assert file_names(picks, here) == {"2": "Old.pdf", "3": "AtrialFibrillation_3.pdf", "4": "Syncope_4.pdf",
                                       "5": "Stroke.pdf", "6": "Stroke_6.pdf"}, file_names(picks, here)
    tox = em.concepts("digoxin toxicity", '("digoxin"[MeSH Terms] OR "digoxin"[All Fields]) AND '
                                          '("toxicity"[MeSH Subheading] OR "toxicity"[All Fields])')
    poisoning = em.concepts("digoxin poisoning", '("digoxin"[MeSH Terms] OR "digoxin"[All Fields]) AND '
                                                 '("poisoning"[MeSH Terms] OR "poisoning"[All Fields])')
    kawasaki = em.concepts("Kawasaki disease", '"mucocutaneous lymph node syndrome"[MeSH Terms] OR ("mucocutaneous"'
                                               '[All Fields] AND "lymph"[All Fields]) OR "kawasaki disease"[All Fields]')
    mlns = em.concepts("Mucocutaneous Lymph Node Syndrome", '"mucocutaneous lymph node syndrome"[MeSH Terms]')
    cap = em.concepts("community-acquired pneumonia", '"community acquired pneumonia"[MeSH Terms] OR "community '
                                                      'acquired pneumonia"[All Fields]')
    pneumonia = em.concepts("pneumonia", '"pneumonia"[MeSH Terms] OR "pneumonia"[All Fields]')
    assert em.same(tox, poisoning) and em.same(kawasaki, mlns), "same concepts not matched"
    assert not em.same(cap, pneumonia), "a broader term taken for the same concepts"
    assert em.counts("13.3.6.2", "pregnancy") and not em.counts("16.4.8", "pregnancy") \
           and not em.counts("16.4.8", "pediatric"), "population rule"
    assert [em.pool(n) for n in ("4.1.3", "2.2.2.3.1", "3.8")] == ["4.1", "2.2.2.3", "3.8"], "pooled under the wrong heading"
    trees = {"melanoma": ["C04.557.465.625.650.510"], "neoplasms": ["C04"], "cysts": ["C04.182"],
             "neurofibroma": ["C04.557.580.600.580"], "pilonidal sinus": ["C04.182.710"]}
    word = lambda *m: [dict(mesh=set(m), sc=set(), sh=set(), text=set())]
    assert em.tumour(word("melanoma"), trees) and not em.tumour(word("pilonidal sinus"), trees) \
           and not em.tumour(word("cysts", "neoplasms", "neurofibroma"), trees), "tumours misjudged"  # "tumor" alone
    assert "chapter" not in EM_SELECT_PROMPT.lower(), "the EM topics' prompt still speaks of a chapter"
    rows = {r["num"]: r for r in em.table()}  # ABEM's PDF, read once
    assert sum("." in n for n in rows) == 1331 and rows["12.9.2"]["title"] == "Nonepileptic seizure" \
           and rows["5.4.1.1.2"]["title"] == "Diabetic ketoacidosis (DKA)" and rows["18.3.4"]["title"] == "Assault" \
           and [rows["1.3.10"][k] for k in ("critical", "emergent", "lower")] == ["", "1", "1"], "EM Model misread"
    assert len(em.items()) == 721 and em.items()[0]["path"] == "Abdominal Wall › Hernias", (len(em.items()), em.items()[0])
    chs = chapters()
    assert len(chs) == 304 and ("29A", "Tracheal Intubation") == chs[28][:2], chs[27:30]
    assert build_query(dict(terms=["syncope"], population="pediatric")) == "((syncope)) AND (child OR infant OR pediatric OR adolescent)"
    query = build_query(dict(terms=["syncope", "caustic ingestion"]))
    assert query == "((syncope) OR (caustic ingestion))", query
    res = search(query)
    esc = eutils("esearch.fcgi", term=f"29562304[uid] AND ({query}) AND {TIER1} AND {filters()}")["esearchresult"]
    assert esc["count"] == "1", "2018 ESC syncope guideline (indexed only as Journal Article) not passing the search"
    assert res["tier"] == 1 and search("(snakebite)")["tier"] == 2, "tier 2 not used"
    assert "30535127" in search("(Kawasaki disease)")["idlist"], "SHARE Kawasaki (a Consensus Statement) not in tier 1"
    cap = eutils("esearch.fcgi", term=f"(37012484[uid] OR 37012080[uid]) AND {TIER1} AND {filters()}")["esearchresult"]
    assert cap["count"] == "2", "2023 ERS/ESICM/ESCMID/ALAT CAP guideline (acronyms only) not passing the European filter"
    nata = eutils("esearch.fcgi", term=f"28722245[uid] AND {TIER2} AND {filters()}")["esearchresult"]["count"]
    assert nata == "1", "2018 NATA obstetric guideline (acronym checked in capitals) not passing the European filter"
    assert not flags(res) and split_terms(["syncope", "caustic ingestion"], res) == "caustic ingestion", res
    fallback = lambda q: set(search(q, fallback=True)["idlist"])
    assert "39001977" in fallback("(vertigo)"), "2024 Italian consensus on acute vertigo in the ED not in the fallback"
    assert "36580567" in fallback("(malaria)"), "WHO guidelines for malaria (a Bookshelf book) not in the fallback"
    assert "41255343" in fallback("(antipsychotic overdose)"), "CTRC recommendations (not typed as guideline) missing"
    assert "31891472" in fallback("(fever)"), "NICE's fever in under 5s (a Bookshelf book) not in the fallback"
    assert summaries(["36580567"])["36580567"]["title"] == "WHO guidelines for malaria", "a book's title not read"
    erj = summaries(["37012080"])["37012080"]  # 2023 ERS/ESICM/ESCMID/ALAT CAP guideline, also in Intensive Care Med
    assert [t["pmid"] for t in twins(erj)] == ["37012484"], twins(erj)
    recs = list(summaries(search("(syncope)")["idlist"]).values())
    syncope = next(c for c in chs if c[1] == "Syncope")
    sel = read_picks(ask(select_prompt(syncope, ["syncope"], recs)), ["syncope"], recs)
    assert [p["pmid"] for p in sel["picks"]] == ["29562304"] and not sel["flags"], sel
    print("ok")
