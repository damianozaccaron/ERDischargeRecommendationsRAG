"""
Streamlit UI for beta users: from the PDF of an Italian Verbale di Pronto Soccorso, anonymised before anything is
translated (or from a note pasted as plain text, which is taken as it is), generates the discharge recommendations
through the API, in Italian, each cited guideline linked to its PDF in Guidelines/ at the page.
"Solo prompt" stops at the prompt instead, for copy-paste into an external LLM app/site.

Run from the project root:  streamlit run code/app.py
"""

import html
import re
import threading
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import quote

import torch
import streamlit as st

from config import (EMBEDDING_MODEL, NER_MODEL, CROSS_ENCODER, RETRIEVAL_K, FINAL_N, FAST_FINAL_N, PROMPT_MAX_CHARS,
                    API, MODEL_NAME, TEMPERATURE, OPENROUTER_KEY)

from storage.storage import load_index_and_metadata, load_vectorizer, load_table_store
from storage.embedding import embed_query, load_embedder
from retrieval.retrieval import retrieve_top_k, load_crossEncoder, reranking_multi_query, retrieve_tables, dense_select
from retrieval.query_building import build_queries, load_ner_model, patient_qualifiers
from generation.output_prod import generate_letter
from generation.prompting import build_prompt, fit_prompt, is_hospitalised, ITALIAN_TRANSLATION, CITATION
from pipeline.translation import deepl_translation, deepl_translation_en_it, llm_translation, llm_translation_en_it
from pipeline.user_input import extract_patient_fields
from pipeline.verbale import parse_verbale

GUIDELINES = Path("Guidelines")


# model loading (cached) 

@st.cache_resource(show_spinner="Caricamento modelli...")
def load_heavy_models():
    """NER and embedder"""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ner_model = load_ner_model(model_name=NER_MODEL, device=device)
    emb_model = load_embedder(EMBEDDING_MODEL)
    return ner_model, emb_model


@st.cache_resource(show_spinner="Caricamento cross-encoder...")
def load_cross_encoder():
    """C-E"""
    return load_crossEncoder(CROSS_ENCODER)


@st.cache_resource(show_spinner="Caricamento indice FAISS e vectorizer...")
def load_index_artifacts():
    """Index, metadata and IDF"""
    vectorizer = load_vectorizer()
    index, metadata = load_index_and_metadata()
    tables, table_index = load_table_store()
    return vectorizer, index, metadata, tables, table_index


def load_pipeline_models():
    ner_model, emb_model = load_heavy_models()
    vectorizer, index, metadata, tables, table_index = load_index_artifacts()
    return ner_model, vectorizer, emb_model, index, metadata, tables, table_index


@st.cache_resource
def guideline_server() -> int:
    """
    The port of a server of Guidelines/ on this machine, so that a citation opens its PDF at the page: the browser
    does not follow a file:// link from the site, and Streamlit serves files only from code/static, behind a flag.
    """
    server = ThreadingHTTPServer(("127.0.0.1", 0), partial(SimpleHTTPRequestHandler, directory=GUIDELINES.resolve()))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server.server_port


# pipeline 

def run_pipeline(italian_note: str, llm_queries: bool, use_cross_encoder: bool, prompt_only: bool,
                 translate_by_llm: bool, translate_in_prompt: bool) -> str:
    """
    The prompt, or the recommendations the generation model writes from it, in Italian only. The note is translated
    by DeepL, or by the generation model when translate_by_llm; the recommendations the same way in a second call, or
    by the generation model in the same call when translate_in_prompt (the prompt only gains it when displayed).
    """
    ner_model, vectorizer, emb_model, index, metadata, tables, table_index = load_pipeline_models()
    cross_encoder = load_cross_encoder() if use_cross_encoder else None
    fallback, dropped_chunks, dropped_tables = [], 0, 0

    with st.status("Elaborazione in corso...", expanded=True) as status:
        st.write("Traduzione della nota clinica...")
        translated_note = (llm_translation(italian_note, api_model=MODEL_NAME) if translate_by_llm
                           else deepl_translation(italian_note))

        st.write("Costruzione delle query...")
        queries, floors, scopes = build_queries(translated_note, ner_model, vectorizer, emb_model, index, metadata,
                                                llm_queries=llm_queries, on_fallback=fallback.append)

        st.write("Retrieval dei chunk rilevanti...")
        all_candidates, ranked = {}, []
        for i, q in enumerate(queries):
            candidates, faiss_ids = retrieve_top_k(
                query_embedding=embed_query(q, embedder=emb_model),
                index=index,
                metadata=metadata,
                # without the cross-encoder the round-robin never goes deeper than the excerpts it keeps
                k=RETRIEVAL_K if use_cross_encoder else FAST_FINAL_N,
                docs=scopes.get(i),
            )
            for cand, fid in zip(candidates, faiss_ids):
                all_candidates.setdefault(fid, {**cand, "retrieved_by": []})["retrieved_by"].append(q)
            ranked.append([all_candidates[fid] for fid in faiss_ids])
        top_k_candidates = list(all_candidates.values())
        if not top_k_candidates:
            raise ValueError("Nessun chunk recuperato dal retrieval top-k.")

        if use_cross_encoder:
            st.write(f"Re-ranking con cross-encoder ({sum(map(len, ranked))} coppie)...")
            final_chunks = reranking_multi_query(
                queries, top_k_candidates, cross_encoder, top_n=FINAL_N, floors=floors, sparse=True,
            )
        else:
            final_chunks = dense_select(ranked, top_n=FAST_FINAL_N)
        if not final_chunks:
            raise ValueError("Nessun chunk selezionato.")

        fields = extract_patient_fields(translated_note)
        final_tables = []
        if not is_hospitalised(fields.get("prognosis", "")):
            st.write("Retrieval delle tabelle...")
            final_tables = retrieve_tables(translated_note, final_chunks, fields.get("diagnosis", ""), patient_qualifiers(fields),
                                           ner_model, emb_model, cross_encoder, tables, table_index)

        if prompt_only:
            st.write("Costruzione del prompt finale...")
            # the Italian block is reserved even when not requested, since it can be switched on after generation
            result, dropped_chunks, dropped_tables = fit_prompt(PROMPT_MAX_CHARS - len(ITALIAN_TRANSLATION), translated_note,
                                                                final_chunks, fields.get("prognosis", ""), final_tables)
            status.update(label="Prompt generato", state="complete", expanded=False)
        else:
            st.write("Generazione delle raccomandazioni...")
            prompt = build_prompt(translated_note, final_chunks, fields.get("prognosis", ""), final_tables)
            # through the API the whole prompt goes in, and max_tokens leaves room for the hidden reasoning (main.py)
            result = generate_letter(prompt + (ITALIAN_TRANSLATION if translate_in_prompt else ""), model=MODEL_NAME,
                                     tokenizer=None, backend="openrouter", temperature=TEMPERATURE, max_tokens=16000,
                                     api_key=OPENROUTER_KEY)
            if translate_in_prompt:
                # the Italian after the English; all of it when the model left the heading out
                _, heading, italian = result.partition("TRADUZIONE IN ITALIANO")
                result = italian.strip() if heading else result
            else:
                st.write("Traduzione delle raccomandazioni...")
                result = (llm_translation_en_it(result, MODEL_NAME, None, "openrouter") if translate_by_llm
                          else deepl_translation_en_it(result))
            status.update(label="Raccomandazioni generate", state="complete", expanded=False)

    # outside the status box, which collapses once done
    if fallback:
        st.warning(f"Generazione delle query con LLM non riuscita ({fallback[0]}): sono state usate le query "
                   "automatiche.")
    if dropped_chunks or dropped_tables:
        st.info(f"Per restare entro {PROMPT_MAX_CHARS} caratteri sono stati esclusi gli ultimi {dropped_chunks} estratti "
                f"e {dropped_tables} tabelle per rilevanza.")
    return result


def with_links(output: str) -> str:
    """
    The output as HTML. A citation whose PDF is in Guidelines/ leaves its line for a superscript at the end of it, the
    guideline's initials (AtrialFibrillation -> AF), that opens the PDF at the first cited page in a new tab; a
    citation without its PDF stays as written.
    """
    lines = []
    for line in output.split("\n"):
        cited = [m for m in CITATION.finditer(line) if (GUIDELINES / f"{m[1]}.pdf").is_file()]
        for m in cited:
            line = line.replace(m[0], "", 1)
        if cited:  # the sources emptied or with a gap: "1 -  - text" -> "1 - text", "1 -  (Clinical note)" -> "1 - (..."
            line = re.sub(r"^(\d+ [-–] )[-–] ", r"\1", re.sub(r" {2,}", " ", line)).rstrip()
            links = " ".join(f'<a href="http://127.0.0.1:{guideline_server()}/{quote(m[1])}.pdf#page={m[2]}" '
                             f'target="_blank" title="{html.escape(m[0])}">'
                             f'{re.sub("[^A-Z]", "", m[1]) or html.escape(m[1])}</a>' for m in cited)
            lines.append(f"{html.escape(line)}<sup>{links}</sup>")
        else:
            lines.append(html.escape(line))
    return '<div style="white-space: pre-wrap">' + "\n".join(lines) + "</div>"


# UI

st.set_page_config(page_title="Generatore di Raccomandazioni alla Dimissione", layout="wide")

st.title("Generatore di Raccomandazioni alla Dimissione")
st.caption(
    "Carica il verbale di Pronto Soccorso, parziale o completo, in formato PDF (o incolla la nota come testo). Il sistema lo anonimizza, recupera "
    "gli estratti delle linee guida rilevanti e genera le raccomandazioni di dimissione, oppure solo il prompt da "
    "incollare in qualsiasi LLM esterno."
)

prompt_only = st.checkbox(
    "Solo prompt",
    value=not API,
    disabled=not API,
    help="Si ferma al prompt, da incollare nell'LLM esterno, senza generare le raccomandazioni. "
         "Sempre attivo quando la generazione non passa dall'API (API = False in config.py).",
)

query_mode = st.radio(
    "Generazione delle query di retrieval",
    ["Automatiche", "Generate da LLM"],
    horizontal=True,
    help="Automatiche: costruite dalla nota tramite regole lessicali automatizzate. Generate da LLM: più accurate, ma richiedono l'accesso a un LLM"
         "; se la chiamata fallisce si usano quelle automatiche.",
)

if prompt_only:
    retrieval_mode = st.radio(
        "Selezione degli estratti",
        ["Rapida", "Accurata (cross-encoder)"],
        index=1,
        horizontal=True,
        help=f"Rapida: Estratti scelti per similarità, pochi secondi. Accurata: un cross-encoder "
             f"riordina i candidati; più preciso, ma senza GPU può essere pesante.",
    )
    translator = None
else:
    # the full output always reranks with the cross-encoder
    retrieval_mode = None
    translator = st.radio(
        "Traduzione",
        ["DeepL", "LLM di generazione"],
        index=1,
        horizontal=True,
        help=f"Traduce la nota in inglese e, con il prompt Standard, le raccomandazioni in italiano. LLM di generazione: "
             f"lo stesso modello che scrive le raccomandazioni ({MODEL_NAME}); più lento.",
    )

prompt_mode = st.radio(
    "Prompt",
    ["Standard", "Con traduzione in italiano"],
    horizontal=True,
    help="Con traduzione in italiano: il prompt chiede all'LLM di aggiungere, dopo le due sezioni in inglese, "
         "la traduzione integrale e fedele in italiano di tutto il testo. Per il solo prompt vale anche per un prompt "
         "già generato; per le raccomandazioni, la traduzione avviene nella stessa chiamata della generazione, al posto "
         "di quella scelta sopra.",
)

button = "Genera prompt" if prompt_only else "Genera raccomandazioni"
italian_note = None

use_text = st.checkbox(
    "Incolla la nota come testo",
    help="Al posto del PDF: la nota clinica in testo libero, con le righe «Diagnosi: ...» e «Prognosi: ...» "
         "in fondo.",
)

if use_text:
    text = st.text_area("Nota clinica", height=400, key="note_text")
    st.caption("Il testo non viene anonimizzato: incollalo senza dati identificativi.")
    if st.button(button, type="primary"):
        if not re.search(r"Diagnosi\s*:[ \t]*\S", text, re.IGNORECASE):
            st.error("Manca la riga «Diagnosi: ...»")
        else:
            # as for the PDF: the Prognosi label ends the diagnosis, and an empty one reads as a home discharge
            has_prognosis = re.search(r"Prognosi\s*:", text, re.IGNORECASE)
            italian_note = text.strip() if has_prognosis else f"{text.strip()}\n\nPrognosi: "
else:
    uploaded = st.file_uploader("Verbale di Pronto Soccorso (PDF)", type=["pdf"])
    if uploaded:
        try:
            # the physician's own advice stays out: it is what the LLM is asked to write
            note, dropped, diagnosis, prognosis, _ = parse_verbale(uploaded.getvalue())
        except ValueError as e:
            st.error(str(e))
        else:
            # keyed by file: a new verbale must not inherit the previous patient's edits or diagnosis
            note = st.text_area(
                "Nota anonimizzata: ",
                value=note, height=400, key=f"note_{uploaded.file_id}",
            )
            st.caption("Sezioni escluse: " + ", ".join(dropped))
            # filled from the verbale's Dimissione when it has one, the prognosis with its Esito
            diagnosis = st.text_input("Diagnosi *", value=diagnosis, key=f"diagnosis_{uploaded.file_id}")
            prognosis = st.text_input("Prognosi", value=prognosis, key=f"prognosis_{uploaded.file_id}",
                                      help="Facoltativa: se vuota si assume la dimissione a domicilio. "
                                           "Scrivi «ricovero» per un paziente ricoverato.")

            if st.button(button, type="primary"):
                if not diagnosis.strip():
                    st.error("Campo obbligatorio mancante: Diagnosi")
                else:
                    # the Prognosi label stays even when empty: it ends the diagnosis for extract_patient_fields,
                    # and an empty prognosis reads as a home discharge downstream (is_hospitalised)
                    italian_note = f"{note.strip()}\n\nDiagnosi: {diagnosis.strip()}\n\nPrognosi: {prognosis.strip()}"

if italian_note:
    try:
        result = run_pipeline(italian_note, llm_queries=query_mode == "Generate da LLM",
                              use_cross_encoder=retrieval_mode != "Rapida", prompt_only=prompt_only,
                              translate_by_llm=translator == "LLM di generazione",
                              translate_in_prompt=prompt_mode == "Con traduzione in italiano")
        st.session_state["result"] = (prompt_only, result)
        st.session_state.pop("edited", None)  # the box would otherwise keep the previous result's edit
    except Exception as e:
        st.error(f"Errore durante l'elaborazione: {e}")

# the last result stays until the next one, whatever the mode switched to since
if "result" in st.session_state:
    is_prompt, result = st.session_state["result"]
    if is_prompt:
        st.subheader("Prompt finale")
        st.caption("Copia il prompt (icona in alto a destra del riquadro) e incollalo nell'LLM desiderato.")
        # appended here rather than when building, so switching the option does not rerun translation and retrieval
        suffix = ITALIAN_TRANSLATION if prompt_mode == "Con traduzione in italiano" else ""
        st.code(result + suffix, language=None, wrap_lines=True)
    else:
        st.subheader("Raccomandazioni")
        st.caption("Bozza da rivedere prima dell'uso. Le sigle in apice aprono il PDF della linea guida citata alla "
                   "pagina indicata, quando il PDF è nella cartella Guidelines/.")
        # the edit replaces the result, which the links are rebuilt from: a citation left as written keeps its link
        if st.toggle("Modifica", help="Rende il testo modificabile, con le citazioni scritte per esteso: lasciate come "
                                      "sono, tornano link quando chiudi la modifica."):
            st.text_area("Raccomandazioni", result, height=600, key="edited", label_visibility="collapsed",
                         on_change=lambda: st.session_state.update(result=(False, st.session_state["edited"])))
        else:
            with st.container(border=True):
                st.html(with_links(result))

# the models load as the page opens rather than on the first generation
if st.runtime.exists():
    load_pipeline_models()
    load_cross_encoder()


# index rebuild
st.divider()

with st.expander("⚙️ Aggiorna indice linee guida"):
    import build_index

    guidelines_dir = GUIDELINES
    guidelines_dir.mkdir(exist_ok=True)
    current_pdfs = sorted(guidelines_dir.glob("*.pdf"))

    st.write(f"Linee guida attualmente nella cartella `Guidelines/`: **{len(current_pdfs)}**")

    uploaded = st.file_uploader(
        "Carica nuove linee guida (PDF)",
        type=["pdf"],
        accept_multiple_files=True,
    )
    if uploaded and st.button("Salva PDF nella cartella Guidelines"):
        saved, skipped = [], []
        for f in uploaded:
            dest = guidelines_dir / f.name
            if dest.exists():
                skipped.append(f.name)
            else:
                dest.write_bytes(f.getbuffer())
                saved.append(f.name)
        if saved:
            st.success(f"Salvati: {', '.join(saved)}")
        if skipped:
            st.warning(f"Già presenti (non sovrascritti): {', '.join(skipped)}")

    st.markdown(
        "**Nota:** la ricostruzione rifà l'indice **da zero** su tutti i PDF nella "
        "cartella (estrazione, chunking, embedding, FAISS, TF-IDF). Con l'intero "
        "corpus richiede diversi minuti. L'app resta inutilizzabile durante il processo."
    )

    confirm = st.checkbox("Conferma")
    if st.button("Ricostruisci indice", type="secondary", disabled=not confirm):
        try:
            with st.status("Ricostruzione indice in corso...", expanded=True) as status:
                st.write("Estrazione, chunking, embedding e salvataggio...")
                build_index.main(guidelines_dir)
                status.update(label="Indice ricostruito", state="complete")
            # invalidate only index artifacts, heavy models stay cached
            load_index_artifacts.clear()
            st.success("Indice aggiornato. Le prossime generazioni useranno il nuovo indice.")
        except Exception as e:
            st.error(f"Errore durante la ricostruzione: {e}")
