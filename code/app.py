"""
Streamlit UI for beta users: builds the final LLM prompt from the PDF of an
Italian Verbale di Pronto Soccorso, anonymised before anything is translated
(or from a note pasted as plain text, which is taken as it is),
and displays it for copy-paste into an external LLM app/site.

Run with:  streamlit run app.py
"""

import re

import torch
import streamlit as st

from config import EMBEDDING_MODEL, NER_MODEL, CROSS_ENCODER, RETRIEVAL_K, FINAL_N, FAST_FINAL_N, PROMPT_MAX_CHARS

from storage.storage import load_index_and_metadata, load_vectorizer, load_table_store
from storage.embedding import embed_query, load_embedder
from retrieval.retrieval import retrieve_top_k, load_crossEncoder, reranking_multi_query, retrieve_tables, dense_select
from retrieval.query_building import build_queries, load_ner_model, patient_qualifiers
from generation.prompting import fit_prompt, is_hospitalised, ITALIAN_TRANSLATION
from pipeline.translation import deepl_translation
from pipeline.user_input import extract_patient_fields
from pipeline.verbale import parse_verbale


# ────────────────────────── model loading (cached) ──────────────────────────

@st.cache_resource(show_spinner="Caricamento modelli...")
def load_heavy_models():
    """NER ed embedder: caricati una volta, mai invalidati."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ner_model = load_ner_model(model_name=NER_MODEL, device=device)
    emb_model = load_embedder(EMBEDDING_MODEL)
    return ner_model, emb_model


@st.cache_resource(show_spinner="Caricamento cross-encoder...")
def load_cross_encoder():
    """Solo in modalità accurata: la modalità rapida non lo carica mai."""
    return load_crossEncoder(CROSS_ENCODER)


@st.cache_resource(show_spinner="Caricamento indice FAISS e vectorizer...")
def load_index_artifacts():
    """Indice, metadata e TF-IDF: invalidati dopo una rebuild dell'indice."""
    vectorizer = load_vectorizer()
    index, metadata = load_index_and_metadata()
    tables, table_index = load_table_store()
    return vectorizer, index, metadata, tables, table_index


def load_pipeline_models():
    ner_model, emb_model = load_heavy_models()
    vectorizer, index, metadata, tables, table_index = load_index_artifacts()
    return ner_model, vectorizer, emb_model, index, metadata, tables, table_index


# ────────────────────────── pipeline (up to prompt) ─────────────────────────

def generate_prompt(italian_note: str, llm_queries: bool, use_cross_encoder: bool) -> str:
    ner_model, vectorizer, emb_model, index, metadata, tables, table_index = load_pipeline_models()
    cross_encoder = load_cross_encoder() if use_cross_encoder else None
    fallback = []

    with st.status("Elaborazione in corso...", expanded=True) as status:
        st.write("Traduzione della nota clinica...")
        translated_note = deepl_translation(italian_note)

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

        st.write("Costruzione del prompt finale...")
        # the Italian block is reserved even when not requested, since it can be switched on after generation
        prompt, dropped_chunks, dropped_tables = fit_prompt(PROMPT_MAX_CHARS - len(ITALIAN_TRANSLATION), translated_note,
                                                            final_chunks, fields.get("prognosis", ""), final_tables)

        status.update(label="Prompt generato", state="complete", expanded=False)

    # outside the status box, which collapses once done
    if fallback:
        st.warning(f"Generazione delle query con LLM non riuscita ({fallback[0]}): il prompt è stato costruito "
                   "con le query automatiche.")
    if dropped_chunks or dropped_tables:
        st.info(f"Per restare entro {PROMPT_MAX_CHARS} caratteri sono stati esclusi gli ultimi {dropped_chunks} estratti "
                f"e {dropped_tables} tabelle per rilevanza.")
    return prompt


# ──────────────────────────────────── UI ────────────────────────────────────

st.set_page_config(page_title="Generatore Prompt - Raccomandazioni di Dimissione", layout="wide")

st.title("Generatore di prompt per raccomandazioni di dimissione")
st.caption(
    "Carica il verbale di Pronto Soccorso in PDF (o incolla la nota come testo). Il sistema lo anonimizza, recupera "
    "gli estratti delle linee guida rilevanti e produce il prompt finale da incollare "
    "nell'LLM generativo di tua scelta."
)

query_mode = st.radio(
    "Generazione delle query di retrieval",
    ["Automatiche", "Generate da LLM"],
    horizontal=True,
    help="Automatiche: costruite dalla nota tramite regole lessicali automatizzate. Generate da LLM: leggermente più accurate, ma richiedono l'accesso a un LLM"
         "; se la chiamata fallisce si usano quelle automatiche.",
)

retrieval_mode = st.radio(
    "Selezione degli estratti",
    ["Rapida", "Accurata (cross-encoder)"],
    index=1,
    horizontal=True,
    help=f"Rapida: Estratti scelti per similarità, pochi secondi. Accurata: un cross-encoder "
         f"riordina i candidati; più preciso, ma senza GPU può essere pesante.",
)

prompt_mode = st.radio(
    "Prompt",
    ["Standard", "Con traduzione in italiano"],
    horizontal=True,
    help="Con traduzione in italiano: il prompt chiede all'LLM di aggiungere, dopo le due sezioni in inglese, "
         "la traduzione integrale e fedele in italiano di tutto il testo. Vale anche per un prompt già generato.",
)

italian_note = None

use_text = st.checkbox(
    "Incolla la nota come testo",
    help="Al posto del PDF: la nota clinica in testo libero, con le righe «Diagnosi: ...» e «Prognosi: ...» "
         "in fondo.",
)

if use_text:
    text = st.text_area("Nota clinica", height=400, key="note_text")
    st.caption("Il testo non viene anonimizzato: incollalo senza dati identificativi.")
    if st.button("Genera prompt", type="primary"):
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

            if st.button("Genera prompt", type="primary"):
                if not diagnosis.strip():
                    st.error("Campo obbligatorio mancante: Diagnosi")
                else:
                    # the Prognosi label stays even when empty: it ends the diagnosis for extract_patient_fields,
                    # and an empty prognosis reads as a home discharge downstream (is_hospitalised)
                    italian_note = f"{note.strip()}\n\nDiagnosi: {diagnosis.strip()}\n\nPrognosi: {prognosis.strip()}"

if italian_note:
    try:
        prompt = generate_prompt(italian_note, llm_queries=query_mode == "Generate da LLM",
                                 use_cross_encoder=retrieval_mode != "Rapida")
        st.session_state["generated_prompt"] = prompt
    except Exception as e:
        st.error(f"Errore durante l'elaborazione: {e}")

if "generated_prompt" in st.session_state:
    st.subheader("Prompt finale")
    st.caption("Copia il prompt (icona in alto a destra del riquadro) e incollalo nell'LLM desiderato.")
    # appended here rather than when building, so switching the option does not rerun translation and retrieval
    suffix = ITALIAN_TRANSLATION if prompt_mode == "Con traduzione in italiano" else ""
    st.code(st.session_state["generated_prompt"] + suffix, language=None, wrap_lines=True)


# ───────────────────── index rebuild (rare operation) ───────────────────────

st.divider()

with st.expander("⚙️ Aggiorna indice linee guida"):
    from pathlib import Path
    import build_index

    guidelines_dir = Path("Guidelines")
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
