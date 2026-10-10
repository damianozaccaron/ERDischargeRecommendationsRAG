import re, torch, time, sys
from pathlib import Path
from collections import defaultdict

from config import (
    MODEL_NAME,
    EMBEDDING_MODEL,
    NER_MODEL,
    CROSS_ENCODER,
    RETRIEVAL_K,
    FINAL_N,
    TEMPERATURE,
    OPENROUTER_KEY,
    API,
    RETRIEVAL_ONLY,
    USE_EXISTING_TRANSLATION,
    QUANT,
    REPO,
    LLM_QUERIES,
    QUERY_MODEL,
    LLM_TRANSLATION,
    OUTPUT_LLM_TRANSLATION,
    OUTPUT_TRANSLATION_IN_PROMPT)

from storage.storage import load_index_and_metadata, load_vectorizer, load_table_store
from storage.embedding import embed_query, load_embedder
from retrieval.retrieval import retrieve_top_k, load_crossEncoder, reranking_multi_query, retrieve_tables

from generation.output_prod import load_model_quant, load_model_transformer, generate_letter

from retrieval.query_building import build_queries, load_ner_model, patient_qualifiers
from generation.prompting import build_prompt, is_hospitalised, ITALIAN_TRANSLATION
from pipeline.translation import deepl_translation, deepl_translation_en_it, llm_translation, llm_translation_en_it
from pipeline.user_input import collect_patient_input, extract_patient_fields
from pipeline.verbale import parse_verbale


def load_models():
    
    start_models = time.time()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading NER pipeline...")
    ner_model = load_ner_model(model_name=NER_MODEL, device=device)
    print("Loading Vectorizer...")
    vectorizer = load_vectorizer()    
    print("Loading embedder...")
    emb_model = load_embedder(EMBEDDING_MODEL)
    print("Loading cross-encoder...")
    cross_encoder = load_crossEncoder(CROSS_ENCODER)
    end_models = time.time()
    print("Loading FAISS index and metadata...")
    index, metadata = load_index_and_metadata()
    print("Loading table store...")
    tables, table_index = load_table_store()
    print(f"Models loaded in {end_models-start_models:.3f} seconds")

    tokenizer = None

    # LLM_QUERIES without the API needs the local model loaded even when only the queries use it
    if API or (RETRIEVAL_ONLY and not (LLM_QUERIES and not QUERY_MODEL)):
        llm_model = None
    elif QUANT:
        # quantized models
        print("Loading LLM...")
        llm_model = load_model_quant(repo=REPO, model_name=MODEL_NAME)
    else:
        # non-quantized models (transformers library)
        print("Loading LLM...")
        device = ("cuda" if torch.cuda.is_available() else "cpu")
        tokenizer, llm_model = load_model_transformer(MODEL_NAME, device)

    return ner_model, vectorizer, emb_model, cross_encoder, index, metadata, llm_model, tokenizer, tables, table_index


def root_of(note: Path) -> Path:
    """The folder of a note's translation and results: Examples for Examples/ITA/AF.txt, Data for Data/ITA/5.txt."""
    return note.parent.parent if note.parent.name == "ITA" else note.parent


def verbale_to_note(pdf: Path) -> Path:
    """
    A Verbale di Pronto Soccorso read into its anonymised Clinical Note, written to ITA/ beside the PDF, and its
    physician's discharge advice to recommendations/, the gold standard the output is compared with. The note never
    holds the advice. Returns the note's path.
    """
    v = parse_verbale(pdf.read_bytes())
    note, rec = pdf.parent / "ITA" / f"{pdf.stem}.txt", pdf.parent / "recommendations" / f"{pdf.stem}_rec.txt"
    for path, text in ((note, f"{v.note}\n\nDiagnosi: {v.diagnosis}\n\nPrognosi: {v.prognosis}\n"),
                       (rec, "Si consiglia:\n" + "".join(f"- {a}\n" for a in v.advice))):
        path.parent.mkdir(exist_ok=True)
        path.write_text(text, encoding="utf-8")
    return note


def main(input_file: Path, ner_model, vectorizer, emb_model, cross_encoder, index, metadata, llm_model, tokenizer, tables, table_index, use_existing_transl = False):
    print(f"\n{'='*50}")
    print(f"Processing: {input_file.name}")
    print(f"{'='*50}")

    # PATIENT DATA COLLECTION
    # italian_data = collect_patient_input()
  
    eng_path = root_of(input_file) / "ENG" / f"translation{input_file.name}"
    if use_existing_transl and eng_path.exists():
        translated_note = eng_path.read_text(encoding="utf-8")
    else:
        start_transl = time.time()
        with open(input_file, "r", encoding="utf-8") as f:
            italian_data = f.read()

        print("Translating clinical note...")
        if LLM_TRANSLATION:
            translated_note = llm_translation(italian_data, llm_model, tokenizer)
        else:
            translated_note = deepl_translation(italian_data)

        eng_path.parent.mkdir(exist_ok=True)
        eng_path.write_text(translated_note, encoding="utf-8")
        print(f"Translation saved to {eng_path}")
        end_transl = time.time()
        print(f"Translation time: {end_transl-start_transl:.3f} seconds")


    # RETRIEVAL
    start_retrieval = time.time()
    print("Building queries...")
    queries, floors, scopes = build_queries(translated_note, ner_model, vectorizer, emb_model, index, metadata, llm_model, tokenizer)

    print("Retrieving relevant chunks...")
    all_candidates = {}
    for i, q in enumerate(queries):
        candidates, faiss_ids = retrieve_top_k(
            query_embedding=embed_query(q, embedder=emb_model),
            index=index,
            metadata=metadata,
            k=RETRIEVAL_K,
            docs=scopes.get(i),
        )
        for cand, fid in zip(candidates, faiss_ids):
            # every query that surfaced it, not just the first: an ablation needs to know the chunk
            # would still have arrived through another query
            all_candidates.setdefault(fid, {**cand, "retrieved_by": []})["retrieved_by"].append(q)

    top_k_candidates = list(all_candidates.values())
    print(f"{len(queries)} queries, {len(top_k_candidates)} candidates")

    if len(top_k_candidates) == 0:
        raise ValueError("No candidates retrieved from top_k retrieval")

    # re-ranking
    print("Reranking...")
    start = time.time()
    final_chunks = reranking_multi_query(queries, top_k_candidates, cross_encoder, top_n=FINAL_N,
                                         floors=floors, sparse=True)
    end = time.time()

    print(f"Total time for re-reranking: {end-start:.3f} seconds")

    if len(final_chunks) == 0:
        raise ValueError("No candidates retrieved from Cross-Encoder")
    
    end_retrieval = time.time()
    print(f"Kept {len(final_chunks)} excerpts of {FINAL_N} "
          f"({final_chunks[0]['final_score']:.2f}..{final_chunks[-1]['final_score']:.2f})")
    print(f"Executed retrieval in {end_retrieval-start_retrieval:.3f} seconds.")

    # TABLES (second hop): only a home discharge prescribes, so admissions skip them
    fields = extract_patient_fields(translated_note)
    prognosis = fields.get("prognosis", "")
    final_tables = []
    if not is_hospitalised(prognosis):
        print("Retrieving tables...")
        final_tables = retrieve_tables(translated_note, final_chunks, fields.get("diagnosis", ""), patient_qualifiers(fields),
                                       ner_model, emb_model, cross_encoder, tables, table_index)

    # LETTER GENERATION
    print("Building prompt...")
    prompt = build_prompt(translated_note, final_chunks, prognosis=prognosis, tables=final_tables)
    translate_in_prompt = OUTPUT_LLM_TRANSLATION and OUTPUT_TRANSLATION_IN_PROMPT
    if translate_in_prompt:
        prompt += ITALIAN_TRANSLATION

    print("Generating recommendations...\n")
    if RETRIEVAL_ONLY:
        output = prompt
    else:
        backend = "local"
        if API:
            backend = "openrouter"
            llm_model = MODEL_NAME
        start_gen = time.time()
        output = generate_letter(prompt=prompt, model=llm_model, tokenizer=tokenizer, backend=backend, 
                                 temperature=TEMPERATURE, max_tokens=4096 if translate_in_prompt else 2048,
                                 api_key=OPENROUTER_KEY)
        end_gen = time.time()
        print(f"Generation time: {end_gen-start_gen:.3f} seconds")

        if OUTPUT_LLM_TRANSLATION and not translate_in_prompt:
            print("Translating recommendations in Italian...\n")
            output += "\n\nTRADUZIONE IN ITALIANO\n\n" + llm_translation_en_it(output, llm_model, tokenizer, backend)

    #output = deepl_translation_en_it(output)

    print("=== GENERATED RECOMMENDATIONS ===\n")

    # provenance, inverted: what each query accounts for in the final set, chunks numbered as in the
    # chunks dump. A query with nothing next to it earned no excerpt and is what an ablation should cut.
    retrieved, scored, floored = defaultdict(list), defaultdict(list), defaultdict(list)
    for n, c in enumerate(final_chunks, 1):
        for q in c.get("retrieved_by", []):
            retrieved[q].append(n)
        scored[c.get("scored_by")].append(n)
        if c.get("floor_query"):
            floored[c["floor_query"]].append(n)
    nums = lambda ns: ", ".join(map(str, ns)) if ns else "none"

    dumped = [f"[floor {floors.get(i, 0)}{' | in ' + ', '.join(sorted(scopes[i])) if i in scopes else ''}] "
              f"retrieved {nums(retrieved[q])} | scored {nums(scored[q])}"
              f"{f' | filled floor with {nums(floored[q])}' if floored[q] else ''}\n{q}"
              for i, q in enumerate(queries)]
    return output, dumped, final_chunks, final_tables

if __name__ == "__main__":

    # notes (.txt), Verbali (.pdf) or folders of them, Examples/ITA by default; a bare name is a note of Examples/ITA
    paths = [Path(a) if Path(a).exists() else Path("Examples/ITA") / a for a in sys.argv[1:]] or [Path("Examples/ITA")]
    inputs = []
    for path in paths:
        if not path.exists():
            print(f"File not found: {path}")
            exit(1)
        inputs += sorted([*path.glob("*.txt"), *path.glob("*.pdf")],
                         key=lambda p: [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", p.name)]) \
            if path.is_dir() else [path]
    if not inputs:
        print(f"No .txt or .pdf files in {', '.join(map(str, paths))}")
        exit(1)
    txt_files = [verbale_to_note(p) if p.suffix.lower() == ".pdf" else p for p in inputs]

    print(f"Found {len(txt_files)} file(s)\n")

    ner_model, vectorizer, emb_model, cross_encoder, index, metadata, model, tokenizer, tables, table_index = load_models()

    start_total = time.time()
    for input_file in txt_files:
        start = time.time()
        output, queries, chunks, final_tables = main(
            input_file=input_file,
            ner_model=ner_model,
            vectorizer=vectorizer,
            emb_model=emb_model,
            cross_encoder=cross_encoder,
            index=index,
            metadata=metadata,
            use_existing_transl=USE_EXISTING_TRANSLATION,
            llm_model=model,
            tokenizer=tokenizer,
            tables=tables,
            table_index=table_index,
        )
        end = time.time()
        print(f"{input_file} processed in {end-start:.3f} seconds.")

        stem = input_file.stem
        results_dir = root_of(input_file) / "results"
        output_path = results_dir / f"output/{stem}_output.txt"
        queries_path = results_dir / f"queries/{stem}_queries.txt"
        for path in (output_path, queries_path):
            path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w", encoding="utf-8") as f:
            f.write(output)

        with open(queries_path, "w", encoding="utf-8") as f:
            for i, q in enumerate(queries, 1):
                f.write(f"--- Query {i} ---\n{q}\n\n")
                
        # Append original recommendations for comparison
        rec_path = root_of(input_file) / "recommendations" / f"{stem}_rec.txt"
        if rec_path.exists() and not RETRIEVAL_ONLY:
            with open(rec_path, "r", encoding="utf-8") as f:
                original_rec = f.read()
            with open(output_path, "a", encoding="utf-8") as f:
                f.write("\n\n" + "=" * 60 + "\n")
                f.write("ORIGINAL RECOMMENDATIONS (for comparison)\n")
                f.write("=" * 60 + "\n")
                f.write(original_rec)

        chunks_path = results_dir / f"chunks/{stem}_chunks.txt"
        chunks_path.parent.mkdir(parents=True, exist_ok=True)
        with open(chunks_path, "w", encoding="utf-8") as f:
            for i, c in enumerate(chunks, 1):
                f.write(f"--- Chunk {i} | {c['doc_id']} | score {c['final_score']:.3f} ---\n"
                        f"retrieved by: {' | '.join(c.get('retrieved_by', []))}\n"
                        f"scored by:    {c.get('scored_by', '?')} ({c.get('rerank_score', float('nan')):.3f})\n"
                        f"floor:        {c.get('floor_query') or '-'}\n\n{c["text"]}\n\n")
            for i, t in enumerate(final_tables, 1):
                f.write(f"--- Table {i} | {t['table_id']} | score {t['score']:.3f} ---\n{t['caption_raw']}\n{t['markdown']}\n\n")

        print(f"Results saved to {output_path} and {queries_path}")
    end_total = time.time()
    print(f"\nAverage execution time: {(end_total-start_total)/len(txt_files):.3f} seconds")