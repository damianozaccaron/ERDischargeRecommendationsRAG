from ingestion.pdf_parsing import extract_folder
from ingestion.chunking import create_chunks
from ingestion.tables import build_tables
from storage.embedding import load_embedder, embed_docs
from storage.storage import build_faiss_index, build_metadata, save_index_and_metadata, sanity_check, save_vectorizer, build_vectorizer, save_table_store
from config import EMBEDDING_MODEL
from pathlib import Path
import argparse, pickle, faiss

def main(rel_path):

    print("Loading embedder...")
    embedder = load_embedder()

    print("Extracting PDFs...")
    data_dir = rel_path
    pages = extract_folder(data_dir)
    print(f"Extracted {len(pages)} pages")

    print("Chunking...")
    chunks = create_chunks(pages)
    sentence_rows, tables, _ = build_tables(pages)
    chunks += sentence_rows  # recommendation rows are text; grid tables get their own store below
    print(f"Extracted {len(chunks)} chunks ({len(sentence_rows)} table rows) and {len(tables)} grid tables")

    print("Embedding...")
    emb_chunks = embed_docs(chunks, embedder)

    print("Building FAISS index...")
    index = build_faiss_index(emb_chunks)
    metadata = build_metadata(emb_chunks)

    sanity_check(index, metadata)

    print("Saving index and metadata...")
    save_index_and_metadata(index, metadata)

    print("Performing TF-IDF...")
    vectorizer = build_vectorizer(chunks)

    print("Saving TF-IDF Vectorizer...")
    save_vectorizer(vectorizer)
    print(f"Vocabulary size: {len(vectorizer.vocabulary_)} terms")

    print("Embedding and saving table store...")
    save_table_store(tables, build_faiss_index(embed_docs(tables, embedder)))

    print("Index successfully built.")


def embed_only():
    """Rebuild the two FAISS indexes with EMBEDDING_MODEL from the chunks saved in Weights/, skipping extraction and
    chunking. Overwrites faiss.index and tables.index; the pickles and the TF-IDF vectorizer are untouched."""
    embedder = load_embedder()
    for store, index in [("metadata.pkl", "faiss.index"), ("tables.pkl", "tables.index")]:
        with open(f"Weights/{store}", "rb") as f:
            chunks = pickle.load(f)
        for c in chunks:
            if "table_surrogate" in c:  # grid tables were embedded on their surrogate, which build_metadata drops as embed_text
                c["embed_text"] = c["table_surrogate"]
        print(f"Embedding {len(chunks)} entries of {store} with {EMBEDDING_MODEL}...")
        faiss.write_index(build_faiss_index(embed_docs(chunks, embedder)), f"Weights/{index}")
    print("Indexes rebuilt.")


def check_chunking(rel_path):
    print("Extracting PDFs...")
    data_dir = rel_path
    pages = extract_folder(data_dir)
    print(f"Extracted {len(pages)} pages")

    print("Chunking...")
    chunks = create_chunks(pages) + build_tables(pages)[0]

    print(f"Extracted {len(chunks)} chunks")

    output = []
    for chunk in chunks:
        output.append(chunk["text"])

    with open('output.txt', 'w', encoding='utf-8') as file:
        file.write("\n\n\n".join(output))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--embedonly", action="store_true", help="re-embed the saved chunks with EMBEDDING_MODEL, no extraction")
    if parser.parse_args().embedonly:
        embed_only()
    else:
        main(Path("./Guidelines"))
