import numpy as np
import faiss, pickle, joblib
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer


def build_faiss_index(chunks):
    """ extract embeddings in the same order as chunks """

    embeddings = np.stack(
        [c["embedding"] for c in chunks]
    ).astype("float32") # turn embeddings into a 2D NumPy array and convert to float32 for faiss (which only accepts float32)

    dim = embeddings.shape[1]
    # create a FAISS index for inner product similarity search (equivalent to cosine similarity if embeddings are normalized, as they are here)
    index = faiss.IndexFlatIP(dim)

    index.add(embeddings)

    return index


def build_metadata(chunks):
    """Everything but the vector (held by FAISS) and the embedding-only text."""
    return [{"kind": "prose", **{k: v for k, v in c.items() if k not in ("embedding", "embed_text")}} for c in chunks]


def sanity_check(index, metadata):
    assert index.ntotal == len(metadata)

def save_index_and_metadata(index, metadata, index_path="Weights/faiss.index", metadata_path="Weights/metadata.pkl"):
    faiss.write_index(index, index_path)

    with open(metadata_path, "wb") as f:
        pickle.dump(metadata, f)


def load_index_and_metadata(index_path="Weights/faiss.index", metadata_path="Weights/metadata.pkl"):

    if not Path(index_path).exists() or not Path(metadata_path).exists():
        raise RuntimeError("FAISS index or metadata not found.")
    
    index = faiss.read_index(index_path)

    with open(metadata_path, "rb") as f:
        metadata = pickle.load(f)

    sanity_check(index, metadata)

    return index, metadata


def save_table_store(tables, index, index_path="Weights/tables.index", store_path="Weights/tables.pkl"):
    """Grid tables (Phase 3 step 2): a flat inner-product index over the table surrogates, position i <-> tables[i]."""
    faiss.write_index(index, index_path)
    with open(store_path, "wb") as f:
        pickle.dump(build_metadata(tables), f)


def load_table_store(index_path="Weights/tables.index", store_path="Weights/tables.pkl"):
    if not Path(index_path).exists() or not Path(store_path).exists():
        raise RuntimeError("Table store not found. Run the indexing pipeline first.")
    index = faiss.read_index(index_path)
    with open(store_path, "rb") as f:
        tables = pickle.load(f)
    sanity_check(index, tables)
    return tables, index


def build_vectorizer(metadata: list[dict]) -> TfidfVectorizer:
    """
    Fit a TfidfVectorizer on the guideline chunk corpus. Stores the IDF weights.
    """

    corpus = [chunk["text"] for chunk in metadata]
 
    vectorizer = TfidfVectorizer(
        stop_words="english",
        token_pattern=r'(?u)[a-zA-Z0-9µ²³]+(?:[-/][a-zA-Z0-9µ²³]+)*', # allows to catch terms such as CHA2DS2-VASc as a single one
        min_df=2,
        max_df=0.85,
    )
 
    vectorizer.fit(corpus)
    return vectorizer


def save_vectorizer(vectorizer, path="Weights/tfidf_vectorizer.joblib"):
    joblib.dump(vectorizer, path)


def load_vectorizer(path="Weights/tfidf_vectorizer.joblib"):
    if not Path(path).exists():
        raise RuntimeError(
            f"TF-IDF vectorizer not found at {path}. "
            "Run the indexing pipeline first to fit and save it."
        )
    return joblib.load(path)
