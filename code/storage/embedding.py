from sentence_transformers import SentenceTransformer
import torch
from typing import List, Dict
from config import EMBEDDING_MODEL


def load_embedder(model_name=EMBEDDING_MODEL):
    emb_model = SentenceTransformer(model_name, device="cuda" if torch.cuda.is_available() else "cpu")
    return emb_model

def embed_docs(chunks: List[Dict], embedder, batch_size=8) -> List[Dict]:
    """
    Embed a list of text chunks using the provided embedder, takes the parameter text from the chunk and returns 
    a list of dicts with the same metadata and the embedding vector added as a new key "embedding".
    """

    texts = [c["embed_text"] if "embed_text" in c else c["text"] for c in chunks]

    embeddings = embedder.encode_document(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    for chunk, emb in zip(chunks, embeddings):
        chunk["embedding"] = emb

    return chunks


def embed_query(query, embedder):
    embedding = embedder.encode_query(
        query,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    return embedding.reshape(1, -1)
