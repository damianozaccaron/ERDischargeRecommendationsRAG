import re
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import requests
import torch
from sentence_transformers import CrossEncoder
from collections import Counter

from ingestion.tables import normalize
from retrieval.query_building import extract_entities
from storage.embedding import embed_query

from config import TABLE_N, OPENROUTER_KEY

TABLE_ANCHOR = "dose regimen criteria score"  # broad query only; a term's query is "<term> dose"
TABLE_MIN_SCORE = 0.05     # reranker probability under which a table is noise, not a weak match
NOTE_TERM_BOOST = 0.2      # a row reached by a medication in the clinical note scores (1 + this) times more
GUIDELINE_BONUS = 0.0      # a table from a guideline with two or more excerpts scores (1 + this) times more; never implemented
LABELLED = re.compile(r"^\W*(?:tables?|box|recommendations?|figure)\b", re.I)

def retrieve_top_k(query_embedding, index, metadata, k=30, docs=None):
    """
    Performs a similarity search in the FAISS index using the query embedding.
    Returns the top-k most similar chunks along with their metadata and scores. With docs, only chunks of those
    guidelines: the search goes deeper and keeps the first k of theirs.
    """

    scores, indices = index.search(query_embedding, k if docs is None else min(index.ntotal, 50 * k))

    results = []
    faiss_ids = []

    for idx, score in zip(indices[0], scores[0]):
        if idx == -1 or (docs is not None and metadata[idx]["doc_id"] not in docs):
            continue
        if len(results) == k:
            break

        item = metadata[idx].copy()
        item["score"] = float(score)

        results.append(item)
        faiss_ids.append(idx)

    return results, faiss_ids


def build_pool(queries: list[tuple[str, int]], embedder, index, metadata, cap: int) -> list[dict]:
    """
    The candidates the cross-encoder will score, from queries paired with the share of the pool each may
    take. Filled round by round rather than by score: a fine query about one drug retrieves at lower
    cosines than a broad query about the diagnosis, so cutting the merged list by score would drop exactly
    the specific chunks the fine queries were asked for. Round-robin instead gives every query its first
    chunk before any query gets its second, and a query stops once it has taken its share.
    """
    ranked = [retrieve_top_k(embed_query(q, embedder), index, metadata, k=k) for q, k in queries]
    pool, taken = {}, [0] * len(queries)
    for depth in range(max((k for _, k in queries), default=0)):
        for i, (candidates, faiss_ids) in enumerate(ranked):
            if len(pool) >= cap:
                return list(pool.values())
            if taken[i] >= queries[i][1] or depth >= len(candidates):
                continue
            pool.setdefault(int(faiss_ids[depth]), {**candidates[depth], "sponsor": queries[i][0]})
            taken[i] += 1
    return list(pool.values())


def mmr_select(
    query_embedding: np.ndarray,
    candidates: list[dict],
    faiss_ids: list[int],
    index,
    top_j=15,
    lambda_=0.5):
    """
    Reranks candidates using Maximal Marginal Relevance (MMR) to balance relevance and diversity.
    Deprecated but could potentially be re-implemented.
    """

    candidate_embeddings = np.stack([index.reconstruct(int(i)) for i in faiss_ids])

    selected = []
    selected_indices = [] # we also keep the indices

    # similarity(query, candidate)
    sim_to_query = candidate_embeddings @ query_embedding.T  # matrix multiplication with result dimensions (k, dim)x(dim,1) -> (k, 1)
    sim_to_query = sim_to_query.flatten() # a list of similarity scores between the query and each candidate, with shape (k,)

    for _ in range(top_j):
        mmr_scores = []

        for i in range(len(candidates)):
            # if the candidate is already selected, we set its MMR score to -inf to exclude it from selection
            if i in selected_indices:
                mmr_scores.append(-np.inf)
                continue
  
            if not selected_indices:
                diversity_penalty = 0
            else:
                # max similarity to already selected chunks
                sim_to_selected = candidate_embeddings[i] @ candidate_embeddings[selected_indices].T
                diversity_penalty = np.max(sim_to_selected)

            mmr_score = (
                lambda_ * sim_to_query[i]
                - (1 - lambda_) * diversity_penalty
            )
            mmr_scores.append(mmr_score)

        best_idx = int(np.argmax(mmr_scores))
        selected_indices.append(best_idx)
        selected.append(candidates[best_idx])

    return selected


class ApiReranker:
    """
    A rerank model of OpenRouter's /rerank endpoint ("qwen/qwen3-reranker-8b" used in the tests) behind the CrossEncoder's predict.
    """
    BATCH = 64

    def __init__(self, model: str):
        self.model, self.cost = model, 0.0

    def rank(self, query: str, docs: list[str]) -> list[float]:
        for wait in (2, 4, 8, 16, 32, 64, None):
            response = requests.post("https://openrouter.ai/api/v1/rerank", timeout=180,
                                     headers={"Authorization": f"Bearer {OPENROUTER_KEY}"},
                                     json={"model": self.model, "query": query, "documents": docs})
            data = response.json()
            if "results" in data:
                self.cost += data.get("usage", {}).get("cost", 0.0)
                scores = [0.0] * len(docs)
                for r in data["results"]:
                    scores[r["index"]] = r["relevance_score"]
                return scores
            if response.status_code not in (429, 500, 502, 503) or wait is None:
                raise RuntimeError(f"{self.model} rerank failed: {data}")
            time.sleep(wait)

    def predict(self, pairs, **_):
        """Scores in the order of the pairs: one request per query and BATCH of its documents, 4 at a time."""
        positions = {}
        for i, (query, _doc) in enumerate(pairs):
            positions.setdefault(query, []).append(i)
        batches = [(q, ids[j:j + self.BATCH]) for q, ids in positions.items() for j in range(0, len(ids), self.BATCH)]
        scores = [0.0] * len(pairs)
        with ThreadPoolExecutor(4) as pool:
            for (_q, ids), got in zip(batches, pool.map(lambda b: self.rank(b[0], [pairs[i][1] for i in b[1]]), batches)):
                for i, s in zip(ids, got):
                    scores[i] = s
        return scores


def load_crossEncoder(model_name: str = "BAAI/bge-reranker-large"):
    """A local cross-encoder by its Hugging Face id, or an OpenRouter rerank model as "openrouter:<its id>"."""
    if model_name.startswith("openrouter:"):
        return ApiReranker(model_name.removeprefix("openrouter:"))
    reranker = CrossEncoder(model_name, activation_fn=torch.nn.Sigmoid(), device="cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        reranker.model.half()  # 3.5x faster on a laptop GPU, same top-10 within 0.003 of the scores
    return reranker

def reranking(query: str, retrieved_chunks: list[dict], reranker: CrossEncoder, top_n: int = 6) -> list[dict]:
    """
    Performs cross-encoder reranking of retrieved chunks based on the query. 
    Deprecated in favor of the more complex multi-query approach.
    """

    texts = [chunk["text"] for chunk in retrieved_chunks]
    ranks = reranker.rank(query, texts, return_documents=True)
    
    # map ranked results back to original chunk dicts to preserve metadata (corpus_id maintains original order)
    ranked_chunks = []
    for rank in ranks[:top_n]:
        original_chunk = retrieved_chunks[rank["corpus_id"]]
        original_chunk["rerank_score"] = rank["score"]
        # chunks already automatically sorted by highest score
        ranked_chunks.append(original_chunk)
    
    return ranked_chunks


def text_overlap(a: str, b: str, threshold: float = 0.8) -> bool:
    """
    Check if two chunk texts are near-duplicates using token overlap.
    """

    tokens_a = set(a.lower().split())
    tokens_b = set(b.lower().split())

    if not tokens_a or not tokens_b:
        return False

    intersection = len(tokens_a & tokens_b)
    smaller = min(len(tokens_a), len(tokens_b))

    return (intersection / smaller) >= threshold


def remove_duplicates(chunks: list[dict]) -> list[dict]:
    """
    Remove near-duplicate chunks, keeping the higher-scored one.
    Assumes chunks are already sorted by score descending.
    """
    kept = []
    for chunk in chunks:
        is_dup = False
        for existing in kept:
            if text_overlap(chunk["text"], existing["text"]):
                is_dup = True
                break
        if not is_dup:
            kept.append(chunk)
    return kept


def identify_primary_source(
    retrieved_chunks: list[dict],
    dominance_ratio: float = 0.4,
) -> set[str]:
    """
    Identify the primary guideline source(s) based on retrieval frequency.
    If a single doc_id accounts for more than dominance_ratio of all retrieved chunks, it is considered the
    primary source. Chunks from other sources receive a penalty during reranking.
    """
    doc_counts = Counter(chunk["doc_id"] for chunk in retrieved_chunks)
    total = len(retrieved_chunks)

    if total == 0:
        return set()

    primary = set()
    for doc_id, count in doc_counts.items():
        if count / total >= dominance_ratio:
            primary.add(doc_id)

    # If no single source dominates, skip penalty entirely
    if not primary:
        return set(doc_counts.keys())

    return primary


def reranking_multi_query(
    queries: list[str],
    retrieved_chunks: list[dict],
    reranker: CrossEncoder,
    top_n = 8,
    category_boost = 0.1,
    source_penalty = 0.15,
    dominance_ratio = 0.35,
    floors: dict[int, int] = None,
    floor_min_score: float = 0.05,
    sparse: bool = False,
) -> list[dict]:
    """
    Rerank chunks by scoring each chunk against each query independently,
    then aggregating with max. Applies category leader boost and text deduplication.

    Pipeline:
      1. Score every (query, chunk) pair with the cross-encoder. With sparse, only the pairs where the query
         retrieved the chunk (chunk["retrieved_by"]): a chunk then needs both encoders to agree, which on the
         example notes chose better excerpts than the full matrix at a quarter of the time, and the cost grows
         with queries x k instead of queries x pool.
      2. For each chunk, take the MAX score across all queries.
      3. Identify the top-scoring chunk per query (category leader) and apply a percentage boost to its max score.
      4. Penalize chunks from non-primary sources
      5. Sort by final score, deduplicate near-identical chunks.
      6. A query with a floor keeps that many of its best chunks in the top-N whatever the other queries
         score, so a patient facet (a comorbidity, a drug of the stay) is represented even when the general
         diagnosis chunks outscore it everywhere. Floors are served strongest query first and fill at most half of
         top_n: a long note lists two dozen facets (a home therapy one drug per line, a nurse's diary), and floors
         filling every slot left a COPD exacerbation without one COPD chunk.
         A chunk under floor_min_score is noise and never fills a floor. The rest of the top-N is filled by
         final score.
    """

    texts = [chunk["text"] for chunk in retrieved_chunks]
    n_chunks = len(texts)
    n_queries = len(queries)

    # Step 1: every (query, chunk) pair, or with sparse only the pairs where the query retrieved the chunk
    cells = [(qi, ci) for qi, q in enumerate(queries) for ci, c in enumerate(retrieved_chunks)
             if not sparse or q in c.get("retrieved_by", ())]
    pairs = [(queries[qi], texts[ci]) for qi, ci in cells]

    # longest first: a batch is padded to its longest pair, so grouping similar lengths cuts the padding the
    # cross-encoder computes (~30% faster on the notes, same scores)
    order = np.argsort([-len(q) - len(t) for q, t in pairs])
    scores = np.empty(len(pairs))
    scores[order] = reranker.predict([pairs[i] for i in order])

    # (n_queries, n_chunks), a pair left unscored counts as 0
    score_matrix = np.zeros((n_queries, n_chunks))
    score_matrix[tuple(zip(*cells))] = scores

    # Step 2
    max_scores = score_matrix.max(axis=0)
    # provenance: which query's score survived the max, for ablation without a qualitative read
    for i, chunk in enumerate(retrieved_chunks):
        chunk["scored_by"] = queries[int(score_matrix[:, i].argmax())]
        chunk["rerank_score"] = float(max_scores[i])
        chunk["floor_query"] = None

    # Step 3
    # For each query, find the chunk that scored highest on that query
    category_leaders = set()
    for q_idx in range(n_queries):
        best_chunk_idx = int(score_matrix[q_idx].argmax())
        category_leaders.add(best_chunk_idx)

    boosted_scores = max_scores.copy()
    for chunk_idx in category_leaders:
        boosted_scores[chunk_idx] *= (1.0 + category_boost)

    # Step 4
    primary_sources = identify_primary_source(retrieved_chunks, dominance_ratio)
    all_doc_ids = set(chunk["doc_id"] for chunk in retrieved_chunks)

    # Only penalize if there are clear dominant sources
    penalty = np.ones(n_chunks)
    if primary_sources < all_doc_ids:
        penalty = np.array([1.0 if c["doc_id"] in primary_sources else 1.0 - source_penalty for c in retrieved_chunks])
        boosted_scores *= penalty
        print(f"  Source penalty: {int((penalty < 1).sum())}/{n_chunks} chunks penalized "
              f"(primary: {primary_sources})")

    # Step 5
    for i, chunk in enumerate(retrieved_chunks):
        chunk["final_score"] = float(boosted_scores[i])

    sorted_chunks = sorted(
        retrieved_chunks,
        key=lambda c: c["final_score"],
        reverse=True,
    )
    final = remove_duplicates(sorted_chunks)

    # Step 6: floors are served strongest query first (by its best source-penalized score) and stop at top_n, so
    # when more queries hold a floor than there are slots, the ones the corpus answers worst give way
    survivors = {id(c): c for c in final}
    weighted = score_matrix * penalty
    kept = []
    for q_idx, floor in sorted((floors or {}).items(), key=lambda f: -weighted[f[0]].max()):
        taken = 0
        for chunk_idx in np.argsort(-weighted[q_idx]):
            c = retrieved_chunks[chunk_idx]
            if weighted[q_idx, chunk_idx] < floor_min_score or taken == floor or len(kept) >= top_n // 2:
                break
            if id(c) not in survivors or c in kept:
                continue
            c["floor_query"] = queries[q_idx]
            kept.append(c)
            taken += 1
    kept += [c for c in final if c not in kept][:max(0, top_n - len(kept))]

    return sorted(kept, key=lambda c: c["final_score"], reverse=True)


def dense_select(ranked: list[list[dict]], top_n: int = 20) -> list[dict]:
    """
    Selection without the cross-encoder, for machines without a GPU. Round-robin over each query's FAISS ranking:
    every query gets its first chunk before any query gets its second, which is what the floors guarantee with
    the cross-encoder. A chunk already kept, or a near-duplicate of one, is skipped.
    """
    kept = []
    for depth in range(max(map(len, ranked), default=0)):
        for chunks in ranked:
            if len(kept) == top_n:
                return kept
            if depth < len(chunks) and not any(text_overlap(chunks[depth]["text"], k["text"]) for k in kept):
                kept.append(chunks[depth])
    return kept


def retrieve_tables(
    note: str,
    excerpts: list[dict],
    diagnosis: str,
    qualifiers: str,
    ner_model,
    embedder,
    reranker: CrossEncoder | None,
    tables: list[dict],
    table_index,
    top_n: int = TABLE_N,
    max_excerpt_terms: int = 8,
    broad_k: int = 5,
    note_boost: float = NOTE_TERM_BOOST,
    guideline_bonus: float = GUIDELINE_BONUS,
) -> list[dict]:
    """
    Second hop, after the text stage: grid tables for the drugs in play. Terms are the note's
    medications plus the medications most often named across the excerpts; a term reaches the rows
    of a table from the excerpts' guidelines when one of its words appears in them (or in the
    caption or header, which reach every row), and a broad diagnosis query adds the rows of the
    closest tables by embedding. The cross-encoder scores every (query, row surrogate) pair, a
    term's query being "<term> dose": a dose table rarely restates the diagnosis, and a query naming
    it scores the rows near zero ("apixaban <diagnosis>" 0.01 on the DOAC dose table, "apixaban dose"
    0.99). A table scores its best row, a note term's rows weigh more than an excerpt term's, and a
    split table takes one slot with its best part. Only tables with a LABELLED caption compete.
    """
    # step 1: term set, deduplicated on the normalized form, first surface form kept for the query
    terms = {}
    for term in extract_entities(note, ner_model).get("MEDICATION", []):
        terms.setdefault(" ".join(normalize(term)), (term, True))
    counts, surface = Counter(), {}
    for ex in excerpts:
        found = {" ".join(normalize(t)): t for t in extract_entities(ex["text"], ner_model, filter_negated=False).get("MEDICATION", [])}
        counts.update(found.keys())
        for key, t in found.items():
            surface.setdefault(key, t)
    for key, _ in counts.most_common(max_excerpt_terms):
        terms.setdefault(key, (surface[key], False))
    terms.pop("", None)

    # step 2: scope to the excerpts' guidelines
    doc_counts = Counter(ex["doc_id"] for ex in excerpts)
    labelled = [i for i, t in enumerate(tables) if LABELLED.match(t["caption_clean"])]
    scoped = [i for i in labelled if tables[i]["doc_id"] in doc_counts]
    if not scoped:
        # the broad query alone: a term's query does not name the diagnosis, so across the whole store
        # a drug name would reach the dose tables of any condition (paediatric doses for an adult)
        print("  [tables] no table from the excerpts' guidelines, broad query over the whole store")
        scoped, terms = labelled, {}

    # step 3: word match, one pair per (term, row) hit
    pairs = {}  # (query, row surrogate) -> (table position, weight)
    for key, (term, from_note) in terms.items():
        words = set(key.split())
        query = f"{term} dose"
        for i in scoped:
            t = tables[i]
            rows = [r for r in t["rows"] if words & (r["key_tokens"] | r["group_tokens"] | r["other_tokens"])]
            if not rows and words & (t["header_tokens"] | set(normalize(t["caption_clean"]))):
                rows = t["rows"]
            for r in rows:
                pairs[(query, r["row_surrogate"])] = (i, 1 + note_boost if from_note else 1.0)

    # step 4: broad query, the rows of the closest scoped tables by embedding
    broad = f"{diagnosis} {TABLE_ANCHOR} {qualifiers}".strip()
    _, ids = table_index.search(embed_query(broad, embedder), len(tables))
    in_scope = set(scoped)
    closest = [int(i) for i in ids[0] if int(i) in in_scope][:broad_k]
    for i in closest:
        for r in tables[i]["rows"]:
            pairs.setdefault((broad, r["row_surrogate"]), (i, 1.0))
    if not pairs:
        return []

    # steps 5-6: one batched cross-encoder call, a table scores its best weighted row. Without the cross-encoder
    # the query is scored against the table's embedding instead, so a row match only decides which tables compete
    if reranker is None:
        vecs = {q: embed_query(q, embedder)[0] for q, _ in pairs}
        scores = [float(vecs[q] @ table_index.reconstruct(i)) for (q, _), (i, _) in pairs.items()]
    else:
        scores = reranker.predict(list(pairs))
    best = {}
    for (i, weight), score in zip(pairs.values(), scores):
        best[i] = max(best.get(i, -np.inf), float(score) * weight)
    for i in best:
        if doc_counts[tables[i]["doc_id"]] >= 2:
            best[i] *= 1 + guideline_bonus
    slots = {}  # one per table id: its best part
    for i in sorted(best, key=best.get, reverse=True):
        slots.setdefault(tables[i]["table_id"], i)
    ranked = [i for i in slots.values() if best[i] >= TABLE_MIN_SCORE][:top_n]
    if closest and doc_counts[tables[closest[0]]["doc_id"]] and len(ranked) < top_n and all(tables[i]["table_id"] != tables[closest[0]]["table_id"] for i in ranked):
        ranked.append(closest[0])
    print(f"  [tables] {len(terms)} terms, {len(pairs)} pairs over {len(best)} tables: {sorted(t for t, _ in terms.values())}")
    return [{**tables[i], "score": best.get(i, 0.0)} for i in ranked]


if __name__ == "__main__":
    # the API reranker regroups the pairs by query and batch: each score must come back to its pair
    class Fake(ApiReranker):
        BATCH = 2
        def rank(self, query, docs):
            return [float(f"{query}{d}") for d in docs]
    pairs = [("1", "1"), ("2", "1"), ("1", "2"), ("1", "3"), ("2", "2"), ("1", "4")]
    assert Fake("x").predict(pairs) == [11, 21, 12, 13, 22, 14]
