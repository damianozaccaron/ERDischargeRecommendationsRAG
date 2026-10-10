"""One check: provenance fields on the reranked chunks. PYTHONPATH=code python -m retrieval.test_provenance"""
import numpy as np
from retrieval.retrieval import reranking_multi_query


class StubEncoder:
    """q0 leads on A and likes C; only q1 prefers B, and B's score never beats C's."""
    SCORES = {("q0", "A"): 0.9, ("q0", "B"): 0.1, ("q0", "C"): 0.7,
              ("q1", "A"): 0.2, ("q1", "B"): 0.3, ("q1", "C"): 0.25}

    def predict(self, pairs):
        return np.array([self.SCORES[(q, t)] for q, t in pairs])


def test():
    chunks = [{"text": t, "doc_id": "d1", "retrieved_by": [f"rq-{t}"]} for t in "ABC"]
    # by final score the top 2 would be A and C; q1's floor of 1 has to pull B in over C
    final = reranking_multi_query(["q0", "q1"], chunks, StubEncoder(), top_n=2, floors={1: 1})

    by_text = {c["text"]: c for c in final}
    assert set(by_text) == {"A", "B"}, f"q1's floor should displace C with B: {list(by_text)}"

    a, b = by_text["A"], by_text["B"]
    assert a["retrieved_by"] == ["rq-A"] and b["retrieved_by"] == ["rq-B"]
    assert a["scored_by"] == "q0" and abs(a["rerank_score"] - 0.9) < 1e-6
    assert b["scored_by"] == "q1" and abs(b["rerank_score"] - 0.3) < 1e-6
    assert a["floor_query"] is None, "A won on score, not on a floor"
    assert b["floor_query"] == "q1", f"B came in through q1's floor, got {b['floor_query']}"
    print("ok")


def test_sparse_floors():
    """Sparse: a chunk is scored only by the queries that retrieved it. Floors: strongest query first, never past top_n."""
    chunks = [{"text": "A", "doc_id": "d1", "retrieved_by": ["q0"]},
              {"text": "B", "doc_id": "d1", "retrieved_by": ["q1"]},
              {"text": "C", "doc_id": "d1", "retrieved_by": ["q1"]}]
    final = reranking_multi_query(["q0", "q1"], chunks, StubEncoder(), top_n=1, floors={1: 1, 0: 1}, sparse=True)

    c = chunks[2]
    assert c["scored_by"] == "q1" and abs(c["rerank_score"] - 0.25) < 1e-6, "q0 never retrieved C, so never scores it"
    assert [x["text"] for x in final] == ["A"], f"q0's floor (0.9) goes before q1's (0.3) and fills the one slot: {final}"
    print("ok")


if __name__ == "__main__":
    test()
    test_sparse_floors()
