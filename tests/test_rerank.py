"""Tests for the cross-encoder rerank client (no live server involved)."""
from __future__ import annotations

import numpy as np

from kb_rag.chunk import Chunk
from kb_rag.index import KbIndex
from kb_rag.rerank import RerankClient, rerank_hybrid


def _chunk(i: int) -> Chunk:
    return Chunk(id=f"klipper/Config_Reference::[s{i}]",
                 doc="Config_Reference", section=f"[s{i}]", source="klipper",
                 breadcrumb="crumb", text=f"body text {i}", tokens=5)


class FakePost:
    """Canned Cohere-shaped /v1/rerank responses keyed by document order."""

    def __init__(self, scores: list[float]):
        self.scores = scores
        self.calls: list[dict] = []

    def __call__(self, url, json=None, timeout=None):
        self.calls.append({"url": url, "json": json})
        docs = json["documents"]
        ranked = sorted(range(len(docs)),
                        key=lambda i: -self.scores[i])[: json.get("top_n")]
        payload = {"results": [{"index": i,
                                "relevance_score": self.scores[i]}
                               for i in ranked]}

        class Resp:
            status_code = 200

            @staticmethod
            def raise_for_status():
                pass

            @staticmethod
            def json():
                return payload

        return Resp()


def test_rerank_client_returns_scores_in_doc_order():
    post = FakePost([1.0, 3.0, 2.0])
    rc = RerankClient("http://unused/v1/rerank", post=post)
    scores = rc.rerank("q", ["a", "b", "c"])
    assert scores == [1.0, 3.0, 2.0]
    assert post.calls[0]["json"]["query"] == "q"


def test_rerank_truncates_docs_to_char_budget():
    post = FakePost([0.0, 0.0])
    rc = RerankClient("http://unused/v1/rerank", max_doc_chars=10, post=post)
    rc.rerank("q", ["x" * 100, "y" * 100])
    sent = post.calls[0]["json"]["documents"]
    assert all(len(d) <= 10 for d in sent)


def test_rerank_hybrid_reorders_top_candidates():
    # hybrid would return c1 (dense winner); rerank prefers c2
    chunks = [_chunk(1), _chunk(2)]
    post = FakePost([-5.0, 4.0])
    rc = RerankClient("http://unused/v1/rerank", post=post)

    class Idx:  # minimal stub: rerank_hybrid only needs .count()-able data
        pass

    def search_fn(k):  # stands in for hybrid_search
        return [type("R", (), {"chunk": chunks[0], "score": 0.9})(),
                type("R", (), {"chunk": chunks[1], "score": 0.8})()]

    out = rerank_hybrid(rc, "q", search_fn, k=2, candidates=2)
    assert [r.chunk.id for r in out] == [chunks[1].id, chunks[0].id]
    assert out[0].score == 4.0  # relevance score, not RRF


def test_rerank_failure_degrades_to_undisrupted_order():
    def boom(url, json=None, timeout=None):
        raise ConnectionError("down")

    chunks = [_chunk(1), _chunk(2)]

    def search_fn(k):
        return [type("R", (), {"chunk": chunks[0], "score": 0.9})(),
                type("R", (), {"chunk": chunks[1], "score": 0.8})()]

    rc = RerankClient("http://unused/v1/rerank", post=boom)
    out = rerank_hybrid(rc, "q", search_fn, k=2, candidates=2)
    assert [r.chunk.id for r in out] == [chunks[0].id, chunks[1].id]
