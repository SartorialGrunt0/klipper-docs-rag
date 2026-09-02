"""Cross-encoder rerank client: llama.cpp --rerank /v1/rerank endpoint.

The bi-encoder (nomic cosine) scores query and chunk independently; a
cross-encoder reads the (query, chunk) pair jointly and is far more
accurate at ~100x the compute — affordable only for rescoring the top
~10 fused candidates down to the final k. Degrades open-circuit: any
rerank failure returns the pre-rerank order rather than failing the
request.

Endpoint shape (Cohere/Jina-compatible, llama.cpp --rerank --pooling
rank): POST {"model", "query", "documents": [...], "top_n"}
-> {"results": [{"index", "relevance_score"}, ...]}
"""
from __future__ import annotations

import json
from typing import Callable, Sequence

import httpx

from kb_rag.index import SearchResult

# bge-reranker-v2-m3 has an 8k-token window shared across query+doc; at
# 10 candidates per request each doc is safely truncated to ~1500 chars
# (chunk target is 256-512 tokens, so this only clips oversized stanzas).
DEFAULT_MAX_DOC_CHARS = 1500


class RerankClient:
    def __init__(self, url: str, model: str = "bge-reranker-v2-m3",
                 max_doc_chars: int = DEFAULT_MAX_DOC_CHARS,
                 timeout: float = 30.0,
                 post: Callable | None = None) -> None:
        self.url = url
        self.model = model
        self.max_doc_chars = max_doc_chars
        self.timeout = timeout
        self._post = post or httpx.post

    def rerank(self, query: str, docs: Sequence[str],
               top_n: int | None = None) -> list[float]:
        """Return one relevance score per doc, in input order.

        llama.cpp returns results sorted by score desc with original
        index; we unsort so callers can zip docs with scores directly.
        """
        payload = {"model": self.model, "query": query,
                   "documents": [d[: self.max_doc_chars] for d in docs]}
        if top_n:
            payload["top_n"] = top_n
        r = self._post(self.url, json=payload, timeout=self.timeout)
        r.raise_for_status()
        body = r.json()
        scores = [float("-inf")] * len(docs)
        for item in body["results"]:
            scores[item["index"]] = float(item["relevance_score"])
        return scores


def rerank_hybrid(client: RerankClient, query: str,
                  search_fn: Callable[[int], list],
                  k: int, candidates: int) -> list[SearchResult]:
    """Rescore search_fn(candidates) with the cross-encoder, return top-k.

    search_fn(n) -> list[SearchResult] (e.g. a partial over
    hybrid_search). On ANY rerank failure the pre-rerank top-k is
    returned unchanged (open-circuit): a down reranker must not take the
    chat path with it. Returned SearchResult.score is the rerank
    relevance score, not the fused RRF score.
    """
    try:
        cands = search_fn(candidates)
        if not cands:
            return []
        scores = client.rerank(query, [c.chunk.text for c in cands])
        order = sorted(range(len(cands)), key=lambda i: -scores[i])
        return [SearchResult(chunk=cands[i].chunk, score=scores[i])
                for i in order[:k]]
    except Exception:
        return search_fn(k)[:k]
