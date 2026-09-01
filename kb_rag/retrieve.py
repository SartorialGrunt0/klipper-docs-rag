"""Hybrid retrieval: FTS5/BM25 sparse leg + dense leg, fused with RRF.

Design (docs/RAG-RESEARCH.md §4.2):
- The sparse leg keeps exact identifiers whole (``heater_fan``, ``tmc2209``,
  ``cycle_time``) which dense-only retrieval occasionally buries; the dense
  leg covers natural-language paraphrase. Reciprocal Rank Fusion (Cormack
  et al., k=60) merges the two rankings without score normalization.
- Query sanitization is strict: every term becomes a quoted FTS5 phrase,
  so no operator characters (``:()*-"```) can ever reach MATCH raw.
"""
from __future__ import annotations

import re
from collections import defaultdict

import numpy as np

from kb_rag.index import KbIndex, SearchResult

_MAX_TERMS = 16  # long queries degrade gracefully; BM25 tops out early anyway
_RRF_K = 60

_TERM_RE = re.compile(r"[A-Za-z0-9._+]+")
_HAS_ALNUM = re.compile(r"[A-Za-z0-9]")
_IDENTISH = re.compile(r"[._+]|[\d]")
# Identifier-like query terms: contains '.' '_' '+' or a digit (heater_fan,
# tmc2209, cycle_time, CANBus... case handled by lowercasing first).
_IDENT_RE = re.compile(r"[a-z0-9]*(?:[._+]|\d)[a-z0-9._+]*")


def fts_query(text: str) -> str:
    """Turn a free-text query into a safe OR-of-quoted-terms FTS5 MATCH string.

    Terms are lowercased (unicode61 indexes lowercase; quoted phrases match
    the index as-is), must contain at least one alphanumeric character, and
    identifier-ish terms (``heater_fan``, ``tmc2209``, ``cycle_time``) are
    ordered first so BM25 weight lands where users mean it. Special
    characters are dropped, never escaped-and-kept.
    Returns "" when no usable terms remain.
    """
    idents: list[str] = []
    words: list[str] = []
    seen: set[str] = set()
    for m in _TERM_RE.finditer(text):
        t = m.group(0).lower()
        if not _HAS_ALNUM.search(t) or t in seen:
            continue
        seen.add(t)
        (idents if _IDENTISH.search(t) else words).append(t)
    ordered = (idents + words)[:_MAX_TERMS]
    if not ordered:
        return ""
    return " OR ".join(f'"{t}"' for t in ordered)


def rrf_fuse(
    rankings: list[list[tuple[str, int]]],
    k_rrf: int = _RRF_K,
    weights: list[float] | None = None,
) -> list[tuple[str, float]]:
    """Reciprocal Rank Fusion over (id, rank) lists from each leg.

    RRF(d) = sum over legs of weight / (k + rank). Weighted variant:
    at corpus scale the dense leg is the strong ranking and the sparse
    leg is a precision booster for exact identifiers, so sparse gets a
    weight < 1 (it must never veto a dense hit with common-word noise).
    Returns (id, score) pairs sorted by fused score descending; ties
    broken by first-seen order.
    """
    if weights is None:
        weights = [1.0] * len(rankings)
    scores: dict[str, float] = defaultdict(float)
    for leg, w in zip(rankings, weights):
        for doc_id, rank in leg:
            scores[doc_id] += w / (k_rrf + rank)
    return sorted(scores.items(), key=lambda kv: -kv[1])


def _fts_search(
    index: KbIndex, query: str, k: int, sources: set[str] | None
) -> list[SearchResult]:
    """BM25 leg straight against chunks_fts (fresh connection per call)."""
    import sqlite3

    con = sqlite3.connect(index.path)
    try:
        sql = (
            "SELECT chunk_id, bm25(chunks_fts) AS s FROM chunks_fts"
            " WHERE chunks_fts MATCH ?"
        )
        args: list = [query]
        if sources:
            sql += " AND source IN (%s)" % ",".join("?" * len(sources))
            args += sorted(sources)
        sql += " ORDER BY s LIMIT ?"  # bm25() is negative-good: ascending
        args.append(k)
        rows = con.execute(sql, args).fetchall()
    except sqlite3.OperationalError:
        return []
    finally:
        con.close()
    out: list[SearchResult] = []
    for chunk_id, _s in rows:
        chunk = index.get(chunk_id)
        if chunk is not None:
            out.append(SearchResult(chunk=chunk, score=0.0))
    return out


def hybrid_search(
    index: KbIndex,
    qvec: np.ndarray | None = None,
    text: str = "",
    k: int = 3,
    sources: set[str] | None = None,
    k_rrf: int = _RRF_K,
    candidate_k: int | None = None,
    dense_weight: float = 1.0,
    sparse_weight: float = 0.5,
) -> list[SearchResult]:
    """Dense + sparse retrieval fused by weighted RRF; top-k SearchResults.

    The sparse leg is *identifier-gated*: only identifier-like query terms
    (containing '.', '_', '+' or a digit — ``heater_fan``, ``tmc2209``,
    ``cycle_time``) feed FTS5, at half weight. Eval (phase 2, n=27) showed
    full prose OR-queries flood fusion with common-word hits that evict
    dense winners (0.926 -> 0.852 recall); at w=0.5 identifier-gated the
    leg recovers parity (0.926) and boosts exact-identifier hits without
    vetoing prose results.

    Either leg alone is fine (missing qvec/text, or no identifiers in
    text, skips that leg). Score in the returned SearchResult is the
    fused RRF score, not a cosine.
    """
    candidate_k = candidate_k or max(3 * k, 10)
    legs: list[list[tuple[str, int]]] = []
    weights: list[float] = []
    by_id: dict[str, SearchResult] = {}

    if qvec is not None:
        dense = index.search_vector(qvec, k=candidate_k, sources=sources)
        legs.append([(r.chunk.id, i + 1) for i, r in enumerate(dense)])
        weights.append(dense_weight)
        for r in dense:
            by_id.setdefault(r.chunk.id, r)

    if text:
        ident_terms = " ".join(_IDENT_RE.findall(text.lower()))
        q = fts_query(ident_terms) if ident_terms else ""
        if q:
            sparse = _fts_search(index, q, candidate_k, sources)
            if sparse:
                legs.append([(r.chunk.id, i + 1) for i, r in enumerate(sparse)])
                weights.append(sparse_weight)
                for r in sparse:
                    by_id.setdefault(r.chunk.id, r)

    if not legs:
        return []

    fused = rrf_fuse(legs, k_rrf=k_rrf, weights=weights)
    out: list[SearchResult] = []
    for chunk_id, score in fused[:k]:
        base = by_id.get(chunk_id)
        if base is not None:
            out.append(SearchResult(chunk=base.chunk, score=score))
    return out
