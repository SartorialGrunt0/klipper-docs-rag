"""Embedding client for llama-server OpenAI-compatible /v1/embeddings.

nomic-embed-text-v1.5 contract: chunks are embedded with the
``search_document: `` prefix, queries with ``search_query: ``. Forgetting a
prefix does NOT error — it silently tanks recall — so the prefixing lives
here, in one place, not in callers.

Vectors are L2-normalized defensively (regardless of server-side
--embd-normalize) so dot product == cosine everywhere downstream.
"""
from __future__ import annotations

from typing import Any, Callable

import httpx
import numpy as np

NOMIC_DOC_PREFIX = "search_document: "
NOMIC_QUERY_PREFIX = "search_query: "

DEFAULT_BASE_URL = "http://192.168.1.135:8100"
DEFAULT_BATCH_SIZE = 8
# llama-server rejects a /v1/embeddings request whose combined input tokens
# exceed the server's --ubatch-size (verified on CachyPC: a 587-real-token
# request 500'd against the default ubatch 512; unit now runs -b 2048 -ub
# 2048). approx_tokens undercounts BPE ~20% on code/JSON-heavy text, so
# 1500 est keeps actual requests under ~1800.
DEFAULT_BATCH_TOKENS = 1500
# llama-server / nomic context is 2048 tokens; 6000 chars is a conservative
# ~4x slack for the technical-doc character mix. Overlong inputs are hard-
# truncated (chunker's 500-token target means this should never trigger;
# it exists so a rogue doc can't 400 the batch).
DEFAULT_MAX_CHARS = 6000


def _post(url: str, *, json: dict[str, Any], timeout: float = 120.0):
    """Module-level seam so tests can monkeypatch the transport."""
    return httpx.post(url, json=json, timeout=timeout)


class EmbedClient:
    def __init__(
        self,
        base_url: str = DEFAULT_BASE_URL,
        model: str = "nomic-embed-text-v1.5",
        batch_size: int = DEFAULT_BATCH_SIZE,
        batch_tokens: int = DEFAULT_BATCH_TOKENS,
        max_chars: int = DEFAULT_MAX_CHARS,
        timeout: float = 120.0,
    ) -> None:
        self.url = base_url.rstrip("/") + "/v1/embeddings"
        self.model = model
        self.batch_size = batch_size
        self.batch_tokens = batch_tokens
        self.max_chars = max_chars
        self.timeout = timeout

    # ── public API ─────────────────────────────────────────────────

    def embed_documents(self, texts: list[str]) -> list[np.ndarray]:
        return self._embed(texts, NOMIC_DOC_PREFIX)

    def embed_query(self, text: str) -> np.ndarray:
        return self._embed([text], NOMIC_QUERY_PREFIX)[0]

    # ── internals ──────────────────────────────────────────────────

    def _embed(self, texts: list[str], prefix: str) -> list[np.ndarray]:
        if not texts:
            return []
        prepared = self._prepare(texts, prefix)
        out: list[np.ndarray] = []
        for batch in self._pack(prepared):
            for emb in self._post_with_split(batch):
                out.append(self._normalize(emb))
        return out

    def _post_with_split(self, batch: list[str]) -> list[list[float]]:
        """Post a batch; on server 500, bisect to isolate the offender."""
        try:
            return self._post_batch(batch)
        except httpx.HTTPStatusError as e:
            if e.response.status_code != 500 or len(batch) == 1:
                raise RuntimeError(
                    f"embedding server rejected one input "
                    f"({len(batch[0])} chars): {e.response.text[:200]}"
                ) from e
            mid = len(batch) // 2
            return (self._post_with_split(batch[:mid])
                    + self._post_with_split(batch[mid:]))

    def _post_batch(self, batch: list[str]) -> list[list[float]]:
        resp = _post(
            self.url,
            json={"model": self.model, "input": batch},
            timeout=self.timeout,
        )
        resp.raise_for_status()
        data = sorted(resp.json()["data"], key=lambda d: d["index"])
        return [d["embedding"] for d in data]

    def _pack(self, texts: list[str]) -> list[list[str]]:
        """Greedy pack: <= batch_size items AND <= batch_tokens est/tokens."""
        from kb_rag.chunk import approx_tokens

        batches: list[list[str]] = []
        cur: list[str] = []
        cur_tok = 0
        for t in texts:
            tok = approx_tokens(t)
            if cur and (len(cur) >= self.batch_size or cur_tok + tok > self.batch_tokens):
                batches.append(cur)
                cur, cur_tok = [], 0
            cur.append(t)
            cur_tok += tok
        if cur:
            batches.append(cur)
        return batches

    def _prepare(self, texts: list[str], prefix: str) -> list[str]:
        prepared = []
        for t in texts:
            t = t.strip()
            if not t.startswith(prefix):
                t = prefix + t
            # cap total length INCLUDING the prefix
            limit = self.max_chars + len(prefix)
            if len(t) > limit:
                t = t[:limit].rstrip()
            prepared.append(t)
        return prepared

    @staticmethod
    def _normalize(vec: list[float]) -> np.ndarray:
        v = np.asarray(vec, dtype=np.float32)
        norm = float(np.linalg.norm(v))
        return v / norm if norm else v
