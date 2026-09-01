"""SQLite-backed chunk + vector store with brute-force dense retrieval.

Corpus size (~1-3K chunks) is the reason there is no vector database:
1246 x 768 float32 is ~3.8 MB — a single numpy matmul beats any ANN index
at this scale, and one file is the whole datastore.

Schema:
  chunks(id TEXT PK, doc, section, source, breadcrumb, tokens INT, text)
  vectors(chunk_id TEXT PK REFERENCES chunks, vec BLOB)  -- float32 LE
  meta(key TEXT PK, value TEXT)                          -- JSON values
"""
from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from kb_rag.chunk import Chunk

_LLM_SCHEMA_HINT = """
CREATE TABLE IF NOT EXISTS chunks (
    id TEXT PRIMARY KEY,
    doc TEXT NOT NULL,
    section TEXT,
    source TEXT NOT NULL,
    breadcrumb TEXT NOT NULL,
    tokens INTEGER NOT NULL,
    text TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS vectors (
    chunk_id TEXT PRIMARY KEY REFERENCES chunks(id) ON DELETE CASCADE,
    vec BLOB NOT NULL
);
CREATE TABLE IF NOT EXISTS meta (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
"""


def _fts_schema() -> str:
    # Sparse leg: FTS5/BM25 over chunk text + section.
    # tokenize keeps '.', '_', '+' as token chars (identifiers like
    # heater_fan, tmc2209, CANBus stay whole); NO porter stemming --
    # stemmers damage exact identifier hits, and the dense leg already
    # covers prose paraphrase.
    return """
CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
    text,
    section,
    chunk_id UNINDEXED,
    doc UNINDEXED,
    source UNINDEXED,
    tokenize = "unicode61 tokenchars '._+'"
);
"""


def _ensure_fts(con: sqlite3.Connection) -> None:
    """Create chunks_fts if missing and backfill it from chunks.

    Makes a phase-1 (dense-only) index file work with the sparse leg
    without a rebuild: FTS content is derived, always safe to regenerate.
    """
    has = con.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table'"
        " AND name='chunks_fts'"
    ).fetchone()
    if has:
        return
    con.execute(_fts_schema())
    con.execute(
        "INSERT INTO chunks_fts (text, section, chunk_id, doc, source)"
        " SELECT text, COALESCE(section,''), id, doc, source FROM chunks"
    )
    con.commit()


@dataclass
class SearchResult:
    chunk: Chunk
    score: float


class KbIndex:
    def __init__(self, path: Path, dim: int | None = None) -> None:
        self.path = Path(path)
        self.dim = dim
        self.chunks: list[Chunk] = []
        self.vectors: np.ndarray = np.zeros((0, dim or 0), dtype=np.float32)
        self.meta: dict = {}
        self._id_index: dict[str, int] = {}

    # ── write path ─────────────────────────────────────────────────

    def save(
        self, chunks: list[Chunk], vectors: np.ndarray, meta: dict
    ) -> None:
        if len(chunks) != vectors.shape[0]:
            raise ValueError(
                f"{len(chunks)} chunks vs {vectors.shape[0]} vectors")
        if chunks and self.dim and vectors.shape[1] != self.dim:
            raise ValueError(
                f"vector dim {vectors.shape[1]} != index dim {self.dim}")
        if chunks and not vectors.flags["C_CONTIGUOUS"]:
            vectors = np.ascontiguousarray(vectors, dtype=np.float32)
        vectors = np.asarray(vectors, dtype=np.float32)

        self.path.parent.mkdir(parents=True, exist_ok=True)
        con = sqlite3.connect(self.path)
        try:
            con.executescript(_LLM_SCHEMA_HINT)
            con.executescript(_fts_schema())
            # full replace: chunking/embedding is deterministic from docs,
            # incremental updates are not worth the complexity at this size
            con.execute("DELETE FROM vectors")
            con.execute("DELETE FROM chunks_fts")
            con.execute("DELETE FROM chunks")
            con.execute("DELETE FROM meta")
            con.executemany(
                "INSERT INTO chunks VALUES (?,?,?,?,?,?,?)",
                [(c.id, c.doc, c.section, c.source, c.breadcrumb, c.tokens,
                  c.text) for c in chunks],
            )
            con.executemany(
                "INSERT INTO vectors VALUES (?,?)",
                [(c.id, vectors[i].tobytes()) for i, c in enumerate(chunks)],
            )
            con.executemany(
                "INSERT INTO chunks_fts (text, section, chunk_id, doc, source)"
                " VALUES (?,?,?,?,?)",
                [(c.text, c.section or "", c.id, c.doc, c.source)
                 for c in chunks],
            )
            con.executemany(
                "INSERT INTO meta VALUES (?,?)",
                [(k, json.dumps(v)) for k, v in meta.items()],
            )
            con.commit()
        finally:
            con.close()
        self.chunks = list(chunks)
        if vectors.ndim == 1:
            vectors = vectors.reshape(1, -1)
        self.vectors = vectors  # (0, N) empty case keeps its trailing dim
        self.meta = dict(meta)
        self._rebuild_id_index()

    # ── read path ──────────────────────────────────────────────────

    @classmethod
    def load(cls, path: Path) -> "KbIndex":
        obj = cls(path)
        con = sqlite3.connect(path)
        try:
            _ensure_fts(con)
            rows = con.execute(
                "SELECT c.id, c.doc, c.section, c.source, c.breadcrumb,"
                " c.tokens, c.text, v.vec"
                " FROM chunks c JOIN vectors v ON v.chunk_id = c.id"
                " ORDER BY c.rowid"
            ).fetchall()
            obj.meta = {
                k: json.loads(v) for k, v in con.execute("SELECT key,value FROM meta")
            }
        finally:
            con.close()
        obj.chunks = [
            Chunk(id=r[0], doc=r[1], section=r[2], source=r[3],
                  breadcrumb=r[4], tokens=r[5], text=r[6])
            for r in rows
        ]
        if rows:
            dim = len(rows[0][7]) // 4
            obj.vectors = np.stack([
                np.frombuffer(r[7], dtype=np.float32) for r in rows
            ]) if rows else np.zeros((0, dim), dtype=np.float32)
            obj.dim = dim
        return obj

    def count(self) -> int:
        return len(self.chunks)

    def search_vector(
        self,
        qvec: np.ndarray,
        k: int = 3,
        sources: set[str] | None = None,
    ) -> list[SearchResult]:
        """Brute-force cosine (vectors are L2-normalized -> dot == cosine)."""
        if not self.chunks:
            return []
        q = np.asarray(qvec, dtype=np.float32).reshape(1, -1)
        if q.shape[1] != self.vectors.shape[1]:
            raise ValueError(
                f"query dim {q.shape[1]} != index dim {self.vectors.shape[1]}")
        scores = (self.vectors @ q.T).ravel()
        order = np.argsort(-scores)
        out: list[SearchResult] = []
        for i in order:
            chunk = self.chunks[int(i)]
            if sources and chunk.source not in sources:
                continue
            out.append(SearchResult(chunk=chunk, score=float(scores[int(i)])))
            if len(out) >= k:
                break
        return out

    def search_text(
        self,
        query: str,
        k: int = 3,
        sources: set[str] | None = None,
    ) -> list[SearchResult]:
        """BM25 (FTS5) keyword leg. Exact identifiers stay whole: the
        tokenizer keeps '.', '_' and '+' as token chars and does NO
        stemming, so 'heater_fan' matches only 'heater_fan'.

        Terms are joined with OR (an all-terms AND dies on any stray word).
        Score is -bm25() so higher == better, consistent with search_vector.
        """
        from kb_rag.retrieve import fts_query

        q = fts_query(query)
        if not q:
            return []
        con = sqlite3.connect(self.path)
        try:
            sql = (
                "SELECT chunk_id, bm25(chunks_fts) AS s FROM chunks_fts"
                " WHERE chunks_fts MATCH ?"
            )
            args: list = [q]
            if sources:
                sql += " AND source IN (%s)" % ",".join("?" * len(sources))
                args += sorted(sources)
            sql += " ORDER BY s LIMIT ?"  # bm25() is negative-good
            args.append(k)
            rows = con.execute(sql, args).fetchall()
        except sqlite3.OperationalError:
            return []
        finally:
            con.close()
        out: list[SearchResult] = []
        for chunk_id, s in rows:
            chunk = self.get(chunk_id)
            if chunk is not None:
                out.append(SearchResult(chunk=chunk, score=-float(s)))
        return out

    def get(self, chunk_id: str) -> Chunk | None:
        self._ensure_id_index()
        i = self._id_index.get(chunk_id)
        return self.chunks[i] if i is not None else None

    # ── internals ──────────────────────────────────────────────────

    def _rebuild_id_index(self) -> None:
        self._id_index = {c.id: i for i, c in enumerate(self.chunks)}

    def _ensure_id_index(self) -> None:
        if not self._id_index and self.chunks:
            self._rebuild_id_index()
