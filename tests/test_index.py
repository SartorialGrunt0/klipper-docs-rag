"""Tests for the SQLite index store + dense retrieval (offline, fake embedder)."""
from __future__ import annotations

import numpy as np
import pytest

from kb_rag.chunk import Chunk
from kb_rag.index import KbIndex


def _chunk(i: int, doc: str = "Config_Reference", section: str | None = None) -> Chunk:
    sec = section or f"sec{i}"
    return Chunk(
        id=f"klipper/{doc}::{sec}", doc=doc, section=sec,
        breadcrumb=f"Klipper Docs > x > {sec}", text=f"[crumb]\n\nbody text {i}",
    )


class FakeEmbedder:
    """Deterministic orthogonal-ish vectors: each text maps to a unit vec
    derived from a stable hash, so similarity is arbitrary but reproducible."""

    def __init__(self, dim: int = 16):
        self.dim = dim

    def embed_documents(self, texts):
        out = []
        for t in texts:
            rng = np.random.RandomState(abs(hash(t)) % 2**31)
            v = rng.rand(self.dim).astype(np.float32)
            out.append(v / np.linalg.norm(v))
        return out

    def embed_query(self, text):
        return self.embed_documents([text])[0]


@pytest.fixture
def tmp_index(tmp_path):
    return KbIndex(tmp_path / "kb.sqlite", dim=16)


def test_save_load_roundtrip(tmp_index):
    chunks = [_chunk(i) for i in range(5)]
    vecs = np.stack([FakeEmbedder().embed_documents([c.text])[0] for c in chunks])
    tmp_index.save(chunks, vecs, meta={"docs_dir": "/x", "embed_model": "nomic"})
    idx = KbIndex.load(tmp_index.path)
    assert idx.count() == 5
    assert idx.meta["embed_model"] == "nomic"
    assert [c.id for c in idx.chunks] == [c.id for c in chunks]
    np.testing.assert_allclose(idx.vectors, vecs, atol=1e-6)


def test_search_returns_scored_chunks(tmp_index):
    chunks = [_chunk(i) for i in range(6)]
    fake = FakeEmbedder()
    vecs = np.stack(fake.embed_documents([c.text for c in chunks]))
    tmp_index.save(chunks, vecs, meta={})
    idx = KbIndex.load(tmp_index.path)
    qv = fake.embed_query(chunks[2].text)
    results = idx.search_vector(qv, k=3)
    assert len(results) == 3
    top = results[0]
    assert top.chunk.id == chunks[2].id          # self is best match
    assert top.score == pytest.approx(1.0, abs=1e-4)
    assert results[0].score >= results[1].score >= results[2].score


def test_search_k_clamps(tmp_index):
    chunks = [_chunk(i) for i in range(3)]
    fake = FakeEmbedder()
    tmp_index.save(chunks, np.stack(fake.embed_documents([c.text for c in chunks])), meta={})
    idx = KbIndex.load(tmp_index.path)
    assert len(idx.search_vector(fake.embed_query("q"), k=10)) == 3


def test_source_filter(tmp_index):
    ck = [_chunk(0, doc="Bed_Mesh", section="Fade")]
    cw = [Chunk(id="extra/Summary::x", doc="Summary", section="x",
                breadcrumb="Extra Docs > Summary > x", text="[c]\n\nextra body",
                source="extra")]
    chunks = ck + cw
    fake = FakeEmbedder()
    tmp_index.save(chunks, np.stack(fake.embed_documents([c.text for c in chunks])), meta={})
    idx = KbIndex.load(tmp_index.path)
    res = idx.search_vector(fake.embed_query("anything"), k=5, sources={"klipper"})
    assert [r.chunk.source for r in res] == ["klipper"]


def test_empty_index_search(tmp_index):
    tmp_index.save([], np.zeros((0, 16), dtype=np.float32), meta={})
    idx = KbIndex.load(tmp_index.path)
    assert idx.search_vector(np.ones(16, dtype=np.float32), k=3) == []


def test_dim_mismatch_rejected(tmp_index):
    with pytest.raises(ValueError):
        tmp_index.save([_chunk(0)], np.zeros((1, 8), dtype=np.float32), meta={})


def test_rebuild_replaces_old(tmp_index):
    fake = FakeEmbedder()
    tmp_index.save([_chunk(0)], np.stack(fake.embed_documents(["a"])), meta={})
    tmp_index.save([_chunk(1), _chunk(2)], np.stack(fake.embed_documents(["b", "c"])),
                   meta={"v": "2"})
    idx = KbIndex.load(tmp_index.path)
    assert idx.count() == 2
    assert idx.meta["v"] == "2"
    assert idx.search_vector(fake.embed_query("b"), k=5)[0].chunk.id == "klipper/Config_Reference::sec1"
