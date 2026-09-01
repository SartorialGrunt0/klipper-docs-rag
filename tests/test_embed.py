"""Tests for the embedding client (offline, transport mocked)."""
from __future__ import annotations

import math

import numpy as np
import pytest

from kb_rag import embed as embed_mod
from kb_rag.embed import EmbedClient, NOMIC_DOC_PREFIX, NOMIC_QUERY_PREFIX


class FakeResponse:
    def __init__(self, payload, status=200):
        self._payload = payload
        self.status_code = status

    def json(self):
        return self._payload

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")


def _vec(seed: int, dim: int = 8) -> list[float]:
    rng = np.random.RandomState(seed)
    v = rng.rand(dim).astype(float)
    return (v / np.linalg.norm(v)).tolist()


class FakePost:
    """Records batch sizes; returns deterministic unit vectors per input."""

    def __init__(self, dim: int = 8):
        self.calls: list[list[str]] = []
        self.dim = dim

    def __call__(self, url, json=None, **kw):
        inputs = json["input"]
        self.calls.append(inputs)
        data = [
            {"index": i, "embedding": _vec(abs(hash(t)) % 1000, self.dim)}
            for i, t in enumerate(inputs)
        ]
        return FakeResponse({"object": "list", "data": data})


@pytest.fixture
def client(monkeypatch):
    fake = FakePost()
    monkeypatch.setattr(embed_mod, "_post", fake)
    return EmbedClient(base_url="http://fake:8100"), fake


def test_query_prefix_applied(client):
    c, fake = client
    v = c.embed_query("what does heater_fan do")
    assert fake.calls[-1] == [NOMIC_QUERY_PREFIX + "what does heater_fan do"]
    assert len(v) == 8


def test_document_prefix_applied(client):
    c, fake = client
    vs = c.embed_documents(["chunk a", "chunk b"])
    assert fake.calls[-1] == [
        NOMIC_DOC_PREFIX + "chunk a",
        NOMIC_DOC_PREFIX + "chunk b",
    ]
    assert len(vs) == 2


def test_no_double_prefix(client):
    c, fake = client
    c.embed_query(NOMIC_QUERY_PREFIX + "already prefixed")
    assert fake.calls[-1] == [NOMIC_QUERY_PREFIX + "already prefixed"]


def test_batching_respects_batch_size(monkeypatch):
    fake = FakePost()
    monkeypatch.setattr(embed_mod, "_post", fake)
    c = EmbedClient(base_url="http://fake:8100", batch_size=3)
    docs = [f"d{i}" for i in range(7)]
    vs = c.embed_documents(docs)
    assert len(vs) == 7
    assert [len(b) for b in fake.calls] == [3, 3, 1]


def test_order_preserved_across_batches(monkeypatch):
    # FakePost hashes text -> vectors; verify per-input identity survives
    fake = FakePost()
    monkeypatch.setattr(embed_mod, "_post", fake)
    c = EmbedClient(base_url="http://fake:8100", batch_size=2)
    docs = ["alpha", "beta", "gamma"]
    vs = c.embed_documents(docs)
    again = c.embed_documents(docs)
    for a, b in zip(vs, again):
        assert np.allclose(a, b)
    # first result matches single-item call of the same text
    single = c.embed_documents(["alpha"])[0]
    assert np.allclose(vs[0], single)


def test_l2_normalize_ensured(client):
    c, _ = client
    v = c.embed_query("x")
    assert abs(float(np.linalg.norm(v)) - 1.0) < 1e-6


def test_server_vectors_renormalized(monkeypatch):
    # server may return non-unit vectors (embd-normalize off): client must
    # normalize so dot product == cosine
    class RawPost:
        def __call__(self, url, json=None, **kw):
            data = [{"index": 0, "embedding": [3.0, 4.0]}]
            return FakeResponse({"object": "list", "data": data})

    monkeypatch.setattr(embed_mod, "_post", RawPost())
    c = EmbedClient(base_url="http://fake:8100")
    v = c.embed_query("x")
    # float32 storage: |norm| tolerance ~1e-6, not exact
    assert abs(math.hypot(*v) - 1.0) < 1e-6


def test_empty_input_short_circuits(monkeypatch):
    fake = FakePost()
    monkeypatch.setattr(embed_mod, "_post", fake)
    c = EmbedClient(base_url="http://fake:8100")
    assert c.embed_documents([]) == []
    assert fake.calls == []


def test_truncates_overlong_input(monkeypatch):
    fake = FakePost()
    monkeypatch.setattr(embed_mod, "_post", fake)
    c = EmbedClient(base_url="http://fake:8100", max_chars=50)
    c.embed_documents(["x" * 500])
    sent = fake.calls[-1][0]
    assert len(sent) <= 50 + len(NOMIC_DOC_PREFIX) + 1
