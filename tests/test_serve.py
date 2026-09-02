"""Tests for the OpenAI-compatible RAG proxy (no live servers involved)."""
from __future__ import annotations

import numpy as np
import pytest

from kb_rag.chunk import Chunk
from kb_rag.index import KbIndex
from kb_rag.serve import CONTEXT_TEMPLATE, VIRTUAL_MODEL, RagService


class FakeEmbed:
    """Deterministic embedder: query -> e_0 (matches chunk 0)."""

    def embed_query(self, text):
        return np.array([1, 0, 0], dtype=np.float32)

    def embed_documents(self, texts):
        return [np.eye(3, dtype=np.float32)[0] for _ in texts]


@pytest.fixture
def service(tmp_path):
    chunks = [
        Chunk(id="klipper/Config_Reference::[heater_fan]",
              doc="Config_Reference", section="[heater_fan]", source="klipper",
              breadcrumb="crumb", text="heater_fan pin heater_temp", tokens=8),
        Chunk(id="klipper/FAQ::x", doc="FAQ", section="x", source="klipper",
              breadcrumb="crumb", text="unrelated prose", tokens=4),
        Chunk(id="klipper/FAQ::y", doc="FAQ", section="y", source="klipper",
              breadcrumb="crumb", text="more unrelated prose", tokens=4),
    ]
    idx = KbIndex(tmp_path / "kb.sqlite", dim=3)
    idx.save(chunks, np.eye(3, dtype=np.float32), meta={})
    return RagService(KbIndex.load(tmp_path / "kb.sqlite"), FakeEmbed(),
                      "http://unused.invalid/v1", "base-model")


def test_virtual_model_name():
    assert VIRTUAL_MODEL == "klipper-expert"


def test_augment_injects_context_system(service):
    msgs, chunks = service.augment(
        [{"role": "user", "content": "heater_fan pin"}])
    assert chunks and chunks[0]["section"] == "[heater_fan]"
    assert msgs[0]["role"] == "system"
    assert "=== CONTEXT ===" in msgs[0]["content"]
    assert "heater_fan pin heater_temp" in msgs[0]["content"]
    assert msgs[-1]["role"] == "user"


def test_augment_merges_existing_system(service):
    msgs, _ = service.augment([
        {"role": "system", "content": "You are terse."},
        {"role": "user", "content": "heater_fan pin"},
    ])
    systems = [m for m in msgs if m["role"] == "system"]
    assert len(systems) == 1
    assert "You are terse." in systems[0]["content"]
    assert "=== CONTEXT ===" in systems[0]["content"]


def test_augment_query_is_last_user_msg(service):
    msgs, chunks = service.augment([
        {"role": "user", "content": "earlier unrelated"},
        {"role": "assistant", "content": "ok"},
        {"role": "user", "content": "heater_fan pin"},
    ])
    assert chunks[0]["section"] == "[heater_fan]"
    # history preserved untouched
    assert msgs[1]["content"] == "earlier unrelated"


def test_augment_multimodal_parts_flattened(service):
    msgs, chunks = service.augment([
        {"role": "user", "content": [
            {"type": "text", "text": "heater_fan pin"},
            {"type": "image_url", "image_url": {"url": "x"}},
        ]},
    ])
    assert chunks[0]["section"] == "[heater_fan]"


def test_context_char_cap(service):
    service.max_context_chars = 50
    msgs, _ = service.augment([{"role": "user", "content": "heater_fan"}])
    sys_txt = msgs[0]["content"]
    block = sys_txt.split("=== CONTEXT ===")[1].split("=== END CONTEXT ===")[0]
    assert len(block.strip()) <= 50


def test_retrieve_shape(service):
    res = service.retrieve("heater_fan", k=1)
    assert len(res) == 1
    assert set(res[0]) == {"doc", "section", "text", "score"}


# --- threshold gate -------------------------------------------------------


class FakeEmbedWeak:
    """Query == corpus centroid direction: whitened top-1 cosine = 0.0.

    (Index chunks are eye(3); their mean direction is (1,1,1)/sqrt3, which
    is exactly what this embedder returns. Raw cosine would be 0.577 —
    the whole point of whitening is that this reads as chit-chat signal.)
    """

    def embed_query(self, text):
        v = np.array([1, 1, 1], dtype=np.float32)
        return v / np.linalg.norm(v)

    def embed_documents(self, texts):
        return [np.eye(3, dtype=np.float32)[0] for _ in texts]


class FakeEmbedDomain:
    """Query aligns with chunk 0's whitened direction: top1 = 1.0."""

    def embed_query(self, text):
        v = np.array([1, -1, 0], dtype=np.float32)
        return v / np.linalg.norm(v)

    def embed_documents(self, texts):
        return [np.eye(3, dtype=np.float32)[0] for _ in texts]


@pytest.fixture
def gated_service(tmp_path):
    """Same index as `service` but with an embedder whose best cosine is
    0.707 and a 0.8 threshold -> gate must block, 0.5 -> must pass."""
    chunks = [
        Chunk(id="klipper/Config_Reference::[heater_fan]",
              doc="Config_Reference", section="[heater_fan]", source="klipper",
              breadcrumb="crumb", text="heater_fan pin heater_temp", tokens=8),
        Chunk(id="klipper/FAQ::x", doc="FAQ", section="x", source="klipper",
              breadcrumb="crumb", text="unrelated prose", tokens=4),
        Chunk(id="klipper/FAQ::y", doc="FAQ", section="y", source="klipper",
              breadcrumb="crumb", text="more unrelated prose", tokens=4),
    ]
    idx = KbIndex(tmp_path / "kb2.sqlite", dim=3)
    idx.save(chunks, np.eye(3, dtype=np.float32), meta={})
    return RagService(KbIndex.load(tmp_path / "kb2.sqlite"), FakeEmbedWeak(),
                      "http://unused.invalid/v1", "base-model",
                      gate_threshold=0.8)


def test_gate_blocks_low_confidence_query(gated_service):
    chunks = gated_service.retrieve("tell me about my summer vacation")
    assert chunks == []


def test_gate_trailing_period_is_not_an_identifier(gated_service):
    # "vacation." must not sneak past the identifier bypass
    chunks = gated_service.retrieve("tell me about my summer vacation.")
    assert chunks == []


def test_gate_augment_passes_through_unchanged(gated_service):
    msgs = [{"role": "user", "content": "tell me about my summer vacation"}]
    out, chunks = gated_service.augment(msgs)
    assert chunks == []
    # no system message injected; history untouched
    assert out == msgs


def test_gate_identifier_bypass(gated_service):
    # below-threshold cosine, but the query names an identifier -> doc
    # intent by construction, sparse leg must not be gated out
    chunks = gated_service.retrieve("what does heater_fan do")
    assert chunks and chunks[0]["section"] == "[heater_fan]"


def test_gate_passes_above_threshold(tmp_path):
    # whitened top1 = 1.0 -> passes any sane threshold
    chunks = [
        Chunk(id="klipper/Config_Reference::[heater_fan]",
              doc="Config_Reference", section="[heater_fan]", source="klipper",
              breadcrumb="crumb", text="heater_fan pin heater_temp", tokens=8),
        Chunk(id="klipper/FAQ::x", doc="FAQ", section="x", source="klipper",
              breadcrumb="crumb", text="unrelated prose", tokens=4),
    ]
    idx = KbIndex(tmp_path / "kb4.sqlite", dim=3)
    idx.save(chunks, np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float32),
             meta={})
    svc = RagService(KbIndex.load(tmp_path / "kb4.sqlite"), FakeEmbedDomain(),
                     "http://unused.invalid/v1", "base-model",
                     gate_threshold=0.8)
    got = svc.retrieve("tell me about my summer vacation")
    assert got


def test_gate_disabled_with_none(tmp_path):
    chunks = [
        Chunk(id="klipper/FAQ::x", doc="FAQ", section="x", source="klipper",
              breadcrumb="crumb", text="unrelated prose", tokens=4),
    ]
    idx = KbIndex(tmp_path / "kb3.sqlite", dim=3)
    idx.save(chunks, np.array([[1, 0, 0]], dtype=np.float32), meta={})
    svc = RagService(KbIndex.load(tmp_path / "kb3.sqlite"), FakeEmbedWeak(),
                     "http://unused.invalid/v1", "base-model",
                     gate_threshold=None)
    assert svc.retrieve("anything at all")  # no gate -> always injects


def test_retrieve_full_reports_gate(gated_service):
    res = gated_service.retrieve_full("tell me about my summer vacation")
    assert res["gate_passed"] is False
    assert res["chunks"] == []
    assert res["top1_cosine"] == pytest.approx(0.0, abs=0.01)
    res2 = gated_service.retrieve_full("heater_fan pin")
    assert res2["gate_passed"] is True


def test_default_service_keeps_gate_active(service):
    # default construction must enable the gate (default threshold 0.40)
    assert service.gate_threshold == pytest.approx(0.40)


def test_domain_hint_bypasses_weak_signal(gated_service):
    # no identifier, whitened signal is 0, but "fan" is a domain noun
    chunks = gated_service.retrieve("the fan is making noise")
    assert chunks


# --- rerank wiring --------------------------------------------------------


class FakeReranker:
    """Reverses candidate order; mimics RerankClient.rerank contract."""
    max_doc_chars = 1500

    def __init__(self):
        self.queries = []

    def rerank(self, query, docs, top_n=None):
        self.queries.append(query)
        # increasing score with index -> reverses the candidate order
        return [float(i + 1) for i in range(len(docs))]


def test_rerank_reorders_hybrid_results(tmp_path):
    chunks = [
        Chunk(id="klipper/Config_Reference::[a]", doc="Config_Reference",
              section="[a]", source="klipper", breadcrumb="c",
              text="first", tokens=2),
        Chunk(id="klipper/Config_Reference::[b]", doc="Config_Reference",
              section="[b]", source="klipper", breadcrumb="c",
              text="second", tokens=2),
    ]
    idx = KbIndex(tmp_path / "kb5.sqlite", dim=3)
    idx.save(chunks, np.array([[1, 0, 0], [0.9, 0.1, 0]],
                              dtype=np.float32), meta={})
    rr = FakeReranker()
    svc = RagService(KbIndex.load(tmp_path / "kb5.sqlite"), FakeEmbed(),
                     "http://unused.invalid/v1", "base-model",
                     gate_threshold=None, rerank_client=rr)
    res = svc.retrieve("anything", k=2)
    # fake reverses: [b] before [a]
    assert [c["section"] for c in res] == ["[b]", "[a]"]
    assert rr.queries == ["anything"]


def test_rerank_down_degrades_to_hybrid(tmp_path):
    chunks = [
        Chunk(id="klipper/Config_Reference::[a]", doc="Config_Reference",
              section="[a]", source="klipper", breadcrumb="c",
              text="first", tokens=2),
    ]
    idx = KbIndex(tmp_path / "kb6.sqlite", dim=3)
    idx.save(chunks, np.array([[1, 0, 0]], dtype=np.float32), meta={})

    class DeadReranker(FakeReranker):
        def rerank(self, query, docs, top_n=None):
            raise ConnectionError("reranker down")

    svc = RagService(KbIndex.load(tmp_path / "kb6.sqlite"), FakeEmbed(),
                     "http://unused.invalid/v1", "base-model",
                     gate_threshold=None, rerank_client=DeadReranker())
    res = svc.retrieve("anything", k=1)
    # open-circuit: hybrid order survives
    assert res and res[0]["section"] == "[a]"
