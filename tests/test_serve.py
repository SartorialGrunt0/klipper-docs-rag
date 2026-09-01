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
