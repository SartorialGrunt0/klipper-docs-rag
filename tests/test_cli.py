"""Tests for the CLI provider wiring (offline, EmbedClient patched).

These pin the standalone contract: no private-network defaults, and the
embedding model name flows CLI -> EmbedClient -> index meta -> query.
"""
from __future__ import annotations

import numpy as np
import pytest

from kb_rag import cli
from kb_rag.chunk import Chunk


class FakeEmbedClient:
    """Records constructor kwargs; returns fixed 8-d unit vectors."""

    instances: list["FakeEmbedClient"] = []

    def __init__(self, base_url=None, model=None, **kw):
        self.base_url = base_url
        self.model = model or "nomic-embed-text-v1.5"
        FakeEmbedClient.instances.append(self)

    def embed_query(self, text):
        v = np.ones(8, dtype=np.float32)
        return v / np.linalg.norm(v)

    def embed_documents(self, texts):
        return [self.embed_query(t) for t in texts]


@pytest.fixture(autouse=True)
def _patch_embed_client(monkeypatch):
    FakeEmbedClient.instances = []
    monkeypatch.setattr("kb_rag.embed.EmbedClient", FakeEmbedClient)


@pytest.fixture
def docs_dir(tmp_path):
    d = tmp_path / "docs"
    d.mkdir()
    (d / "Config_Reference.md").write_text(
        "# Config Reference\n\n"
        "## heater_fan\n\n"
        "This section configures a fan that turns on with the heater.\n\n"
        "### Parameters\n\n"
        "- **pin**: Pin the fan is connected to.\n"
        "- **heater_temp**: Temperature above which the fan runs.\n",
        encoding="utf-8",
    )
    return d


def test_defaults_have_no_private_network_urls():
    """Standalone default must not hardcode a LAN address."""
    assert "192.168" not in cli.DEFAULT_EMBED_URL
    from kb_rag import embed
    assert "192.168" not in embed.DEFAULT_BASE_URL


def test_build_passes_embed_url_and_model(tmp_path, docs_dir, monkeypatch):
    state = tmp_path / "kb.sqlite"
    rc = cli.main([
        "build", str(docs_dir), "--state", str(state),
        "--embed-url", "http://provider.test:9999",
        "--embed-model", "my-embed-v1",
    ])
    assert rc == 0
    client = FakeEmbedClient.instances[-1]
    assert client.base_url == "http://provider.test:9999"
    assert client.model == "my-embed-v1"
    from kb_rag.index import KbIndex
    idx = KbIndex.load(state)
    assert idx.meta["embed_model"] == "my-embed-v1"
    assert idx.meta["embed_url"] == "http://provider.test:9999"


def test_query_uses_index_meta_embed_model(tmp_path, docs_dir, monkeypatch):
    state = tmp_path / "kb.sqlite"
    assert cli.main([
        "build", str(docs_dir), "--state", str(state),
        "--embed-url", "http://provider.test:9999",
        "--embed-model", "my-embed-v1",
    ]) == 0
    FakeEmbedClient.instances.clear()
    rc = cli.main(["query", str(state), "heater fan pin"])
    assert rc == 0
    client = FakeEmbedClient.instances[-1]
    assert client.model == "my-embed-v1"
    assert client.base_url == "http://provider.test:9999"


def test_query_embed_model_flag_overrides_meta(tmp_path, docs_dir):
    state = tmp_path / "kb.sqlite"
    assert cli.main([
        "build", str(docs_dir), "--state", str(state),
        "--embed-url", "http://provider.test:9999",
    ]) == 0
    FakeEmbedClient.instances.clear()
    rc = cli.main([
        "query", str(state), "heater fan pin",
        "--embed-model", "explicit-override",
    ])
    assert rc == 0
    assert FakeEmbedClient.instances[-1].model == "explicit-override"
