"""Tests for the sparse (FTS5/BM25) leg and RRF hybrid fusion."""
from __future__ import annotations

import numpy as np
import pytest

from kb_rag.chunk import Chunk
from kb_rag.index import KbIndex
from kb_rag.retrieve import fts_query, rrf_fuse, hybrid_search


def _c(i: int, section: str, text: str, doc: str = "Config_Reference",
       source: str = "klipper") -> Chunk:
    return Chunk(id=f"{source}/{doc}::{section}", doc=doc, section=section,
                 breadcrumb=f"crumb {section}", text=text, source=source)


@pytest.fixture
def seeded(tmp_path):
    """Index with 4 chunks; deterministic vectors: chunk i == unit vec e_i."""
    chunks = [
        _c(0, "[heater_fan]", "Heater cooling fan pin heater_temp fan speed"),
        _c(1, "[extruder]", "Extruder rotation_distance nozzle heater block"),
        _c(2, "[fan]", "Part cooling fan cycle_time hardware_pwm tachometer"),
        _c(3, "FAQ", "Why does the raspberry pi reboot during prints"),
    ]
    vecs = np.eye(4, dtype=np.float32)
    idx = KbIndex(tmp_path / "kb.sqlite", dim=4)
    idx.save(chunks, vecs, meta={})
    return KbIndex.load(tmp_path / "kb.sqlite")


# ── query sanitization ──────────────────────────────────────────────


def test_fts_query_or_of_terms():
    q = fts_query("how does heater_fan cycle_time work?")
    assert q == '"heater_fan" OR "cycle_time" OR "how" OR "does" OR "work"'


def test_fts_query_escapes_specials():
    # quotes/colons/parens/asterisks must never reach MATCH raw
    q = fts_query('pin: PB0 "quoted" (group) near* -not')
    assert all(ch not in q for ch in ":()*-")
    assert '"pin"' in q and '"pb0"' in q


def test_fts_query_empty():
    assert fts_query("... !!! ???") == ""


def test_fts_query_term_cap():
    q = fts_query(" ".join(f"w{i}" for i in range(50)))
    assert q.count(" OR ") <= 15


# ── sparse leg ──────────────────────────────────────────────────────


def test_bm25_finds_identifier(seeded):
    res = seeded.search_text("heater_fan", k=3)
    assert res
    assert res[0].chunk.section == "[heater_fan]"
    assert res[0].score >= res[1].score if len(res) > 1 else True


def test_bm25_ranks_exact_identifier_above_prose(seeded):
    res = seeded.search_text("tachometer cycle_time", k=4)
    assert res[0].chunk.section == "[fan]"


def test_bm25_or_semantics_partial_match(seeded):
    # AND semantics would return nothing for a mixed query; OR must not
    res = seeded.search_text("heater_fan raspberry reboot", k=3)
    secs = {r.chunk.section for r in res}
    assert "[heater_fan]" in secs and "FAQ" in secs


def test_bm25_empty_query(seeded):
    assert seeded.search_text("... ???", k=3) == []


def test_bm25_source_filter(seeded):
    res = seeded.search_text("fan", k=4, sources={"kwc"})
    assert res == []


# ── RRF ─────────────────────────────────────────────────────────────


def test_rrf_math():
    # standard k=60: rank1 -> 1/61, rank2 -> 1/62
    a = [("x", 1), ("y", 2)]           # leg A rankings (id, rank)
    b = [("y", 1), ("z", 2)]           # leg B rankings
    fused = dict(rrf_fuse([a, b], k_rrf=60))
    assert fused["y"] == pytest.approx(1/62 + 1/61)
    assert fused["x"] == pytest.approx(1/61)
    assert fused["z"] == pytest.approx(1/62)
    # y appears in both legs -> strictly best
    assert max(fused, key=fused.get) == "y"


def test_rrf_single_leg_degrades_to_ranking():
    fused = rrf_fuse([[("a", 1), ("b", 2)]])
    assert [item[0] for item in fused] == ["a", "b"]


def test_rrf_empty_legs():
    assert rrf_fuse([[], []]) == []


# ── hybrid search ───────────────────────────────────────────────────


def test_hybrid_combines_legs(seeded):
    # query vector identical to chunk 3 (FAQ) but text terms point to fan
    qv = np.array([0, 0, 0, 1], dtype=np.float32)
    res = hybrid_search(seeded, qvec=qv, text="tachometer cycle_time", k=2)
    ids = [r.chunk.section for r in res]
    # dense wants FAQ, sparse wants [fan]; both must surface in top-2
    assert "FAQ" in ids and "[fan]" in ids


def test_hybrid_agreement_boosts(seeded):
    # both legs agree on [heater_fan] -> it must rank first
    qv = np.array([1, 0, 0, 0], dtype=np.float32)
    res = hybrid_search(seeded, qvec=qv, text="heater_fan heater_temp", k=3)
    assert res[0].chunk.section == "[heater_fan]"
    # agreeing result outscores a single-leg result
    assert res[0].score > res[1].score


def test_hybrid_source_filter(seeded):
    qv = np.array([1, 0, 0, 0], dtype=np.float32)
    res = hybrid_search(seeded, qvec=qv, text="heater_fan", k=3,
                        sources={"klipper"})
    assert all(r.chunk.source == "klipper" for r in res)


def test_hybrid_respects_k(seeded):
    qv = np.ones(4, dtype=np.float32) / 2
    assert len(hybrid_search(seeded, qvec=qv, text="fan", k=1)) == 1
