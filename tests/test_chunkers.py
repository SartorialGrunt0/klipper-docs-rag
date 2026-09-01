"""Tests for the structure-aware chunkers (phase 0)."""
from __future__ import annotations

from kb_rag.chunk import Chunk, approx_tokens
from kb_rag.chunkers import chunk_document, chunk_stanza_file, chunk_headings_file
from kb_rag.chunkers.gcodes import chunk_gcodes_file

# ── fixtures ────────────────────────────────────────────────────────

CONFIG_REF_SAMPLE = """\
# Configuration reference

This document describes the configuration options.

## Micro-controller configuration

### Format of micro-controller pin names

Pins are named `PA1` style with optional inversion via `!`.

### [mcu]

This section configures the main MCU.

#serial:
#   The serial port to connect to the MCU.
#   This parameter must be provided.
#baud: 250000
#   The baud rate to use. The default is 250000.

### [stepper_x]

Steps per mm for the X axis.

#step_pin:
#   Step pin.
#rotation_distance: 40
#   Distance the belt travels for one full rotation.

## Section example format

Stanzas repeat config sections use the `section name` syntax.
"""

GCODES_SAMPLE = """\
# G-Codes

This document describes the commands that Klipper supports.

## G-Code commands

Klipper supports the following standard G-Code commands:
- Move (G0 or G1): `G1 [X<pos>]`

## Additional Commands

Extended commands follow a similar format.

### [bed_mesh]

Support for meshes.

#### BED_MESH_CALIBRATE

Calibrate the mesh. This may probe the bed.

#### BED_MESH_CLEAR

Clear the current mesh.

### [adxl345]

Support for the ADXL345 accelerometer.

#### ACCELEROMETER_MEASURE

Take a measurement.
"""

HEADINGS_SAMPLE = """\
# Bed Mesh

Intro paragraph about bed mesh.

## Basic Configuration

### Rectangular Beds

Rectangular bed config text here.

### Round beds

Round bed config text here.

## Advanced Configuration

Advanced config text.

### Mesh Interpolation

Interpolation text.
"""


def _long_param_stanza(name: str, n_params: int, desc_words: int = 30) -> str:
    lines = [f"### [{name}]", "", "Stanza intro."]
    for i in range(n_params):
        lines.append(f"#param_{i}:")
        # realistic Klipper style: several terminated sentences
        line = ("description word " * (desc_words // 3)).strip() + "."
        lines.append("#   " + line)
    return "\n".join(lines)


# ── approx_tokens ───────────────────────────────────────────────────


def test_approx_tokens_basic():
    assert approx_tokens("") == 1
    assert approx_tokens("a" * 40) == 10


# ── stanza chunker (Config_Reference style) ─────────────────────────


def test_stanza_one_chunk_per_section():
    chunks = chunk_stanza_file(CONFIG_REF_SAMPLE, "Config_Reference")
    sections = [c.section for c in chunks]
    assert "[mcu]" in sections
    assert "[stepper_x]" in sections
    # each stanza is exactly one chunk
    assert sections.count("[mcu]") == 1
    assert sections.count("[stepper_x]") == 1


def test_stanza_chunk_text_and_breadcrumb():
    chunks = chunk_stanza_file(CONFIG_REF_SAMPLE, "Config_Reference")
    mcu = next(c for c in chunks if c.section == "[mcu]")
    assert mcu.text.startswith("[Klipper Docs > Config Reference > [mcu]]")
    assert "#baud: 250000" in mcu.text
    assert "[stepper_x]" not in mcu.text  # no bleed from sibling stanza
    assert mcu.doc == "Config_Reference"
    assert mcu.id == "klipper/Config_Reference::[mcu]"


def test_stanza_preamble_and_non_stanza_headings_kept():
    chunks = chunk_stanza_file(CONFIG_REF_SAMPLE, "Config_Reference")
    # the pin-names subsection + trailing "## Section example format" survive
    texts = " ".join(c.text for c in chunks)
    assert "Pins are named `PA1` style" in texts
    assert "Stanzas repeat config sections" in texts
    # every chunk is attributable to something
    assert all(c.section is not None for c in chunks)


def test_stanza_oversized_splits_on_params():
    big = f"# Config\n\n{ _long_param_stanza('huge', 20) }\n"
    chunks = chunk_stanza_file(big, "Config_Reference", max_tokens=400)
    parts = [c for c in chunks if c.section == "[huge]"]
    assert len(parts) > 1
    # parts are numbered and contiguous-unique
    assert [c.part for c in parts] == list(range(len(parts)))
    assert [c.id for c in parts] == [
        f"klipper/Config_Reference::[huge]#part{i}" for i in range(len(parts))
    ]
    # every param lands in exactly one part
    joined = "\n".join(c.text for c in parts)
    for i in range(20):
        assert joined.count(f"#param_{i}:") == 1
    # parts carry breadcrumb too
    assert all(c.text.startswith("[Klipper Docs > Config Reference > [huge]") for c in parts)
    # no part exceeds the cap (within one param block's worth of slack)
    assert all(c.tokens <= 500 for c in parts)


# ── gcodes chunker ──────────────────────────────────────────────────


def test_gcodes_splits_on_commands():
    chunks = chunk_gcodes_file(GCODES_SAMPLE, "G-Codes")
    sections = [c.section for c in chunks]
    assert "BED_MESH_CALIBRATE" in sections
    assert "ACCELEROMETER_MEASURE" in sections
    assert sections.count("BED_MESH_CLEAR") == 1


def test_gcodes_command_chunk_has_family_context():
    chunks = chunk_gcodes_file(GCODES_SAMPLE, "G-Codes")
    cal = next(c for c in chunks if c.section == "BED_MESH_CALIBRATE")
    assert "[bed_mesh]" in cal.breadcrumb
    assert "Klipper Docs" in cal.text  # breadcrumb prefix embedded
    assert "BED_MESH_CLEAR" not in cal.text


def test_gcodes_intro_not_lost():
    chunks = chunk_gcodes_file(GCODES_SAMPLE, "G-Codes")
    texts = " ".join(c.text for c in chunks)
    assert "standard G-Code commands" in texts
    assert "Extended commands follow" in texts


# ── headings chunker (prose docs) ───────────────────────────────────


def test_headings_splits_on_heading_boundaries():
    chunks = chunk_headings_file(HEADINGS_SAMPLE, "Bed_Mesh")
    assert len(chunks) >= 4
    # heading text is present in breadcrumbs
    crumbs = [c.breadcrumb for c in chunks]
    assert any("Mesh Interpolation" in b for b in crumbs)
    assert any("Rectangular Beds" in b for b in crumbs)


def test_headings_nested_context_included():
    chunks = chunk_headings_file(HEADINGS_SAMPLE, "Bed_Mesh")
    interp = next(c for c in chunks if "Mesh Interpolation" in c.breadcrumb)
    assert "Advanced Configuration" in interp.breadcrumb  # parent heading kept


def test_headings_oversized_block_paragraph_split():
    filler = ("This sentence explains bed mesh fading behaviour in detail. " * 60).strip()
    doc = f"# Bed Mesh\n\n## Mesh Fade\n\n{filler}\n"
    chunks = chunk_headings_file(doc, "Bed_Mesh", max_tokens=400)
    fade = [c for c in chunks if "Mesh Fade" in c.breadcrumb]
    assert len(fade) > 1
    assert all(c.tokens <= 500 for c in fade)
    # overlap: consecutive parts share at least one sentence
    sentences = [s for s in fade[0].text.split(". ") if len(s) > 20]
    assert any(s in fade[1].text for s in sentences[-2:])


# ── document dispatch ───────────────────────────────────────────────


def test_chunk_document_dispatch():
    s = chunk_document(CONFIG_REF_SAMPLE, "Config_Reference")
    g = chunk_document(GCODES_SAMPLE, "G-Codes")
    p = chunk_document(HEADINGS_SAMPLE, "Bed_Mesh")
    assert any(c.section == "[mcu]" for c in s)
    assert any(c.section == "BED_MESH_CALIBRATE" for c in g)
    assert all(c.section is not None for c in p)
    for c in s + g + p:
        assert isinstance(c, Chunk)
        assert c.tokens > 0


def test_chunk_kwc_doc_source_tag():
    chunks = chunk_document(CONFIG_REF_SAMPLE, "Klipper_Docs_AI_Summary", source="kwc")
    assert all(c.source == "kwc" for c in chunks)
    assert all(c.id.startswith("kwc/") for c in chunks)
    assert "[Klipper Docs" not in chunks[0].breadcrumb  # KWC crumb, not official
    assert "KWC Docs" in chunks[0].breadcrumb
