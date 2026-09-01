"""Stanza chunker: one chunk per ``### [section]`` block.

Targets Config_Reference.md (and KWC summaries shaped like it): the doc's
value unit is a config stanza, exactly the insight KWC's DocIndex already
encodes with CONFIG_SECTION_HEADER_RE. Non-stanza content (doc intro,
format discussion, ``##``-level sections) is preserved as its own chunks
anchored to the nearest heading so nothing silently drops.

Oversized stanzas split on ``#param:`` block boundaries so a parameter's
name and description never straddle two chunks.
"""
from __future__ import annotations

import re

from kb_rag.chunk import Chunk, approx_tokens
from kb_rag.chunkers.common import enforce_cap, find_headings, pack_blocks, split_param_blocks

DEFAULT_MAX_TOKENS = 500


def _breadcrumb(display: str, section: str, source: str) -> str:
    root = "KWC Docs" if source == "kwc" else "Klipper Docs"
    return f"{root} > {display} > {section}"


def _emit(
    *, source: str, doc: str, display: str, section: str, body: str,
    max_tokens: int, seen: dict[str, int] | None = None,
) -> list[Chunk]:
    body = body.strip("\n").strip()
    if not body:
        return []
    crumb = _breadcrumb(display, section, source)
    prefix = f"[{crumb}]\n\n"
    base_id = f"{source}/{doc}::{section}"
    if seen is not None:
        n = seen.get(base_id, 0)
        seen[base_id] = n + 1
        if n:
            base_id = f"{base_id}#dup{n}"
    if approx_tokens(prefix + body) <= max_tokens:
        return [Chunk(
            id=base_id, doc=doc, section=section,
            breadcrumb=crumb, text=prefix + body, source=source,
        )]
    body_budget = max(64, max_tokens - approx_tokens(prefix))
    parts = enforce_cap(pack_blocks(split_param_blocks(body), body_budget), body_budget)
    if len(parts) <= 1:
        return [Chunk(
            id=base_id, doc=doc, section=section,
            breadcrumb=crumb, text=prefix + body, source=source,
        )]
    return [Chunk(
        id=f"{base_id}#part{i}", doc=doc, section=section,
        breadcrumb=crumb, text=prefix + p, source=source, part=i,
    ) for i, p in enumerate(parts)]


def chunk_stanza_file(
    text: str, doc: str, source: str = "klipper", max_tokens: int = DEFAULT_MAX_TOKENS,
) -> list[Chunk]:
    display = doc.replace("_", " ")
    chunks: list[Chunk] = []
    seen: dict[str, int] = {}

    # Walk every heading; a segment runs from its heading to the next
    # heading of *any* level. Stanza headers (### [x]) own their segment;
    # other headers own theirs too (they may hold prose or nothing).
    headings = find_headings(text)
    if not headings:
        return _emit(source=source, doc=doc, display=display,
                     section=display, body=text, max_tokens=max_tokens,
                     seen=seen)

    # content before the first heading -> attach to doc title
    if text[: headings[0].start()].strip():
        title = headings[0].group(2) if headings[0].group(1) == "#" else display
        chunks.extend(_emit(
            source=source, doc=doc, display=display, section=title,
            body=text[: headings[0].start()], max_tokens=max_tokens,
            seen=seen))

    for i, m in enumerate(headings):
        title = m.group(2).strip()
        # A heading owns text up to the NEXT heading of ANY level: parent
        # ## sections with only ### children emit nothing (their subtree is
        # covered by the children), avoiding duplicated content.
        end = headings[i + 1].start() if i + 1 < len(headings) else len(text)
        body = text[m.end():end]
        # stanza header or plain header: the header text IS the section
        chunks.extend(_emit(
            source=source, doc=doc, display=display, section=title,
            body=body, max_tokens=max_tokens, seen=seen))
    return chunks
