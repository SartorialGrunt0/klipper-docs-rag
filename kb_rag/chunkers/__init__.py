"""Chunkers: structure-aware splitting of Klipper docs.

Dispatch rules (docs/RAG-RESEARCH.md §4.1):
- Config_Reference* / KWC summaries shaped like it  -> stanza chunker
- G-Codes                                          -> command chunker
- everything else                                  -> headings chunker
"""
from __future__ import annotations

from kb_rag.chunk import Chunk, approx_tokens
from kb_rag.chunkers.stanza import chunk_stanza_file
from kb_rag.chunkers.headings import chunk_headings_file
from kb_rag.chunkers.gcodes import chunk_gcodes_file

__all__ = [
    "Chunk", "approx_tokens",
    "chunk_document", "chunk_stanza_file", "chunk_headings_file",
    "chunk_gcodes_file",
]

_STANZA_DOCS = {"Config_Reference", "Status_Reference"}
_GCODE_DOCS = {"G-Codes"}


def chunk_document(
    text: str, doc: str, source: str = "klipper", max_tokens: int = 500,
) -> list[Chunk]:
    """Chunk one markdown document, choosing the strategy by doc name."""
    if doc in _STANZA_DOCS:
        return chunk_stanza_file(text, doc, source=source, max_tokens=max_tokens)
    if doc in _GCODE_DOCS:
        return chunk_gcodes_file(text, doc, source=source, max_tokens=max_tokens)
    return chunk_headings_file(text, doc, source=source, max_tokens=max_tokens)
