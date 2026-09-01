"""Heading-aware recursive chunker for prose docs (Bed_Mesh.md style).

Chunks at ``##``/``###`` boundaries, keeping the heading path as
breadcrumb context; blocks over budget are packed on paragraph boundaries
with sentence overlap. Section-level headings (### under ##) keep their
parent in the breadcrumb so a query for "mesh fade" still retrieves the
chunk with "Bed Mesh > Advanced Configuration" context.
"""
from __future__ import annotations

import re

from kb_rag.chunk import Chunk, approx_tokens
from kb_rag.chunkers.common import enforce_cap, find_headings, split_prose

DEFAULT_MAX_TOKENS = 500


def _breadcrumb(display: str, path: list[str], source: str) -> str:
    root = "KWC Docs" if source == "kwc" else "Klipper Docs"
    return " > ".join([root, display, *path])


def _emit(
    *, source: str, doc: str, display: str, path: list[str], body: str,
    max_tokens: int, seen: dict[str, int],
) -> list[Chunk]:
    body = body.strip("\n").strip()
    if not body:
        return []
    crumb = _breadcrumb(display, path, source)
    section = path[-1] if path else display
    prefix = f"[{crumb}]\n\n"
    base_id = f"{source}/{doc}::{section}"
    n = seen.get(base_id, 0)
    seen[base_id] = n + 1
    if n:
        base_id = f"{base_id}#dup{n}"
    chunks: list[Chunk] = []
    body_budget = max(64, max_tokens - approx_tokens(prefix))
    parts = [body] if approx_tokens(prefix + body) <= max_tokens else enforce_cap(split_prose(body, body_budget), body_budget)
    for part_idx, p in enumerate(parts):
        cid = base_id if len(parts) == 1 else f"{base_id}#part{part_idx}"
        chunks.append(Chunk(
            id=cid, doc=doc, section=section, breadcrumb=crumb,
            text=prefix + p, source=source, part=part_idx,
        ))
    return chunks


def chunk_headings_file(
    text: str, doc: str, source: str = "klipper", max_tokens: int = DEFAULT_MAX_TOKENS,
) -> list[Chunk]:
    display = doc.replace("_", " ")
    chunks: list[Chunk] = []
    seen: dict[str, int] = {}

    headings = find_headings(text)
    if not headings:
        return _emit(source=source, doc=doc, display=display, path=[doc],
                     body=text, max_tokens=max_tokens, seen=seen)

    # heading path stack: index = level-1
    stack: list[tuple[int, str]] = []  # (level, title)

    def path_titles() -> list[str]:
        return [t for _, t in stack]

    # preamble before first heading
    if text[: headings[0].start()].strip():
        chunks.extend(_emit(source=source, doc=doc, display=display,
                            path=[headings[0].group(2).strip()],
                            body=text[: headings[0].start()],
                            max_tokens=max_tokens, seen=seen))

    for i, m in enumerate(headings):
        level = len(m.group(1))
        title = m.group(2).strip()
        while stack and stack[-1][0] >= level:
            stack.pop()
        stack.append((level, title))
        end = headings[i + 1].start() if i + 1 < len(headings) else len(text)
        body = text[m.end():end]
        chunks.extend(_emit(source=source, doc=doc, display=display,
                            path=path_titles(), body=body,
                            max_tokens=max_tokens, seen=seen))
    return chunks
