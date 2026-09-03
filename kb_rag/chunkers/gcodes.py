"""G-Codes.md chunker: one chunk per command (``#### COMMAND``).

G-Codes.md nests commands as ``### [config_section]`` / ``#### COMMAND``.
The retrievable unit is a command, but a command stanza alone ("Support for
meshes.") loses family context — so each command chunk keeps its parent
``[section]`` intro line in the breadcrumb and carries the section's intro
text as a one-line preamble when present.
"""
from __future__ import annotations

import re

from kb_rag.chunk import Chunk, approx_tokens
from kb_rag.chunkers.common import enforce_cap, find_headings, split_prose
from kb_rag.chunkers.stanza import _emit  # reuse emit semantics

DEFAULT_MAX_TOKENS = 500


def chunk_gcodes_file(
    text: str, doc: str = "G-Codes", source: str = "klipper",
    max_tokens: int = DEFAULT_MAX_TOKENS,
) -> list[Chunk]:
    display = doc.replace("_", " ")
    chunks: list[Chunk] = []
    seen: dict[str, int] = {}

    headings = find_headings(text)
    if not headings:
        return []

    section_intro: str | None = None  # intro text of current ### [section]
    stack: list[tuple[int, str]] = []

    for i, m in enumerate(headings):
        level = len(m.group(1))
        title = m.group(2).strip()
        end = headings[i + 1].start() if i + 1 < len(headings) else len(text)
        body = text[m.end():end].strip("\n").strip()

        while stack and stack[-1][0] >= level:
            stack.pop()
        stack.append((level, title))

        if title.startswith("[") and title.endswith("]"):
            # config-section header: keep intro for child commands, emit
            # the section's own description as a chunk of its own
            section_intro = body.split("\n")[0] if body else None
            if body:
                chunks.extend(_emit(source=source, doc=doc, display=display,
                                    section=title, body=body,
                                    max_tokens=max_tokens, seen=seen))
            continue

        path = [t for _, t in stack]
        if level >= 4:
            # a command: breadcrumb includes the parent [section]
            crumb_root = "Extra Docs" if source == "extra" else "Klipper Docs"
            crumb = " > ".join([crumb_root, display, *path])
            prefix = f"[{crumb}]\n\n"
            eff_max = max_tokens
            base_id = f"{source}/{doc}::{title}"
            n = seen.get(base_id, 0)
            seen[base_id] = n + 1
            if n:
                base_id = f"{base_id}#dup{n}"
            if approx_tokens(prefix + body) <= eff_max:
                chunks.append(Chunk(id=base_id, doc=doc, section=title,
                                    breadcrumb=crumb, text=prefix + body,
                                    source=source))
            else:
                body_budget = max(64, eff_max - approx_tokens(prefix))
                parts = enforce_cap(split_prose(body, body_budget), body_budget)
                for pi, p in enumerate(parts):
                    chunks.append(Chunk(
                        id=f"{base_id}#part{pi}", doc=doc,
                        section=title, breadcrumb=crumb, text=prefix + p,
                        source=source, part=pi))
        else:
            if body:
                chunks.extend(_emit(source=source, doc=doc, display=display,
                                    section=title, body=body,
                                    max_tokens=max_tokens, seen=seen))
    return chunks
