"""Shared splitting primitives used by all chunkers.

Two granularities:
- ``split_param_blocks``: stanza bodies (Config_Reference style `#param:`
  lines) split at parameter boundaries so a param's name+description never
  straddles two chunks.
- ``split_prose``: paragraph packing with sentence-level overlap, used for
  prose blocks and any single block still over budget after param split.
"""
from __future__ import annotations

import re

# A commented-out example parameter line, e.g. "#baud: 250000" or "#pin:".
# Description lines are "#   text" (hash + spaces), which do NOT match.
PARAM_START_RE = re.compile(r"^#\w[\w]*\s*:")

# Real markdown headings: 1-6 '#', EXACTLY one space, then text starting
# with a non-space, non-'#' char. This deliberately does NOT match Klipper
# doc conventions inside stanza bodies:
#   "#param: 40"   -> no space after '#'
#   "#   description text"  -> three spaces (comment descriptions)
HEADING_RE = re.compile(r"^(#{1,6}) (?=[^#\s])(.*?)\s*$", re.MULTILINE)
FENCE_RE = re.compile(r"^\s*(```|~~~)")


def find_headings(text: str):
    """HEADING_RE matches that are outside fenced code blocks.

    Klipper docs embed example configs inside ``` fences whose '# comment'
    lines would otherwise parse as level-1 headings and shred stanzas.
    Returned matches keep HEADING_RE's group numbering (1=hashes, 2=title).
    """
    fence_spans: list[tuple[int, int]] = []
    in_fence = False
    marker = None
    start = 0
    for m in re.finditer(r"^\s*(```|~~~)", text, re.MULTILINE):
        if not in_fence:
            in_fence, marker, start = True, m.group(1), m.start()
        elif m.group(1) == marker:
            in_fence = False
            fence_spans.append((start, m.end()))
    if in_fence:
        fence_spans.append((start, len(text)))
    out = []
    for m in HEADING_RE.finditer(text):
        if not any(s <= m.start() < e for s, e in fence_spans):
            out.append(m)
    return out


def approx_tokens(text: str) -> int:
    from kb_rag.chunk import approx_tokens as _t

    return _t(text)


def split_param_blocks(body: str) -> list[str]:
    """Split a stanza body into (intro + each ``#param:`` block).

    Returns one block per parameter plus a leading intro block (may be
    empty-ish; callers filter). Never splits inside a parameter block.
    """
    blocks: list[str] = []
    current: list[str] = []
    for line in body.split("\n"):
        if PARAM_START_RE.match(line):
            blocks.append("\n".join(current))
            current = [line]
        else:
            current.append(line)
    blocks.append("\n".join(current))
    return [b.strip("\n") for b in blocks if b.strip()]


def _sentences(text: str) -> list[str]:
    # split after terminal punctuation + whitespace, keeping the punctuation
    parts = re.split(r"(?<=[.!?])\s+", text)
    return [p for p in parts if p.strip()]


def _pack_lines(text: str, max_tokens: int) -> list[str]:
    """Last-resort split: pack raw lines under max_tokens (tables, dumps)."""
    parts: list[str] = []
    current: list[str] = []
    current_tok = 0
    for line in text.split("\n"):
        line_tok = approx_tokens(line)
        if current_tok + line_tok > max_tokens and current:
            parts.append("\n".join(current))
            current, current_tok = [], 0
        current.append(line)
        current_tok += line_tok
    if current:
        parts.append("\n".join(current))
    return parts


def split_prose(body: str, max_tokens: int) -> list[str]:
    """Pack paragraphs into <= max_tokens parts, splitting on sentences.

    When a part rolls over, the last sentence of the previous part is
    repeated at the head of the next (cheap overlap so boundary chunks stay
    quotable).
    """
    paragraphs = [p for p in re.split(r"\n\s*\n", body) if p.strip()]
    parts: list[str] = []
    current: list[str] = []
    current_tok = 0

    def flush(overlap: str | None) -> None:
        nonlocal current, current_tok
        if current:
            parts.append("\n\n".join(current))
        if overlap:
            current = [overlap]
            current_tok = approx_tokens(overlap)
        else:
            current = []
            current_tok = 0

    for para in paragraphs:
        para = para.strip()
        if approx_tokens(para) <= max_tokens:
            if current_tok + approx_tokens(para) > max_tokens and current:
                last = _sentences(current[-1])
                flush(last[-1] if last else None)
            current.append(para)
            current_tok += approx_tokens(para)
        else:
            # paragraph itself oversized -> sentence-pack it
            if current:
                last = _sentences(current[-1])
                flush(last[-1] if last else None)
            buf: list[str] = []
            buf_tok = 0
            for sent in _sentences(para):
                sent = sent.strip()
                if approx_tokens(sent) > max_tokens:
                    # no sentence ends inside (tables, shell dumps):
                    # fall back to line packing
                    if buf:
                        parts.append(" ".join(buf))
                        buf, buf_tok = [], 0
                    parts.extend(_pack_lines(sent, max_tokens))
                    continue
                if buf_tok + approx_tokens(sent) > max_tokens and buf:
                    parts.append(" ".join(buf))
                    # carry the final sentence across as overlap
                    buf = [buf[-1], sent] if buf[-1] != sent else [sent]
                    buf_tok = approx_tokens(buf[0]) + approx_tokens(sent)
                else:
                    buf.append(sent)
                    buf_tok += approx_tokens(sent)
            if buf:
                current = buf
                current_tok = buf_tok
    flush(None)
    return [p for p in parts if p.strip()]


def enforce_cap(parts: list[str], budget: int) -> list[str]:
    """Hard guarantee: no part exceeds budget tokens.

    Packing heuristics (overlap carry, cross-paragraph accounting) can
    overshoot by a sentence; anything still over gets line-packed.
    """
    out: list[str] = []
    for p in parts:
        if approx_tokens(p) <= budget:
            out.append(p)
        else:
            out.extend(_pack_lines(p, budget))
    return out


def pack_blocks(blocks: list[str], max_tokens: int) -> list[str]:
    """Greedy-pack pre-split blocks (e.g. param blocks) under max_tokens.

    Any single block larger than max_tokens is further prose-split
    (sentence granularity) so nothing escapes the cap.
    """
    parts: list[str] = []
    current: list[str] = []
    current_tok = 0
    for block in blocks:
        if approx_tokens(block) > max_tokens:
            if current:
                parts.append("\n".join(current))
                current, current_tok = [], 0
            parts.extend(split_prose(block, max_tokens))
            continue
        if current_tok + approx_tokens(block) > max_tokens and current:
            parts.append("\n".join(current))
            current, current_tok = [], 0
        current.append(block)
        current_tok += approx_tokens(block)
    if current:
        parts.append("\n".join(current))
    return parts
