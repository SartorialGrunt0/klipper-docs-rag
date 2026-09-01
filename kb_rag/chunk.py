"""Chunk model shared by all chunkers."""
from __future__ import annotations

from dataclasses import dataclass, field


def approx_tokens(text: str) -> int:
    """Cheap, deterministic token estimate (~4 chars/token).

    Used for chunk-size budgeting only, never for billing or prompts.
    Klipper docs are dense with short identifiers so the true ratio is
    slightly lower; budgeting against a *higher* estimate keeps chunks
    conservatively small.
    """
    return max(1, (len(text) + 3) // 4)


@dataclass
class Chunk:
    """One retrievable unit of documentation.

    ``text`` already contains the breadcrumb prefix; embedding it verbatim
    is intentional (structured-doc retrieval win, see docs/RAG-RESEARCH.md
    §4.1).
    """

    id: str                 # stable: f"{source}/{doc}::{section_or_heading}[#part]"
    doc: str                # filename stem, e.g. "Config_Reference"
    text: str               # breadcrumb prefix + body
    source: str = "klipper"  # klipper | kwc | config
    section: str | None = None   # e.g. "[heater_fan]" or "BED_MESH_CALIBRATE"
    breadcrumb: str = ""         # e.g. "Klipper Docs > Config Reference > [heater_fan]"
    part: int = 0                # sub-split index of the same logical section
    tokens: int = field(default=0, compare=False)

    def __post_init__(self) -> None:
        if not self.tokens:
            self.tokens = approx_tokens(self.text)
