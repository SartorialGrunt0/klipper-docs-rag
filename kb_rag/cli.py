"""kb-rag CLI — phase 0: corpus chunking stats + dump.

Usage:
    kb-rag stats <docs-dir> [--kwc-dir DIR] [--max-tokens N] [--json]
    kb-rag dump  <docs-dir> (--section NAME | --doc NAME) [--json]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from kb_rag.chunkers import chunk_document


def _iter_docs(docs_dir: Path, source: str):
    for path in sorted(docs_dir.glob("*.md")):
        yield path.stem, path.read_text(encoding="utf-8", errors="replace"), source


def _collect(docs_dir: Path, kwc_dir: Path | None, max_tokens: int):
    chunks = []
    for stem, text, source in _iter_docs(docs_dir, "klipper"):
        chunks.extend(chunk_document(text, stem, source=source, max_tokens=max_tokens))
    if kwc_dir and kwc_dir.is_dir():
        for stem, text, source in _iter_docs(kwc_dir, "kwc"):
            chunks.extend(chunk_document(text, stem, source=source, max_tokens=max_tokens))
    return chunks


def cmd_stats(args: argparse.Namespace) -> int:
    docs_dir = Path(args.docs_dir).expanduser()
    if not docs_dir.is_dir():
        print(f"error: no such directory: {docs_dir}", file=sys.stderr)
        return 2
    chunks = _collect(docs_dir, Path(args.kwc_dir).expanduser() if args.kwc_dir else None,
                      args.max_tokens)
    n = len(chunks)
    if n == 0:
        print("error: no markdown docs found", file=sys.stderr)
        return 2
    toks = sorted(c.tokens for c in chunks)
    # chunker contract: split bodies fit max_tokens; total = body + prefix.
    # Prefixes (breadcrumb + brackets + join blank) are <64 tokens, so a
    # 64-token slack is the real hard cap; anything over it is a bug.
    hard_cap = args.max_tokens + 64
    over = [c for c in chunks if c.tokens > hard_cap]
    by_doc: dict[str, int] = {}
    for c in chunks:
        by_doc[c.doc] = by_doc.get(c.doc, 0) + 1
    ids = [c.id for c in chunks]
    dupes = len(ids) - len(set(ids))
    stanza_chunks = sum(1 for c in chunks if c.section and c.section.startswith("["))
    stats = {
        "docs": len(by_doc),
        "chunks": n,
        "tokens_approx_total": sum(toks),
        "tokens_min": toks[0],
        "tokens_median": toks[n // 2],
        "tokens_p95": toks[int(n * 0.95)],
        "tokens_max": toks[-1],
        "over_max_tokens": len(over),
        "hard_cap": hard_cap,
        "duplicate_ids": dupes,
        "stanza_chunks": stanza_chunks,
        "splits": sum(1 for c in chunks if "#" in c.id.split("::", 1)[-1]),
    }
    if args.json:
        print(json.dumps(stats, indent=2))
        return 0
    for k, v in stats.items():
        print(f"{k:22} {v}")
    print("\nper-doc chunk counts:")
    for doc, count in sorted(by_doc.items(), key=lambda kv: -kv[1])[:15]:
        print(f"  {doc:45} {count}")
    if over:
        print("\nover-cap chunks (first 10):", file=sys.stderr)
        for c in over[:10]:
            print(f"  {c.id} ({c.tokens} tok)", file=sys.stderr)
    return 1 if (over or dupes) else 0


def cmd_dump(args: argparse.Namespace) -> int:
    docs_dir = Path(args.docs_dir).expanduser()
    if not docs_dir.is_dir():
        print(f"error: no such directory: {docs_dir}", file=sys.stderr)
        return 2
    chunks = _collect(docs_dir, Path(args.kwc_dir).expanduser() if args.kwc_dir else None,
                      args.max_tokens)
    if args.section:
        needle = args.section.lower().removeprefix("[").removesuffix("]")
        chunks = [
            c for c in chunks
            if c.section and needle in c.section.lower().removeprefix("[").removesuffix("]")
        ]
    elif args.doc:
        chunks = [c for c in chunks if c.doc == args.doc]
    if args.json:
        print(json.dumps([{
            "id": c.id, "doc": c.doc, "section": c.section,
            "breadcrumb": c.breadcrumb, "tokens": c.tokens, "text": c.text,
        } for c in chunks], indent=2))
        return 0
    for c in chunks:
        print(f"=== {c.id}  [{c.tokens} tok]")
        print(c.text)
        print()
    print(f"({len(chunks)} chunks)", file=sys.stderr)
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="kb-rag", description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

    ps = sub.add_parser("stats", help="chunk-count and size statistics for a docs dir")
    ps.add_argument("docs_dir")
    ps.add_argument("--kwc-dir", help="extra dir of KWC-authored docs")
    ps.add_argument("--max-tokens", type=int, default=500)
    ps.add_argument("--json", action="store_true")
    ps.set_defaults(fn=cmd_stats)

    pd = sub.add_parser("dump", help="print chunks matching a section or doc")
    pd.add_argument("docs_dir")
    pd.add_argument("--kwc-dir")
    pd.add_argument("--max-tokens", type=int, default=500)
    pd.add_argument("--section", help="substring of chunk section, e.g. heater_fan")
    pd.add_argument("--doc", help="exact doc stem, e.g. Bed_Mesh")
    pd.add_argument("--json", action="store_true")
    pd.set_defaults(fn=cmd_dump)

    args = p.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    raise SystemExit(main())
