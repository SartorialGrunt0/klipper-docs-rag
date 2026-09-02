"""kb-rag CLI — phase 0: corpus chunking stats + dump.

Usage:
    kb-rag stats <docs-dir> [--kwc-dir DIR] [--max-tokens N] [--json]
    kb-rag dump  <docs-dir> (--section NAME | --doc NAME) [--json]
    kb-rag build <docs-dir> [--kwc-dir DIR] [--state PATH] [--embed-url URL]
    kb-rag query <state-path> "text query" [-k N] [--json]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

from kb_rag.chunkers import chunk_document

DEFAULT_STATE = Path(os.environ.get("KB_STATE", str(Path.home() / "klipper-rag-state" / "kb.sqlite")))
DEFAULT_EMBED_URL = os.environ.get("KB_EMBED_URL", "http://127.0.0.1:8080")
DEFAULT_EMBED_MODEL = os.environ.get("KB_EMBED_MODEL", "nomic-embed-text-v1.5")


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


def cmd_build(args: argparse.Namespace) -> int:
    docs_dir = Path(args.docs_dir).expanduser()
    if not docs_dir.is_dir():
        print(f"error: no such directory: {docs_dir}", file=sys.stderr)
        return 2
    chunks = _collect(docs_dir, Path(args.kwc_dir).expanduser() if args.kwc_dir else None,
                      args.max_tokens)
    if not chunks:
        print("error: no markdown docs found", file=sys.stderr)
        return 2
    from kb_rag.embed import EmbedClient
    from kb_rag.index import KbIndex

    client = EmbedClient(base_url=args.embed_url, model=args.embed_model)
    # probe once, early — fail before burning CPU on chunking output
    try:
        probe = client.embed_query("probe")
    except Exception as e:  # noqa: BLE001 - surface any transport error
        print(f"error: embedding server unreachable at {args.embed_url}: {e}",
              file=sys.stderr)
        return 3
    dim = len(probe)

    t0 = time.monotonic()
    vectors = _stack_rows(client.embed_documents([c.text for c in chunks]), dim)
    idx = KbIndex(Path(args.state), dim=dim)
    idx.save(chunks, vectors, meta={
        "docs_dir": str(docs_dir),
        "kwc_dir": str(args.kwc_dir) if args.kwc_dir else None,
        "embed_url": args.embed_url,
        "embed_model": client.model,
        "built_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "build_seconds": round(time.monotonic() - t0, 1),
    })
    print(f"built {idx.count()} chunks dim={dim} "
          f"in {idx.meta['build_seconds']}s -> {args.state}")
    return 0


def _stack_rows(vecs, dim: int):
    import numpy as np
    arr = np.stack([np.asarray(v, dtype=np.float32) for v in vecs])
    bad = np.abs(np.linalg.norm(arr, axis=1) - 1.0) > 1e-3
    if bad.any():
        raise RuntimeError(f"{int(bad.sum())} vectors not unit-norm")
    assert arr.shape[1] == dim
    return arr


def cmd_query(args: argparse.Namespace) -> int:
    from kb_rag.embed import EmbedClient
    from kb_rag.index import KbIndex

    state = Path(args.state).expanduser()
    if not state.is_file():
        print(f"error: no index at {state} (run kb-rag build first)", file=sys.stderr)
        return 2
    idx = KbIndex.load(state)
    embed_url = args.embed_url or idx.meta.get("embed_url") or DEFAULT_EMBED_URL
    embed_model = args.embed_model or idx.meta.get("embed_model") or DEFAULT_EMBED_MODEL
    client = EmbedClient(base_url=embed_url, model=embed_model)

    t0 = time.monotonic()
    qv = client.embed_query(args.query)
    embed_ms = int((time.monotonic() - t0) * 1000)
    sources = set(args.source.split(",")) if args.source else None
    t1 = time.monotonic()
    results = idx.search_vector(qv, k=args.k, sources=sources)
    search_ms = int((time.monotonic() - t1) * 1000)

    if args.json:
        print(json.dumps([{
            "id": r.chunk.id, "section": r.chunk.section,
            "breadcrumb": r.chunk.breadcrumb, "score": round(r.score, 4),
            "text": r.chunk.text,
        } for r in results], indent=2))
        return 0
    for r in results:
        print(f"--- {r.score:.4f}  {r.chunk.id}")
        snippet = r.chunk.text[:240].replace(chr(10), " ")
        print(f"    {snippet}...")
    print(f"(embed {embed_ms}ms + search {search_ms}ms, {idx.count()} chunks)",
          file=sys.stderr)
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

    pb = sub.add_parser("build", help="chunk + embed a docs dir into the index")
    pb.add_argument("docs_dir")
    pb.add_argument("--kwc-dir")
    pb.add_argument("--max-tokens", type=int, default=500)
    pb.add_argument("--state", default=str(DEFAULT_STATE))
    pb.add_argument("--embed-url", default=DEFAULT_EMBED_URL)
    pb.add_argument("--embed-model", default=DEFAULT_EMBED_MODEL,
                    help="embedding model name served by --embed-url "
                         "(must stay the same for query/serve; changing it "
                         "requires a rebuild)")
    pb.set_defaults(fn=cmd_build)

    pq = sub.add_parser("query", help="dense semantic query against the index")
    pq.add_argument("state")
    pq.add_argument("query")
    pq.add_argument("-k", type=int, default=3)
    pq.add_argument("--source", help="comma list: klipper,kwc")
    pq.add_argument("--embed-url", default=None)
    pq.add_argument("--embed-model", default=None,
                    help="override the model recorded at build time")
    pq.add_argument("--json", action="store_true")
    pq.set_defaults(fn=cmd_query)

    args = p.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    raise SystemExit(main())
