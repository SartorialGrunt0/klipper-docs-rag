"""Validate gold labels against the built index (no network needed)."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from kb_rag.index import KbIndex

idx = KbIndex.load(str(Path.home() / "klipper-rag-state/kb.sqlite"))
docs = {c.doc for c in idx.chunks}
gold = [json.loads(l) for l in open(Path(__file__).parent / "gold/queries.jsonl")
        if l.strip()]
bad = 0
for item in gold:
    for doc, prefix in item["gold"]:
        if doc not in docs:
            print(f"BAD DOC  {item['id']}: {doc} not in corpus")
            bad += 1
            continue
        ok = any(c.doc == doc and prefix.lower() in (c.section or "").lower()
                 for c in idx.chunks)
        if not ok:
            print(f"BAD SECT {item['id']}: {doc}::{prefix}")
            bad += 1
print("invalid gold entries:", bad)
corpus = "\n".join(c.text for c in idx.chunks)
for needle in ("Timer too close", "already triggered", "dockable",
               "Common_Errors", "z_tilt", "sdcard_loop"):
    print(needle, "->", needle in corpus)
