#!/usr/bin/env python3
"""Standalone RAG demo: same model, same question — cold vs RAG-augmented.

Pipeline mirrors the intended KWC wiring exactly:
  query -> kb_rag.hybrid_search (embeds via 135:8100) -> top-k chunks
        -> injected as CONTEXT block -> chat model on 135:8080

KWC talks to qwen3.5-9b; we demo on gemma-4-12b (currently loaded —
avoids a ~10s model swap mid-demo). Swap CHAT_MODEL to compare later.
"""
import json
import sys
import time
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parent))
from kb_rag.index import KbIndex          # noqa: E402
from kb_rag.embed import EmbedClient      # noqa: E402
from kb_rag.retrieve import hybrid_search # noqa: E402

CHAT_URL = "http://192.168.1.135:8080/v1/chat/completions"
CHAT_MODEL = "gemma-4-12b"
STATE = Path.home() / "klipper-rag-state/kb.sqlite"

QUESTIONS = [
    {
        "id": "probe-z-offset",
        "q": "My probe triggers too late and the first layer squishes. Which config parameter do I adjust and in which direction?",
        "expect": ["z_offset"],
    },
    {
        "id": "multi-mcu",
        "q": "Can I use more than one micro-controller board in one printer?",
        "expect": ["mcu"],
    },
    {
        "id": "gcode-macro-variable",
        "q": "How do I declare a persistent variable in a gcode_macro section and change it at runtime?",
        "expect": ["SET_GCODE_VARIABLE"],
    },
]

SYSTEM = (
    "You are a Klipper 3D-printer firmware expert helping the user edit "
    "their printer.cfg. Be concise and concrete: name exact config "
    "sections, parameters, and gcode commands."
)

AUGMENT = (
    "Answer using ONLY the CONTEXT excerpts from the Klipper docs below. "
    "Cite the doc/section for each claim. If the context doesn't answer "
    "a part, say so.\n\n=== CONTEXT ===\n{context}\n=== END CONTEXT ==="
)


def ask(messages, max_tokens=1200):
    r = httpx.post(
        CHAT_URL,
        json={
            "model": CHAT_MODEL,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": 0.3,
        },
        timeout=240.0,
    )
    r.raise_for_status()
    msg = r.json()["choices"][0]["message"]
    # reasoning models can burn the whole budget in reasoning_content;
    # fall back to the tail of reasoning so the demo shows something
    content = (msg.get("content") or "").strip()
    if not content and msg.get("reasoning_content"):
        content = "[answer only in reasoning] " + msg["reasoning_content"].strip()
    return content


def main():
    idx = KbIndex.load(STATE)
    client = EmbedClient(base_url=idx.meta["embed_url"])

    for item in QUESTIONS:
        q = item["q"]
        print("=" * 72)
        print(f"Q [{item['id']}]: {q}")
        print(f"   gold-ish terms: {item['expect']}")

        t0 = time.time()
        qv = client.embed_query(q)
        hits = hybrid_search(idx, qvec=qv, text=q, k=3)
        retrieve_ms = (time.time() - t0) * 1000

        print(f"\n--- retrieved in {retrieve_ms:.0f} ms "
              f"(embed+FTS5+dense+RRF) ---")
        for i, h in enumerate(hits, 1):
            crumb = f"{h.chunk.doc}::{h.chunk.section}"
            print(f"  [{i}] {crumb}  (rrf={h.score:.4f})")
            print("      " + h.chunk.text.replace("\n", " ")[:140])

        context = "\n\n".join(
            f"--- {h.chunk.doc} :: {h.chunk.section} ---\n{h.chunk.text}"
            for h in hits
        )

        # cold answer
        t1 = time.time()
        cold = ask([{"role": "system", "content": SYSTEM},
                    {"role": "user", "content": q}])
        cold_ms = (time.time() - t1) * 1000
        # rag answer
        t2 = time.time()
        rag = ask([{"role": "system", "content": SYSTEM},
                   {"role": "user",
                    "content": AUGMENT.format(context=context) + "\n\n" + q}])
        rag_ms = (time.time() - t2) * 1000

        def score(ans):
            low = ans.lower()
            return " / ".join(
                f"{t}:{'HIT' if t.lower() in low else 'miss'}"
                for t in item["expect"])

        print(f"\n--- COLD ({cold_ms:.0f} ms) — terms: {score(cold)} ---")
        print(cold.strip()[:600])
        print(f"\n--- RAG  ({rag_ms:.0f} ms) — terms: {score(rag)} ---")
        print(rag.strip()[:600])
        print()


if __name__ == "__main__":
    main()
