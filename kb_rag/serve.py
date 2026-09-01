"""OpenAI-compatible RAG proxy: exposes virtual model 'klipper-expert'.

Any OpenAI client (KWC, Open WebUI, curl) can select model
"klipper-expert"; this server retrieves from the klipper-docs index,
injects a CONTEXT block, and forwards to the real chat model on the
llama.cpp server. No tool calling needed — RAG happens for every prompt.

Endpoints:
  GET  /v1/models             -> virtual 'klipper-expert' + passthrough list
  POST /v1/chat/completions   -> RAG-augmented chat (model=klipper-expert)
  POST /retrieve              -> {"query":..., "k":...} raw chunks (dev)
  GET  /health                -> ok

Run (on the CachyPC, next to llama-server and the embed server):
  python -m kb_rag.serve --state ~/klipper-rag-state/kb.sqlite \
      --chat-url http://127.0.0.1:8080/v1 --base-model gemma-4-12b --port 8090

Stdlib http.server + httpx only: the dependency list stays closed.
"""
from __future__ import annotations

import argparse
import json
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Protocol

import httpx
import numpy as np

from kb_rag.embed import EmbedClient
from kb_rag.index import KbIndex
from kb_rag.retrieve import hybrid_search


class QueryEmbedder(Protocol):
    """Anything with embed_query(text) -> vector (EmbedClient or a fake)."""

    def embed_query(self, text: str) -> "np.ndarray": ...

VIRTUAL_MODEL = "klipper-expert"
SYSTEM_PREAMBLE = (
    "You are a Klipper 3D-printer firmware expert helping the user with "
    "printer.cfg, macros, and troubleshooting."
)
CONTEXT_TEMPLATE = (
    "{user_system}\n\n"
    "Use the CONTEXT excerpts from the Klipper docs when relevant and cite "
    "the doc/section; if they don't cover the question, answer from your "
    "own knowledge and say the docs context didn't cover it.\n\n"
    "=== CONTEXT ===\n{context}\n=== END CONTEXT ==="
)


class RagService:
    def __init__(self, index: KbIndex, embed: QueryEmbedder,
                 chat_url: str, base_model: str, k: int = 3,
                 max_context_chars: int = 4000) -> None:
        self.index = index
        self.embed = embed
        self.chat_url = chat_url.rstrip("/")
        self.base_model = base_model
        self.k = k
        self.max_context_chars = max_context_chars
        self._lock = threading.Lock()  # index is read-only; embed is not reentrant-safe

    def retrieve(self, query: str, k: int | None = None) -> list[dict]:
        with self._lock:
            qv = self.embed.embed_query(query)
        hits = hybrid_search(self.index, qvec=qv, text=query,
                             k=k or self.k)
        return [{"doc": h.chunk.doc, "section": h.chunk.section,
                 "text": h.chunk.text, "score": h.score} for h in hits]

    def augment(self, messages: list[dict]) -> tuple[list[dict], list[dict]]:
        """Return (forwarded_messages, retrieved_chunks)."""
        msgs = list(messages)
        # last user message is the query
        query = next((m["content"] for m in reversed(msgs)
                      if m.get("role") == "user"), "")
        if isinstance(query, list):  # multimodal parts; keep text only
            query = " ".join(p.get("text", "") for p in query
                             if isinstance(p, dict))
        chunks = self.retrieve(query)
        if not chunks:
            return msgs, []
        context = "\n\n".join(
            f"--- {c['doc']} :: {c['section']} ---\n{c['text']}"
            for c in chunks)[: self.max_context_chars]
        out = []
        seen_system = False
        for m in msgs:
            if m.get("role") == "system" and not seen_system:
                out.append({"role": "system",
                            "content": CONTEXT_TEMPLATE.format(
                                user_system=m.get("content", SYSTEM_PREAMBLE),
                                context=context)})
                seen_system = True
            else:
                out.append(m)
        if not seen_system:
            out.insert(0, {"role": "system",
                           "content": CONTEXT_TEMPLATE.format(
                               user_system=SYSTEM_PREAMBLE,
                               context=context)})
        return out, chunks

    def chat(self, payload: dict) -> dict:
        messages, chunks = self.augment(payload.get("messages", []))
        fwd = {**payload, "model": self.base_model, "messages": messages}
        fwd.pop("stream", None)  # non-streaming only for now
        r = httpx.post(f"{self.chat_url}/chat/completions", json=fwd,
                       timeout=300.0)
        r.raise_for_status()
        data = r.json()
        data["model"] = VIRTUAL_MODEL
        data["rag"] = {"chunks": [
            {"doc": c["doc"], "section": c["section"], "score": c["score"]}
            for c in chunks]}
        return data


class Handler(BaseHTTPRequestHandler):
    service: RagService  # injected

    def _json(self, code: int, obj: dict) -> None:
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):  # noqa: N802
        if self.path == "/health":
            self._json(200, {"ok": True, "chunks": self.service.index.count()})
        elif self.path.startswith("/v1/models"):
            try:
                r = httpx.get(f"{self.service.chat_url}/models", timeout=10.0)
                models = r.json().get("data", [])
            except Exception:
                models = []
            names = [m["id"] for m in models]
            self._json(200, {"object": "list", "data": [
                {"id": VIRTUAL_MODEL, "object": "model",
                 "owned_by": "klipper-docs-rag",
                 "base": self.service.base_model, "passthrough": names}]})
        else:
            self._json(404, {"error": "not found"})

    def do_POST(self):  # noqa: N802
        length = int(self.headers.get("Content-Length", 0))
        try:
            payload = json.loads(self.rfile.read(length))
        except (json.JSONDecodeError, ValueError):
            self._json(400, {"error": "invalid json"})
            return
        try:
            if self.path == "/v1/chat/completions":
                model = payload.get("model", "")
                if model not in (VIRTUAL_MODEL, f"klipper-rag/{VIRTUAL_MODEL}"):
                    # plain passthrough for real model names
                    r = httpx.post(f"{self.service.chat_url}/chat/completions",
                                   json=payload, timeout=300.0)
                    self._json(r.status_code, r.json())
                else:
                    self._json(200, self.service.chat(payload))
            elif self.path == "/retrieve":
                res = self.service.retrieve(payload.get("query", ""),
                                            payload.get("k"))
                self._json(200, {"results": res})
            else:
                self._json(404, {"error": "not found"})
        except httpx.HTTPStatusError as e:
            self._json(e.response.status_code,
                       {"error": f"upstream: {e.response.text[:300]}"})
        except Exception as e:  # keep the server alive; report cleanly
            self._json(502, {"error": f"{type(e).__name__}: {e}"})

    def log_message(self, format, *args):  # noqa: A002 quieter default log
        sys.stderr.write("%s %s\n" % (self.address_string(), format % args))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--state", required=True)
    ap.add_argument("--chat-url", default="http://127.0.0.1:8080/v1")
    ap.add_argument("--base-model", default="gemma-4-12b")
    ap.add_argument("--embed-url", default=None)
    ap.add_argument("--port", type=int, default=8090)
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("-k", type=int, default=3)
    args = ap.parse_args()

    index = KbIndex.load(Path(args.state))
    embed = EmbedClient(base_url=args.embed_url or index.meta["embed_url"])
    Handler.service = RagService(index, embed, args.chat_url,
                                 args.base_model, k=args.k)
    print(f"klipper-expert on :{args.port} -> {args.chat_url} "
          f"({args.base_model}), index={index.count()} chunks", flush=True)
    ThreadingHTTPServer((args.host, args.port), Handler).serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
