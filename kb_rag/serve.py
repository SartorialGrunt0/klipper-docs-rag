"""OpenAI-compatible RAG proxy: exposes virtual model 'klipper-expert'.

Any OpenAI client (Open WebUI, LibreChat, curl, SDKs) can select model
"klipper-expert"; this server retrieves from the klipper-docs index,
injects a CONTEXT block, and forwards to the real chat model on the
upstream provider. No tool calling needed — RAG happens for every prompt.

Endpoints:
  GET  /v1/models             -> virtual 'klipper-expert' + passthrough list
  POST /v1/chat/completions   -> RAG-augmented chat (model=klipper-expert)
  POST /retrieve              -> {"query":..., "k":...} raw chunks (dev)
  GET  /health                -> ok

Run (next to your embedding server, after `kb-rag build`):
  python -m kb_rag.serve --state ~/.klipper-rag/kb.sqlite \
      --chat-url http://127.0.0.1:8080/v1 --base-model <chat-model> --port 8090

Stdlib http.server + httpx only: the dependency list stays closed.
"""
from __future__ import annotations

import argparse
import json
import re
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
# --- topic gate -----------------------------------------------------------
# Blocks doc injection for off-topic turns (chit-chat) so they don't burn
# the context budget or drag the model into doc-talk.
#
# Signal = *centroid-whitened* top-1 cosine. nomic embeddings are
# anisotropic: every vector sits in one narrow cone, so raw cosine floors
# at ~0.5 even for "hello" (measured on the live index: chit-chat t1
# 0.50-0.70, domain questions 0.63-0.86 — overlapping, undecidable).
# Removing the corpus-mean direction separates them: measured chit-chat
# tops out at 0.35 ("what's the weather like"), while gold-set queries
# without any intent-word bypass floor at 0.49. Threshold 0.40 sits in
# that gap; vague domain questions below it are protected by DOMAIN_HINTS.
DEFAULT_GATE_THRESHOLD = 0.40
# Words that make a query domain-relevant by construction even when the
# embedding signal is weak (vague questions: "the fan won't turn off").
# Only unambiguous Klipper-domain nouns go here; generic words dilute the
# gate (an edit-verb regex on the client side handles edit flows).
DOMAIN_HINTS = frozenset({
    "klipper", "printer", "printer.cfg", "mcu", "gcode", "macro", "jinja",
    "extruder", "stepper", "heater", "heater_fan", "part_fan", "fan",
    "probe", "bltouch", "tmc2209", "tmc2240", "tmc", "driver", "drivers",
    "firmware", "moonraker", "mainsail", "fluidd", "bed_mesh", "mesh",
    "homing", "endstop", "retraction", "pressure_advance", "input_shaper",
    "shaper", "z_tilt", "screws_tilt", "delta", "resonance", "ringing",
    "ratros", "voron", "ebbccan", "katapult", "klippy", "gcodes",
    "kinematics", "stealthburner", "hotkey", "spider",
})
# Cross-encoder rescoring: 10 fused candidates -> k (eval-earned: hybrid
# R@3 0.857/MRR 0.694 -> 0.898/0.770 at p50 74ms, eval-earned).
RERANK_CANDIDATES = 10
# Stricter than retrieve._IDENT_RE: separators must sit *between* alphanums
# (no trailing "vacation." false positives), digits must be inside a word.
# heater_fan / tmc2209 / cycle_time / can0 match; prose does not.
_IDENT_WORD_RE = re.compile(
    r"[a-z0-9]+(?:[._+][a-z0-9]+)+|(?:\b|\A)[a-z]*\d[a-z0-9._+]*(?:\b|\Z)")
SYSTEM_PREAMBLE = (
    "You are a Klipper 3D-printer firmware expert helping the user with "
    "printer.cfg, macros, and troubleshooting."
)
CONTEXT_TEMPLATE = (
    "{user_system}\n\n"
    "Use the CONTEXT excerpts from the Klipper docs when relevant and cite "
    "the doc/section. If the context doesn't cover part of the question and "
    "document-lookup tools (search_klipper_docs, get_config_reference_section, "
    "read_klipper_doc) are available, USE THEM for that part instead of "
    "answering from memory. Only fall back to your own knowledge when no "
    "tool can help, and say the docs context didn't cover it.\n\n"
    "=== CONTEXT ===\n{context}\n=== END CONTEXT ==="
)


class RagService:
    def __init__(self, index: KbIndex, embed: QueryEmbedder,
                 chat_url: str, base_model: str, k: int = 3,
                 max_context_chars: int = 4000,
                 gate_threshold: float | None = DEFAULT_GATE_THRESHOLD,
                 rerank_client=None,
                 staleness=None) -> None:
        self.index = index
        self.embed = embed
        self.chat_url = chat_url.rstrip("/")
        self.base_model = base_model
        self.k = k
        self.max_context_chars = max_context_chars
        self.gate_threshold = gate_threshold or None
        self.rerank = rerank_client  # None or RerankClient
        self.staleness = staleness  # None or UpstreamChecker
        self._lock = threading.Lock()  # index is read-only; embed is not reentrant-safe
        self._wmat: np.ndarray | None = None  # centroid-whitened chunk vectors

    def _whitened_top1(self, qv: np.ndarray) -> float:
        """Top-1 cosine after removing the corpus-mean direction.

        Corrects nomic's anisotropy (raw cosines compress into 0.5-0.9
        regardless of topic). Computed against a lazily built whitened
        copy of the chunk matrix; falls back to raw cosine if the
        correction degenerates.
        """
        V = self.index.vectors
        mu = V.mean(axis=0)
        n = float(np.linalg.norm(mu))
        if not n:
            return float((V @ qv).max()) if len(V) else 0.0
        mu = mu / n
        if self._wmat is None:
            w = V - np.outer(V @ mu, mu)
            norms = np.linalg.norm(w, axis=1, keepdims=True)
            self._wmat = w / np.where(norms == 0, 1, norms)
        qw = qv - mu * float(qv @ mu)
        qn = float(np.linalg.norm(qw))
        if not qn:
            return 0.0
        return float((self._wmat @ (qw / qn)).max())

    def retrieve_full(self, query: str, k: int | None = None) -> dict:
        """Hybrid retrieval with a topic-confidence gate.

        Gate signal = centroid-whitened best dense cosine (RRF scores are
        rank-fusion values, not comparable to a threshold). A query passes
        when its whitened top-1 clears the threshold, it names an
        identifier (``heater_fan``, ``tmc2209``), or it contains a known
        domain word (``fan``, ``mesh``...): those are doc intent by
        construction, and the embedding signal undershoots on short or
        vague queries. Fails closed: gate ON by default.
        """
        with self._lock:
            qv = self.embed.embed_query(query)
        top1 = self._whitened_top1(qv)
        ql = query.lower()
        tokens = set(re.findall(r"[a-z0-9._+]+", ql))
        has_intent = bool(_IDENT_WORD_RE.findall(ql)) or bool(
            DOMAIN_HINTS.intersection(tokens))
        if self.gate_threshold is not None and not has_intent \
                and top1 < self.gate_threshold:
            return {"chunks": [], "gate_passed": False,
                    "top1_cosine": top1, "ident": False}
        kk = k or self.k
        if self.rerank is not None:
            from kb_rag.rerank import rerank_hybrid
            hits = rerank_hybrid(
                self.rerank, query,
                lambda n: hybrid_search(self.index, qvec=qv, text=query,
                                        k=n),
                k=kk, candidates=RERANK_CANDIDATES)
        else:
            hits = hybrid_search(self.index, qvec=qv, text=query, k=kk)
        return {"chunks": [{"doc": h.chunk.doc, "section": h.chunk.section,
                            "text": h.chunk.text, "score": h.score}
                           for h in hits],
                "gate_passed": True, "top1_cosine": top1,
                "ident": has_intent}

    def retrieve(self, query: str, k: int | None = None) -> list[dict]:
        return self.retrieve_full(query, k)["chunks"]

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
        data["rag"] = {"gate_passed": bool(chunks), "chunks": [
            {"doc": c["doc"], "section": c["section"], "score": c["score"]}
            for c in chunks]}
        if chunks and self.staleness is not None:
            # docs-grounded answer + demonstrably-stale corpus => footnote
            note = self.staleness.footnote()
            if note:
                choice = (data.get("choices") or [{}])[0]
                msg = choice.get("message") or {}
                if isinstance(msg.get("content"), str) and msg["content"]:
                    choice.setdefault("message", msg)["content"] = \
                        msg["content"] + note
                    data["rag"]["stale"] = True
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
            body = {"ok": True, "chunks": self.service.index.count(),
                    "docs_version": self.service.index.meta.get("docs_version")}
            if self.service.staleness is not None:
                body["stale"] = bool(self.service.staleness.footnote())
            self._json(200, body)
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
                res = self.service.retrieve_full(
                    payload.get("query", ""), payload.get("k"))
                self._json(200, {"results": res["chunks"],
                                 "gate_passed": res["gate_passed"],
                                 "top1_cosine": res["top1_cosine"]})
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
    ap.add_argument("--base-model", required=True,
                    help="chat model served by --chat-url that answers "
                         "RAG-augmented requests")
    ap.add_argument("--embed-url", default=None)
    ap.add_argument("--embed-model", default=None,
                    help="embedding model name; defaults to the one "
                         "recorded in the index at build time")
    ap.add_argument("--port", type=int, default=8090)
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("-k", type=int, default=3)
    ap.add_argument("--gate-threshold", type=float,
                    default=DEFAULT_GATE_THRESHOLD,
                    help="min top-1 dense cosine to inject docs; 0 disables")
    ap.add_argument("--rerank-url", default=None,
                    help="llama.cpp --rerank endpoint (e.g. "
                         "http://127.0.0.1:8101/v1/rerank); unset disables")
    ap.add_argument("--check-upstream", dest="check_upstream",
                    action="store_true", default=True,
                    help="cached staleness check vs upstream Klipper; "
                         "stale RAG appends a docs-update footnote (default)")
    ap.add_argument("--no-check-upstream", dest="check_upstream",
                    action="store_false",
                    help="never contact upstream; no staleness footnote")
    args = ap.parse_args()

    index = KbIndex.load(Path(args.state))
    embed = EmbedClient(
        base_url=args.embed_url or index.meta["embed_url"],
        model=args.embed_model or index.meta.get("embed_model"),
    )
    reranker = None
    if args.rerank_url:
        from kb_rag.rerank import RerankClient
        reranker = RerankClient(args.rerank_url)
    staleness = None
    if args.check_upstream:
        from kb_rag.version import UpstreamChecker
        staleness = UpstreamChecker(
            index.meta.get("docs_version"),
            cache_path=Path(args.state).parent / "upstream.json")
    Handler.service = RagService(index, embed, args.chat_url,
                                 args.base_model, k=args.k,
                                 gate_threshold=args.gate_threshold,
                                 rerank_client=reranker,
                                 staleness=staleness)
    print(f"klipper-expert on :{args.port} -> {args.chat_url} "
          f"({args.base_model}), index={index.count()} chunks, "
          f"gate={args.gate_threshold or 'OFF'}, "
          f"rerank={args.rerank_url or 'OFF'}", flush=True)
    ThreadingHTTPServer((args.host, args.port), Handler).serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
