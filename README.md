# klipper-docs-rag

Retrieval Augmented Generation (RAG) for the Klipper docs.

Standalone hybrid-retrieval service (SQLite FTS5 BM25 + dense embeddings,
RRF fusion) over the Klipper firmware documentation, designed to serve
[Klipper-Wire-Configurator](https://github.com/SartorialGrunt0/Klipper-Wire-Configurator)'s
AI chat. Embeddings are served by a co-located
[`llama-server --embedding`](https://github.com/ggml-org/llama.cpp) process;
no vector database, no heavyweight frameworks.

- **Design & research:** [`docs/RAG-RESEARCH.md`](docs/RAG-RESEARCH.md)
- **Status:** Phases 0–2 + serve proxy. **Corpus is official Klipper docs
  only** — KWC-authored docs are excluded by building without `--kwc-dir`
  (they outranked primary docs on macro/Jinja queries; removing them
  raised dense Recall@3 0.835 → 0.867). Eval n=98: dense 0.867 / hybrid
  0.857 / sparse-only 0.553 / **hybrid+rerank 0.898 (MRR 0.770, p50
  74ms)** — the reranker is eval-earned and enabled via `--rerank-url`.
  Topic gate: centroid-whitened cosine + intent bypass. OpenAI-compatible
  `klipper-expert` proxy.

## Quick start (dev)

```bash
python3 -m venv .venv
.venv/bin/pip install -e '.[dev]'
.venv/bin/pytest -q
.venv/bin/kb-rag stats ~/klipper/docs
.venv/bin/kb-rag dump ~/klipper/docs --section heater_fan
```

`kb-rag` expects a Klipper checkout's `docs/` directory (or the KWC bundled
copy) as its corpus root.

## Reranker service (optional, eval-earned)

`bge-reranker-v2-m3` served by llama.cpp for cross-encoder rescoring of
the top-10 fused candidates. On the CachyPC:

```bash
# one-time: model download
curl -sL -o ~/models/bge-reranker-v2-m3-q8_0.gguf \
  https://huggingface.co/lj027/bge-reranker-v2-m3-Q8_0-GGUF/resolve/main/bge-reranker-v2-m3-q8_0.gguf

# run (currently started via nohup; see systemd note below)
~/apps/llama.cpp/build/bin/llama-server \
  -m ~/models/bge-reranker-v2-m3-q8_0.gguf \
  --embedding --pooling rank --rerank \
  --host 127.0.0.1 --port 8101 --ctx-size 4096

# smoke test — first doc should score clearly highest
curl -s http://127.0.0.1:8101/v1/rerank -H 'Content-Type: application/json' \
  -d '{"model":"bge","query":"what parameters does heater_fan take",
       "documents":["[heater_fan] pin heater_temp","unrelated pasta text"]}'
```

Bind is `127.0.0.1` on purpose: the host firewall drops LAN access to
non-published ports (which is why remote probes time out rather than get
refused). Cross-host use needs an SSH tunnel (`ssh -L 8101:127.0.0.1:8101`)
or a firewall port opening — prefer the tunnel.

**systemd (TODO for the box's admin / default Hermes profile):** the
server above does not survive reboot. Install
`systemd/klipper-rerank.service` and `systemctl --user enable --now`
it (or system-level), then add `--rerank-url http://127.0.0.1:8101/v1/rerank`
to the klipper-rag proxy's ExecStart. Retrieval degrades open-circuit —
a dead reranker silently falls back to plain hybrid order.
