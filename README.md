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
  0.857 / sparse-only 0.553; the identifier-gated sparse leg trades ~1 pt
  overall recall for conceptual-bucket gains and exact-identifier boosts.
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
