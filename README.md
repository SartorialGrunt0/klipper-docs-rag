# klipper-docs-rag

Retrieval Augmented Generation (RAG) for the [Klipper 3D printer firmware
documentation](https://www.klipper3d.org/Config_Reference.html).

`klipper-docs-rag` is a standalone service that turns the official Klipper
docs into a searchable knowledge base and serves answers through an
**OpenAI-compatible proxy**. Point any OpenAI client — chat UI, SDK, or
custom app — at the proxy and select the virtual model `klipper-expert`:
every prompt is automatically grounded in the relevant documentation
sections, cited and injected, with no tool calling required.

- **Design & research:** [`docs/RAG-RESEARCH.md`](docs/RAG-RESEARCH.md)
- **Install:** [`install.sh`](install.sh) · **Remove:** [`uninstall.sh`](uninstall.sh)
- **Dependencies:** `httpx` + `numpy` + Python stdlib. No frameworks, no
  vector database, no telemetry.

## What it does

Ask *"What parameters does `[heater_fan]` take?"* and, instead of a model
guessing from training data, the service:

1. **retrieves** the top-k doc chunks for the question (hybrid search, below),
2. **gates** off-topic traffic (chit-chat gets no injected context, so it
   costs nothing and never derails),
3. **injects** the retrieved chunks as a cited `CONTEXT` block, and
4. **forwards** to your chat model, returning the answer plus a
   `rag.chunks` provenance field.

## Design

```
                 ┌────────────────────────────────────────────┐
 OpenAI client ──►  proxy (kb_rag.serve, :PORT)               │
 /v1/chat/       │   model=klipper-expert ──► retrieve+inject─┼──► your chat
 completions     │   other model names ─────► passthrough     │    provider
                 │   GET /v1/models · POST /retrieve · /health│
                 └───────────────┬────────────────────────────┘
                                 │
                 hybrid retrieval over kb.sqlite
                 ┌───────────────┴────────────────────────────┐
                 │ dense leg : embeddings (nomic-embed-text)  │──► your embed
                 │ sparse leg: SQLite FTS5 (BM25)             │    provider
                 │ fusion    : weighted RRF (k=60, 1.0/0.5)   │
                 │ rerank    : optional bge-reranker-v2-m3    │
                 └────────────────────────────────────────────┘
```

- **Corpus:** official Klipper docs only (a Klipper checkout's `docs/`).
  Docs are split by structure — config stanzas (`[heater_fan]` and friends),
  headings, G-code reference — into ≤500-token chunks with breadcrumb
  context, so a hit tells you exactly where it came from.
- **Hybrid retrieval:** a dense embedding leg (via your OpenAI-compatible
  `/v1/embeddings`) catches paraphrase; a SQLite FTS5 BM25 leg keeps exact
  identifiers (`heater_fan`, `tmc2209`, `cycle_time`) whole. Rankings are
  merged with weighted Reciprocal Rank Fusion. The index is one SQLite
  file — brute-force cosine over ~1k chunks is milliseconds on a Raspberry
  Pi, so there is no vector database.
- **Topic gate:** embedding-space similarity alone can't tell "hello" from
  a vague printer question, so the gate scores the *centroid-whitened*
  top-1 cosine (default threshold 0.40) with an identifier/domain-word
  bypass. Off-topic turns pass through to your chat model untouched.
- **Optional rerank:** a bge-reranker-v2-m3 cross-encoder rescoring the
  fused top-10 measurably improves answer retrieval (Recall@3 0.857 →
  0.898, MRR 0.694 → 0.770 on a 98-query gold set, at ~75 ms p50). It
  degrades open-circuit: an unreachable reranker silently falls back to
  plain hybrid order.
- **Eval harness:** `eval/run_eval.py` A/B-tests dense/sparse/hybrid (and
  `run_rerank_eval.py` the reranker) against a labeled gold set, so
  retrieval changes are measured, not guessed.

Measured on the gold set (n=98, k=3): dense Recall@3 **0.867**, hybrid
0.857, sparse-only 0.553, hybrid+rerank **0.898**. Retrieval latency is
30–150 ms per query plus one embedding round-trip.

## Prerequisites

- **Python ≥ 3.11** with a stdlib `sqlite3` that has FTS5 enabled (default
  on virtually all distros; verify: `python3 -c "import sqlite3;
  sqlite3.connect(':memory:').execute('CREATE VIRTUAL TABLE t USING fts5(x))"`).
- **git** (to fetch the Klipper docs corpus if you don't have a checkout).
- **An OpenAI-compatible provider with two capabilities:**
  - **embeddings** — a `/v1/embeddings` endpoint serving an embedding
    model. Recommended: [`nomic-embed-text-v1.5`](https://huggingface.co/nomic-ai/nomic-embed-text-v1.5)
    via [`llama-server --embedding --pooling mean --embd-normalize 2`](https://github.com/ggml-org/llama.cpp)
    (the chunker and index are calibrated for nomic's `search_document:` /
    `search_query:` prefixing, which this service applies for you).
  - **chat** — a `/v1/chat/completions` endpoint serving any instruct
    model large enough to follow doc-grounded instructions (≥7B-class
    recommended for local serving).
    Both can be the same llama.cpp host (each model needs its own
    `llama-server` process), the same hosted provider, or a mix.
  - If using llama.cpp locally, build it once
    ([build instructions](https://github.com/ggml-org/llama.cpp#build));
    a ~4k-token embedding context (`-ub 2048`) is sufficient.
- **Network:** the machine running the proxy must reach both provider
  endpoints; clients must reach the proxy port you choose.
- **Optional (reranker):** a `llama-server` build supporting `--rerank`
  and the [`bge-reranker-v2-m3` GGUF](https://huggingface.co/lj027/bge-reranker-v2-m3-Q8_0-GGUF)
  (~330 MB; `install.sh` downloads it).

## Setup

### Installer (recommended)

Start your embedding server (and chat server, if separate), then:

```bash
git clone https://github.com/SartorialGrunt0/klipper-docs-rag
cd klipper-docs-rag
./install.sh
```

The installer prompts for:

1. **Embedding provider base URL** (default `http://127.0.0.1:8100`) and
   **embedding model name** (default `nomic-embed-text-v1.5`) — live-probed
   against `/v1/embeddings` before anything is written;
2. **Chat provider base URL** and **chat model name** — the model the
   proxy answers with;
3. **Proxy port** (default `8090`);
4. **Optional path to a Klipper `docs/` dir** — press Enter to accept the
   default: an existing checkout at `~/klipper/docs` is auto-detected,
   otherwise Klipper is shallow-cloned into `~/.klipper-rag/klipper`.
   (Also settable non-interactively with `--docs-dir DIR`.)

It then creates a private virtualenv at `~/.klipper-rag/venv`, fetches the
Klipper docs (existing checkout, or a shallow clone), builds the index to
`~/.klipper-rag/kb.sqlite`, writes `~/.klipper-rag/env`, and installs a
systemd *user* service (`klipper-rag-proxy`) that starts the proxy now and
on login. Where systemd isn't available, it prints an equivalent manual run
command. Add `--with-reranker` to also set up the optional reranker service,
`--prefix DIR` to relocate, and `--yes` to accept all defaults.

### Manual

```bash
python3 -m venv .venv && .venv/bin/pip install -e .

# build the index from a Klipper checkout's docs/ dir
.venv/bin/kb-rag build ~/klipper/docs \
    --embed-url http://127.0.0.1:8100 --embed-model nomic-embed-text-v1.5 \
    --state ~/.klipper-rag/kb.sqlite

# run the proxy
.venv/bin/python -m kb_rag.serve \
    --state ~/.klipper-rag/kb.sqlite \
    --chat-url http://127.0.0.1:8080/v1 --base-model <your-chat-model> \
    --port 8090
# optional: --rerank-url http://127.0.0.1:8101/v1/rerank
#           --gate-threshold 0   (disable the topic gate)
```

### Use it

```bash
curl -s http://127.0.0.1:8090/health
curl -s http://127.0.0.1:8090/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"klipper-expert","max_tokens":800,"messages":[
        {"role":"user","content":"What parameters does [heater_fan] take?"}]}'
```

In any OpenAI-compatible client, add a provider with base URL
`http://<host>:<port>/v1` and model `klipper-expert`. Other model names
sent to the proxy pass through to your chat provider untouched. `POST
/retrieve {"query": "...", "k": 3}` returns raw chunks for your own
pipelines.

Note for reasoning-style models: they can spend the whole budget in a
reasoning channel — clients should request `max_tokens ≥ ~800`.

### Updating the corpus

Re-run the build over a refreshed checkout (`git pull` the Klipper repo,
then `kb-rag build ...` / re-run `install.sh`). Keep `--embed-model`
constant: vectors from different embedding models don't mix — changing it
rebuilds the whole index.

## Removal

```bash
./uninstall.sh
```

Stops and disables the `klipper-rag-proxy` (and `klipper-rerank`, if
installed) user services, removes their unit files, and — after
confirmation — deletes `~/.klipper-rag` (venv, index, cloned docs). It
touches nothing else: your embedding/chat servers and any Klipper checkout
you supplied predate the install.

## Development

```bash
python3 -m venv .venv && .venv/bin/pip install -e '.[dev]'
.venv/bin/pytest -q                 # unit tests (offline, transports mocked)
.venv/bin/kb-rag stats ~/klipper/docs
.venv/bin/kb-rag dump ~/klipper/docs --section heater_fan
.venv/bin/python eval/run_eval.py --state ~/.klipper-rag/kb.sqlite \
    --mode dense|sparse|hybrid      # retrieval A/B against the gold set
```

## License

GPL-3.0-or-later. The indexed documentation remains under its own terms
(github.com/Klipper3d/klipper).
