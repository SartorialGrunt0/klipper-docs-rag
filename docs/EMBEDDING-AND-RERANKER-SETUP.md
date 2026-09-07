# Setting up the embedding model and reranker

This document explains how to set up the two model services that
`klipper-docs-rag` depends on:

| Service | Required? | Model | Serves | Used for |
| --- | --- | --- | --- | --- |
| **Embedding server** | Yes | `nomic-embed-text-v1.5` | `POST /v1/embeddings` | Indexing every chunk + embedding each query (the dense retrieval leg and the topic gate) |
| **Reranker** | Optional | `bge-reranker-v2-m3` | `POST /v1/rerank` | Rescoring the fused top-10 candidates down to the final k |

Both are plain [llama.cpp](https://github.com/ggml-org/llama.cpp)
`llama-server` processes; `install.sh` wires them up for you. This guide
covers what those commands actually do, why each flag matters, how to run
the models on a different machine than the proxy, and how to verify and
troubleshoot them. If you only want the happy path, run `./install.sh`
and skim [Verification](#verification).

## Quick path (installer)

```bash
# 1. start the embedding server (required before install.sh — it live-probes it)
llama-server -m ~/models/nomic-embed-text-v1.5-Q8_0.gguf \
  --embedding --pooling mean --embd-normalize 2 \
  -b 2048 -ub 2048 \
  --host 127.0.0.1 --port 8100 \
  --sleep-idle-seconds 300   # unload from VRAM after 5 min idle

# 2. install; add --with-reranker to set up the reranker service too
cd klipper-docs-rag
./install.sh --with-reranker
```

The installer prompts for provider URLs and model names, probes
`/v1/embeddings` before writing anything, builds the index, and installs
systemd user services (`klipper-rag-proxy`, and `klipper-rerank` with
`--with-reranker`). For unattended runs the prompts accept these env
overrides: `EMBED_URL`, `EMBED_MODEL`, `CHAT_URL`, `BASE_MODEL`, `PORT`,
`DOCS_DIR` (plus `--yes`).

---

## The embedding model

### Why this model

The chunker and index are **calibrated for
[`nomic-embed-text-v1.5`](https://huggingface.co/nomic-ai/nomic-embed-text-v1.5)**:

- Nomic's retrieval contract prefixes documents with `search_document: `
  and queries with `search_query: `. `kb_rag/embed.py` applies both
  prefixes for you — you never add them yourself, and you must not use a
  provider that mangles them.
- It produces **768-dimensional** vectors; the index stores them
  side-by-side and scores with cosine (vectors are re-normalized
  client-side defensively, so dot product == cosine throughout).
- The topic gate's default threshold (0.40 on centroid-whitened cosine)
  was tuned against nomic's embedding geometry.

You *can* point the build at another embedding model
(`--embed-model <name>`), but everything — gate threshold included —
should then be re-evaluated (`eval/run_eval.py`), and changing the model
on an existing index requires a full rebuild: vectors from different
models don't mix. Keep `--embed-model` constant across rebuilds.

### Download the GGUF

Any GGUF export of the model works. With the Hugging Face CLI:

```bash
hf download nomic-ai/nomic-embed-text-v1.5-GGUF \
  nomic-embed-text-v1.5.Q8_0.gguf --local-dir ~/models
```

(or `curl -fL` from any mirror repo; F16 is ~275 MB, Q8_0 ~half that
with no measurable recall loss for this corpus.)

### Serve it

```bash
llama-server -m ~/models/nomic-embed-text-v1.5-Q8_0.gguf \
  --embedding --pooling mean --embd-normalize 2 \
  -b 2048 -ub 2048 \
  --host 127.0.0.1 --port 8100 \
  --sleep-idle-seconds 300
```

What each flag is doing:

- `--sleep-idle-seconds 300` — after 300 s with no request the model is
  unloaded from VRAM (llama-server stays up, `/props` reports
  `is_sleeping`); the next request reloads it automatically (~0.1–1 s for
  these small models, plus the ~9 s first-load if the process just
  restarted). This keeps the embedder at ~0 VRAM when unused instead of
  reserving ~460 MiB permanently. Optional; drop it for always-hot.

- `--embedding --pooling mean` — serves the model as an embedder with
  mean pooling over token embeddings. Required.
- `--embd-normalize 2` — L2-normalize server-side. The client also
  normalizes defensively, but keep this on: it matches how the index was
  built.
- `-b 2048 -ub 2048` — nomic's context is 2048 tokens, and llama.cpp
  rejects an `/v1/embeddings` request whose combined input exceeds
  `--ubatch-size` (the default 512 will 500 the build step on real
  chunks). The client batches conservatively, but these flags are what
  make batching legal.
- `--host 127.0.0.1` — binds locally. If the *proxy* runs on a different
  machine, bind to `0.0.0.0` (or an interface address) instead and open
  the port in your firewall — see [Running models on another
  machine](#running-models-on-another-machine).

RAM/footprint is trivial (~0.5 GB); the model runs fine even on a
Raspberry Pi, though embedding the ~1k-chunk corpus on a Pi 3B+ takes
minutes instead of ~1–2 minutes on a modern machine.

### Tell the index about it

The build step is where the embedding provider gets baked in:

```bash
kb-rag build <klipper-docs-dir> \
  --state ~/.klipper-rag/kb.sqlite \
  --embed-url http://127.0.0.1:8100 --embed-model nomic-embed-text-v1.5
```

`embed_url` and `embed_model` are recorded in the index metadata, and
`kb_rag.serve` reads them back — so once built, the proxy needs no
embedding flags unless you're overriding (`--embed-url`, `--embed-model`,
or `KB_EMBED_URL` / `KB_EMBED_MODEL` env). If you move the embedding
server to another host/port later, either re-run `install.sh` or restart
the proxy with `--embed-url <new-url>` (the model name must stay the
same, or rebuild).

---

## The reranker (optional)

### What it buys you

A cross-encoder reads the *(query, chunk)* pair jointly, instead of
scoring each side independently like the embedding model. Applied to the
hybrid fused top-10, on the 98-query gold set:

| Metric | Hybrid | Hybrid + rerank |
| --- | --- | --- |
| Recall@3 | 0.857 | **0.898** |
| MRR | 0.694 | **0.770** |

at ~75 ms p50 (p90 ~92 ms). It is *optional by design*: the retrieval
path **degrades open-circuit** — if the reranker is down, misconfigured,
or slow to answer, `kb_rag/rerank.py` silently falls back to the plain
hybrid order rather than failing the request. You can install, stop, or
break it at any time without touching the proxy.

### Download the GGUF

```bash
mkdir -p ~/models
curl -fL -o ~/models/bge-reranker-v2-m3-q8_0.gguf \
  https://huggingface.co/lj027/bge-reranker-v2-m3-Q8_0-GGUF/resolve/main/bge-reranker-v2-m3-q8_0.gguf
```

(~330 MB. `install.sh --with-reranker` downloads this itself if missing.)

### Serve it

```bash
llama-server -m ~/models/bge-reranker-v2-m3-q8_0.gguf \
  --embedding --pooling rank --rerank \
  --host 127.0.0.1 --port 8101 --ctx-size 4096 \
  --sleep-idle-seconds 300
```

- `--sleep-idle-seconds 300` — same idle-unload behavior as the embedder
  (~570 MiB → ~0 VRAM when unused; reload on the next rerank request).
  The rerank step is only reached on gated-in RAG queries, so this model
  is idle most of the time.
- `--rerank` needs a llama.cpp build recent enough to support it (the
  flag errors on older builds — update if so). `--pooling rank` selects
  the cross-encoder ranking head.
- `--ctx-size 4096` covers the 8k-token bge window as used here: the
  client truncates each candidate document to 1500 chars and scores up
  to 10 candidates, which fits comfortably.
- Endpoint shape is Cohere/Jina-compatible: `POST {"model", "query",
  "documents": [...], "top_n"}` → `{"results": [{"index",
  "relevance_score"}, ...]}`.

### Install as a service

`install.sh --with-reranker` prompts for the `llama-server` binary path,
fills in `systemd/klipper-rerank.service.template`, and enables the
`klipper-rerank` user unit on `127.0.0.1:8101`. To do it by hand:

```bash
sed -e "s|__LLAMA_SERVER__|/path/to/llama-server|" \
    -e "s|__GGUF__|$HOME/models/bge-reranker-v2-m3-q8_0.gguf|" \
  systemd/klipper-rerank.service.template \
  > ~/.config/systemd/user/klipper-rerank.service
systemctl --user daemon-reload
systemctl --user enable --now klipper-rerank
```

### Point the proxy at it

The reranker is opt-in per proxy instance:

```bash
kb_rag.serve --state ~/.klipper-rag/kb.sqlite \
  --chat-url <chat-provider>/v1 --base-model <chat-model> --port 8090 \
  --rerank-url http://127.0.0.1:8101/v1/rerank
```

`--rerank-url` unset disables reranking entirely. The proxy prints
`rerank=<url|OFF>` in its startup line — check
`journalctl --user -u klipper-rag-proxy` to confirm it is wired in. If you installed with
`--with-reranker`, install.sh already added the flag to the proxy unit;
adding it later means re-running `install.sh --with-reranker` (or
editing `~/.config/systemd/user/klipper-rag-proxy.service` and
`systemctl --user restart klipper-rag-proxy`).

To confirm the reranker is actually *improving* your corpus before
keeping it:

```bash
.venv/bin/python eval/run_rerank_eval.py
```

---

## Running models on another machine

The proxy only needs HTTP reach to both services — the models can live
anywhere on your network (or in the cloud). Two rules:

1. **Bind to a reachable interface** on the model host
   (`--host 0.0.0.0` or a specific interface address) and allow the port
   through its firewall.
2. **Firewalls that *drop* rather than reject** are the classic failure:
   requests hang until timeout instead of refusing. If the embedding
   probe in `install.sh` hangs at "Probing …", suspect a dropped port,
   then run the probe curl *on the model host itself* to bisect
   server-side vs network-side.

Example layout (models on a workstation, proxy on a Pi):

```bash
# workstation
llama-server -m nomic-embed-text-v1.5-Q8_0.gguf \
  --embedding --pooling mean --embd-normalize 2 -b 2048 -ub 2048 \
  --host 0.0.0.0 --port 8100 --sleep-idle-seconds 300
llama-server -m bge-reranker-v2-m3-q8_0.gguf \
  --embedding --pooling rank --rerank \
  --host 0.0.0.0 --port 8101 --sleep-idle-seconds 300

# Pi
EMBED_URL=http://192.168.x.x:8100 ./install.sh   # and reranker via --rerank-url
```

---

## Verification

Check each layer independently:

```bash
# 1. embedding server answers (raw probe — same call install.sh makes)
curl -s http://127.0.0.1:8100/v1/embeddings \
  -H 'Content-Type: application/json' \
  -d '{"model":"nomic-embed-text-v1.5","input":["search_query: probe"]}' \
  | head -c 200
# expect: {"data":[{"embedding":[-0.018..., ...]}...]}  (768 floats)

# 2. reranker answers
curl -s http://127.0.0.1:8101/v1/rerank \
  -H 'Content-Type: application/json' \
  -d '{"model":"bge-reranker-v2-m3","query":"what does [heater_fan] do?",
       "documents":["[heater_fan] section configures a fan ...",
                    "blather about unrelated topic"],"top_n":2}'
# expect: {"results":[{"index":0,"relevance_score":...},...]}

# 3. proxy sees the index and chat path
curl -s http://127.0.0.1:8090/health
# expect: {"ok":true,"chunks":<n>,"docs_version":{...},"stale":false}
# (rerank wiring shows in the proxy startup line, not /health)

# 4. end to end
curl -s http://127.0.0.1:8090/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"klipper-expert","max_tokens":800,"messages":[
        {"role":"user","content":"What parameters does [heater_fan] take?"}]}'
```

A quicker retrieval-only check (no chat model needed) is
`POST /retrieve {"query": "...", "k": 3}` against the proxy.

## Troubleshooting

| Symptom | Cause / fix |
| --- | --- |
| `install.sh` dies at "embedding probe failed" | Embedding server not up, wrong URL/model name, or a firewall silently dropping the port (probe hangs). curl it from the proxy host, then from the model host. |
| Build step 500s from `/v1/embeddings` | `--ubatch-size` too small on the embedding server — run with `-b 2048 -ub 2048`. |
| Retrieval quality collapses after switching providers | Provider is stripping/ignoring nomic's `search_query:` / `search_document:` prefixes, or you changed embedding models without rebuilding. Silent — verify via the raw probe above and rebuild if needed. |
| Answers unchanged after starting a reranker | Proxy started without `--rerank-url` (check the startup line in `journalctl --user -u klipper-rag-proxy`, `rerank=OFF`), or the proxy unit wasn't rewritten — re-run `install.sh --with-reranker`. |
| `llama-server: unknown option --rerank` | llama.cpp build too old; rebuild/update. |
| Reranker up but scores ignored | Only the *order* changes, never the candidates: a chunk that never enters the fused top-10 can't be reranked up. That's a retrieval/chunking issue, not a reranker issue. |
| Everything fine but answers stale | Not a model problem — see Version alignment in the README; re-run `install.sh`. |

## See also

- [`README.md`](../README.md) — overall design, install, version alignment
- [`docs/RAG-RESEARCH.md`](RAG-RESEARCH.md) — the normative design and
  eval research behind the retrieval stack
