# Setup brief: embedding + rerank servers on CachyPC

**For:** Clifford's default Hermes profile (or Clifford directly).
**From:** coder profile, 2026-08-31. Part of the `klipper-docs-rag` project
(repo: https://github.com/SartorialGrunt0/klipper-docs-rag — see
`docs/RAG-RESEARCH.md` for why these servers exist).
**Goal:** two additional llama-server processes on CachyPC: an embedding
server on **:8100** and (optional, phase 3) a rerank server on **:8101**,
alongside the existing chat router on :8080. Zero interference with existing
services.

## Verified host facts (probed via SSH from the Pi, do not re-guess)

- Host `Cachy-Server`, user `clifgall`, **fish is the login shell** — wrap
  multi-command lines in `bash -c`.
- GPU: **RTX 4070 Ti SUPER, 16 GB, ~2 GB used** (chat router) → ample VRAM.
- llama.cpp build: `/home/clifgall/apps/llama.cpp/build/bin/llama-server`
  (commit a4a4c51, 2026-08-12). **`--embedding`, `--pooling`, `--rerank`
  all confirmed present** in this build. Do NOT update llama.cpp for this —
  the build works and the chat router depends on it.
- Existing: `llama-server --models-preset ~/.config/llama-router/my-models.ini
  --host 0.0.0.0 --port 8080 --models-max 1` — leave it alone; it's the KWC
  chat backend.
- Disk: 370 GB free on /home. Ports 8100/8101: free.

## What to do

### 1. Download models (Q8_0 GGUFs, into ~/models/)

```bash
mkdir -p ~/models
# primary embedder (nomic-embed-text-v1.5, 768-dim, 2048 ctx)
curl -L -o ~/models/nomic-embed-text-v1.5-Q8_0.gguf \
  https://huggingface.co/nomic-ai/nomic-embed-text-v1.5-GGUF/resolve/main/nomic-embed-text-v1.5.Q8_0.gguf
# optional later (phase-3 reranker — download now, don't run yet):
# curl -L -o ~/models/bge-reranker-v2-m3-Q8_0.gguf \
#   https://huggingface.co/gpustack/bge-reranker-v2-m3-GGUF/resolve/main/bge-reranker-v2-m3-Q8_0.gguf
# optional A/B arm (bge-m3 dense+sparse):
# curl -L -o ~/models/bge-m3-Q8_0.gguf \
#   https://huggingface.co/unsloth/bge-m3-GGUF/resolve/main/bge-m3-Q8_0.gguf
```

(If a URL 404s on a repo reorg, search HF for the model + GGUF and grab the
Q8_0 file; sizes ≈ 270 MB nomic, 600 MB bge-m3, 1.1 GB reranker. nomic
filename may carry a `-v1.5.Q8_0` or `F16`-style suffix — take Q8_0.)

### 2. Smoke test

```bash
bash -c '/home/clifgall/apps/llama.cpp/build/bin/llama-server \
  -m ~/models/nomic-embed-text-v1.5-Q8_0.gguf \
  --embedding --pooling mean --embd-normalize 2 \
  --host 0.0.0.0 --port 8100 --flash-attn on &'
sleep 3
curl -s http://localhost:8100/v1/embeddings -H 'Content-Type: application/json' \
  -d '{"model":"nomic","input":"search_query: what does heater_fan do"}' \
  | head -c 300
```

Expected: JSON `{"object":"list","data":[{"embedding":[...768 floats...]}`.
Also check `curl -s localhost:8100/props | head -c 300` reports pooling mean
and a context ≥ 2048.

### 3. Make it persistent (systemd user unit)

`~/.config/systemd/user/llama-embed.service`:

```ini
[Unit]
Description=llama.cpp embedding server (nomic-embed-text-v1.5) for klipper-rag
After=network.target

[Service]
ExecStart=/home/clifgall/apps/llama.cpp/build/bin/llama-server \
  -m /home/clifgall/models/nomic-embed-text-v1.5-Q8_0.gguf \
  --embedding --pooling mean --embd-normalize 2 \
  --host 0.0.0.0 --port 8100 --flash-attn on
Restart=on-failure
RestartSec=5

[Install]
WantedBy=default.target
```

```bash
systemctl --user daemon-reload
systemctl --user enable --now llama-embed
loginctl enable-linger $USER   # survives logout/reboot without a session
```

### 4. Definition of done (verify all, then report back)

- [ ] `curl -s localhost:8100/health` → `{"status":"ok"}`
- [ ] `/v1/embeddings` call returns **768**-float vectors for both a
      `search_document: ...` and a `search_query: ...` input (prefixes are
      the caller's job — nomic needs them; server just stores text)
- [ ] Two different inputs produce different vectors; same input twice
      produces identical vectors (determinism)
- [ ] Chat router on :8080 still healthy (`curl -s localhost:8080/health`),
      no llama-server restarts in its logs
- [ ] `nvidia-smi` shows the new process using < 1 GB VRAM
- [ ] Unit survives `systemctl --user restart llama-embed` + host reboot
- [ ] Report back: exact model file + sha256, llama-server version string,
      output of the smoke test

### 5. NOT in scope

- Don't touch the :8080 router, its preset file, or its model files.
- Don't update/rebuild llama.cpp.
- Don't start the reranker yet (phase 3; eval decides).
- No auth/firewall changes — LAN-trust model, same as :8080. If anything
  about that changes, note it in the report rather than improvising.

**Gotcha for whoever wires clients later:** nomic requires matching
`search_document: `/`search_query: ` prefixes at index vs query time —
forgetting one silently tanks retrieval, it does not error.
