# RAG over Klipper Documentation — Research & Implementation Plan

**Date:** 2026-08-31
**Status:** Research complete; plan pending approval
**Scope:** Standalone service ("klipper-rag"), separate repo. KWC changes limited to one optional integration point.

---

## 1. Executive summary

- **What RAG buys KWC's AI chat:** today, doc retrieval only happens if the
  chat model *chooses* to call `search_klipper_docs`, and that tool is a
  keyword/TF inverted index. Two failure classes: (a) the small local models
  (qwen 4b/9b) sometimes answer from parametric memory instead of calling
  tools; (b) keyword search misses semantic/phrasing mismatches ("how do I
  stop overshoot at corners" → `[input_shaper]`). RAG fixes (b) outright and
  (a) partially — a semantic retriever that works *without model cooperation*
  is the actual prize.
- **Recommended architecture:** **Hybrid retrieval** (BM25/FTS5 + dense
  vectors fused with Reciprocal Rank Fusion), heading-aware + stanza-level
  chunking with breadcrumb metadata, optional cross-encoder rerank. Hybrid is
  not optional polish here: Klipper queries are *identifier-heavy*
  (`heater_fan`, `pressure_advance`, `SET_GCODE_VARIABLE`) — pure dense
  retrieval is *worse* than today's keyword index for exactly those, and pure
  keyword is what we have now. The union, fused via RRF, strictly dominates.
- **Serving:** everything stays on the llama.cpp host (192.168.1.135).
  `llama-server` natively serves embeddings (`--embedding`, OpenAI-compatible
  `/v1/embeddings`) and rerankers (`--rerank --pooling rank`,
  `/v1/rerank`, Cohere/Jina-shaped) — verified against the upstream server
  README on 2026-08-31. A second small `llama-server` process for the
  embedding model on another port costs ~0.3 GB RAM and a few hundred MB
  of VRAM.
- **Embedding model:** `nomic-embed-text-v1.5` GGUF (768-dim, 2048-token
  window, Q8 ≈ 270 MB) as the default; `bge-m3` (dim 1024, 8192 window, also
  emits sparse weights) as the upgrade candidate. Both run comfortably
  CPU-only at this corpus size.
- **Corpus facts (measured):** 56 official Klipper docs ≈ 934 KB, 7 KWC docs
  ≈ 106 KB, 280 example `.cfg` files. Whole corpus ≈ **~250–300K tokens** —
  far too big for the 4k context budget, trivially small for an index
  (~1.5–3K chunks; brute-force cosine over that is sub-millisecond; **no
  vector DB needed** — SQLite holds vectors as blobs).
- **Delivery to KWC:** a standalone HTTP service exposing
  `POST /retrieve {query} → top-k chunks` (+ `/search` for humans). KWC's
  already-built, currently-disabled `KWC_AUTO_SEARCH` injection slot
  (4000-char cap, edit-gating regex) becomes its first consumer; the MCP
  `search_klipper_docs` tool can also point at it. Both paths are A/B-testable
  against the existing `scripts/ai_chat_accuracy_test.py` harness.
- **Success gate before any integration:** Recall@3 on a labeled query set —
  hybrid RAG ≥ keyword baseline on identifier queries (must not regress) and
  clearly better on semantic queries. Eval budget ≈ 100 queries, deterministic
  `must_contain` grading first, LLM-judge only for reference comparisons.

---

## 2. Current state (ground truth, verified in repo 2026-08-31)

KWC (`~/Klipper-Wire-Configurator`) today:

| Component | Location | Notes |
| --- | --- | --- |
| Doc corpus | `backend/klipper_paths.py` → installed klipper `docs/` else bundled `reference/reference_docs/klipper_docs/` | 56 files, ~934 KB; `Config_Reference.md` = 213 KB; `G-Codes.md` = 79 KB |
| KWC-authored docs | `reference/kwc_docs/` | 7 files, ~106 KB |
| Example configs | `reference/config/` (bundled, categorized subdirs) | 280 `.cfg` files |
| Retriever | `backend/mcp_server.py` `DocIndex` | hand-rolled inverted index, word TF counts, per-section sub-index for `### [section]` stanzas |
| Tools | `search_klipper_docs`, `read_doc`, `get_config_reference_section`, `list_klipper_docs` | MCP tool-calling; model must choose to call |
| Injection slot | `backend/api/ai_routes.py` `AUTO_SEARCH_FALLBACK_MAX_CHARS = 4000` | disabled by default (`KWC_AUTO_SEARCH=0`); edit-request gating regex exists because doc injection during edits degrades drafts (verified 2026-08) |
| Eval harness | `scripts/ai_chat_accuracy_test.py` | drives real `/ai/chat`; per-question pass/fail; reports tool names + tool-turn counts |
| Context budget | `maxTokens: 4096` default | injected docs compete with config files + conversation |

Two important observations from this audit:

1. **The pre-injection plumbing already exists and was deliberately
   disabled** — because the keyword retriever behind it wasn't good enough to
   justify burning 4k chars of a 4k-token budget on possibly-irrelevant docs.
   RAG is the missing quality upgrade for that slot, not a new architecture
   KWC must adopt.
2. **The section-level scoring in `DocIndex` already encodes the single best
   domain insight** (chunk at `[section]` granularity, not whole-file). Any
   chunker we build should reuse that structure.

Prior decision context (2026-08-18): RAG was written off because (a) docs are
small and (b) the app must run on a Pi 3B+/1GB. Both reasons are now stale
for this standalone design: the *service* runs on the llama.cpp host (135),
and the Pi only ever makes one cheap HTTP call. The tool-calling approach
stayed because "models that use tools consistently" — which is exactly the
assumption this project relaxes.

---

## 3. RAG primer (the 20% that matters here)

RAG (Lewis et al., 2020, arXiv:2005.11401): retrieve relevant passages at
query time, stuff them into the prompt, instruct the model to answer from
them. The pipeline:

```
corpus ──► chunking ──► embedding ──► index            (offline, at build)
query  ──► (rewrite?) ─► embed ──► ANN/brute-force ──► top-k ──► (rerank?) ──► prompt injection   (online)
```

Design axes:

- **Sparse retrieval** (BM25 / SQLite FTS5): exact term matching. Perfect for
  `CANBus_Troubleshooting` or `rotation_distance`; blind to paraphrase.
- **Dense retrieval** (embedding vectors + cosine): handles paraphrase
  ("ringing on the print" → input shaper docs); weak on rare identifiers and
  exact names.
- **Hybrid + fusion:** run both, merge rankings. **RRF**
  (`score(d) = Σ 1/(k + rank_i(d))`, k=60) is the standard because it fuses
  *ranks*, sidestepping the incomparable score scales of BM25 (0..∞) vs
  cosine (0..1).
- **Cross-encoder reranking:** a second model scores each (query, chunk) pair
  jointly; much more accurate than bi-encoder cosine, ~100× the compute —
  fine for rescoring top-10 → top-3.
- **Delivery:** pre-injection (no model cooperation, costs context always),
  tool-calling (context-efficient, depends on model cooperation — current
  KWC), agentic/Self-RAG (retrieve-evaluate-repeat; overkill and loop-prone
  for 9b-class models on 4k context).

Corpus-size context: ~250–300K tokens of Klipper docs is *far* below what
"just stuff the whole corpus into a 128k context window" needs — it doesn't
fit 4k and would cost a prefill of every turn even at 128k — but it *is*
small enough that we can keep the entire chunk corpus in one SQLite file,
embed it in seconds, rebuild on every docs update, and skip every
vector-database dependency (no Qdrant/Chroma/pgvector).

---

## 4. What RAG looks like for Klipper docs

### 4.1 Chunking (the highest-leverage decision)

Structure-aware, three chunk families:

1. **`Config_Reference.md` → one chunk per `### [section]` stanza** (and
   `####`-parameter blocks where the stanza exceeds ~500 tokens). This is
   the doc KWC questions hit most ("what params does `[heater_fan]` take?").
   Metadata: `doc=Config_Reference`, `section=[heater_fan]`, breadcrumb
   `Klipper Docs > Config Reference > [heater_fan]`.
2. **Prose docs** (`Bed_Mesh.md`, `Resonance_Compensation.md`,
   `Multi_MCU_Homing.md`, troubleshooting docs…) → **heading-aware recursive
   splitting**: split at `##`/`###`; only sub-split blocks that exceed the
   token target (target **256–512 tokens**, overlap ~15% only when a block is
   force-split). The 512-token target matters because embedding quality
   degrades ("dilutes") when you stuff long mixed-topic blocks into one
   vector, even in 8k-window models.
3. **`G-Codes.md` → one chunk per G-code command block** (it's a flat list of
   `### G28`-style entries; naturally parameter-level).

**Every chunk gets a breadcrumb prefix** (`[Klipper Docs > Config Reference >
[stepper]] …`) prepended before embedding. This is the single cheapest known
quality win for structured-doc RAG (Anthropic's "Contextual Retrieval"
finding in the small-corpus limit): an isolated stanza like
`rotation_distance: See rotary delta...` is meaningless without its heading
context, and the prefix costs nothing but index time.

Example configs (280 `.cfg`): index them as a *separate source* with
breadcrumb `Example config > Printer/{vendor}/{file}`, but **exclude from the
default injection set** — include only when the query names a board/printer
model. They're retrieval fodder for "give me an EBBCan cfg" questions.

Estimated chunk count: **~1.5–3K chunks** total.

### 4.2 Retrieval

- **Sparse leg:** SQLite **FTS5** (BM25 built-in, `bm25()` ranking; zero
  dependencies — Python stdlib `sqlite3` ships it). Porter stemming ON;
  Klipper identifiers survive stemming fine (`heater_fan`, `mcu`).
- **Dense leg:** chunk vectors from the embedding service; brute-force
  cosine in numpy over ~3K × 768 floats (<1 ms, ~9 MB). **No ANN library and
  no vector DB.** This corpus size is the joke-free reason to keep the whole
  thing boring.
- **Fusion:** RRF, k=60, top-10 candidates per leg.
- **Rerank (phase 3, optional):** `bge-reranker-v2-m3` GGUF served by
  llama.cpp `--rerank --pooling rank` on `/v1/rerank`; rescoring top-10 →
  top-3. On a corpus this small, rerank value is *measurable but probably
  modest* — earn it via eval before adopting.
- **Threshold gate:** drop injection entirely when best fused score / rerank
  score is below threshold → chit-chat and pure-edit turns get zero doc
  noise (solves the "irrelevant docs pollute drafts" failure KWC already
  verified empirically; complements the existing edit-verb regex).

### 4.3 Delivery to KWC (both, gated — A/B decides)

- **Path A — pre-injection (the "without relying on tool calling" goal):**
  KWC re-enables its auto-search slot, but the retriever call goes to the RAG
  service instead of `DocIndex`. Inject top-2/3 chunks (≤ ~1200 tokens of the
  4k budget) only when the threshold gate passes *and* the edit regex says
  it's a question, not an edit. Zero model cooperation required — fixes
  failure class (a).
- **Path B — smarter tool:** repoint MCP `search_klipper_docs` at the RAG
  service (`POST /search` returns the same snippet shape). When the model
  *does* call tools, it gets strictly better recall. Cheap to do, keep both.

Path A is the differentiator; Path B is insurance. They share the service and
the eval.

### 4.4 What NOT to do

- **Vector DBs / LangChain / LlamaIndex:** 3K chunks + SQLite + ~300 lines of
  Python. Every framework dependency here is negative value on a
  Pi-adjacent, self-hosted, version-pinned stack.
- **GraphRAG / knowledge graphs:** construction cost and query latency for
  negligible recall gain over hybrid on 63 files.
- **Fine-tuning the chat model on Klipper docs:** bakes facts that change with
  every Klipper release (Config_Changes.md exists precisely because of this);
  retrieval stays fresh with a git pull + re-embed.
- **Full-corpus injection:** ~250K+ tokens; doesn't fit, and prefill cost per
  turn is absurd at 4k budgets.
- **Agentic Self-RAG:** extra tool rounds against a 4k budget on 9b models —
  more loop surface, less room for the answer.

---

## 5. Serving on the llama.cpp host (verified facts, 2026-08-31)

From the upstream `llama.cpp/tools/server/README.md` (master):

- `--embedding` restricts/serves embeddings; OpenAI-compatible
  **`/v1/embeddings`** plus legacy `/embedding`, `/embeddings`.
- `--pooling {none,mean,cls,last,rank}`; `--embd-normalize` (L2).
- **`--rerank` enables a reranking endpoint at `/v1/rerank`, `/v1/reranking`,
  `/rerank`** (Cohere/Jina-shaped), requires
  `--embedding --pooling rank` and a reranker model such as
  `bge-reranker-v2-m3`. *(Note: an earlier internal research note claimed
  llama.cpp has no rerank support — that was wrong; corrected here.)*
- Multiple models = multiple `llama-server` processes (one per port). The
  embedding model is tiny; co-residency with qwen3.5-9b is a non-issue on any
  box running the chat model (CPU-only serving is viable for this load: one
  ~200-token query embedding per chat turn).

### Model choice

| Model | Dim | Window | Size (Q8) | Notes |
| --- | --- | --- | --- | --- |
| **nomic-embed-text-v1.5** ← default | 768 | 2048 | ~270 MB | Needs prompt prefixes: `search_document: ` for chunks, `search_query: ` for queries. Mean pooling, L2-normalize. |
| bge-m3 (upgrade candidate) | 1024 | 8192 | ~600 MB | Also emits *sparse* lexical weights → could replace FTS5 with learned-sparse, one model for hybrid. Heavier; second iteration. |
| qwen3-embedding-0.6b/4b | 1024 | 32k | ~0.6/2.3 GB | MTEB leaders; check GGUF pooling support on the host's llama.cpp build version before committing. |
| all-MiniLM-L6-v2 | 384 | 512 | ~45 MB | Only as a smoke-test placeholder. |

Exact commands (on 135):

```bash
# embedding service, alongside the existing chat server on :8080
llama-server -m models/nomic-embed-text-v1.5-Q8_0.gguf \
  --embedding --pooling mean --embd-normalize 2 \
  --host 0.0.0.0 --port 8100

# later, if eval earns a reranker (separate process/port)
llama-server -m models/bge-reranker-v2-m3-Q8_0.gguf \
  --embedding --pooling rank --rerank \
  --host 0.0.0.0 --port 8101

# smoke tests
curl -s localhost:8100/v1/embeddings -H 'Content-Type: application/json' \
  -d '{"model":"nomic","input":"search_query: what does heater_fan do"}' | head -c 200
```

Gotchas checklist: nomic prefixes must be applied at *both* index and query
time (forgetting one silently tanks recall); pin one llama.cpp build —
embedding-pooling regressions have happened upstream; normalize vectors once
at index time and use dot product.

---

## 6. Implementation plan — `klipper-rag` (standalone repo)

### 6.1 Shape

```
klipper-rag/
├── README.md                  # runbook: install, systemd, rebuild, smoke tests
├── docs/RAG-RESEARCH.md       # this document
├── pyproject.toml             # deps: fastapi, uvicorn, httpx, numpy   (4 total)
├── kb_rag/
│   ├── chunkers/              # stanza.py (### [x]), headings.py (recursive), gcodes.py
│   ├── embed.py               # httpx client → llama-server /v1/embeddings (+ prefixes, batching, cache)
│   ├── index.py               # build: docs dir → chunks table + vectors blob in state/kb.sqlite
│   ├── retrieve.py            # FTS5 leg + dense leg + RRF + gate → top-k
│   ├── rerank.py              # optional /v1/rerank client (phase 3)
│   ├── server.py              # FastAPI: POST /retrieve, GET /search?q=, POST /reindex, GET /health, GET /stats
│   └── config.py              # env: KB_DOCS_DIR, KB_EMBED_URL, KB_RERANK_URL, port
├── eval/
│   ├── build_gold.py          # synthetic query gen from chunks (LLM-assisted) + qrels.jsonl
│   ├── run_eval.py            # Recall@3 / Precision@3 / MRR; keyword-baseline comparison
│   └── gold/queries.jsonl     # curated + generated, human spot-checked
├── systemd/klipper-rag.service (+ klipper-embed.service snippet)
└── scripts/rebuild.sh         # git -C klipper pull && kb-rag reindex (cron-able)
```

- **Ingestion source:** the klipper repo checkout on the host (`~/klipper`,
  git pull → reindex → fresh docs for free; same "installed docs" KWC already
  prefers, so both systems answer from the same revision).
- **API contracts:**
  - `POST /retrieve {query, k=3, max_chars=1200, sources=["klipper","kwc"]}`
    → `{chunks: [{text, doc, section, breadcrumb, score, source}], gate: bool, latency_ms}`
  - `POST /search` → snippet list shaped like KWC's current tool output (drop-in
    for Path B).
  - `POST /reindex` → rebuilds; cheap enough to run per docs update.
- **Freshness model:** docs change with Klipper releases → staleness is a
  *git pull*, not a retrain. `rebuild.sh` on cron (weekly) + manual.
  Reindex target: < 60 s wall for 3K chunks at CPU batch embedding.
- **Deployment:** on 135 next to the chat model; systemd units; state in
  `state/kb.sqlite` (single file, backup = copy). The service is LAN-only;
  no auth needed initially (same trust zone as the chat server), optionally
  `--api-key` (llama-server supports it) or a static bearer token on
  klipper-rag if it ever leaves the LAN.

### 6.2 Phases

| Phase | Deliverable | Exit criterion | Manual test |
| --- | --- | --- | --- |
| **0 — Skeleton + chunker** | repo, chunkers for all 3 families, chunk stats CLI (`kb-rag stats`) | ~2–3K chunks; every `### [section]` of Config_Reference exactly one chunk; no chunk > 700 tokens | `kb-rag dump --section heater_fan` eyeball; spot-check 10 random chunks vs source |
| **1 — Embed + index + dense retrieval** | embed service up on 135, index build, `/retrieve` (dense-only), gold set v1 (≥50 labeled queries) | dense Recall@3 ≥ 80% overall; **identifier-query subset measured vs keyword baseline (expect possible loss — that's why phase 2 exists)** | `curl :8090/retrieve` with 10 handwritten queries; inspect breadcrumb/scores |
| **2 — Hybrid (FTS5 + RRF)** | FTS5 leg, RRF fusion, threshold gate | hybrid ≥ keyword baseline on identifier bucket AND ≥ dense on semantic bucket; Recall@3 ≥ 90%; miss-rate ≤ 10% | A/B CLI: `eval/run_eval.py --mode keyword\|dense\|hybrid` side-by-side report |
| **3 — Rerank (optional)** | bge-reranker on :8101, top-10→top-3 | MRR improves ≥ 5 pts; latency Δ ≤ 500 ms | re-run eval with `--rerank` flag |
| **4 — KWC integration A/B** | Path A behind `KWC_AUTO_SEARCH=2`-style env switch (RAG URL); Path B repointed tool | accuracy harness: correctness improvement ≥ threshold on question-category queries with no regression on edit flows; latency Δ ≤ 1.5 s | run `scripts/ai_chat_accuracy_test.py` against: baseline / Path A / Path B; plus manual Trident-context questions in the real UI |
| **5 — Freshness + docs** | systemd units, rebuild cron, README runbook | fresh-clone → `pip install && ./scripts/rebuild.sh && systemctl start` works end-to-end | fresh-clone drill on this Pi |

Phase 4 is the *only* phase touching KWC, and it touches one env-gated code
path, not the tool surface. Everything before it is provably safe (standalone).

### 6.3 Dependency justification

- `fastapi` + `uvicorn`: team-standard HTTP layer (KWC backend precedent).
- `httpx`: call the embedding service.
- `numpy`: cosine over 3K vectors.
- SQLite FTS5, argparse, jsonl: **stdlib**. Nothing else. (No sentence-transformers,
  no torch, no vector DB, no LangChain.)

---

## 7. Evaluation plan

**Retrieval metrics** (offline, cheap, deterministic — the primary gate):
- **Recall@3** (does the gold chunk make top-3) — primary.
- **Precision@3** (noise proxy: junk in 4k budget = degraded answers).
- **MRR** (tuning signal for chunking/rerank decisions).
- NDCG: skip — order-of-3 isn't worth it.

**Gold set construction:** iterate chunks, LLM-generates 3–5 candidate
questions per section + which section answers it (`qrels.jsonl`), human
spot-checks ~10%. Target N≈100, four buckets sized ~25/25/25/25:
*parameter lookup / conceptual / troubleshooting / macro-Jinja*. Troubleshooting
queries should include error strings verbatim ("Timer too close", "MCU 'can0'
shutdown") — those are FTS-heavy, paraphrase is dense-heavy, so the bucket
split directly tests the hybrid thesis.

**Answer-level metrics** (phase 4, harness extension): deterministic
`must_contain` / `forbidden_claims` grading first (the KWC harness
philosophy — conditional pass, exact-fact checklist, no LLM-judge flakiness);
LLM-as-judge with reference answer + required-facts checklist only where
checklists can't express the judgment. Judge bias mitigations: never
self-grade with the same model family without a human calibration batch
(~20 items) first.

**A/B design (phase 4):** same 100 queries →
A = current tool-calling keyword (status quo), B = RAG pre-injection,
C = RAG-backed tool. Metrics: correctness pass-rate per bucket, tool-call
rate & retrieval-miss rate, wall-clock Δ, prompt-token cost Δ. Success:
B or C ≥ +15 pts correctness on question buckets, no regression on edit flows
(gating must keep edits doc-free), latency Δ ≤ 1.5 s. N=100 is enough for
per-bucket direction (±~10 pts noise band per 25-query bucket; treat
sub-bucket differences <15 pts as noise, overall differences ≥10 pts as
signal).

**Seeded golden queries** (corrected against actual docs — starts the gold
set; `Macro_Guide.md` does not exist in the corpus; macros live in
`Config_Reference.md` `[gcode_macro]` + `Multi_MCU_Homing.md` examples):

| Category | Query | Gold location | Key facts |
| --- | --- | --- | --- |
| Parameter | What does `step_timeout` do? | Config_Reference `[stepper]` | default 12.0 s; time allowed to reach next step |
| Parameter | What parameters does `[heater_fan]` take? | Config_Reference `[heater_fan]` | `heater`, `fan`, `temperature`, `heater_temp`, `fan_speed` |
| Parameter | What is `rotation_distance`? | Config_Reference `[extruder]`/`[stepper_x]` | default 100; steps→mm calibration |
| Conceptual | How does input shaping reduce ringing? | Resonance_Compensation.md | MZv/EI shapers, resonant frequencies, `shaper_freq_x/y` |
| Conceptual | How does Klipper's host/MCU split work? | Code_Overview.md / Intro.md | host plans moves, MCU step-generates in real time |
| Troubleshooting | What causes "Timer too close"? | Common_Errors.md | MCU can't keep up: serial/CAN latency, congested USB |
| Troubleshooting | Klipper says "MCU 'EBB' shutdown: Communication timeout" — where do I look? | CANBus_Troubleshooting.md | bit-stuffing, bkrates, termination |
| Macro/Jinja | How do I store state between macro calls? | Config_Reference `[gcode_macro]` | `variable_*` + `SAVE_VARIABLE` to persist |
| Macro/Jinja | How do I pass parameters to a macro? | Config_Reference `[gcode_macro]` | `params.X`, called as `MY_MACRO X=1` |
| Parameter | When do I use `[verify_heater]` `max_heater_temp`? | Config_Reference `[verify_heater]` | heating/cooling sanity checks |

(Plus the existing KWC `Klipper_GCode_Macro_AI_Summary.md` as a KWC-source
bucket once Path A lands.)

---

## 8. Risks & mitigations

| Risk | Impact | Mitigation |
| --- | --- | --- |
| Injection eats the 4k budget (config + convo already there) | degraded edit drafts, truncation | gate + ≤1200-token hard cap + edit regex (slot already implements) |
| Dense-only regresses identifier recall vs today's TF index | "RAG is worse" verdict at phase 1 | **do not ship dense-only**; hybrid is the shippable minimum (phase 2 is the MVP) |
| Small models ignore the injected-context instruction | hallucination persists | prompt-side "answer from CONTEXT first, cite doc/section"; A/B measures it |
| nomic prefix/pooling misconfig at index vs query time | silent recall collapse | smoke test in README with expected top-hit; pin llama.cpp build |
| Docs drift between KWC's copy and RAG's `~/klipper` | inconsistent answers between paths A/B | both point at the same installed klipper docs dir; `/stats` exposes corpus revision |
| qwen3.5-9b router (145) hangs mid-eval | flaky A/B numbers | probe host before eval runs (standing practice); 135 is primary for eval |
| Scope creep toward frameworks | unmaintainable side project | §4.4 "what NOT to do" is normative; deps list in §6.3 is closed |

## 9. Key references

- RAG original: Lewis et al. 2020, arXiv:2005.11401
- RRF: Cormack, Clarke & Buettcher 2009 (SIGIR), *Reciprocal Rank Fusion*
- llama.cpp server endpoints: `ggml-org/llama.cpp/tools/server/README.md`
  (`--embedding`, `--pooling rank`, `--rerank`, `/v1/embeddings`, `/v1/rerank`) — verified 2026-08-31
- SQLite FTS5 + `bm25()`: sqlite.org/fts5.html
- nomic-embed-text-v1.5 model card (prefixes, pooling): HuggingFace nomic-ai/nomic-embed-text-v1.5
- bge-m3 (dense+sparse multi-function): BAAI/bge-m3 model card, arXiv:2402.03216
- Contextual chunk prefixes: Anthropic *Introducing Contextual Retrieval* (2024)
- Cross-encoder reranking: bge-reranker-v2-m3 model card
- Self-RAG (why not here): Asai et al. 2023, arXiv:2310.11511
- Retrieval metrics methodology: BEIR (Thakur et al. 2021, arXiv:2104.08663)
- LLM-judge biases: Zheng et al. 2023 (MT-Bench/Chatbot Arena), *Judging LLM-as-a-Judge*
