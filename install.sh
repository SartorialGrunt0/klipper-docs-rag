#!/usr/bin/env bash
#
# install.sh — set up klipper-docs-rag end to end:
#   1. check prerequisites
#   2. create a virtualenv and install the package
#   3. obtain the Klipper docs corpus (existing checkout or shallow clone)
#   4. build the search index against your embedding provider
#   5. install & start the OpenAI-compatible proxy as a systemd user service
#      (falls back to printing a run command where systemd is unavailable)
#
# The proxy exposes a virtual model "klipper-expert": point any
# OpenAI-compatible client at http://<host>:<port>/v1 and select it.
#
# Usage:
#   ./install.sh [--prefix DIR] [--docs-dir DIR] [--with-reranker] [--yes]
#
# Flags:
#   --prefix DIR       install root (default: ~/.klipper-rag)
#   --docs-dir DIR     optional path to a Klipper docs/ dir; when omitted,
#                      an existing checkout is auto-detected, otherwise
#                      Klipper is shallow-cloned under the install prefix
#   --with-reranker    also set up the optional bge-reranker service
#   --yes              accept defaults for every prompt (needs the required
#                      values to be sensible; prompts still print them)
#
set -euo pipefail

PREFIX="${HOME}/.klipper-rag"
WITH_RERANKER=0
ASSUME_YES=0

usage() { grep -E '^#( |$)' "$0" | sed 's/^# \{0,1\}//'; exit 0; }

while [[ $# -gt 0 ]]; do
  case "$1" in
    --prefix) PREFIX="$2"; shift 2 ;;
    --docs-dir) DOCS_DIR="$2"; export DOCS_DIR; shift 2 ;;
    --with-reranker) WITH_RERANKER=1; shift ;;
    --yes) ASSUME_YES=1; shift ;;
    -h|--help) usage ;;
    *) echo "unknown flag: $1 (see --help)" >&2; exit 2 ;;
  esac
done

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STATE_DIR="${PREFIX}"
STATE_DB="${STATE_DIR}/kb.sqlite"
VENV="${PREFIX}/venv"
ENV_FILE="${STATE_DIR}/env"

say()  { printf '\033[1;34m==>\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33mwarn:\033[0m %s\n' "$*" >&2; }
die()  { printf '\033[1;31merror:\033[0m %s\n' "$*" >&2; exit 1; }

# ---------------------------------------------------------------------------
# 1. Prerequisites
# ---------------------------------------------------------------------------
say "Checking prerequisites"

command -v python3 >/dev/null || die "python3 not found (needs Python >= 3.11)"
PY_OK=$(python3 -c 'import sys; print(1 if sys.version_info >= (3, 11) else 0)')
[[ "$PY_OK" == "1" ]] || die "python3 is too old ($(python3 -V 2>&1)); needs >= 3.11"
command -v git >/dev/null || die "git not found"
sqlite3_has_fts=$(python3 -c 'import sqlite3; c=sqlite3.connect(":memory:"); c.execute("CREATE VIRTUAL TABLE t USING fts5(x)"); print("yes")' 2>/dev/null || echo no)
[[ "$sqlite3_has_fts" == "yes" ]] || die "Python's sqlite3 lacks FTS5 support (rebuild Python against SQLite >= 3.9 with FTS5)"

HAS_SYSTEMD=0
# NB: 'is-system-running' exits non-zero on 'degraded' — it can't gate here.
if command -v systemctl >/dev/null && systemctl --user show-environment >/dev/null 2>&1; then
  HAS_SYSTEMD=1
fi

say "Python:    $(python3 -V 2>&1)"
say "Repo:      ${REPO_DIR}"
say "Prefix:    ${PREFIX}"
say "systemd user session: $([[ $HAS_SYSTEMD == 1 ]] && echo available || echo 'NOT available (proxy will get a manual run command)')"

# ---------------------------------------------------------------------------
# 2. Provider prompts
# ---------------------------------------------------------------------------
prompt() { # prompt NAME DEFAULT QUESTION
  # Precedence: pre-set env var of NAME > interactive input > default.
  local name="$1" def="$2" q="$3" val
  if [[ -n "${!name:-}" ]]; then
    val="${!name}"
  elif [[ $ASSUME_YES == 1 ]]; then
    val="$def"
  elif [[ -t 0 ]]; then
    read -r -p "  ${q}$([[ -n "$def" ]] && printf ' [%s]' "$def") " val || true
    val="${val:-$def}"
  else
    val="$def"
  fi
  [[ -n "$val" ]] || die "${name} is required"
  printf -v "$name" '%s' "$val"
}

normalize_base() { # strip trailing slashes and an trailing /v1 (we append what we need)
  local u="$1"
  u="${u%/}"; u="${u%/}"; u="${u%/v1}"
  printf '%s' "$u"
}

probe_embeddings() { # URL MODEL -> 0 if /v1/embeddings answers
  local url="$1" model="$2"
  python3 - "$url" "$model" <<'PY'
import json, sys, urllib.request
url, model = sys.argv[1].rstrip("/") + "/v1/embeddings", sys.argv[2]
req = urllib.request.Request(url, data=json.dumps(
    {"model": model, "input": ["search_query: probe"]}).encode(),
    headers={"Content-Type": "application/json"})
try:
    with urllib.request.urlopen(req, timeout=20) as r:
        d = json.load(r)
    v = d["data"][0]["embedding"]
    sys.exit(0 if isinstance(v, list) and v else 1)
except Exception as e:
    print(f"    probe failed: {e}", file=sys.stderr)
    sys.exit(1)
PY
}

probe_chat() { # URL MODEL -> 0 if /v1/models lists the model (warn-only check)
  local url="$1" model="$2"
  python3 - "$url" "$model" <<'PY'
import json, sys, urllib.request
url = sys.argv[1].rstrip("/") + "/v1/models"
try:
    with urllib.request.urlopen(url, timeout=20) as r:
        names = [m.get("id") for m in json.load(r).get("data", [])]
    sys.exit(0 if not names or sys.argv[2] in names else 1)
except Exception:
    sys.exit(2)  # /models unavailable: neither pass nor fail
PY
}

say "OpenAI-compatible provider settings"
say "  The service needs TWO capabilities from your provider:"
say "    - embeddings  (/v1/embeddings)  for indexing and queries"
say "    - chat        (/v1/chat/completions) for answering"
say "  llama.cpp serves each with its own 'llama-server' process; hosted"
say "  providers (OpenAI, etc.) serve both from one base URL."

prompt EMBED_URL "http://127.0.0.1:8100" \
  "Embedding provider base URL (serving /v1/embeddings):"
EMBED_URL="$(normalize_base "$EMBED_URL")"
prompt EMBED_MODEL "nomic-embed-text-v1.5" \
  "Embedding model name:"
say "  Probing ${EMBED_URL}/v1/embeddings with model '${EMBED_MODEL}'..."
if ! probe_embeddings "$EMBED_URL" "$EMBED_MODEL"; then
  die "embedding probe failed — start your embedding server (e.g. 'llama-server -m <embed.gguf> --embedding --pooling mean --embd-normalize 2 --port 8100') and re-run"
fi

prompt CHAT_URL "${EMBED_URL}" \
  "Chat provider base URL (serving /v1/chat/completions):"
CHAT_URL="$(normalize_base "$CHAT_URL")"
prompt BASE_MODEL "gemma-4-12b" \
  "Chat model name (the model the proxy answers with):"
probe="$(probe_chat "$CHAT_URL" "$BASE_MODEL"; echo $?)"
if [[ "$probe" == "1" ]]; then
  warn "'${BASE_MODEL}' was not found in ${CHAT_URL}/v1/models — continuing anyway; fix the name if requests come back as errors"
fi

prompt PORT "8090" "Port to serve the proxy on:"
[[ "$PORT" =~ ^[0-9]+$ ]] && (( PORT >= 1 && PORT <= 65535 )) || die "port must be 1-65535"

# ---------------------------------------------------------------------------
# 3. Corpus
# ---------------------------------------------------------------------------
say "Klipper docs corpus"
# Optional: if the user defines no path (prompt or --docs-dir / DOCS_DIR env),
# fall back to the default path: an existing checkout is auto-detected,
# otherwise Klipper is shallow-cloned under the install prefix.
DEFAULT_DOCS=""
for d in "$HOME/klipper/docs" "$HOME/klipper/klipper/docs"; do
  [[ -f "$d/Config_Reference.md" ]] && DEFAULT_DOCS="$d" && break
done
[[ -n "$DEFAULT_DOCS" ]] || DEFAULT_DOCS="${PREFIX}/klipper/docs"
prompt DOCS_DIR "$DEFAULT_DOCS" \
  "Optional path to a Klipper docs/ dir (Enter = default$([[ "$DEFAULT_DOCS" == "${PREFIX}/klipper/docs" ]] && printf ', clones Klipper there)' || printf ', uses it as-is)')"
if [[ "$DOCS_DIR" != "$DEFAULT_DOCS" ]]; then
  [[ -f "$DOCS_DIR/Config_Reference.md" ]] \
    || die "no Klipper docs at $DOCS_DIR (Config_Reference.md not found) — point at a Klipper checkout's docs/ dir, or press Enter at the prompt for the default"
elif [[ ! -f "$DOCS_DIR/Config_Reference.md" ]]; then
  say "No Klipper docs at ${DOCS_DIR} — shallow-cloning Klipper into ${DOCS_DIR%/docs} ..."
  mkdir -p "${DOCS_DIR%/docs}"
  git clone --depth 1 https://github.com/Klipper3d/klipper "${DOCS_DIR%/docs}"
  [[ -f "$DOCS_DIR/Config_Reference.md" ]] || die "clone completed but no Config_Reference.md at $DOCS_DIR"
fi

# ---------------------------------------------------------------------------
# 4. venv + build
# ---------------------------------------------------------------------------
say "Creating virtualenv at ${VENV}"
mkdir -p "$PREFIX"
python3 -m venv "$VENV"
say "Installing klipper-docs-rag"
"$VENV/bin/pip" install --quiet --upgrade pip
"$VENV/bin/pip" install --quiet -e "$REPO_DIR"

if [[ -f "$STATE_DB" ]]; then
  say "Existing index found at ${STATE_DB} — rebuilding over it"
fi
say "Building the index (chunking + embedding; ~1-2 min on a modern machine)"
"$VENV/bin/kb-rag" build "$DOCS_DIR" --state "$STATE_DB" \
  --embed-url "$EMBED_URL" --embed-model "$EMBED_MODEL"

# ---------------------------------------------------------------------------
# 5. Proxy service
# ---------------------------------------------------------------------------
cat > "$ENV_FILE" <<EOF
# klipper-docs-rag — written by install.sh on $(date -Iseconds)
KB_STATE=${STATE_DB}
KB_EMBED_URL=${EMBED_URL}
KB_EMBED_MODEL=${EMBED_MODEL}
KB_CHAT_URL=${CHAT_URL}/v1
KB_CHAT_MODEL=${BASE_MODEL}
KB_PROXY_PORT=${PORT}
EOF

RERANK_URL=""
if [[ $WITH_RERANKER == 1 ]]; then
  say "Optional reranker (--with-reranker)"
  say "The reranker is a bge-reranker-v2-m3 cross-encoder served by llama.cpp"
  say "with '--embedding --pooling rank --rerank'. Retrieval degrades"
  say "open-circuit: an unreachable reranker silently falls back to plain"
  say "hybrid order, so this step can be redone later."
  RERANK_BIN=""
  prompt RERANK_BIN "$HOME/apps/llama.cpp/build/bin/llama-server" \
    "Path to a llama-server binary built with --rerank support (empty = skip):"
  if [[ -n "$RERANK_BIN" && -x "$RERANK_BIN" ]]; then
    RERANK_GGUF="$HOME/models/bge-reranker-v2-m3-q8_0.gguf"
    if [[ ! -f "$RERANK_GGUF" ]]; then
      say "Downloading bge-reranker-v2-m3-q8_0.gguf (~330 MB)"
      mkdir -p "$(dirname "$RERANK_GGUF")"
      curl -fL -o "$RERANK_GGUF" \
        "https://huggingface.co/lj027/bge-reranker-v2-m3-Q8_0-GGUF/resolve/main/bge-reranker-v2-m3-q8_0.gguf"
    fi
    mkdir -p "$HOME/.config/systemd/user"
    sed -e "s|__LLAMA_SERVER__|$RERANK_BIN|" -e "s|__GGUF__|$RERANK_GGUF|" \
      "$REPO_DIR/systemd/klipper-rerank.service.template" \
      > "$HOME/.config/systemd/user/klipper-rerank.service"
    if [[ $HAS_SYSTEMD == 1 ]]; then
      systemctl --user daemon-reload
      systemctl --user enable --now klipper-rerank
      RERANK_URL="http://127.0.0.1:8101/v1/rerank"
      say "Reranker service started on 127.0.0.1:8101"
    fi
  else
    warn "llama-server not found — skipping reranker"
  fi
fi

mkdir -p "$HOME/.config/systemd/user"
EXECSTART="$VENV/bin/python -m kb_rag.serve --state $STATE_DB --chat-url ${CHAT_URL}/v1 --base-model $BASE_MODEL --port $PORT"
[[ -n "$RERANK_URL" ]] && EXECSTART="$EXECSTART --rerank-url $RERANK_URL"
sed "s|__EXECSTART__|$EXECSTART|" \
  "$REPO_DIR/systemd/klipper-rag-proxy.service.template" \
  > "$HOME/.config/systemd/user/klipper-rag-proxy.service"

if [[ $HAS_SYSTEMD == 1 ]]; then
  say "Installing systemd user service 'klipper-rag-proxy'"
  systemctl --user daemon-reload
  systemctl --user enable --now klipper-rag-proxy
  healthy=0
  for _ in 1 2 3 4 5 6 7 8 9 10; do
    sleep 1
    curl -fsS -m 3 "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1 && { healthy=1; break; }
  done
  if [[ $healthy == 1 ]]; then
    say "Proxy healthy on port ${PORT}"
  else
    warn "proxy not answering /health — check: journalctl --user -u klipper-rag-proxy -n 50"
  fi
else
  say "systemd unavailable — run the proxy manually with:"
  echo
  echo "  $VENV/bin/python -m kb_rag.serve \\"
  echo "      --state $STATE_DB --chat-url ${CHAT_URL}/v1 \\"
  echo "      --base-model $BASE_MODEL --port $PORT$([[ -n "$RERANK_URL" ]] && printf ' \\\n      --rerank-url %s' "$RERANK_URL")"
  echo
fi

say "Done."
echo
echo "  Proxy base URL : http://127.0.0.1:${PORT}/v1"
echo "  Virtual model  : klipper-expert"
echo "  Index          : $STATE_DB"
echo "  Settings file  : $ENV_FILE"
echo
echo "  Try it:"
echo "    curl -s http://127.0.0.1:${PORT}/health"
echo "    curl -s http://127.0.0.1:${PORT}/v1/chat/completions -H 'Content-Type: application/json' -d '{\"model\":\"klipper-expert\",\"messages\":[{\"role\":\"user\",\"content\":\"What parameters does [heater_fan] take?\"}]}'"
echo
echo "  Remove everything: $REPO_DIR/uninstall.sh"
