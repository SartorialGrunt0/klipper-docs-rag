#!/usr/bin/env bash
#
# uninstall.sh — remove everything install.sh created:
#   - systemd user services (klipper-rag-proxy, klipper-rerank)
#   - the install prefix (venv, index, cloned docs) — confirm before deleting
#
# Usage: ./uninstall.sh [--yes] [--purge-logs]
#
set -euo pipefail

PREFIX="${HOME}/.klipper-rag"
ASSUME_YES=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --prefix) PREFIX="$2"; PREFIX_EXPLICIT=1; shift 2 ;;
    --yes) ASSUME_YES=1; shift ;;
    --purge-logs) PURGE_LOGS=1; shift ;;
    -h|--help) grep -E '^#( |$)' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "unknown flag: $1 (see --help)" >&2; exit 2 ;;
  esac
done

say() { printf '\033[1;34m==>\033[0m %s\n' "$*"; }

# If the proxy unit was installed to a custom prefix, derive it so a plain
# './uninstall.sh' still cleans up, unless --prefix was given explicitly.
UNIT_FILE="$HOME/.config/systemd/user/klipper-rag-proxy.service"
if [[ "${PREFIX_EXPLICIT:-0}" == "0" && -f "$UNIT_FILE" ]]; then
  detected=$(sed -n 's|.*--state \(.*\)/kb\.sqlite.*|\1|p' "$UNIT_FILE" | head -1)
  [[ -n "$detected" && -d "$detected" ]] && PREFIX="$detected"
fi

confirm() {
  [[ $ASSUME_YES == 1 ]] && return 0
  read -r -p "$1 [y/N] " a || return 1
  [[ "$a" == "y" || "$a" == "Y" ]]
}

# 1. services
for unit in klipper-rag-proxy klipper-rerank; do
  if command -v systemctl >/dev/null && systemctl --user list-unit-files "${unit}.service" >/dev/null 2>&1 \
     && [[ -f "$HOME/.config/systemd/user/${unit}.service" ]]; then
    say "Stopping and disabling ${unit}.service"
    systemctl --user disable --now "$unit" 2>/dev/null || true
    rm -f "$HOME/.config/systemd/user/${unit}.service"
  else
    say "${unit}.service not installed — skipping"
  fi
done
if command -v systemctl >/dev/null && systemctl --user show-environment >/dev/null 2>&1; then
  systemctl --user daemon-reload 2>/dev/null || true
fi

# 2. install prefix
if [[ -d "$PREFIX" ]]; then
  say "The install prefix contains the venv, the search index (kb.sqlite),"
  say "the env file, and any Klipper checkout cloned during install."
  if confirm "Delete ${PREFIX} ?"; then
    rm -rf "$PREFIX"
    say "Removed ${PREFIX}"
  else
    say "Kept ${PREFIX} (remove it later with: rm -rf ${PREFIX})"
  fi
else
  say "${PREFIX} not present — skipping"
fi

if [[ "${PURGE_LOGS:-0}" == "1" ]] && command -v journalctl >/dev/null; then
  say "Purging journal entries for klipper-rag units"
  journalctl --user --rotate >/dev/null 2>&1 || true
  journalctl --user --vacuum-time=1s >/dev/null 2>&1 || true
fi

say "Done. Nothing else to remove: the package was installed into the"
say "prefix's own virtualenv, and your embedding/chat servers predate it."
