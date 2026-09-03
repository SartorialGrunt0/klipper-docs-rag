"""Version provenance + upstream staleness footnote.

The index records WHICH Klipper version its docs came from (git describe
of the source checkout). At serve time, a cheap cached check of upstream
Klipper master lets the proxy append a one-line footnote to RAG answers
when the corpus has drifted behind:

    Klipper docs note: the latest upstream docs are at <X>, while this RAG
    index was built at <Y>. Re-run install.sh to refresh the RAG.

Design constraints: never nags on failure (offline boxes, rate limits),
fetches are TTL-cached to disk, and the check never blocks a request more
than once per TTL.
"""
from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path

GITHUB_API = "https://api.github.com/repos/Klipper3d/klipper"
DEFAULT_TTL_SECONDS = 6 * 3600  # one upstream probe per 6h per cache file
USER_AGENT = "klipper-docs-rag"


# --------------------------------------------------------------------------
# provenance: which version is a docs checkout?
# --------------------------------------------------------------------------

def docs_version(docs_dir) -> dict | None:
    """git-describe the repo containing `docs_dir`. None if not a git repo.

    Returns {"describe", "sha", "tag", "commits_since_tag"} — e.g.
    {"describe": "v0.13.0-734-gfe4eb865", "sha": "<full>", "tag": "v0.13.0",
     "commits_since_tag": "734"}.

    A shallow clone carries no tags, which would make git-describe return a
    bare sha; fetch --depth 1 --tags first (best effort — offline or
    tagless-origin repos degrade to the pre-fetch describe).
    """
    if not docs_dir:
        return None
    p = Path(docs_dir)

    def git(*args: str) -> str | None:
        try:
            r = subprocess.run(("git", "-C", str(p), *args),
                               capture_output=True, text=True, timeout=60)
        except (OSError, subprocess.TimeoutExpired):
            return None
        return r.stdout.strip() if r.returncode == 0 else None

    sha = git("rev-parse", "HEAD")
    if not sha:
        return None
    describe = git("describe", "--tags", "--always") or sha[:7]
    if describe == sha[:7]:
        # describe went bare: shallow clone without tags. One cheap tag
        # fetch, then describe again. Offline/forked origins stay bare.
        git("fetch", "--quiet", "--depth", "1", "--tags", "origin")
        describe = git("describe", "--tags", "--always") or describe
    # "v0.13.0-734-gabc1234" -> tag "v0.13.0", N "734"; plain tags/shas stand.
    tag, n = describe, "0"
    parts = describe.split("-")
    if len(parts) >= 3 and parts[-1].startswith("g") and parts[-2].isdigit():
        tag, n = "-".join(parts[:-2]), parts[-2]
    return {"describe": describe, "sha": sha, "tag": tag,
            "commits_since_tag": n}


# --------------------------------------------------------------------------
# staleness decision
# --------------------------------------------------------------------------

def is_stale(build_version: dict, upstream_sha: str | None) -> bool:
    """True when the built docs sha differs from upstream master's sha.

    Quiet by default: unknown on either side means NOT stale — we only
    claim drift we can actually demonstrate.
    """
    if not upstream_sha or not build_version:
        return False
    local = str(build_version.get("sha") or "")
    if not local:
        return False
    n = min(len(local), len(upstream_sha))
    return local[:n] != upstream_sha[:n]


def footnote_text(upstream: dict, build_version: dict) -> str:
    """Markdown footnote appended to the answer.

    Upstream label: the latest release tag, EXCEPT when the build sits on
    the same tag (e.g. a master checkout N commits past the tag) — there
    the tag says nothing and the short sha is the honest label. Falls back
    to the short sha when upstream has no tag.
    """
    up_sha = str(upstream.get("sha") or "")[:7]
    up_tag = upstream.get("tag") or ""
    build_tag = build_version.get("tag") or ""
    label = up_tag if (up_tag and up_tag != build_tag) else (up_sha or up_tag)
    build_label = build_version.get("describe") or build_tag or "unknown"
    return ("\n\n---\nKlipper docs note: the latest upstream docs are at "
            f"**{label}**, while this RAG index was built at "
            f"**{build_label}**. Re-run `install.sh` to refresh the RAG.")


# --------------------------------------------------------------------------
# serve-time checker
# --------------------------------------------------------------------------

def fetch_upstream_github() -> tuple[str | None, str | None]:
    """(master_sha, latest_tag) from the GitHub API. Raises on any failure."""
    import httpx

    r = httpx.get(f"{GITHUB_API}/git/ref/heads/master", timeout=10.0,
                  headers={"User-Agent": USER_AGENT})
    r.raise_for_status()
    sha = r.json()["object"]["sha"]
    tag = None
    try:  # tag is cosmetic in the footnote; sha is the authority
        rt = httpx.get(f"{GITHUB_API}/tags?per_page=1", timeout=10.0,
                       headers={"User-Agent": USER_AGENT})
        rt.raise_for_status()
        data = rt.json()
        tag = data[0]["name"] if data else None
    except Exception:  # noqa: BLE001
        pass
    return sha, tag


class UpstreamChecker:
    """Cached, fail-quiet staleness check.

    footnote() -> str: the appended-markdown footnote when demonstrably
    stale, else "". Upstream probes are cached in `cache_path` (JSON) and
    rate-limited to one per ttl_seconds; failures are cached too so an
    offline box never pays per-request network costs.
    """

    def __init__(self, build_version: dict | None,
                 fetch_upstream=fetch_upstream_github,
                 cache_path: Path | None = None,
                 ttl_seconds: int = DEFAULT_TTL_SECONDS) -> None:
        self.build_version = build_version or {}
        self._fetch = fetch_upstream
        self.cache_path = Path(cache_path) if cache_path else None
        self.ttl = ttl_seconds
        self._last_attempt: float = 0.0

    def _load_cache(self) -> dict | None:
        if not self.cache_path or not self.cache_path.is_file():
            return None
        try:
            return json.loads(self.cache_path.read_text())
        except (json.JSONDecodeError, OSError):
            return None

    def _save_cache(self, cache: dict) -> None:
        if not self.cache_path:
            return
        try:
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            self.cache_path.write_text(json.dumps(cache))
        except OSError:
            pass  # read-only home: degrade to memory-only caching

    def _upstream(self) -> dict | None:
        now = time.time()
        cache = self._load_cache()
        if cache and self.cache_path is not None \
                and (now - cache.get("checked_at", 0)) < self.ttl:
            return cache
        if (now - self._last_attempt) < min(self.ttl, 60) and cache:
            return cache  # recent failed attempt: don't hammer
        self._last_attempt = now
        try:
            sha, tag = self._fetch()
        except Exception:  # noqa: BLE001 — offline/rate-limited: stay quiet
            self._save_cache({**(cache or {}), "checked_at": now})
            return cache
        new = {"checked_at": now, "upstream_sha": sha, "upstream_tag": tag}
        self._save_cache(new)
        return new

    def footnote(self) -> str:
        if not self.build_version.get("sha"):
            return ""  # pre-versioning index: nothing to compare, stay quiet
        up = self._upstream()
        if not up or not up.get("upstream_sha"):
            return ""
        if not is_stale(self.build_version, up["upstream_sha"]):
            return ""
        return footnote_text({"tag": up.get("upstream_tag"),
                              "sha": up["upstream_sha"]},
                             self.build_version)
