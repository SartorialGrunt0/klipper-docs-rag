"""Tests for version provenance + upstream staleness footnote (offline)."""
from __future__ import annotations

import json
import subprocess

import pytest

from kb_rag import version
from kb_rag.version import (
    UpstreamChecker, docs_version, footnote_text, is_stale,
)


# ---------- docs_version (git describe provenance) ----------

@pytest.fixture
def git_repo(tmp_path):
    """A tiny repo with one tagged commit and one commit past the tag."""
    r = tmp_path / "klipper"
    (r / "docs").mkdir(parents=True)
    env = ["git", "-C", str(r)]

    def git(*a):
        subprocess.run([*env, *a], check=True, capture_output=True,
                       env={"GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t",
                            "GIT_COMMITTER_NAME": "t",
                            "GIT_COMMITTER_EMAIL": "t", "HOME": str(tmp_path)})

    (r / "docs" / "a.md").write_text("one\n")
    git("init", "-q", "-b", "master")
    git("add", ".")
    git("commit", "-qm", "first")
    git("tag", "v0.13.0")
    (r / "docs" / "b.md").write_text("two\n")
    git("add", ".")
    git("commit", "-qm", "second")
    return r


def test_docs_version_describes_tag_distance(git_repo):
    v = docs_version(git_repo / "docs")
    assert v["describe"].startswith("v0.13.0-1-g")
    assert len(v["sha"]) == 40
    assert v["tag"] == "v0.13.0"
    assert v["commits_since_tag"] == "1"


def test_docs_version_exact_tag(git_repo):
    subprocess.run(["git", "-C", str(git_repo), "reset", "--hard", "-q",
                    "v0.13.0"], check=True)
    v = docs_version(git_repo / "docs")
    assert v["describe"] == "v0.13.0"
    assert v["tag"] == "v0.13.0"


def test_docs_version_not_a_repo(tmp_path):
    d = tmp_path / "plain"
    d.mkdir()
    assert docs_version(d) is None
    assert docs_version(None) is None


# ---------- staleness decision ----------

def test_is_stale_same_sha():
    sha = "f0892d82b0f1c1228454f09eb508eddde2250f4b"
    assert is_stale({"sha": sha}, sha) is False
    assert is_stale({"sha": sha[:7]}, sha) is False  # abbreviated both ways


def test_is_stale_different_sha():
    assert is_stale({"sha": "aaaaaaa" + "0" * 33}, "b" * 40) is True


def test_is_stale_no_local_version():
    assert is_stale({}, "b" * 40) is False  # nothing to compare: stay quiet


def test_is_stale_no_upstream():
    assert is_stale({"sha": "a" * 40}, None) is False


# ---------- footnote text ----------

def test_footnote_text():
    t = footnote_text({"tag": "v0.14.0", "sha": "f0892d82b0f1c1228454f09eb508eddde2250f4b"},
                      {"tag": "v0.13.0", "describe": "v0.13.0-734-gfe4eb865"})
    assert "v0.14.0" in t and "v0.13.0-734-gfe4eb865" in t
    assert t.startswith("\n\n---\n")
    assert "install.sh" in t


def test_footnote_text_same_tag_uses_sha():
    # build is 734 commits past v0.13.0; upstream tag is still v0.13.0 —
    # labeling upstream "v0.13.0" would read as "not behind". Use the sha.
    t = footnote_text({"tag": "v0.13.0", "sha": "f0892d82b0f1c1228454f09eb508eddde2250f4b"},
                      {"tag": "v0.13.0", "describe": "v0.13.0-734-gfe4eb865"})
    assert "f0892d8" in t
    assert "**v0.13.0**, while" not in t


def test_footnote_text_untagged_upstream():
    t = footnote_text({"tag": "", "sha": "f0892d82b0f1c1228454f09eb508eddde2250f4b"},
                      {"tag": "v0.13.0", "describe": "v0.13.0"})
    assert "f0892d8" in t


# ---------- UpstreamChecker (fetch monkeypatched) ----------

class FakeUpstream:
    def __init__(self, sha="b" * 40, tag="v0.14.0", fail=False):
        self.sha, self.tag, self.fail = sha, tag, fail
        self.calls = 0

    def __call__(self):
        self.calls += 1
        if self.fail:
            raise RuntimeError("network down")
        return self.sha, self.tag


def test_checker_stale_footnote(tmp_path):
    fake = FakeUpstream()
    c = UpstreamChecker(build_version={"sha": "a" * 40, "describe": "v0.13.0-1-gx"},
                        fetch_upstream=fake, cache_path=tmp_path / "up.json")
    assert c.footnote().count("v0.14.0") == 1
    assert fake.calls == 1
    # cached: second call does not refetch
    c.footnote()
    assert fake.calls == 1


def test_checker_fresh_no_footnote(tmp_path):
    fake = FakeUpstream(sha="a" * 40)
    c = UpstreamChecker(build_version={"sha": "a" * 40},
                        fetch_upstream=fake, cache_path=tmp_path / "up.json")
    assert c.footnote() == ""


def test_checker_network_failure_quiet(tmp_path):
    fake = FakeUpstream(fail=True)
    c = UpstreamChecker(build_version={"sha": "a" * 40},
                        fetch_upstream=fake, cache_path=tmp_path / "up.json")
    assert c.footnote() == ""  # never nags on a failed check
    # failure is not cached as fresh forever, but also not refetched instantly
    c.footnote()
    assert fake.calls <= 2


def test_checker_no_build_version_quiet(tmp_path):
    fake = FakeUpstream()
    c = UpstreamChecker(build_version=None,
                        fetch_upstream=fake, cache_path=tmp_path / "up.json")
    assert c.footnote() == ""
    assert fake.calls == 0  # don't even probe


def test_checker_cache_persists(tmp_path):
    cache = tmp_path / "up.json"
    UpstreamChecker(build_version={"sha": "a" * 40},
                    fetch_upstream=FakeUpstream(), cache_path=cache).footnote()
    data = json.loads(cache.read_text())
    assert data["upstream_sha"].startswith("b")
    # a fresh checker instance reuses the on-disk cache
    c2 = UpstreamChecker(build_version={"sha": "a" * 40},
                         fetch_upstream=FakeUpstream(fail=True),
                         cache_path=cache)
    assert "v0.14.0" in c2.footnote()
