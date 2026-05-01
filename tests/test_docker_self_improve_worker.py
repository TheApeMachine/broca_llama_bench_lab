"""Unit tests for Docker self-improve worker (mocked Docker / no network)."""

from __future__ import annotations

import subprocess
from pathlib import Path
from unittest.mock import MagicMock

import pytest

import core.broca as broca_mod
from conftest import make_stub_llm_pair
from core.broca import BrocaMind
from core.docker_self_improve_worker import (
    SelfImproveConfig,
    SelfImproveDockerWorker,
    _clean_github_repo_url,
    _extract_json_object,
    _resolve_repo_url,
)


class _FakeTok:
    def __init__(self, inner):
        self.inner = inner


@pytest.fixture
def fake_host_loader(monkeypatch: pytest.MonkeyPatch):
    class _H:
        cfg = type("C", (), {"d_model": 8})()

        def __init__(self):
            self.llm, self._stub = make_stub_llm_pair()

        def add_graft(self, *a, **k):
            pass

    def _make():
        h = _H()
        monkeypatch.setattr(
            broca_mod,
            "load_llama_broca_host",
            lambda *args, **kwargs: (h, _FakeTok(h._stub)),
        )
        return h

    return _make


def test_extract_json_object_fenced() -> None:
    raw = 'Prefix\n```json\n{"task_summary": "x", "unified_diff": ""}\n```\n'
    d = _extract_json_object(raw)
    assert d["task_summary"] == "x"
    assert d["unified_diff"] == ""


def test_extract_json_object_plain() -> None:
    d = _extract_json_object('{"a": 1}')
    assert d["a"] == 1


def test_extract_json_object_braces_inside_string() -> None:
    raw = '{"task_summary": "hello {world}", "unified_diff": ""}'
    d = _extract_json_object(raw)
    assert d["task_summary"] == "hello {world}"
    assert d["unified_diff"] == ""


def test_clean_github_repo_url() -> None:
    assert _clean_github_repo_url("git@github.com:org/repo.git") == "https://github.com/org/repo.git"
    assert (
        _clean_github_repo_url("https://x-access-token:secret@github.com/org/repo.git")
        == "https://github.com/org/repo.git"
    )


def test_resolve_repo_url_explicit() -> None:
    assert _resolve_repo_url("https://example.com/r.git") == "https://example.com/r.git"


def test_empty_diff_skips_docker(tmp_path: Path, fake_host_loader, monkeypatch: pytest.MonkeyPatch) -> None:
    fake_host_loader()
    mind = BrocaMind(seed=0, db_path=tmp_path / "m.sqlite", namespace="ut", device="cpu", hf_token=False)
    cfg = SelfImproveConfig(enabled=True, interval_s=60.0)
    w = SelfImproveDockerWorker(mind, config=cfg)
    monkeypatch.setenv("GITHUB_TOKEN", "t0ken")
    monkeypatch.setenv("BROCA_SELF_IMPROVE_REPO", "https://github.com/a/b.git")
    w.docker_binary = "/bin/docker"

    called: list[object] = []

    def _no_docker(**kwargs: object) -> subprocess.CompletedProcess[str]:  # noqa: ARG001
        called.append(True)
        return subprocess.CompletedProcess(args=[], returncode=0, stdout="", stderr="")

    monkeypatch.setattr(w, "_plan_patch", lambda _m: {"task_summary": "noop", "unified_diff": ""})
    monkeypatch.setattr(w, "_run_docker_cycle", _no_docker)
    w._run_once("run-empty-diff")
    assert called == []


def test_docker_invoked_with_patch(tmp_path: Path, fake_host_loader, monkeypatch: pytest.MonkeyPatch) -> None:
    fake_host_loader()
    mind = BrocaMind(seed=0, db_path=tmp_path / "m2.sqlite", namespace="ut2", device="cpu", hf_token=False)
    cfg = SelfImproveConfig(enabled=True, interval_s=60.0)
    w = SelfImproveDockerWorker(mind, config=cfg)
    monkeypatch.setenv("GITHUB_TOKEN", "t0ken")
    monkeypatch.setenv("BROCA_SELF_IMPROVE_REPO", "https://github.com/a/b.git")
    w.docker_binary = "/bin/docker"

    cap: dict[str, object] = {}

    def _fake_cycle(**kwargs: object) -> subprocess.CompletedProcess[str]:
        cap.update(kwargs)
        return subprocess.CompletedProcess(args=[], returncode=0, stdout="ok", stderr="")

    tiny_diff = (
        "--- a/README.md\n+++ b/README.md\n@@ -1,1 +1,2 @@\n x\n+y\n"
    )
    monkeypatch.setattr(
        w,
        "_plan_patch",
        lambda _m: {"task_summary": "doc tweak", "unified_diff": tiny_diff},
    )
    monkeypatch.setattr(w, "_run_docker_cycle", _fake_cycle)
    w._run_once("run-patch")
    assert "patch_text" in cap
    assert cap["patch_text"] == tiny_diff
    assert "branch_name" in cap and str(cap["branch_name"]).startswith("broca/self-improve-")


def test_start_stop_thread(tmp_path: Path, fake_host_loader, monkeypatch: pytest.MonkeyPatch) -> None:
    fake_host_loader()
    mind = BrocaMind(seed=0, db_path=tmp_path / "m3.sqlite", namespace="ut3", device="cpu", hf_token=False)
    cfg = SelfImproveConfig(enabled=True, interval_s=3600.0)
    w = SelfImproveDockerWorker(mind, config=cfg)
    monkeypatch.setattr(w, "_run_once_safe", MagicMock())
    w.start()
    assert w.running
    w.stop(timeout=2.0)
    assert not w.running


def test_mind_wiring_start_stop(tmp_path: Path, fake_host_loader, monkeypatch: pytest.MonkeyPatch) -> None:
    fake_host_loader()
    mind = BrocaMind(seed=0, db_path=tmp_path / "m4.sqlite", namespace="ut4", device="cpu", hf_token=False)
    monkeypatch.setenv("GITHUB_TOKEN", "x")
    # Patch before the background thread's first loop iteration (it calls _run_once_safe immediately).
    monkeypatch.setattr(SelfImproveDockerWorker, "_run_once_safe", MagicMock())
    worker = mind.start_self_improve_worker(interval_s=3600.0, enabled=True)
    assert worker is mind._self_improve_worker
    mind.stop_self_improve_worker()
