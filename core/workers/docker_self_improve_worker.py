"""Background worker: Docker-isolated clone, validate, branch, and PR workflow.

This is independent of :class:`CognitiveBackgroundWorker` (DMN). It uses the same
``SubstrateController`` for (1) substrate-biased planning via :meth:`SubstrateController.chat_reply`
and (2) persistent outcomes via semantic claims, reflections, journal, and Hopfield.

**Opt-in:** set ``BROCA_SELF_IMPROVE=1`` or pass ``--self-improve`` to
``python -m core chat``. Requires ``GITHUB_TOKEN`` with ``repo``
scope, ``docker`` on PATH, and ``BROCA_SELF_IMPROVE_REPO`` (or a resolveable
``git`` remote).

Environment (optional)::

    BROCA_SELF_IMPROVE_REPO — HTTPS git URL (default: ``git remote get-url origin``)
    BROCA_SELF_IMPROVE_BASE_BRANCH — default ``main``
    BROCA_SELF_IMPROVE_INTERVAL_S — seconds between cycles (default ``3600``)
    BROCA_SELF_IMPROVE_DOCKER_IMAGE — default ``python:3.11-slim``
    BROCA_SELF_IMPROVE_DOCKER_NETWORK — default ``bridge`` (clone needs network)
    BROCA_SELF_IMPROVE_DOCKER_MEMORY — default ``4g``
    BROCA_SELF_IMPROVE_DOCKER_CPUS — default ``2``
    BROCA_SELF_IMPROVE_TIMEOUT_S — default ``1800``
    BROCA_SELF_IMPROVE_RUN_DEMO — if ``1``, run ``python -m core.demo --mode broca`` after pytest
    BROCA_SELF_IMPROVE_RUN_PAPER — if ``1``, run ``python -m core.paper`` (``refresh_paper_experiments``)
        after a **successful** Docker validation cycle (local benchmark/TeX deps as for ``make paper-bench``)
"""

from __future__ import annotations

import json
import logging
import os
import re
import shutil
import stat
import subprocess
import tempfile
import textwrap
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..cognition.substrate import CognitiveFrame, SubstrateController
from ..frame.continuous_frame import stable_sketch

logger = logging.getLogger(__name__)

_RUNNER_BASH = textwrap.dedent(
    r"""
    set -euo pipefail
    export DEBIAN_FRONTEND=noninteractive
    apt-get update -qq
    apt-get install -y --no-install-recommends git ca-certificates curl \
      python3-venv >/dev/null

    ARCH=$(uname -m)
    case "$ARCH" in
      x86_64) GH_ARCH=amd64 ;;
      aarch64|arm64) GH_ARCH=arm64 ;;
      *) echo "unsupported arch: $ARCH" >&2; exit 2 ;;
    esac
    GH_VER=2.63.2
    curl -fsSL "https://github.com/cli/cli/releases/download/v${GH_VER}/gh_${GH_VER}_linux_${GH_ARCH}.tar.gz" \
      | tar xz -C /tmp
    mv "/tmp/gh_${GH_VER}_linux_${GH_ARCH}/bin/gh" /usr/local/bin/gh

    {
      echo '#!/bin/sh'
      echo 'printf "%s\n" "${GITHUB_TOKEN:-}"'
    } > /tmp/git-askpass.sh
    chmod 700 /tmp/git-askpass.sh
    export GIT_ASKPASS=/tmp/git-askpass.sh
    export GIT_TERMINAL_PROMPT=0

    rm -rf /work/repo
    if ! git clone --depth 1 --branch "${BASE_BRANCH}" "${REPO_URL}" /work/repo 2>/dev/null; then
      echo "git clone failed for branch ${BASE_BRANCH} (network, auth, or branch missing)" >&2
      exit 5
    fi
    cd /work/repo
    if ! git rev-parse --verify "refs/heads/${BASE_BRANCH}" >/dev/null 2>&1; then
      echo "git: branch ${BASE_BRANCH} not checked out after clone" >&2
      exit 5
    fi

    git checkout -b "${BRANCH_NAME}"
    if [[ -s /context/patch.diff ]]; then
      git apply /context/patch.diff || { echo "git apply failed" >&2; exit 3; }
    else
      echo "empty patch; nothing to validate" >&2
      exit 0
    fi

    python -m pip install --upgrade pip setuptools wheel -q
    python -m pip install -e ".[test]" -q

    pytest -q
    echo "pytest_ok" >> /context/outcomes.log

    if [[ "${RUN_SELF_IMPROVE_DEMO:-0}" == "1" ]]; then
      python -m pip install -e ".[benchmark]" -q
      python -m core.demo --mode broca --seed 0
      echo "demo_ok" >> /context/outcomes.log
    fi

    git config user.email "${GIT_AUTHOR_EMAIL:-broca-self-improve@users.noreply.github.com}"
    git config user.name "${GIT_AUTHOR_NAME:-Broca self-improve worker}"
    git add -A
    git commit -m "${COMMIT_MESSAGE}" || { echo "nothing to commit" >&2; exit 4; }

    git push -u origin "${BRANCH_NAME}"

    export GH_TOKEN="${GITHUB_TOKEN}"
    gh pr create \
      --title "${PR_TITLE}" \
      --body-file /context/pr_body.md \
      --base "${BASE_BRANCH}" \
      --head "${BRANCH_NAME}"

    echo "pr_created" >> /context/outcomes.log
    """
).strip()


def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _safe_float(raw: str | None, default: float) -> float:
    if raw is None or not str(raw).strip():
        return float(default)
    try:
        return float(str(raw).strip())
    except ValueError:
        logger.warning("Ignoring invalid float env value %r; using default %s", raw, default)
        return float(default)


def _clean_github_repo_url(url: str) -> str:
    """Return an https GitHub URL without embedded credentials (GIT_ASKPASS supplies token)."""

    u = url.strip()
    if not u:
        return u
    if u.startswith("git@github.com:"):
        path = u[len("git@github.com:") :].strip()
        if not path.endswith(".git"):
            path = f"{path}.git"
        return f"https://github.com/{path}"
    u = re.sub(r"^https://[^/@]+@", "https://", u, count=1)
    return u


def _resolve_repo_url(explicit: str | None) -> str | None:
    if explicit and explicit.strip():
        return explicit.strip()
    env = os.environ.get("BROCA_SELF_IMPROVE_REPO", "").strip()
    if env:
        return env
    try:
        out = subprocess.run(
            ["git", "remote", "get-url", "origin"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        if out.returncode == 0 and out.stdout.strip():
            return out.stdout.strip()
    except (OSError, subprocess.TimeoutExpired):
        logger.exception("_resolve_repo_url: git remote failed")
    return None


def _extract_json_object(text: str) -> dict[str, Any]:
    """Parse first JSON object from model output (optionally inside a fenced block)."""

    s = text.strip()
    fence = re.search(r"```(?:json)?\s*(\{[\s\S]*?\})\s*```", s, re.IGNORECASE)
    if fence:
        s = fence.group(1).strip()
    brace = s.find("{")
    if brace < 0:
        return json.loads(s)
    tail = s[brace:]
    for i, ch in enumerate(tail):
        if ch != "}":
            continue
        candidate = tail[: i + 1]
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            continue
    return json.loads(tail)


@dataclass
class SelfImproveConfig:
    """Configuration for :class:`SelfImproveDockerWorker`."""

    enabled: bool = field(default_factory=lambda: _env_bool("BROCA_SELF_IMPROVE", False))
    interval_s: float = field(
        default_factory=lambda: max(
            60.0,
            _safe_float(os.environ.get("BROCA_SELF_IMPROVE_INTERVAL_S"), 3600.0),
        )
    )
    repo_url: str | None = None
    base_branch: str = field(
        default_factory=lambda: os.environ.get("BROCA_SELF_IMPROVE_BASE_BRANCH", "main").strip() or "main"
    )
    docker_image: str = field(
        default_factory=lambda: os.environ.get("BROCA_SELF_IMPROVE_DOCKER_IMAGE", "python:3.11-slim").strip()
    )
    docker_network: str = field(
        default_factory=lambda: os.environ.get("BROCA_SELF_IMPROVE_DOCKER_NETWORK", "bridge").strip()
    )
    docker_memory: str = field(
        default_factory=lambda: os.environ.get("BROCA_SELF_IMPROVE_DOCKER_MEMORY", "4g").strip()
    )
    docker_cpus: str = field(
        default_factory=lambda: os.environ.get("BROCA_SELF_IMPROVE_DOCKER_CPUS", "2").strip()
    )
    timeout_s: float = field(
        default_factory=lambda: _safe_float(os.environ.get("BROCA_SELF_IMPROVE_TIMEOUT_S"), 1800.0)
    )
    run_arch_demo: bool = field(default_factory=lambda: _env_bool("BROCA_SELF_IMPROVE_RUN_DEMO", False))
    run_paper_refresh: bool = field(default_factory=lambda: _env_bool("BROCA_SELF_IMPROVE_RUN_PAPER", False))
    max_new_tokens_plan: int = 1024


class SelfImproveDockerWorker:
    """Daemon thread that plans a change, runs Docker validation, and opens a PR."""

    def __init__(
        self,
        mind: SubstrateController,
        *,
        config: SelfImproveConfig | None = None,
    ):
        self.mind = mind
        self.config = config if config is not None else SelfImproveConfig()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.iterations = 0
        self.iterations_lock = threading.Lock()
        self.last_error: str | None = None
        self.last_summary: str | None = None
        self.docker_binary = shutil.which("docker")

    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def get_iterations(self) -> int:
        with self.iterations_lock:
            return int(self.iterations)

    def start(self) -> None:
        if self.running:
            return
        if not self.config.enabled:
            logger.info("SelfImproveDockerWorker.start: disabled (BROCA_SELF_IMPROVE not set)")
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, name="broca-self-improve", daemon=True)
        self._thread.start()
        logger.info("SelfImproveDockerWorker.start: interval=%.1fs", self.config.interval_s)

    def stop(self, timeout: float = 5.0) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=max(0.0, float(timeout)))
        logger.info("SelfImproveDockerWorker.stop")

    def _memory_context_text(self) -> str:
        lines: list[str] = []
        try:
            claims = self.mind.memory.claims(subject="docker_self_improve")
            for c in claims[-20:]:
                lines.append(
                    f"- semantic: {c['predicate']}={c['object']!r} status={c['status']} "
                    f"conf={c['confidence']:.2f} ev={c['evidence']}"
                )
        except Exception:
            logger.exception("_memory_context_text: claims failed")
        try:
            for r in self.mind.memory.reflections(kind="self_improve_run")[-12:]:
                lines.append(f"- reflection: {r['summary']!r} ev={r['evidence']}")
        except Exception:
            logger.exception("_memory_context_text: reflections failed")
        return "\n".join(lines) if lines else "(no prior docker self-improve memory)"

    def _record_outcome(
        self,
        *,
        run_id: str,
        ok: bool,
        summary: str,
        evidence: dict[str, Any],
        branch: str | None,
    ) -> None:
        try:
            self.mind.memory.record_claim(
                "docker_self_improve",
                "outcome",
                "success" if ok else "failure",
                confidence=1.0 if ok else 0.35,
                status="observed",
                evidence={**evidence, "run_id": run_id, "branch": branch},
            )
        except Exception:
            logger.exception("_record_outcome: record_claim failed")
        try:
            self.mind.memory.record_reflection(
                "self_improve_run",
                "docker_self_improve",
                "cycle",
                summary[:2048],
                evidence,
                dedupe_key=f"{run_id}:{int(time.time())}",
            )
        except Exception:
            logger.exception("_record_outcome: record_reflection failed")
        try:
            utterance = f"[docker-self-improve] {summary[:500]}"
            frame = CognitiveFrame(
                "maintenance",
                subject="docker_self_improve",
                answer="success" if ok else "failure",
                confidence=1.0 if ok else 0.4,
                evidence={**evidence, "run_id": run_id, "branch": branch or ""},
            )
            self.mind.journal.append(utterance, frame)
        except Exception:
            logger.exception("_record_outcome: journal failed")
        try:
            ut = stable_sketch(f"self_improve:{summary[:256]}")
            trip = stable_sketch(f"outcome:{'ok' if ok else 'fail'}:{branch or 'none'}:{run_id[:8]}")
            self.mind.remember_hopfield(
                ut,
                trip,
                metadata={"kind": "docker_self_improve", "run_id": run_id},
            )
        except Exception:
            logger.exception("_record_outcome: hopfield failed")

    def _plan_patch(self, memory_blob: str) -> dict[str, Any] | None:
        instructions = textwrap.dedent(
            f"""
            You are an autonomous maintainer for this Python repository (Mosaic).
            Prior runs and outcomes from persistent substrate memory:
            {memory_blob}

            Propose ONE small, safe improvement: bugfix, clearer error handling, test coverage,
            docstring, or refactor limited to a few files. You MUST include at least one new or
            updated test if you change behavior.

            Reply with a single JSON object only (no prose), optionally wrapped in ```json fences.
            Schema:
            {{
              "task_summary": "one line",
              "unified_diff": "complete unified diff text (git format) OR empty string to skip this cycle",
              "notes": "optional string"
            }}

            The unified_diff must apply with `git apply` from the repository root after cloning
            the default branch. Use paths like a/core/foo.py (prefix a/).
            If there is nothing worthwhile to do, set unified_diff to "".
            """
        ).strip()
        _frame, reply = self.mind.chat_reply(
            [{"role": "user", "content": instructions}],
            max_new_tokens=int(self.config.max_new_tokens_plan),
            do_sample=True,
            temperature=0.5,
            top_p=0.9,
        )
        try:
            return _extract_json_object(reply)
        except (json.JSONDecodeError, ValueError, TypeError) as exc:
            logger.warning("SelfImproveDockerWorker: JSON parse failed: %s reply_preview=%r", exc, reply[:240])
            return None

    def _run_docker_cycle(
        self,
        *,
        repo_url: str,
        patch_text: str,
        branch_name: str,
        pr_title: str,
        pr_body: str,
        commit_message: str,
    ) -> subprocess.CompletedProcess[str]:
        docker_bin = self.docker_binary
        if docker_bin is None:
            raise RuntimeError("docker_binary is not set (docker CLI not found on PATH)")
        token = os.environ.get("GITHUB_TOKEN", "").strip()
        if not token:
            raise RuntimeError("GITHUB_TOKEN is not set")

        safe_repo = _clean_github_repo_url(repo_url)
        gh_env_path: str | None = None
        try:
            with tempfile.TemporaryDirectory(prefix="broca_self_improve_") as td:
                root = Path(td)
                (root / "patch.diff").write_text(patch_text, encoding="utf-8")
                (root / "runner.sh").write_text("#!/usr/bin/env bash\n" + _RUNNER_BASH + "\n", encoding="utf-8")
                (root / "pr_body.md").write_text(pr_body, encoding="utf-8")
                (root / "outcomes.log").write_text("", encoding="utf-8")

                with tempfile.NamedTemporaryFile(
                    mode="w",
                    prefix="broca_gh_token_",
                    suffix=".env",
                    delete=False,
                    encoding="utf-8",
                ) as gh_env:
                    gh_env.write(f"GITHUB_TOKEN={token}\n")
                    gh_env.flush()
                    gh_env_path = gh_env.name
                os.chmod(gh_env_path, stat.S_IRUSR | stat.S_IWUSR)

                env_pairs = [
                    f"REPO_URL={safe_repo}",
                    f"BRANCH_NAME={branch_name}",
                    f"BASE_BRANCH={self.config.base_branch}",
                    f"PR_TITLE={pr_title}",
                    f"COMMIT_MESSAGE={commit_message}",
                    f"RUN_SELF_IMPROVE_DEMO={'1' if self.config.run_arch_demo else '0'}",
                ]
                cmd: list[str] = [
                    docker_bin,
                    "run",
                    "--rm",
                    "--network",
                    self.config.docker_network,
                    "--memory",
                    self.config.docker_memory,
                    "--cpus",
                    self.config.docker_cpus,
                    "--env-file",
                    gh_env_path,
                    "-v",
                    f"{root.resolve()}:/context",
                    "-w",
                    "/work",
                ]
                for kv in env_pairs:
                    cmd.extend(["-e", kv])
                cmd.extend(
                    [
                        self.config.docker_image,
                        "bash",
                        "-lc",
                        "bash /context/runner.sh 2>&1 | tee /context/docker.log; exit ${PIPESTATUS[0]}",
                    ]
                )
                return subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=float(self.config.timeout_s),
                    check=False,
                )
        finally:
            if gh_env_path is not None:
                try:
                    os.unlink(gh_env_path)
                except OSError:
                    logger.warning("Could not remove temporary GitHub env file %s", gh_env_path)

    def _loop(self) -> None:
        while not self._stop.is_set():
            if self.config.enabled:
                self._run_once_safe()
            if self._stop.wait(timeout=float(self.config.interval_s)):
                break

    def _run_once_safe(self) -> None:
        run_id = str(uuid.uuid4())
        with self.iterations_lock:
            self.iterations += 1
            it = int(self.iterations)
        try:
            self.mind.event_bus.publish("self_improve.cycle_start", {"run_id": run_id, "iteration": it})
        except Exception:
            logger.exception("self-improve: cycle_start publish failed")
        try:
            self._run_once(run_id)
            try:
                self.mind.event_bus.publish(
                    "self_improve.cycle_complete",
                    {"run_id": run_id, "iteration": it, "summary": self.last_summary, "error": self.last_error},
                )
            except Exception:
                logger.exception("self-improve: cycle_complete publish failed")
        except Exception as exc:
            self.last_error = str(exc)
            logger.exception("SelfImproveDockerWorker cycle failed")
            self._record_outcome(
                run_id=run_id,
                ok=False,
                summary=f"uncaught error: {exc}",
                evidence={"error": str(exc)},
                branch=None,
            )
            try:
                self.mind.event_bus.publish(
                    "self_improve.cycle_complete",
                    {"run_id": run_id, "iteration": it, "summary": None, "error": str(exc)},
                )
            except Exception:
                logger.exception("self-improve: cycle_complete publish failed")

    def _run_once(self, run_id: str) -> None:
        self.last_error = None
        if self.docker_binary is None:
            raise RuntimeError("docker CLI not found on PATH")

        token = os.environ.get("GITHUB_TOKEN", "").strip()
        if not token:
            raise RuntimeError("GITHUB_TOKEN not set (needs repo scope for push and gh pr create)")

        repo_url = _resolve_repo_url(self.config.repo_url)
        if not repo_url:
            raise RuntimeError("no repo URL (set BROCA_SELF_IMPROVE_REPO or run inside a git clone)")

        memory_blob = self._memory_context_text()
        plan = self._plan_patch(memory_blob)
        if not plan:
            self._record_outcome(
                run_id=run_id,
                ok=False,
                summary="planner produced no valid JSON",
                evidence={},
                branch=None,
            )
            return

        raw_diff = plan.get("unified_diff")
        diff = "" if raw_diff is None else str(raw_diff)
        if diff.strip() and not diff.endswith("\n"):
            diff += "\n"
        summary = str(plan.get("task_summary") or "self-improve").strip()
        self.last_summary = summary
        if not diff.strip():
            logger.info("SelfImproveDockerWorker: empty diff; skipping Docker")
            self._record_outcome(
                run_id=run_id,
                ok=True,
                summary=f"no change proposed: {summary}",
                evidence={"notes": plan.get("notes")},
                branch=None,
            )
            return

        safe_slug = re.sub(r"[^a-zA-Z0-9._-]+", "-", summary.lower())[:40].strip("-") or "task"
        branch_name = f"broca/self-improve-{safe_slug}-{run_id[:8]}"
        pr_title = f"[broca-self-improve] {summary[:120]}"
        pr_body = textwrap.dedent(
            f"""
            Automated proposal from Broca **self-improve** worker (`{run_id}`).

            ### Summary
            {summary}

            ### Notes
            {plan.get("notes") or "—"}

            ### Memory context (abbrev.)
            {memory_blob[:2000]}

            _Validation: `pip install -e ".[test]"`, `pytest -q`{"`, `python -m core.demo --mode broca`" if self.config.run_arch_demo else ""} inside Docker._
            """
        ).strip()

        proc = self._run_docker_cycle(
            repo_url=repo_url,
            patch_text=diff,
            branch_name=branch_name,
            pr_title=pr_title,
            pr_body=pr_body,
            commit_message=f"self-improve: {summary[:72]}",
        )
        ok = proc.returncode == 0
        log_tail = (proc.stdout or "")[-4000:]
        if not ok:
            self.last_error = f"docker exit {proc.returncode}: {log_tail}"
        self._record_outcome(
            run_id=run_id,
            ok=ok,
            summary=summary if ok else f"{summary} (docker failed)",
            evidence={
                "returncode": proc.returncode,
                "log_tail": log_tail,
                "branch": branch_name,
            },
            branch=branch_name,
        )
        if ok and self.config.run_paper_refresh:
            try:
                from ..paper.harness import refresh_paper_experiments

                refresh_paper_experiments()
            except Exception:
                logger.exception("self-improve: refresh_paper_experiments failed after successful Docker cycle")
