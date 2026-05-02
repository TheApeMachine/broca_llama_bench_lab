"""Execute synthesized native tools inside ephemeral Docker containers.

This lifts the restriction of the in-process :class:`ToolSandbox` (no imports,
no I/O) while keeping synthesis input out of the host process. Tools still
must expose ``def <name>(values: dict) -> object`` at module top level.

Environment (optional)::

    BROCA_TOOL_DOCKER_IMAGE — default ``python:3.11-slim``
    BROCA_TOOL_DOCKER_NETWORK — default ``none``
    BROCA_TOOL_DOCKER_MEMORY — default ``512m``
    BROCA_TOOL_DOCKER_CPUS — default ``1.0``
    BROCA_TOOL_TIMEOUT_S — default ``30``
"""

from __future__ import annotations

import json
import logging
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Mapping

from ..natives.native_tools import SandboxResult, ToolSandbox, ToolSynthesisError

logger = logging.getLogger(__name__)

_RUNNER_HEADER = """
import importlib.util
import json
import sys

def _main():
    spec = importlib.util.spec_from_file_location("tool_impl", "/work/tool_impl.py")
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    fn = getattr(mod, {func_name!r})
    raw = sys.stdin.read() or "{{}}"
    vals = json.loads(raw)
    out = fn(vals)
    json.dump({{"ok": True, "result": out}}, sys.stdout, default=str)
    sys.stdout.write("\\n")

if __name__ == "__main__":
    _main()
"""


class _DockerRelaxedValidator:
    """Minimal structural checks for Docker-backed tools (AST only, no local exec)."""

    @staticmethod
    def validate(tree: Any, function_name: str) -> None:
        import ast

        if not tree.body:
            raise ToolSynthesisError("tool source is empty")
        defs: list[ast.FunctionDef | ast.AsyncFunctionDef] = []
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                defs.append(node)
            elif (
                isinstance(node, ast.Expr)
                and isinstance(node.value, ast.Constant)
                and isinstance(node.value.value, str)
            ):
                continue
            elif isinstance(node, (ast.Import, ast.ImportFrom)):
                continue
            elif isinstance(node, ast.Assign | ast.AnnAssign | ast.AugAssign):
                continue
            else:
                raise ToolSynthesisError(
                    f"unsupported top-level node in Docker tool: {type(node).__name__}"
                )
        target = next((d for d in defs if d.name == function_name), None)
        if target is None:
            raise ToolSynthesisError(
                f"expected a function named {function_name!r}; found {[d.name for d in defs]!r}"
            )


class DockerToolSandbox(ToolSandbox):
    """Parse and verify tool sources locally; execute inside Docker."""

    def __init__(
        self,
        *,
        max_source_chars: int = 16_384,
        image: str | None = None,
        network: str | None = None,
        memory: str | None = None,
        cpus: str | None = None,
        timeout_s: float | None = None,
    ):
        super().__init__(max_source_chars=max_source_chars)
        import os

        self.docker_binary = shutil.which("docker")
        self.image = image or os.environ.get("BROCA_TOOL_DOCKER_IMAGE", "python:3.11-slim").strip()
        self.network = network or os.environ.get("BROCA_TOOL_DOCKER_NETWORK", "none").strip()
        self.memory = memory or os.environ.get("BROCA_TOOL_DOCKER_MEMORY", "512m").strip()
        self.cpus = cpus or os.environ.get("BROCA_TOOL_DOCKER_CPUS", "1.0").strip()
        self.timeout_s = float(timeout_s or os.environ.get("BROCA_TOOL_TIMEOUT_S", "30"))

    def compile(self, source: str, function_name: str) -> SandboxResult:
        if self.docker_binary is None:
            raise ToolSynthesisError(
                "DockerToolSandbox requires the docker CLI on PATH; install Docker Desktop / Engine or disable BROCA_USE_DOCKER_TOOLS"
            )
        import ast

        if not isinstance(source, str):
            raise ToolSynthesisError("source must be a string")
        if len(source) > self.max_source_chars:
            raise ToolSynthesisError(f"source exceeds maximum length {self.max_source_chars}")
        try:
            tree = ast.parse(source, mode="exec")
        except SyntaxError as exc:
            raise ToolSynthesisError(f"syntax error in tool source: {exc}") from exc

        _DockerRelaxedValidator.validate(tree, function_name)

        def _runner(values: Mapping[str, Any]) -> Any:
            return _docker_invoke(
                self.docker_binary,
                source,
                function_name,
                dict(values),
                image=self.image,
                network=self.network,
                memory=self.memory,
                cpus=self.cpus,
                timeout_s=self.timeout_s,
            )

        return SandboxResult(fn=_runner, source=source, function_name=function_name)


def _docker_invoke(
    docker_bin: str,
    source: str,
    function_name: str,
    values: dict[str, Any],
    *,
    image: str,
    network: str,
    memory: str,
    cpus: str,
    timeout_s: float,
) -> Any:
    runner_body = _RUNNER_HEADER.format(func_name=function_name)
    with tempfile.TemporaryDirectory(prefix="broca_tool_") as td:
        root = Path(td)
        (root / "tool_impl.py").write_text(source, encoding="utf-8")
        (root / "runner.py").write_text(runner_body, encoding="utf-8")
        work = root.resolve()
        cmd = [
            docker_bin,
            "run",
            "--rm",
            "-i",
            "--network",
            network,
            "--memory",
            memory,
            "--cpus",
            cpus,
            "-v",
            f"{work}:/work:ro",
            image,
            "python",
            "/work/runner.py",
        ]
        payload = json.dumps(values, default=str).encode("utf-8")
        try:
            proc = subprocess.run(
                cmd,
                input=payload,
                capture_output=True,
                timeout=timeout_s,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            raise ToolSynthesisError(f"Docker tool timed out after {timeout_s}s") from exc
        except OSError as exc:
            raise ToolSynthesisError(f"failed to spawn docker: {exc}") from exc
        if proc.returncode != 0:
            err = proc.stderr.decode("utf-8", errors="replace").strip()
            raise ToolSynthesisError(
                f"docker tool exited {proc.returncode}: {err or proc.stdout.decode('utf-8', errors='replace')}"
            )
        out = proc.stdout.decode("utf-8", errors="replace").strip()
        if not out:
            raise ToolSynthesisError("docker tool produced empty stdout")
        try:
            envelope = json.loads(out.splitlines()[-1])
        except json.JSONDecodeError as exc:
            raise ToolSynthesisError(f"invalid tool JSON output: {out[:512]!r}") from exc
        if not envelope.get("ok"):
            raise ToolSynthesisError(f"tool reported failure: {envelope!r}")
        return envelope.get("result")
