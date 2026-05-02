from __future__ import annotations

from unittest import mock

import pytest

from core.system.sandbox import DockerToolSandbox
from core.natives.native_tools import ToolSynthesisError


def test_docker_sandbox_requires_cli():
    sb = DockerToolSandbox()
    sb.docker_binary = None
    with pytest.raises(ToolSynthesisError, match="docker CLI"):
        sb.compile("def f(values):\n    return 1\n", "f")


def test_docker_sandbox_invokes_runner(monkeypatch: pytest.MonkeyPatch, tmp_path):
    sb = DockerToolSandbox()
    sb.docker_binary = "/bin/docker"

    captured: dict = {}

    def fake_run(cmd, **kwargs):
        captured["cmd"] = cmd
        captured["input"] = kwargs.get("input")
        return mock.Mock(returncode=0, stdout=b'{"ok": true, "result": 42}\n', stderr=b"")

    monkeypatch.setattr("core.system.sandbox.subprocess.run", fake_run)

    class FakeTmp:
        def __enter__(self_inner):
            return str(tmp_path)

        def __exit__(self_inner, *a):
            return None

    monkeypatch.setattr("core.system.sandbox.tempfile.TemporaryDirectory", lambda **k: FakeTmp())

    res = sb.compile(
        "def tool(values):\n    return int(values['x']) + 1\n",
        "tool",
    )
    out = res.fn({"x": 41})
    assert out == 42
    assert captured["cmd"][0] == "/bin/docker"
    assert captured["input"] is not None
    assert b'"x": 41' in captured["input"]


def test_docker_relaxed_validator_allows_import():
    import ast

    from core.system.sandbox import _DockerRelaxedValidator

    src = "import math\n\ndef tool(values):\n    return int(math.sqrt(values['x']))\n"
    tree = ast.parse(src)
    _DockerRelaxedValidator.validate(tree, "tool")
