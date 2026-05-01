"""Tests for the native tool registry, sandbox, and SCM attachment.

Synthesized tools are persisted Python source. Each test composes a fresh
SQLite-backed registry against a tmp path, exercises a piece of the
pipeline, and asserts on observable outputs (verified flag, SCM
``add_endogenous`` side effects, sandbox rejection, persistence
round-trip).
"""

from __future__ import annotations

import pytest

from asi_broca_core.causal import FiniteSCM
from asi_broca_core.native_tools import (
    NativeTool,
    NativeToolRegistry,
    SandboxResult,
    ToolSandbox,
    ToolSynthesisError,
)


# ---------------------------------------------------------------------------
# Sandbox compilation
# ---------------------------------------------------------------------------


def test_sandbox_compiles_simple_function():
    sandbox = ToolSandbox()
    result = sandbox.compile(
        """
def double(values):
    return 2 * values["x"]
""",
        function_name="double",
    )
    assert isinstance(result, SandboxResult)
    assert result.fn({"x": 5}) == 10
    assert result.function_name == "double"


def test_sandbox_rejects_imports():
    sandbox = ToolSandbox()
    with pytest.raises(ToolSynthesisError):
        sandbox.compile(
            """
import os
def bad(values):
    return os.getcwd()
""",
            function_name="bad",
        )


def test_sandbox_rejects_dunder_attribute_access():
    sandbox = ToolSandbox()
    with pytest.raises(ToolSynthesisError):
        sandbox.compile(
            """
def evil(values):
    return values.__class__.__bases__[0].__subclasses__()
""",
            function_name="evil",
        )


def test_sandbox_rejects_top_level_statements():
    sandbox = ToolSandbox()
    with pytest.raises(ToolSynthesisError):
        sandbox.compile(
            """
x = 5
def f(values):
    return x
""",
            function_name="f",
        )


def test_sandbox_rejects_eval_and_exec():
    sandbox = ToolSandbox()
    with pytest.raises(ToolSynthesisError):
        sandbox.compile(
            """
def f(values):
    return eval('1+1')
""",
            function_name="f",
        )


def test_sandbox_rejects_when_target_function_missing():
    sandbox = ToolSandbox()
    with pytest.raises(ToolSynthesisError):
        sandbox.compile(
            """
def other_name(values):
    return 1
""",
            function_name="missing_name",
        )


def test_sandbox_rejects_oversize_source():
    sandbox = ToolSandbox(max_source_chars=100)
    big = "def f(values):\n    return " + " + ".join(["1"] * 200)
    with pytest.raises(ToolSynthesisError):
        sandbox.compile(big, function_name="f")


def test_sandbox_allows_module_docstring():
    sandbox = ToolSandbox()
    result = sandbox.compile(
        '''
"""harmless docstring"""
def f(values):
    return 1
''',
        function_name="f",
    )
    assert result.fn({}) == 1


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------


def test_sandbox_verify_accepts_in_domain_outputs():
    sandbox = ToolSandbox()
    fn = sandbox.compile(
        """
def f(v):
    return 1 if v["x"] > 0 else 0
""",
        function_name="f",
    ).fn
    outs = ToolSandbox.verify(
        fn,
        domain=[0, 1],
        sample_inputs=[{"x": -1}, {"x": 0}, {"x": 1}],
    )
    assert outs == [0, 0, 1]


def test_sandbox_verify_rejects_out_of_domain_output():
    sandbox = ToolSandbox()
    fn = sandbox.compile(
        """
def f(v):
    return 99
""",
        function_name="f",
    ).fn
    with pytest.raises(ToolSynthesisError):
        ToolSandbox.verify(fn, domain=[0, 1], sample_inputs=[{"x": 0}])


def test_sandbox_verify_rejects_when_fn_raises():
    sandbox = ToolSandbox()
    fn = sandbox.compile(
        """
def f(v):
    return v["missing_key"]
""",
        function_name="f",
    ).fn
    with pytest.raises(ToolSynthesisError):
        ToolSandbox.verify(fn, domain=[0, 1], sample_inputs=[{"x": 0}])


def test_sandbox_verify_requires_at_least_one_sample():
    sandbox = ToolSandbox()
    fn = sandbox.compile("def f(v):\n    return 0\n", function_name="f").fn
    with pytest.raises(ToolSynthesisError):
        ToolSandbox.verify(fn, domain=[0], sample_inputs=[])


# ---------------------------------------------------------------------------
# Registry persistence
# ---------------------------------------------------------------------------


def test_registry_synthesize_persists_tool(tmp_path):
    db = tmp_path / "tools.sqlite"
    reg = NativeToolRegistry(db, namespace="t")
    tool = reg.synthesize(
        "is_positive",
        "def is_positive(v):\n    return 1 if v['x'] > 0 else 0\n",
        parents=("x",),
        domain=(0, 1),
        sample_inputs=[{"x": -1}, {"x": 1}],
        description="returns 1 for positive x",
    )
    assert tool.verified
    assert tool.id is not None
    assert reg.count() == 1
    fetched = reg.get("is_positive")
    assert fetched is not None
    assert fetched.parents == ("x",)
    assert fetched.domain == (0, 1)
    assert fetched.fn is not None
    assert fetched.fn({"x": 7}) == 1
    assert fetched.fn({"x": -3}) == 0


def test_registry_refuses_overwrite_without_flag(tmp_path):
    db = tmp_path / "tools.sqlite"
    reg = NativeToolRegistry(db, namespace="t")
    src = "def f(v):\n    return 0\n"
    reg.synthesize("f", src, parents=(), domain=(0,), sample_inputs=[{}])
    with pytest.raises(ToolSynthesisError):
        reg.synthesize("f", src, parents=(), domain=(0,), sample_inputs=[{}])


def test_registry_overwrite_replaces_source(tmp_path):
    db = tmp_path / "tools.sqlite"
    reg = NativeToolRegistry(db, namespace="t")
    reg.synthesize(
        "constant",
        "def constant(v):\n    return 0\n",
        parents=(),
        domain=(0, 1),
        sample_inputs=[{}],
    )
    reg.synthesize(
        "constant",
        "def constant(v):\n    return 1\n",
        parents=(),
        domain=(0, 1),
        sample_inputs=[{}],
        overwrite=True,
    )
    fresh = reg.get("constant")
    assert fresh is not None
    assert fresh.fn({}) == 1


def test_registry_remove_deletes_persisted_row(tmp_path):
    db = tmp_path / "tools.sqlite"
    reg = NativeToolRegistry(db, namespace="t")
    reg.synthesize("f", "def f(v):\n    return 0\n", parents=(), domain=(0,), sample_inputs=[{}])
    assert reg.count() == 1
    assert reg.remove("f") is True
    assert reg.count() == 0
    assert reg.remove("f") is False


def test_registry_namespace_isolation(tmp_path):
    db = tmp_path / "tools.sqlite"
    reg_a = NativeToolRegistry(db, namespace="a")
    reg_b = NativeToolRegistry(db, namespace="b")
    reg_a.synthesize("f", "def f(v):\n    return 0\n", parents=(), domain=(0,), sample_inputs=[{}])
    assert reg_a.count() == 1
    assert reg_b.count() == 0


def test_registry_round_trip_rehydrates_callable(tmp_path):
    db = tmp_path / "tools.sqlite"
    reg = NativeToolRegistry(db, namespace="t")
    reg.synthesize(
        "is_positive",
        "def is_positive(v):\n    return 1 if v['x'] > 0 else 0\n",
        parents=("x",),
        domain=(0, 1),
        sample_inputs=[{"x": 1}, {"x": -1}],
    )

    # Re-instantiate the registry against the same path — simulates a fresh process.
    reg2 = NativeToolRegistry(db, namespace="t")
    tools = reg2.all_tools()
    assert len(tools) == 1
    assert tools[0].fn is not None
    assert tools[0].fn({"x": 4}) == 1
    assert tools[0].fn({"x": -4}) == 0


# ---------------------------------------------------------------------------
# SCM attachment
# ---------------------------------------------------------------------------


def test_attach_to_scm_registers_endogenous_equation(tmp_path):
    db = tmp_path / "tools.sqlite"
    reg = NativeToolRegistry(db, namespace="t")
    reg.synthesize(
        "rains_today",
        "def rains_today(v):\n    return 1 if v['humidity'] >= 1 else 0\n",
        parents=("humidity",),
        domain=(0, 1),
        sample_inputs=[{"humidity": 0}, {"humidity": 1}],
    )
    scm = FiniteSCM(domains={})
    n_attached = reg.attach_to_scm(scm)
    assert n_attached == 1
    assert "rains_today" in scm.equations
    # The auto-declared parent is endogenous (pass-through of its own noise) so
    # Pearl-style do() interventions rewrite its equation as expected.
    assert "humidity" in scm.equations
    assert scm.domains["humidity"] == (0, 1)
    assert "U_humidity" in scm.exogenous
    # The equation must be evaluable through the SCM's standard pipeline.
    p = scm.probability({"rains_today": 1}, interventions={"humidity": 1})
    assert p == 1.0
    p0 = scm.probability({"rains_today": 1}, interventions={"humidity": 0})
    assert p0 == 0.0


def test_attach_to_scm_skips_unknown_parents_when_disallowed(tmp_path):
    db = tmp_path / "tools.sqlite"
    reg = NativeToolRegistry(db, namespace="t")
    reg.synthesize(
        "f",
        "def f(v):\n    return 0\n",
        parents=("missing_parent",),
        domain=(0, 1),
        sample_inputs=[{"missing_parent": 0}],
    )
    scm = FiniteSCM(domains={})
    n = reg.attach_to_scm(scm, allow_unknown_parents=False)
    assert n == 0
    assert "f" not in scm.equations


def test_attach_to_scm_supports_intervention_via_native_tool(tmp_path):
    """End-to-end: a synthesized tool can be intervened on by the SCM."""

    db = tmp_path / "tools.sqlite"
    reg = NativeToolRegistry(db, namespace="t")
    # Add: a tool that says "alarm fires whenever both bell parents fire".
    reg.synthesize(
        "alarm",
        "def alarm(v):\n    return 1 if v['fire'] == 1 and v['smoke'] == 1 else 0\n",
        parents=("fire", "smoke"),
        domain=(0, 1),
        sample_inputs=[
            {"fire": 0, "smoke": 0},
            {"fire": 1, "smoke": 0},
            {"fire": 0, "smoke": 1},
            {"fire": 1, "smoke": 1},
        ],
    )
    scm = FiniteSCM(domains={})
    reg.attach_to_scm(scm)
    # Intervene on smoke alone (fire varies under prior).
    p_alarm_smoke1 = scm.probability({"alarm": 1}, interventions={"smoke": 1, "fire": 1})
    p_alarm_smoke0 = scm.probability({"alarm": 1}, interventions={"smoke": 0, "fire": 1})
    assert p_alarm_smoke1 == 1.0
    assert p_alarm_smoke0 == 0.0


def test_synthesized_tool_wraps_runtime_failure_with_fallback(tmp_path):
    """If the synthesized fn raises at runtime, the wrapper must coerce to a domain default."""

    db = tmp_path / "tools.sqlite"
    reg = NativeToolRegistry(db, namespace="t")
    # Cheekily verify with one input set, then call with another that triggers KeyError.
    reg.synthesize(
        "lookup",
        "def lookup(v):\n    return v['x']\n",
        parents=("x",),
        domain=(0, 1),
        sample_inputs=[{"x": 0}, {"x": 1}],
    )
    scm = FiniteSCM(domains={})
    reg.attach_to_scm(scm)
    # The wrapped equation handles missing keys by returning the domain's first value (0).
    out = scm.equations["lookup"].fn({"x": 1})
    assert out == 1
    out_missing = scm.equations["lookup"].fn({})
    assert out_missing == 0


def test_attach_to_scm_rejects_non_scm():
    reg = NativeToolRegistry(":memory:", namespace="t")
    with pytest.raises(TypeError):
        reg.attach_to_scm(object())


# ---------------------------------------------------------------------------
# End-to-end through BrocaMind-style helpers (via direct registry; no LLM needed)
# ---------------------------------------------------------------------------


def test_full_synthesis_pipeline_describes_real_dependency(tmp_path):
    """Synthesize a tool, register it on a fresh SCM, run the do-calculus."""

    db = tmp_path / "tools.sqlite"
    reg = NativeToolRegistry(db, namespace="weather_lab")

    # Tool: indoor humidity is high when the door is closed.
    tool = reg.synthesize(
        "humidity_high",
        "def humidity_high(v):\n    return 1 if v['door_closed'] == 1 else 0\n",
        parents=("door_closed",),
        domain=(0, 1),
        sample_inputs=[{"door_closed": 0}, {"door_closed": 1}],
        description="indoor humidity follows door state",
    )
    assert tool.verified

    scm = FiniteSCM(domains={})
    reg.attach_to_scm(scm)

    # Open door → humidity 0; closed door → humidity 1.
    p_high_closed = scm.probability({"humidity_high": 1}, interventions={"door_closed": 1})
    p_high_open = scm.probability({"humidity_high": 1}, interventions={"door_closed": 0})
    assert p_high_closed == 1.0
    assert p_high_open == 0.0
    # ATE is the full effect.
    assert (p_high_closed - p_high_open) == 1.0
