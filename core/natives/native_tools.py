"""Native Tool Synthesis — generated code as endogenous causal nodes.

When the substrate cannot answer a question with its existing causal model,
it synthesizes a new function (a "tool") that fills the gap.  Unlike
classical LLM tool-use — where the function lives outside the model and is
invoked once — a synthesized native tool is registered as an
``EndogenousEquation`` inside the :class:`FiniteSCM`, becoming a permanent
part of the substrate's mathematical world model.  The next time the
substrate runs ``do(·)`` interventions or counterfactuals, those tools are
queried as natively as any hand-coded structural equation.

The pipeline:

1.  **Synthesize.**  An LLM (or, in tests, a deterministic source string)
    produces Python source defining ``def fn(values: dict) -> object``.
2.  **Sandbox-compile.**  ``ToolSandbox`` compiles the source under a
    restricted ``__builtins__`` namespace and wraps it so the output is
    coerced into the declared domain.
3.  **Verify.**  ``ToolSandbox.verify`` runs the function on
    ``sample_inputs`` and checks every output lies in the declared domain.
    Optionally a dedicated split-conformal predictor on channel
    ``native_tool_output`` inspects the empirical label histogram from those
    runs — when calibration is warm and the conformal prediction set size is
    not exactly one, synthesis aborts before the tool gains SCM authority.
4.  **Persist.**  ``NativeToolRegistry`` writes the verified tool to
    SQLite.
5.  **Attach.**  ``NativeToolRegistry.attach_to_scm`` calls
    ``scm.add_endogenous(...)`` for every persisted tool, growing the SCM's
    structural graph at runtime. Attached equations carry an online conformal
    martingale; exchangeability drift quarantines the tool and exposes its SCM
    node as exogenous uncertainty.

Optional **Docker** execution (``BROCA_USE_DOCKER_TOOLS``) uses
:class:`core.system.sandbox.DockerToolSandbox`: tools run in ``python:3.11-slim``
with imports and the standard library available, still without host ``exec``.
"""

from __future__ import annotations

import ast
import inspect
import json
import logging
import os
import sqlite3
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from collections import Counter
from typing import Any, Callable, Mapping, Sequence

from ..calibration.conformal import ConformalPredictor, OnlineConformalMartingale


logger = logging.getLogger(__name__)


class ToolSynthesisError(Exception):
    """Raised when source compilation, sandboxing, or verification fails."""


# ---------------------------------------------------------------------------
# Sandbox
# ---------------------------------------------------------------------------


_SAFE_BUILTIN_NAMES: tuple[str, ...] = (
    "abs",
    "all",
    "any",
    "bool",
    "dict",
    "divmod",
    "enumerate",
    "filter",
    "float",
    "frozenset",
    "int",
    "isinstance",
    "issubclass",
    "len",
    "list",
    "map",
    "max",
    "min",
    "pow",
    "range",
    "reversed",
    "round",
    "set",
    "sorted",
    "str",
    "sum",
    "tuple",
    "zip",
)


def _build_safe_builtins() -> dict[str, Any]:
    """Construct the restricted ``__builtins__`` namespace exposed to synthesized tools."""

    safe: dict[str, Any] = {}
    import builtins as _builtins

    for name in _SAFE_BUILTIN_NAMES:
        if hasattr(_builtins, name):
            safe[name] = getattr(_builtins, name)
    return safe


class _ASTValidator(ast.NodeVisitor):
    """Reject obviously dangerous AST nodes before we compile.

    Synthesized tools are arithmetic / branching only.  We forbid imports,
    function attribute access on dunder names, ``exec``/``eval``, ``open``,
    ``__class__``, and other escape hatches.
    """

    _FORBIDDEN_NAMES = {
        "exec",
        "eval",
        "compile",
        "open",
        "input",
        "globals",
        "locals",
        "vars",
        "dir",
        "getattr",
        "setattr",
        "delattr",
        "__import__",
        "breakpoint",
        "help",
        "memoryview",
        "object",
        "super",
        "type",
    }

    def __init__(self) -> None:
        self.errors: list[str] = []

    def visit_Import(self, node: ast.Import) -> None:  # noqa: N802
        self.errors.append("import statements are not permitted in synthesized tools")

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:  # noqa: N802
        self.errors.append("import statements are not permitted in synthesized tools")

    def visit_Attribute(self, node: ast.Attribute) -> None:  # noqa: N802
        if isinstance(node.attr, str) and node.attr.startswith("__") and node.attr.endswith("__"):
            self.errors.append(f"dunder attribute access {node.attr!r} is not permitted")
        self.generic_visit(node)

    def visit_Subscript(self, node: ast.Subscript) -> None:  # noqa: N802
        sl = node.slice
        index_t = getattr(ast, "Index", None)
        if index_t is not None and isinstance(sl, index_t):  # type: ignore[arg-type]
            sl = getattr(sl, "value", sl)
        if isinstance(sl, ast.Constant) and isinstance(sl.value, str):
            nm = sl.value
            if nm.startswith("__") or nm.endswith("__"):
                self.errors.append(f"dunder attribute access {nm!r} is not permitted")
        self.generic_visit(node)

    def visit_JoinedStr(self, node: ast.JoinedStr) -> None:  # noqa: N802
        self.generic_visit(node)

    def visit_Name(self, node: ast.Name) -> None:  # noqa: N802
        if node.id in self._FORBIDDEN_NAMES:
            self.errors.append(f"name {node.id!r} is not permitted")
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:  # noqa: N802
        # Catch calls like ``getattr(x, "__class__")``.
        func = node.func
        if isinstance(func, ast.Name) and func.id in self._FORBIDDEN_NAMES:
            self.errors.append(f"call to forbidden builtin {func.id!r}")
        self.generic_visit(node)


@dataclass
class SandboxResult:
    fn: Callable[[Mapping[str, Any]], Any]
    source: str
    function_name: str


class ToolSandbox:
    """Compiles untrusted source into a callable suitable for use as an endogenous equation.

    The expected source defines a single top-level function whose name is
    passed as ``function_name`` and which has the signature
    ``def function_name(values: dict) -> object``.  All other top-level
    statements are forbidden.

    This sandbox only restricts namespaces and parses AST—it does **not**
    isolate CPU time or memory. Malicious **synchronous** code can still infinite-loop
    or allocate until the OS intervenes; do not rely on subprocess-level containment.
    """

    def __init__(self, *, max_source_chars: int = 4096):
        self.max_source_chars = int(max_source_chars)

    def compile(self, source: str, function_name: str) -> SandboxResult:
        if not isinstance(source, str):
            raise ToolSynthesisError("source must be a string")
        if len(source) > self.max_source_chars:
            raise ToolSynthesisError(
                f"source exceeds maximum length {self.max_source_chars}"
            )
        try:
            tree = ast.parse(source, mode="exec")
        except SyntaxError as exc:
            raise ToolSynthesisError(f"syntax error in tool source: {exc}") from exc

        if not tree.body:
            raise ToolSynthesisError("tool source is empty")

        validator = _ASTValidator()
        validator.visit(tree)
        if validator.errors:
            raise ToolSynthesisError("; ".join(validator.errors))

        # Allow only function defs / async defs and (optionally) module docstring at top level.
        defs: list[ast.FunctionDef | ast.AsyncFunctionDef] = []
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                defs.append(node)
            elif (
                isinstance(node, ast.Expr)
                and isinstance(node.value, ast.Constant)
                and isinstance(node.value.value, str)
            ):
                continue  # module docstring
            else:
                raise ToolSynthesisError(
                    "synthesized tool source may only contain function or async function definitions"
                )
        if not defs:
            raise ToolSynthesisError("no function definition found in tool source")
        target = next((d for d in defs if d.name == function_name), None)
        if target is None:
            raise ToolSynthesisError(
                f"expected a function named {function_name!r}; found {[d.name for d in defs]!r}"
            )

        safe_builtins = _build_safe_builtins()
        namespace: dict[str, Any] = {"__builtins__": safe_builtins}
        try:
            code = compile(tree, filename=f"<synthesized:{function_name}>", mode="exec")
            exec(code, namespace)
        except Exception as exc:
            raise ToolSynthesisError(f"failed to compile synthesized source: {exc}") from exc
        fn = namespace.get(function_name)
        if not callable(fn):
            raise ToolSynthesisError(f"compiled namespace has no callable {function_name!r}")
        if inspect.iscoroutinefunction(fn):
            raise ToolSynthesisError(
                f"compiled tool {function_name!r} is an async def; "
                "synthesized tools must be ordinary synchronous callables for verification and SCM use",
            )
        return SandboxResult(fn=fn, source=source, function_name=function_name)

    @staticmethod
    def verify(
        fn: Callable[[Mapping[str, Any]], Any],
        *,
        domain: Sequence[Any],
        sample_inputs: Sequence[Mapping[str, Any]],
    ) -> list[Any]:
        """Run ``fn`` on every ``sample_input`` and assert the output lies in ``domain``.

        Returns the list of produced outputs so the caller can inspect coverage.
        """

        if not sample_inputs:
            raise ToolSynthesisError("at least one sample input is required for verification")
        domain_elems = list(domain)
        try:
            domain_set = set(domain_elems)
        except TypeError as exc:
            bad: list[str] = []
            for elt in domain_elems:
                try:
                    hash(elt)
                except TypeError:
                    bad.append(f"{elt!r} ({type(elt).__name__})")
            detail = "; ".join(bad) if bad else repr(exc)
            raise ToolSynthesisError(
                f"domain elements must be hashable for membership checks ({detail})",
            ) from exc
        outputs: list[Any] = []
        for i, sample in enumerate(sample_inputs):
            try:
                out = fn(dict(sample))
            except Exception as exc:
                raise ToolSynthesisError(
                    f"tool raised on sample {i}: {sample} -> {exc!r}"
                ) from exc
            if out not in domain_set:
                raise ToolSynthesisError(
                    f"tool returned {out!r} for sample {i}; not in declared domain {domain_elems!r}"
                )
            outputs.append(out)
        return outputs


def native_tool_domain_label(value: Any) -> str:
    """Stable categorical key for conformal prediction over tool codomains."""

    if isinstance(value, bool):
        return f"bool:{str(value).lower()}"
    if value is None:
        return "none"
    if isinstance(value, (int, float)):
        return f"num:{repr(value)}"
    if isinstance(value, str):
        return f"str:{value}"
    try:
        return json.dumps(value, sort_keys=True, default=str)
    except TypeError:
        return repr(value)


def full_domain_empirical_distribution(
    domain: Sequence[Any], outputs: Sequence[Any]
) -> dict[str, float]:
    """Map every declared codomain element to its empirical frequency under verifier samples."""

    if not outputs:
        return {}
    counts = Counter(native_tool_domain_label(o) for o in outputs)
    n = float(len(outputs))
    dist: dict[str, float] = {}
    for v in domain:
        lab = native_tool_domain_label(v)
        dist[lab] = counts.get(lab, 0) / n
    return dist


def tool_output_nonconformity_scores(
    domain: Sequence[Any],
    calibration_outputs: Sequence[Any],
) -> tuple[list[float], dict[str, float]]:
    """Nonconformity scores induced by a tool's verifier output histogram."""

    dist = full_domain_empirical_distribution(domain, calibration_outputs)
    if not dist:
        raise ToolSynthesisError("tool drift monitor requires verifier outputs")
    scores = [
        1.0 - float(dist.get(native_tool_domain_label(output), 0.0))
        for output in calibration_outputs
    ]
    return scores, dist


def tool_output_nonconformity_score(value: Any, distribution: Mapping[str, float]) -> float:
    """Nonconformity score for one live tool output under the verifier law."""

    return 1.0 - float(distribution.get(native_tool_domain_label(value), 0.0))


def assert_singleton_conformal_for_tool_outputs(
    predictor: ConformalPredictor,
    domain: Sequence[Any],
    outputs: Sequence[Any],
) -> None:
    """Raise :class:`ToolSynthesisError` when conformal ambiguity is too high for SCM attachment.

    Builds an empirical law over the **full declared domain** — zeros on never-seen
    values — and requires a singleton conformal prediction set.  Until the
    predictor's calibration list reaches ``min_calibration`` this is a no-op so
    cold starts preserve legacy behaviour.
    """

    if len(predictor) < predictor.min_calibration:
        return
    dist = full_domain_empirical_distribution(domain, outputs)
    active = {k: v for k, v in dist.items() if v > 0.0}
    if not active:
        raise ToolSynthesisError(
            "conformal tool gate: empty verifier histogram after masking zero masses"
        )
    cset = predictor.predict_set(active)
    if cset.set_size != 1:
        raise ToolSynthesisError(
            "conformal tool gate: verifier behaviour is epistemically ambiguous "
            f"(prediction set size {cset.set_size}, labels={cset.labels!r}); "
            "refusing SCM attachment"
        )


def tool_sandbox_from_env() -> ToolSandbox:
    """Return :class:`DockerToolSandbox` when ``BROCA_USE_DOCKER_TOOLS`` is set, else in-process sandbox."""

    flag = os.environ.get("BROCA_USE_DOCKER_TOOLS", "").strip().lower()
    if flag in ("1", "true", "yes", "on"):
        from ..system.sandbox import DockerToolSandbox

        logger.info("native tools: DockerToolSandbox enabled via BROCA_USE_DOCKER_TOOLS")
        return DockerToolSandbox()
    return ToolSandbox()


# ---------------------------------------------------------------------------
# Native tool registry
# ---------------------------------------------------------------------------


@dataclass
class NativeTool:
    """A verified, persisted native tool — one node in the substrate's causal graph.

    ``rehydrated`` is True once a live sandbox callable exists in this process
    (successful ``registry.get(..., rehydrate=True)`` or fresh ``synthesize``).
    Rehydrate failures clear ``verified`` so ``verified`` does not falsely imply ``fn``.
    """

    name: str
    source: str
    function_name: str
    parents: tuple[str, ...]
    domain: tuple
    fn: Callable[[Mapping[str, Any]], Any] | None = None
    verified: bool = False
    rehydrated: bool = False
    sample_inputs: tuple[dict, ...] = field(default_factory=tuple)
    sample_outputs: tuple = field(default_factory=tuple)
    description: str = ""
    created_at: float = 0.0
    id: int | None = None

    def callable_or_raise(self) -> Callable[[Mapping[str, Any]], Any]:
        if self.fn is None:
            raise ToolSynthesisError(
                f"NativeTool {self.name!r} has no compiled callable; call rehydrate()"
            )
        return self.fn

    def domain_coerce(self, value: Any) -> Any:
        if value in self.domain:
            return value
        raise ToolSynthesisError(
            f"tool {self.name!r} produced {value!r} outside domain {self.domain!r}"
        )


class NativeToolRegistry:
    """Persistent store of synthesized tools that can be rehydrated into an SCM.

    Survives the lifetime of a process: a tool synthesized today is still a
    causal node tomorrow.  Tools are keyed by ``(namespace, name)``; the
    registry refuses to overwrite a verified tool unless the caller passes
    ``overwrite=True``.
    """

    def __init__(
        self,
        path: str | Path,
        *,
        namespace: str = "main",
        sandbox: ToolSandbox | None = None,
    ):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.namespace = str(namespace)
        self.sandbox = sandbox if sandbox is not None else tool_sandbox_from_env()
        self._db_lock = threading.RLock()
        self._conn: sqlite3.Connection | None = None
        self._init_schema()

    def _init_schema(self) -> None:
        with self._db_lock:
            con = self._lazy_open()
            con.execute(
                """
                CREATE TABLE IF NOT EXISTS native_tools (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    namespace TEXT NOT NULL,
                    name TEXT NOT NULL,
                    source TEXT NOT NULL,
                    function_name TEXT NOT NULL,
                    parents_json TEXT NOT NULL,
                    domain_json TEXT NOT NULL,
                    sample_inputs_json TEXT NOT NULL,
                    sample_outputs_json TEXT NOT NULL,
                    description TEXT NOT NULL,
                    verified INTEGER NOT NULL,
                    created_at REAL NOT NULL,
                    UNIQUE(namespace, name)
                )
                """
            )
            con.execute(
                "CREATE INDEX IF NOT EXISTS idx_native_tools_namespace ON native_tools(namespace)"
            )

    def _lazy_open(self) -> sqlite3.Connection:
        """Open the shared SQLite connection once (caller holds ``self._db_lock``)."""

        if self._conn is None:
            # DMN / background workers call count()/list from non-main threads; same pattern as
            # PersistentSemanticMemory in broca.py (check_same_thread=False + lock).
            self._conn = sqlite3.connect(self.path, timeout=5.0, check_same_thread=False)
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.isolation_level = None
        return self._conn

    def synthesize(
        self,
        name: str,
        source: str,
        *,
        function_name: str | None = None,
        parents: Sequence[str],
        domain: Sequence[Any],
        sample_inputs: Sequence[Mapping[str, Any]],
        description: str = "",
        overwrite: bool = False,
        conformal_predictor: ConformalPredictor | None = None,
    ) -> NativeTool:
        """Compile, sandbox, verify, and persist a native tool in one shot.

        Raises :class:`ToolSynthesisError` if any stage fails.
        """

        fn_name = function_name or name
        compiled = self.sandbox.compile(source, fn_name)
        domain_t = tuple(domain)
        if not domain_t:
            raise ToolSynthesisError(
                f"native tool {name!r}: domain must declare at least one allowed output "
                "(empty domains are ambiguous for SCM verification)",
            )
        # Verify produces outputs as a side-effect; we keep the recorded outputs for telemetry.
        outputs = self.sandbox.verify(
            compiled.fn, domain=domain_t, sample_inputs=list(sample_inputs)
        )
        if conformal_predictor is not None:
            assert_singleton_conformal_for_tool_outputs(conformal_predictor, domain_t, outputs)
        tool = NativeTool(
            name=name,
            source=source,
            function_name=fn_name,
            parents=tuple(parents),
            domain=domain_t,
            fn=compiled.fn,
            verified=True,
            rehydrated=True,
            sample_inputs=tuple(dict(s) for s in sample_inputs),
            sample_outputs=tuple(outputs),
            description=description,
            created_at=time.time(),
        )
        existing = self.get(name)
        if existing is not None and existing.verified and not overwrite:
            raise ToolSynthesisError(
                f"tool {name!r} already exists in namespace {self.namespace!r}; pass overwrite=True"
            )
        self._upsert_row(tool)
        logger.info(
            "NativeToolRegistry.synthesize: ns=%s name=%s parents=%s domain=%s description=%r",
            self.namespace,
            name,
            list(tool.parents),
            list(tool.domain),
            description,
        )
        return tool

    def _upsert_row(self, tool: NativeTool) -> None:
        domain_repr = self._serialize_domain(tool.domain)
        sample_inputs_repr = self._serialize_samples(tool.sample_inputs)
        sample_outputs_repr = self._serialize_outputs(tool.sample_outputs)
        parents_json = json.dumps(list(tool.parents))
        created_at_f = float(tool.created_at or time.time())
        with self._db_lock:
            con = self._lazy_open()
            row = con.execute(
                """
                INSERT INTO native_tools(namespace, name, source, function_name, parents_json,
                    domain_json, sample_inputs_json, sample_outputs_json, description, verified, created_at)
                VALUES (?,?,?,?,?,?,?,?,?,?,?)
                ON CONFLICT(namespace, name) DO UPDATE SET
                    source=excluded.source,
                    function_name=excluded.function_name,
                    parents_json=excluded.parents_json,
                    domain_json=excluded.domain_json,
                    sample_inputs_json=excluded.sample_inputs_json,
                    sample_outputs_json=excluded.sample_outputs_json,
                    description=excluded.description,
                    verified=excluded.verified
                RETURNING id
                """,
                (
                    self.namespace,
                    tool.name,
                    tool.source,
                    tool.function_name,
                    parents_json,
                    domain_repr,
                    sample_inputs_repr,
                    sample_outputs_repr,
                    tool.description,
                    int(bool(tool.verified)),
                    created_at_f,
                ),
            ).fetchone()
            if row is None:
                raise ToolSynthesisError(
                    f"native tool upsert produced no RETURNING row for namespace={self.namespace!r}, "
                    f"name={tool.name!r}",
                )
            tool.id = int(row[0])

    @staticmethod
    def _serialize_domain(domain: Sequence[Any]) -> str:
        return json.dumps(
            [{"type": type(v).__name__, "repr": v} for v in domain], default=str
        )

    @staticmethod
    def _deserialize_domain(blob: str) -> tuple:
        items = json.loads(blob)
        out: list[Any] = []
        for entry in items:
            tname = entry.get("type")
            v = entry.get("repr")
            if tname == "int":
                out.append(int(v))
            elif tname == "float":
                out.append(float(v))
            elif tname == "bool":
                # JSON/manual edits rarely preserve strict bool types; normalize before SCM checks / hashing.
                if isinstance(v, bool):
                    bv = v
                elif isinstance(v, str) and v in ("true", "True", "false", "False"):
                    bv = v in ("true", "True")
                elif isinstance(v, int):
                    bv = bool(v)
                elif isinstance(v, str):
                    try:
                        iv = int(v)
                    except ValueError as ive:
                        raise ToolSynthesisError(
                            f"cannot coerce serialized bool payload {v!r} ({type(v).__name__}); "
                            f"non-numeric string for int coercion"
                        ) from ive
                    bv = bool(iv)
                else:
                    raise ToolSynthesisError(
                        f"cannot coerce serialized bool payload {v!r} (got {type(v).__name__})"
                    )
                out.append(bv)
            elif tname == "NoneType":
                out.append(None)
            else:
                out.append(v)
        return tuple(out)

    @staticmethod
    def _serialize_samples(samples: Sequence[Mapping[str, Any]]) -> str:
        return json.dumps([dict(s) for s in samples], default=str)

    @staticmethod
    def _deserialize_samples(blob: str) -> tuple[dict, ...]:
        return tuple(dict(s) for s in json.loads(blob))

    @staticmethod
    def _serialize_outputs(outputs: Sequence[Any]) -> str:
        return json.dumps([{"type": type(v).__name__, "repr": v} for v in outputs], default=str)

    @staticmethod
    def _deserialize_outputs(blob: str) -> tuple:
        return NativeToolRegistry._deserialize_domain(blob)

    def _row_to_tool(self, row: tuple, *, rehydrate: bool) -> NativeTool:
        (
            rid,
            _ns,
            name,
            source,
            function_name,
            parents_json,
            domain_json,
            sample_inputs_json,
            sample_outputs_json,
            description,
            verified,
            created_at,
        ) = row
        parents = tuple(json.loads(parents_json))
        domain = self._deserialize_domain(domain_json)
        sample_inputs = self._deserialize_samples(sample_inputs_json)
        sample_outputs = self._deserialize_outputs(sample_outputs_json)
        tool = NativeTool(
            id=int(rid),
            name=str(name),
            source=str(source),
            function_name=str(function_name),
            parents=parents,
            domain=domain,
            fn=None,
            verified=bool(int(verified)),
            rehydrated=False,
            sample_inputs=sample_inputs,
            sample_outputs=sample_outputs,
            description=str(description),
            created_at=float(created_at),
        )
        if rehydrate:
            try:
                compiled = self.sandbox.compile(tool.source, tool.function_name)
                tool.fn = compiled.fn
                tool.rehydrated = True
            except ToolSynthesisError:
                logger.exception(
                    "NativeToolRegistry: failed to rehydrate tool %s; leaving fn=None",
                    tool.name,
                )
                tool.fn = None
                tool.rehydrated = False
                tool.verified = False
        return tool

    def get(self, name: str, *, rehydrate: bool = True) -> NativeTool | None:
        with self._db_lock:
            con = self._lazy_open()
            row = con.execute(
                """
                SELECT id, namespace, name, source, function_name, parents_json, domain_json,
                       sample_inputs_json, sample_outputs_json, description, verified, created_at
                FROM native_tools WHERE namespace=? AND name=?
                """,
                (self.namespace, name),
            ).fetchone()
        return self._row_to_tool(row, rehydrate=rehydrate) if row is not None else None

    def all_tools(self, *, rehydrate: bool = True) -> list[NativeTool]:
        with self._db_lock:
            con = self._lazy_open()
            rows = con.execute(
                """
                SELECT id, namespace, name, source, function_name, parents_json, domain_json,
                       sample_inputs_json, sample_outputs_json, description, verified, created_at
                FROM native_tools WHERE namespace=? ORDER BY id ASC
                """,
                (self.namespace,),
            ).fetchall()
        return [self._row_to_tool(row, rehydrate=rehydrate) for row in rows]

    def remove(self, name: str) -> bool:
        with self._db_lock:
            con = self._lazy_open()
            cur = con.execute(
                "DELETE FROM native_tools WHERE namespace=? AND name=?",
                (self.namespace, name),
            )
        return int(cur.rowcount or 0) > 0

    def mark_unverified(self, name: str, *, reason: str, evidence: Mapping[str, Any]) -> None:
        with self._db_lock:
            con = self._lazy_open()
            cur = con.execute(
                """
                UPDATE native_tools
                SET verified=0, description=description || ?
                WHERE namespace=? AND name=?
                """,
                (
                    "\n[quarantined] " + json.dumps(
                        {"reason": str(reason), "evidence": dict(evidence)},
                        sort_keys=True,
                        default=str,
                    ),
                    self.namespace,
                    str(name),
                ),
            )
        if int(cur.rowcount or 0) <= 0:
            raise ToolSynthesisError(f"cannot quarantine unknown native tool {name!r}")

    def count(self) -> int:
        with self._db_lock:
            con = self._lazy_open()
            row = con.execute(
                "SELECT COUNT(*) FROM native_tools WHERE namespace=?",
                (self.namespace,),
            ).fetchone()
        return int(row[0]) if row else 0

    # ----------------------- SCM integration -----------------------

    def attach_to_scm(
        self,
        scm,
        *,
        allow_unknown_parents: bool = True,
        strict_tool_wrappers: bool = False,
        on_tool_drift: Callable[[NativeTool, Mapping[str, Any]], None] | None = None,
    ) -> int:
        """Register every verified tool as an endogenous equation on ``scm``.

        Tools whose parents reference variables not yet declared on the SCM
        are skipped unless ``allow_unknown_parents=True`` — in that case the
        missing parents are auto-declared as exogenous binary variables so
        the SCM remains evaluable.

        Returns the number of tools attached.
        """

        from ..causal import FiniteSCM

        if not isinstance(scm, FiniteSCM):
            raise TypeError("attach_to_scm: scm must be a FiniteSCM")

        attached = 0
        for tool in self.all_tools(rehydrate=True):
            if not tool.verified or tool.fn is None:
                continue
            if tool.name in scm.equations:
                scm.update_endogenous(
                    tool.name,
                    fn=self._wrap_for_scm(
                        tool,
                        scm=scm,
                        registry=self,
                        strict=strict_tool_wrappers,
                        on_tool_drift=on_tool_drift,
                    ),
                    domain=list(tool.domain),
                    parents=tuple(tool.parents),
                )
                attached += 1
                continue

            missing = [p for p in tool.parents if p not in scm.domains]
            if missing and not allow_unknown_parents:
                logger.debug(
                    "NativeToolRegistry.attach_to_scm: skipping %s; missing parents=%s",
                    tool.name,
                    missing,
                )
                continue
            for p in missing:
                # Declare the missing parent as endogenous so Pearl-style do(p=v)
                # interventions actually rewrite its structural equation. Each
                # endogenous parent is a pass-through of its own dedicated
                # exogenous noise variable, so the auto-declaration looks just
                # like an ordinary binary variable from the SCM's perspective.
                noise = f"U_{p}"
                if noise not in scm.exogenous:
                    scm.add_exogenous(noise, [0, 1], {0: 0.5, 1: 0.5})
                if p not in scm.equations:
                    scm.add_endogenous(p, [0, 1], [noise], (lambda noise=noise: lambda v: v[noise])())
                logger.debug(
                    "NativeToolRegistry.attach_to_scm: auto-declared endogenous parent %s for %s (noise=%s)",
                    p,
                    tool.name,
                    noise,
                )
            scm.add_endogenous(
                tool.name,
                list(tool.domain),
                list(tool.parents),
                self._wrap_for_scm(
                    tool,
                    scm=scm,
                    registry=self,
                    strict=strict_tool_wrappers,
                    on_tool_drift=on_tool_drift,
                ),
            )
            attached += 1
            logger.info(
                "NativeToolRegistry.attach_to_scm: attached %s parents=%s domain=%s",
                tool.name,
                list(tool.parents),
                list(tool.domain),
            )
        return attached

    @staticmethod
    def _wrap_for_scm(
        tool: NativeTool,
        *,
        scm,
        registry: "NativeToolRegistry",
        strict: bool = False,
        on_tool_drift: Callable[[NativeTool, Mapping[str, Any]], None] | None = None,
    ) -> Callable[[dict], Any]:
        """Wrap ``tool.fn`` for SCM queries with live conformal drift quarantine."""

        if not tool.domain:
            raise ToolSynthesisError(
                f"native tool {tool.name!r} has empty domain; cannot synthesize SCM wrapper"
            )
        name = tool.name
        fn = tool.callable_or_raise()
        calibration_scores, verifier_distribution = tool_output_nonconformity_scores(
            tool.domain,
            tool.sample_outputs,
        )
        monitor = OnlineConformalMartingale(
            calibration_scores,
            alpha=1.0 / float(len(calibration_scores) + 1),
        )

        def quarantine(reason: str, evidence: Mapping[str, Any]) -> None:
            payload = {
                "tool": name,
                "reason": str(reason),
                "verifier_distribution": dict(verifier_distribution),
                **dict(evidence),
            }
            try:
                scm.detach_endogenous_as_exogenous(name)
            except ValueError:
                logger.debug("NativeTool %s already detached from SCM", name)
            registry.mark_unverified(name, reason=reason, evidence=payload)
            if on_tool_drift is not None:
                on_tool_drift(tool, payload)

        def _wrapped(values: dict) -> Any:
            try:
                out = fn(values)
            except Exception as exc:
                quarantine("runtime_exception", {"error": repr(exc)})
                raise ToolSynthesisError(
                    f"native tool {name!r} raised during SCM evaluation"
                ) from exc
            try:
                coerced = tool.domain_coerce(out)
            except ToolSynthesisError as exc:
                quarantine("out_of_domain_output", {"output": repr(out)})
                raise exc
            score = tool_output_nonconformity_score(coerced, verifier_distribution)
            drift = monitor.update(score)
            if bool(drift["drifted"]):
                quarantine("conformal_drift", {"output": repr(coerced), **drift})
                raise ToolSynthesisError(
                    f"native tool {name!r} violated live conformal martingale "
                    f"(p={float(drift['p_value']):.6f}, M={float(drift['martingale']):.6f})"
                )
            return coerced

        return _wrapped


__all__ = [
    "ToolSynthesisError",
    "ToolSandbox",
    "SandboxResult",
    "tool_sandbox_from_env",
    "NativeTool",
    "NativeToolRegistry",
    "native_tool_domain_label",
    "full_domain_empirical_distribution",
    "tool_output_nonconformity_score",
    "tool_output_nonconformity_scores",
    "assert_singleton_conformal_for_tool_outputs",
]
