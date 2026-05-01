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
4.  **Persist.**  ``NativeToolRegistry`` writes the verified tool to
    SQLite.
5.  **Attach.**  ``NativeToolRegistry.attach_to_scm`` calls
    ``scm.add_endogenous(...)`` for every persisted tool, growing the SCM's
    structural graph at runtime.

The sandbox is intentionally restrictive — no imports, no I/O, no
attribute access on disallowed objects — because the substrate may
synthesize tools from inputs (LLM completions) it does not fully trust.
"""

from __future__ import annotations

import ast
import json
import logging
import sqlite3
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence


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
    "True",
    "False",
    "None",
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

        # Allow only function defs and (optionally) module docstring at the top level.
        if not tree.body:
            raise ToolSynthesisError("tool source is empty")
        defs: list[ast.FunctionDef] = []
        for node in tree.body:
            if isinstance(node, ast.FunctionDef):
                defs.append(node)
            elif (
                isinstance(node, ast.Expr)
                and isinstance(node.value, ast.Constant)
                and isinstance(node.value.value, str)
            ):
                continue  # module docstring
            else:
                raise ToolSynthesisError(
                    "synthesized tool source may only contain function definitions"
                )
        if not defs:
            raise ToolSynthesisError("no function definition found in tool source")
        target = next((d for d in defs if d.name == function_name), None)
        if target is None:
            raise ToolSynthesisError(
                f"expected a function named {function_name!r}; found {[d.name for d in defs]!r}"
            )
        validator = _ASTValidator()
        validator.visit(tree)
        if validator.errors:
            raise ToolSynthesisError("; ".join(validator.errors))

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
        domain_set = list(domain)
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
                    f"tool returned {out!r} for sample {i}; not in declared domain {domain_set!r}"
                )
            outputs.append(out)
        return outputs


# ---------------------------------------------------------------------------
# Native tool registry
# ---------------------------------------------------------------------------


@dataclass
class NativeTool:
    """A verified, persisted native tool — one node in the substrate's causal graph."""

    name: str
    source: str
    function_name: str
    parents: tuple[str, ...]
    domain: tuple
    fn: Callable[[Mapping[str, Any]], Any] | None = None
    verified: bool = False
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
        self.sandbox = sandbox or ToolSandbox()
        self._init_schema()

    def _connect(self) -> sqlite3.Connection:
        con = sqlite3.connect(self.path)
        con.execute("PRAGMA journal_mode=WAL")
        return con

    def _init_schema(self) -> None:
        with self._connect() as con:
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
    ) -> NativeTool:
        """Compile, sandbox, verify, and persist a native tool in one shot.

        Raises :class:`ToolSynthesisError` if any stage fails.
        """

        fn_name = function_name or name
        compiled = self.sandbox.compile(source, fn_name)
        domain_t = tuple(domain)
        # Verify produces outputs as a side-effect; we keep the recorded outputs for telemetry.
        outputs = self.sandbox.verify(
            compiled.fn, domain=domain_t, sample_inputs=list(sample_inputs)
        )
        tool = NativeTool(
            name=name,
            source=source,
            function_name=fn_name,
            parents=tuple(parents),
            domain=domain_t,
            fn=compiled.fn,
            verified=True,
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
        with self._connect() as con:
            row = con.execute(
                "SELECT id FROM native_tools WHERE namespace=? AND name=?",
                (self.namespace, tool.name),
            ).fetchone()
            if row is None:
                cur = con.execute(
                    """
                    INSERT INTO native_tools(namespace, name, source, function_name, parents_json,
                        domain_json, sample_inputs_json, sample_outputs_json, description, verified, created_at)
                    VALUES (?,?,?,?,?,?,?,?,?,?,?)
                    """,
                    (
                        self.namespace,
                        tool.name,
                        tool.source,
                        tool.function_name,
                        json.dumps(list(tool.parents)),
                        domain_repr,
                        sample_inputs_repr,
                        sample_outputs_repr,
                        tool.description,
                        int(bool(tool.verified)),
                        float(tool.created_at or time.time()),
                    ),
                )
                tool.id = int(cur.lastrowid)
            else:
                tool.id = int(row[0])
                con.execute(
                    """
                    UPDATE native_tools SET source=?, function_name=?, parents_json=?,
                        domain_json=?, sample_inputs_json=?, sample_outputs_json=?,
                        description=?, verified=? WHERE id=?
                    """,
                    (
                        tool.source,
                        tool.function_name,
                        json.dumps(list(tool.parents)),
                        domain_repr,
                        sample_inputs_repr,
                        sample_outputs_repr,
                        tool.description,
                        int(bool(tool.verified)),
                        tool.id,
                    ),
                )

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
                out.append(bool(v))
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
            sample_inputs=sample_inputs,
            sample_outputs=sample_outputs,
            description=str(description),
            created_at=float(created_at),
        )
        if rehydrate:
            try:
                compiled = self.sandbox.compile(tool.source, tool.function_name)
                tool.fn = compiled.fn
            except ToolSynthesisError:
                logger.exception(
                    "NativeToolRegistry: failed to rehydrate tool %s; leaving fn=None",
                    tool.name,
                )
        return tool

    def get(self, name: str, *, rehydrate: bool = True) -> NativeTool | None:
        with self._connect() as con:
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
        with self._connect() as con:
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
        with self._connect() as con:
            cur = con.execute(
                "DELETE FROM native_tools WHERE namespace=? AND name=?",
                (self.namespace, name),
            )
        return int(cur.rowcount or 0) > 0

    def count(self) -> int:
        with self._connect() as con:
            row = con.execute(
                "SELECT COUNT(*) FROM native_tools WHERE namespace=?",
                (self.namespace,),
            ).fetchone()
        return int(row[0]) if row else 0

    # ----------------------- SCM integration -----------------------

    def attach_to_scm(self, scm, *, allow_unknown_parents: bool = True) -> int:
        """Register every verified tool as an endogenous equation on ``scm``.

        Tools whose parents reference variables not yet declared on the SCM
        are skipped unless ``allow_unknown_parents=True`` — in that case the
        missing parents are auto-declared as exogenous binary variables so
        the SCM remains evaluable.

        Returns the number of tools attached.
        """

        from .causal import FiniteSCM

        if not isinstance(scm, FiniteSCM):
            raise TypeError("attach_to_scm: scm must be a FiniteSCM")

        attached = 0
        for tool in self.all_tools(rehydrate=True):
            if not tool.verified or tool.fn is None:
                continue
            if tool.name in scm.equations:
                # Already attached; refresh the function pointer so a re-synthesized tool wins.
                scm.equations[tool.name] = scm.equations[tool.name].__class__(
                    name=tool.name, parents=tuple(tool.parents), fn=self._wrap_for_scm(tool)
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
                self._wrap_for_scm(tool),
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
    def _wrap_for_scm(tool: NativeTool) -> Callable[[dict], Any]:
        """Wrap a tool's ``fn`` so the SCM gets a deterministic, domain-bound callable.

        Any exception inside the synthesized function is converted into the
        domain's first value — a "safe default" that keeps SCM evaluation
        non-fatal.  Failures are still logged so they show up in DMN
        telemetry.
        """

        fallback = tool.domain[0] if tool.domain else 0
        domain_set = set(tool.domain)
        name = tool.name
        fn = tool.callable_or_raise()

        def _wrapped(values: dict) -> Any:
            try:
                out = fn(values)
            except Exception:
                logger.exception("NativeTool %s raised; using fallback %r", name, fallback)
                return fallback
            if out not in domain_set:
                logger.warning(
                    "NativeTool %s produced %r outside domain %r; using fallback %r",
                    name,
                    out,
                    tool.domain,
                    fallback,
                )
                return fallback
            return out

        return _wrapped


__all__ = [
    "ToolSynthesisError",
    "ToolSandbox",
    "SandboxResult",
    "NativeTool",
    "NativeToolRegistry",
]
