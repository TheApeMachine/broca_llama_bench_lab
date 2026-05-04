"""Stability diagnostics for categorical PC causal discovery."""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Mapping, Sequence

from ..causal.causal_discovery import pc_algorithm


@dataclass(frozen=True)
class EdgeStability:
    """Bootstrap frequency for one discovered adjacency/orientation."""

    edge: tuple[str, str]
    kind: str
    frequency: float

    def as_dict(self) -> dict[str, object]:
        return {"edge": list(self.edge), "kind": self.kind, "frequency": self.frequency}


@dataclass(frozen=True)
class CausalDiscoveryStabilityReport:
    """How stable PC-discovered edges are under row resampling."""

    n_rows: int
    n_bootstrap: int
    variables: tuple[str, ...]
    edges: tuple[EdgeStability, ...] = field(default_factory=tuple)
    warnings: tuple[str, ...] = field(default_factory=tuple)

    @property
    def status(self) -> str:
        if self.warnings:
            return "warn"
        weak = [edge for edge in self.edges if edge.frequency < 0.5]
        return "unstable" if weak else "pass"

    def as_dict(self) -> dict[str, object]:
        return {
            "n_rows": self.n_rows,
            "n_bootstrap": self.n_bootstrap,
            "variables": list(self.variables),
            "edges": [edge.as_dict() for edge in self.edges],
            "warnings": list(self.warnings),
            "status": self.status,
        }


class CausalDiscoveryStability:
    """Bootstrap PC and report edge/orientation frequencies."""

    def evaluate(
        self,
        rows: Sequence[Mapping[str, object]],
        variables: Sequence[str] | None = None,
        *,
        n_bootstrap: int = 20,
        sample_fraction: float = 0.8,
        alpha: float = 0.05,
        max_conditioning_size: int | None = 2,
        seed: int = 0,
    ) -> CausalDiscoveryStabilityReport:
        row_list = [dict(row) for row in rows]
        vars_tuple = tuple(variables or sorted({str(k) for row in row_list for k in row}))
        warnings: list[str] = []
        if len(row_list) < max(8, 2 * len(vars_tuple)):
            warnings.append("too few rows for stable PC discovery; treat edges as hypotheses only")
        if len(vars_tuple) < 2:
            return CausalDiscoveryStabilityReport(len(row_list), 0, vars_tuple, warnings=tuple(warnings))
        rng = random.Random(seed)
        counts: dict[tuple[str, str, str], int] = {}
        n = max(1, int(n_bootstrap))
        sample_size = max(1, int(round(len(row_list) * max(0.05, min(1.0, sample_fraction)))))
        for _ in range(n):
            sample = [row_list[rng.randrange(len(row_list))] for _ in range(sample_size)]
            graph = pc_algorithm(sample, vars_tuple, alpha=alpha, max_conditioning_size=max_conditioning_size)
            for u, v in graph.directed_edges:
                counts[("directed", str(u), str(v))] = counts.get(("directed", str(u), str(v)), 0) + 1
            for edge in graph.undirected_edges:
                a, b = sorted(str(x) for x in edge)
                counts[("undirected", a, b)] = counts.get(("undirected", a, b), 0) + 1
        edges = tuple(
            EdgeStability(edge=(a, b), kind=kind, frequency=count / n)
            for (kind, a, b), count in sorted(counts.items())
        )
        return CausalDiscoveryStabilityReport(
            n_rows=len(row_list),
            n_bootstrap=n,
            variables=vars_tuple,
            edges=edges,
            warnings=tuple(warnings),
        )
