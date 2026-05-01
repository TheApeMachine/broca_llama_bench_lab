"""Constraint-based causal discovery (PC algorithm) for the substrate.

Given a stream of joint observations of categorical variables, the PC
algorithm (Spirtes & Glymour, 1991) recovers the equivalence class of the
underlying directed acyclic graph by iteratively removing edges whose
endpoints are conditionally independent given some subset of their neighbors,
then orienting v-structures.

This is the algorithm the substrate needs to *grow* its SCM autonomously: every
time the user discusses a new domain, the DMN can re-run PC over the
observations stored in semantic memory and replace ``mind.scm`` with a freshly
discovered SCM whose structure matches the user's actual data.

Implementation notes:

* Conditional independence is tested with the G² statistic (likelihood-ratio
  chi-square) on contingency tables — the standard CI test for categorical
  data and the one most consistent with PC's asymptotic guarantees.
* Variables are inferred from the observation dataframe. The user supplies
  observations as ``list[dict[str, object]]``; categorical levels are derived.
* The orientation phase implements both v-structure detection and Meek's three
  orientation rules (R1-R3) to make as much of the CPDAG directed as possible.
* The output is convertible to a :class:`FiniteSCM` via :func:`build_scm_from_skeleton`,
  with conditional probability tables fitted from the observation data.
"""

from __future__ import annotations

import logging
import math
from collections import Counter
from dataclasses import dataclass, field
from itertools import combinations
from typing import Iterable, Mapping, Sequence

from .causal import FiniteSCM

logger = logging.getLogger(__name__)

# Resolution for inverse-CDF sampling of discrete exogenous U_v in fitted SCMs.
SCM_EXOGENOUS_DOMAIN_SIZE = 10_000


@dataclass
class DiscoveredGraph:
    """Output of the PC algorithm."""

    variables: list[str]
    domains: dict[str, tuple]
    undirected_edges: set[frozenset]
    directed_edges: set[tuple[str, str]]
    separating_sets: dict[frozenset, frozenset] = field(default_factory=dict)

    def __post_init__(self) -> None:
        p: dict[str, list[str]] = {v: [] for v in self.variables}
        c: dict[str, list[str]] = {v: [] for v in self.variables}
        for u, w in self.directed_edges:
            if w in p:
                p[w].append(u)
            if u in c:
                c[u].append(w)
        self._parents_of: dict[str, tuple[str, ...]] = {v: tuple(sorted(set(p[v]))) for v in self.variables}
        self._children_of: dict[str, tuple[str, ...]] = {v: tuple(sorted(set(c[v]))) for v in self.variables}

    def parents(self, v: str) -> list[str]:
        return list(self._parents_of.get(v, ()))

    def children(self, v: str) -> list[str]:
        return list(self._children_of.get(v, ()))


def _g_squared_independence(
    rows: Sequence[Mapping[str, object]],
    x: str,
    y: str,
    z: Sequence[str],
    *,
    alpha: float = 0.05,
) -> tuple[bool, float, float]:
    """Test ``X ⊥ Y | Z`` with the G² likelihood-ratio statistic.

    Returns ``(independent, g_squared, p_value_approx)``. The chi² CDF used
    here is the survival function via the regularized upper incomplete gamma
    function — implemented by hand so the module has no scipy dependency.
    """

    if not rows:
        return True, 0.0, 1.0
    n = len(rows)
    z_vals = list(z)

    def cell_counter(keys: Sequence[str]) -> Counter:
        c: Counter = Counter()
        for r in rows:
            try:
                c[tuple(r[k] for k in keys)] += 1
            except KeyError:
                continue
        return c

    n_xyz = cell_counter([x, y] + z_vals)
    n_xz = cell_counter([x] + z_vals)
    n_yz = cell_counter([y] + z_vals)
    n_z = cell_counter(z_vals)

    g = 0.0
    for (xv, yv, *zvals), nxyz in n_xyz.items():
        zv = tuple(zvals)
        nxz = n_xz[(xv,) + zv]
        nyz = n_yz[(yv,) + zv]
        nz = n_z[zv] if zvals else n
        if nxz == 0 or nyz == 0 or nz == 0:
            continue
        expected = nxz * nyz / nz
        if expected <= 0.0:
            continue
        g += 2.0 * nxyz * math.log(nxyz / expected)

    x_levels = len({r[x] for r in rows if x in r})
    y_levels = len({r[y] for r in rows if y in r})
    df_per_z = max(0, (x_levels - 1) * (y_levels - 1))
    if z_vals:
        df_z_count = 1
        for zvar in z_vals:
            df_z_count *= len({r[zvar] for r in rows if zvar in r})
        df_z_count = max(1, df_z_count)
    else:
        df_z_count = 1
    df = df_per_z * df_z_count
    p = _chi2_sf(g, df) if df > 0 else 1.0
    independent = bool(p >= alpha)
    return independent, float(g), float(p)


def _chi2_sf(stat: float, df: int) -> float:
    """Survival function (1 - CDF) of the chi-square distribution.

    Implemented via the regularized upper incomplete gamma function ``Q(s, x)``
    so the module has no dependency on scipy. Numerically stable for the small
    statistic / df values that PC's CI test produces.
    """

    if stat <= 0.0:
        return 1.0
    if df <= 0:
        return 1.0
    s = df / 2.0
    x = stat / 2.0
    # Q(s, x) computed via either continued fraction (x > s + 1) or series.
    if x < s + 1.0:
        # Series expansion of P(s, x), then 1 - P.
        term = 1.0 / s
        total = term
        for k in range(1, 200):
            term *= x / (s + k)
            total += term
            if term < 1e-12 * abs(total):
                break
        p = total * math.exp(-x + s * math.log(max(x, 1e-300)) - math.lgamma(s))
        return float(max(0.0, 1.0 - p))
    # Lentz's algorithm for continued fraction of Q.
    fpmin = 1e-300
    b = x + 1.0 - s
    c = 1.0 / fpmin
    d = 1.0 / b
    h = d
    for i in range(1, 200):
        an = -i * (i - s)
        b += 2.0
        d = an * d + b
        if abs(d) < fpmin:
            d = fpmin
        c = b + an / c
        if abs(c) < fpmin:
            c = fpmin
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < 1e-12:
            break
    q = h * math.exp(-x + s * math.log(max(x, 1e-300)) - math.lgamma(s))
    return float(max(0.0, q))


def _directed_parents_to(directed: set[tuple[str, str]], v: str) -> set[str]:
    return {u for (u, w) in directed if w == v}


def _meek_try_rule1(
    variables: Sequence[str],
    edges: set[frozenset],
    directed: set[tuple[str, str]],
    u: str,
    v: str,
    edge: frozenset,
) -> bool:
    for w in variables:
        if w in (u, v):
            continue
        if (w, u) in directed and frozenset((w, v)) not in edges and (w, v) not in directed and (v, w) not in directed:
            if (u, v) not in directed and (v, u) not in directed:
                directed.add((u, v))
                edges.discard(edge)
                logger.info("pc_algorithm.orient.R1: %s -> %s (chain via %s)", u, v, w)
                return True
    return False


def _meek_try_rule2(
    variables: Sequence[str],
    edges: set[frozenset],
    directed: set[tuple[str, str]],
    u: str,
    v: str,
    edge: frozenset,
) -> bool:
    for w in variables:
        if w in (u, v):
            continue
        if (u, w) in directed and (w, v) in directed:
            directed.add((u, v))
            edges.discard(edge)
            logger.info("pc_algorithm.orient.R2: %s -> %s (path via %s)", u, v, w)
            return True
    return False


def _meek_try_rule3(
    variables: Sequence[str],
    edges: set[frozenset],
    directed: set[tuple[str, str]],
    u: str,
    v: str,
    edge: frozenset,
) -> bool:
    parents_v = _directed_parents_to(directed, v)
    for w in parents_v:
        if w == u:
            continue
        if frozenset((u, w)) not in edges:
            continue
        if (w, v) not in directed:
            continue
        for x in parents_v:
            if x in (u, v, w):
                continue
            if frozenset((u, x)) not in edges:
                continue
            if (x, v) not in directed:
                continue
            wx_edge = frozenset((w, x)) in edges
            wx_dir = (w, x) in directed or (x, w) in directed
            if wx_edge or wx_dir:
                continue
            if (u, v) in directed or (v, u) in directed:
                continue
            directed.add((u, v))
            edges.discard(edge)
            logger.info("pc_algorithm.orient.R3: %s -> %s (colliders via %s, %s)", u, v, w, x)
            return True
    return False


def pc_algorithm(
    rows: Sequence[Mapping[str, object]],
    variables: Sequence[str] | None = None,
    *,
    alpha: float = 0.05,
    max_conditioning_size: int | None = None,
) -> DiscoveredGraph:
    """Run the PC algorithm and return the partial DAG it discovers."""

    if not rows:
        return DiscoveredGraph(variables=list(variables or ()), domains={}, undirected_edges=set(), directed_edges=set())

    if variables is None:
        variables = sorted({k for r in rows for k in r.keys()})

    domains: dict[str, tuple] = {}
    for v in variables:
        levels = sorted({r[v] for r in rows if v in r}, key=lambda x: (str(type(x).__name__), str(x)))
        domains[v] = tuple(levels)

    edges: set[frozenset] = {frozenset((a, b)) for a, b in combinations(variables, 2)}
    sep_sets: dict[frozenset, frozenset] = {}

    def neighbors(v: str) -> set[str]:
        out: set[str] = set()
        for e in edges:
            if v in e:
                other = next(iter(e - {v}))
                out.add(other)
        return out

    n = len(variables)
    max_l = n - 2 if max_conditioning_size is None else min(int(max_conditioning_size), n - 2)
    for l in range(0, max(0, max_l) + 1):
        to_remove: list[tuple[str, str, frozenset]] = []
        for a, b in combinations(variables, 2):
            edge = frozenset((a, b))
            if edge not in edges:
                continue
            adj_a = neighbors(a) - {b}
            adj_b = neighbors(b) - {a}
            separated = False
            if len(adj_a) >= l:
                for z in combinations(sorted(adj_a), l):
                    independent, g, p = _g_squared_independence(rows, a, b, z, alpha=alpha)
                    logger.debug(
                        "pc_algorithm.CI: %s ⊥ %s | %s -> indep=%s g=%.3f p=%.4f",
                        a,
                        b,
                        list(z),
                        independent,
                        g,
                        p,
                    )
                    if independent:
                        to_remove.append((a, b, frozenset(z)))
                        separated = True
                        break
            if not separated and len(adj_b) >= l:
                for z in combinations(sorted(adj_b), l):
                    independent, g, p = _g_squared_independence(rows, a, b, z, alpha=alpha)
                    logger.debug(
                        "pc_algorithm.CI: %s ⊥ %s | %s -> indep=%s g=%.3f p=%.4f",
                        a,
                        b,
                        list(z),
                        independent,
                        g,
                        p,
                    )
                    if independent:
                        to_remove.append((a, b, frozenset(z)))
                        break
        for a, b, z in to_remove:
            edges.discard(frozenset((a, b)))
            sep_sets[frozenset((a, b))] = z
            logger.info("pc_algorithm.skeleton: removed edge %s — %s | sep_set=%s", a, b, sorted(z))

    # Orient v-structures: A — C — B with A and B not adjacent and C not in sep(A, B).
    directed: set[tuple[str, str]] = set()
    for a, b in combinations(variables, 2):
        if frozenset((a, b)) in edges:
            continue
        sep = sep_sets.get(frozenset((a, b)), frozenset())
        for c in (neighbors(a) & neighbors(b)) - {a, b}:
            if c in sep:
                continue
            directed.add((a, c))
            directed.add((b, c))
            logger.info("pc_algorithm.orient.v_structure: %s -> %s <- %s", a, c, b)

    # Meek rules R1-R3: orient remaining edges where forced.
    changed = True
    while changed:
        changed = False
        for edge in list(edges):
            a, b = sorted(edge)
            for u, v in [(a, b), (b, a)]:
                if _meek_try_rule1(variables, edges, directed, u, v, edge):
                    changed = True
                    break
                if _meek_try_rule2(variables, edges, directed, u, v, edge):
                    changed = True
                    break
                if _meek_try_rule3(variables, edges, directed, u, v, edge):
                    changed = True
                    break
            if changed:
                break

    return DiscoveredGraph(
        variables=list(variables),
        domains=domains,
        undirected_edges=edges,
        directed_edges=directed,
        separating_sets=sep_sets,
    )


def build_scm_from_skeleton(graph: DiscoveredGraph, rows: Sequence[Mapping[str, object]]) -> FiniteSCM:
    """Fit a categorical FiniteSCM from a discovered partial DAG.

    Conditional probability tables are estimated by Laplace-smoothed maximum
    likelihood on ``rows``. Undirected edges in the skeleton are oriented in
    topological order over the variable list as a tie-breaker — a Meek-rule
    pass already removed every edge whose orientation was forced.
    """

    scm = FiniteSCM(domains={v: tuple(graph.domains[v]) for v in graph.variables})

    parents_of: dict[str, list[str]] = {v: [] for v in graph.variables}
    for u, v in graph.directed_edges:
        if v in parents_of:
            parents_of[v].append(u)

    seen_dir: set[tuple[str, str]] = set(graph.directed_edges)
    for edge in sorted(graph.undirected_edges, key=lambda e: tuple(sorted(e))):
        a, b = sorted(edge, key=lambda v: graph.variables.index(v))
        if (a, b) in seen_dir:
            continue
        seen_dir.add((a, b))
        parents_of[b].append(a)

    fitted: dict[str, dict[tuple, dict[object, float]]] = {}
    for v in graph.variables:
        ps = parents_of[v]
        cpt: dict[tuple, Counter] = {}
        for r in rows:
            if v not in r:
                continue
            try:
                key = tuple(r[p] for p in ps)
            except KeyError:
                continue
            cpt.setdefault(key, Counter())[r[v]] += 1
        smoothed: dict[tuple, dict[object, float]] = {}
        for key, counts in cpt.items():
            denom = sum(counts.values()) + len(graph.domains[v]) * 1.0
            smoothed[key] = {value: (counts.get(value, 0) + 1.0) / denom for value in graph.domains[v]}
        fitted[v] = smoothed

    # Each variable becomes endogenous, deterministic on parents + exogenous random tape U_v.
    for v in graph.variables:
        ps = parents_of[v]
        u_name = f"U_{v}"
        domain = graph.domains[v]
        scm.add_exogenous(u_name, list(range(SCM_EXOGENOUS_DOMAIN_SIZE)))
        cpt = fitted.get(v, {})

        def make_fn(var=v, parents=ps, table=cpt, dom=domain, u=u_name):
            def fn(values: dict[str, object]) -> object:
                key = tuple(values[p] for p in parents)
                row = table.get(key)
                if row is None:
                    logger.warning(
                        "build_scm_from_skeleton.CPT: missing CPT key=%r for node=%r; using dom[0]=%r as fallback",
                        key,
                        var,
                        dom[0],
                    )
                    return dom[0]
                u_val = int(values[u]) % SCM_EXOGENOUS_DOMAIN_SIZE
                cumulative = 0.0
                scale = float(SCM_EXOGENOUS_DOMAIN_SIZE)
                for value in dom:
                    p = row[value]
                    cumulative += float(p) * scale
                    if u_val < cumulative:
                        return value
                return dom[-1]

            return fn

        scm.add_endogenous(v, list(domain), [u_name] + ps, make_fn())

    logger.info(
        "build_scm_from_skeleton: vars=%s directed=%s undirected=%s",
        graph.variables,
        sorted(graph.directed_edges),
        [tuple(sorted(e)) for e in sorted(graph.undirected_edges, key=lambda e: tuple(sorted(e)))],
    )
    return scm
