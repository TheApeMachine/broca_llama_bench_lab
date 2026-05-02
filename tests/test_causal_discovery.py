from __future__ import annotations

import random

from core.causal.causal_discovery import (
    DiscoveredGraph,
    _chi2_sf,
    _g_squared_independence,
    build_scm_from_skeleton,
    local_predicate_cluster,
    pc_algorithm,
    project_rows_to_variables,
)


def _generate_chain_data(n: int, seed: int = 0) -> list[dict[str, int]]:
    """Sample from X -> Y -> Z (so X ⊥ Z | Y)."""

    rng = random.Random(seed)
    rows = []
    for _ in range(n):
        x = rng.randint(0, 1)
        y = x if rng.random() < 0.85 else 1 - x
        z = y if rng.random() < 0.85 else 1 - y
        rows.append({"X": x, "Y": y, "Z": z})
    return rows


def _generate_v_structure_data(n: int, seed: int = 0) -> list[dict[str, int]]:
    """Sample from X -> Z <- Y where X ⊥ Y but X ⊥ Y | Z is FALSE."""

    rng = random.Random(seed)
    rows = []
    for _ in range(n):
        x = rng.randint(0, 1)
        y = rng.randint(0, 1)
        # Z is the XOR-flavored function so conditioning on Z creates dependence.
        z_base = x + y
        z = z_base if rng.random() < 0.9 else (z_base + 1) % 3
        rows.append({"X": x, "Y": y, "Z": z})
    return rows


def test_chi2_sf_matches_known_values():
    # Chi-square with 1 df at x=3.84 has approximately 0.05 survival mass.
    p = _chi2_sf(3.841, 1)
    assert 0.04 < p < 0.06
    p2 = _chi2_sf(0.0, 1)
    assert abs(p2 - 1.0) < 1e-6


def test_independence_test_detects_dependent_pair():
    rng = random.Random(0)
    rows = []
    for _ in range(200):
        x = rng.randint(0, 1)
        y = x if rng.random() < 0.85 else 1 - x
        rows.append({"X": x, "Y": y})
    indep, g, p = _g_squared_independence(rows, "X", "Y", [], alpha=0.05)
    # Highly correlated => not independent.
    assert not indep, f"correlated pair flagged independent (p={p:.4f}, g={g:.3f})"


def test_independence_test_finds_independence_under_separator():
    rows = _generate_chain_data(800)
    indep, g, p = _g_squared_independence(rows, "X", "Z", ["Y"], alpha=0.05)
    assert indep, f"chain X -> Y -> Z should give X ⊥ Z | Y (p={p:.4f})"


def test_pc_recovers_chain_skeleton():
    rows = _generate_chain_data(800)
    graph = pc_algorithm(rows, ["X", "Y", "Z"], alpha=0.05)
    edges = graph.undirected_edges | {frozenset(e) for e in graph.directed_edges}
    assert frozenset(("X", "Y")) in edges
    assert frozenset(("Y", "Z")) in edges
    assert (
        frozenset(("X", "Z")) not in edges
    ), f"PC failed to remove X-Z edge; graph={graph}"


def test_pc_orients_v_structure():
    rows = _generate_v_structure_data(1500)
    graph = pc_algorithm(rows, ["X", "Y", "Z"], alpha=0.05)
    # The X — Z and Y — Z edges should be directed *into* Z.
    incoming_z = [u for (u, v) in graph.directed_edges if v == "Z"]
    assert (
        "X" in incoming_z and "Y" in incoming_z
    ), f"v-structure not oriented: directed={graph.directed_edges} undirected={graph.undirected_edges}"


def test_pc_handles_independent_variables():
    rng = random.Random(42)
    rows = [{"A": rng.randint(0, 1), "B": rng.randint(0, 1)} for _ in range(400)]
    graph = pc_algorithm(rows, ["A", "B"], alpha=0.05)
    # Truly independent variables yield no edges.
    assert frozenset(("A", "B")) not in graph.undirected_edges
    assert ("A", "B") not in graph.directed_edges
    assert ("B", "A") not in graph.directed_edges


def test_build_scm_from_skeleton_runs_simulation():
    rows = _generate_chain_data(400)
    graph = pc_algorithm(rows, ["X", "Y", "Z"], alpha=0.05)
    scm = build_scm_from_skeleton(graph, rows)
    assert "X" in scm.endogenous_names
    p = scm.probability({"Z": 1}, given={}, interventions={"X": 1})
    # Intervening on the chain root should still bias Z because Y depends on X.
    p0 = scm.probability({"Z": 1}, given={}, interventions={"X": 0})
    assert (
        abs(p - p0) > 0.05
    ), f"do(X=1) and do(X=0) gave same Z=1 probability ({p:.3f} vs {p0:.3f})"


def test_local_predicate_cluster_trims_wide_schemas():
    letters = [chr(ord("a") + i) for i in range(12)]
    rows: list[dict[str, str]] = []
    for i in range(15):
        r = {p: f"v_{i}_{p}" for p in letters}
        rows.append(r)
    cluster = local_predicate_cluster(rows, max_variables=6, rng=random.Random(2))
    assert len(cluster) == 6
    assert len(set(cluster)) == 6
    projected = project_rows_to_variables(rows, cluster)
    assert len(projected) == len(rows)
    assert all(len(r) == 6 for r in projected)


def test_project_rows_to_variables_drops_sparse_rows():
    rows = [
        {"A": 1, "B": 2},
        {"A": 0},
        {"C": 3, "D": 4},
    ]
    out = project_rows_to_variables(rows, ["A", "B", "C", "D"])
    assert len(out) == 2
