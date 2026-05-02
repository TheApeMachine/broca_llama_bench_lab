from __future__ import annotations

from typing import Iterable, Mapping, Sequence


class CausalDAG:
    """Directed graph utilities for d-separation and adjustment-set search."""

    def __init__(self, parents: Mapping[str, Sequence[str]]) -> None:
        self.parents: dict[str, list[str]] = {k: list(v) for k, v in parents.items()}

    def descendants(self, node: str) -> set[str]:
        children = self._children_adjacency()
        out: set[str] = set()
        stack = list(children.get(node, []))

        while stack:
            cur = stack.pop()

            if cur in out:
                continue

            out.add(cur)
            stack.extend(children.get(cur, []))

        return out

    def remove_outgoing_from(self, nodes: Iterable[str]) -> CausalDAG:
        blocked = set(nodes)
        updated = {child: [p for p in ps if p not in blocked] for child, ps in self.parents.items()}
        return CausalDAG(updated)

    def directed_paths(self, start: str, end: str, *, max_paths: int | None = None) -> list[list[str]]:
        children = self._children_adjacency()
        paths: list[list[str]] = []
        stack = [(start, [start])]

        while stack:
            cur, path = stack.pop()

            if cur == end:
                paths.append(path)
                if max_paths is not None and len(paths) >= max_paths:
                    return paths
                continue

            for nxt in children.get(cur, []):
                if nxt not in path:
                    stack.append((nxt, path + [nxt]))

        return paths

    def d_separated(self, x: str | Iterable[str], y: str | Iterable[str], z: Iterable[str], *, max_simple_paths: int | None = None) -> bool:
        xs = {x} if isinstance(x, str) else set(x)
        ys = {y} if isinstance(y, str) else set(y)
        conditioned = set(z)
        conditioned_or_desc = set(conditioned)
        for z_node in conditioned:
            conditioned_or_desc.update(self.descendants(z_node))

        for a in xs:
            for b in ys:
                paths = self.simple_paths_between(a, b, max_paths=max_simple_paths)

                for path in paths:
                    if len(path) > 1 and self.path_active(path, conditioned, conditioned_or_desc):
                        return False

        return True

    def simple_paths_between(self, start: str, end: str, *, max_len: int | None = None, max_paths: int | None = None) -> list[list[str]]:
        """Enumerate simple paths; stops and returns when ``max_paths`` paths are found (truncated enumeration)."""

        nb = self._undirected_neighbor_sets()
        max_len_eff = max_len if max_len is not None else len(nb) + 1
        paths: list[list[str]] = []
        stack = [(start, [start])]

        while stack:
            cur, path = stack.pop()

            if len(path) > max_len_eff:
                continue

            if cur == end:
                paths.append(path)

                if max_paths is not None and len(paths) >= max_paths:
                    return paths

                continue

            for nxt in nb.get(cur, ()):
                if nxt not in path:
                    stack.append((nxt, path + [nxt]))

        return paths

    def path_active(self, path: Sequence[str], conditioned: set[str], conditioned_or_desc: set[str]) -> bool:
        for i in range(1, len(path) - 1):
            a, b, c = path[i - 1], path[i], path[i + 1]
            collider = self.has_arrow(self.parents, a, b) and self.has_arrow(self.parents, c, b)

            if collider:
                if b not in conditioned_or_desc:
                    return False

            elif b in conditioned:
                return False

        return True

    def _children_adjacency(self) -> dict[str, list[str]]:
        children: dict[str, list[str]] = {n: [] for n in self.parents}

        for child, ps in self.parents.items():
            for p in ps:
                children.setdefault(p, []).append(child)

        return children

    def _undirected_neighbor_sets(self) -> dict[str, set[str]]:
        nodes = set(self.parents)

        for ps in self.parents.values():
            nodes.update(ps)

        nb = {n: set() for n in nodes}

        for child, ps in self.parents.items():
            for p in ps:
                nb[p].add(child)
                nb[child].add(p)

        return nb

    @staticmethod
    def has_arrow(parents: Mapping[str, Sequence[str]], src: str, dst: str) -> bool:
        return src in parents.get(dst, ())
