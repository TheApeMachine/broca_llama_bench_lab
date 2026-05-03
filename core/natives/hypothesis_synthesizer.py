"""HypothesisSynthesizer — turns foraging EFE recommendations into real SCM tools.

When the tool-foraging EFE math says ``synthesize_tool``, the substrate has to
*actually* synthesize. Until now the recommendation was a dangling reflection
with no actor consuming it, leaving the ``native_tools`` table empty. This
synthesizer closes the loop in-process: it deterministically authors a
conjunction hypothesis ``hyp_<a>_AND_<b>`` over two SCM endogenous variables it
hasn't already covered, compiles + verifies it through the existing sandbox /
conformal pipeline, and persists the resulting tool into the registry.

Conjunction hypotheses are the smallest non-trivial SCM nodes the substrate can
add unilaterally — they're deterministic, finite-domain, and verifiable against
sample inputs. As PC algorithm discovery adds new endogenous variables across
sessions, the synthesizer covers the growing pair space. There is no LLM-author
step here; the tool source is generated from a fixed template, so every
synthesis succeeds or surfaces a real registry / sandbox failure.
"""

from __future__ import annotations

import logging
import textwrap
from itertools import combinations
from typing import Any, Sequence

logger = logging.getLogger(__name__)


_SAMPLE_BINARY_INPUTS = ({"a": 0, "b": 0}, {"a": 0, "b": 1}, {"a": 1, "b": 0}, {"a": 1, "b": 1})


class HypothesisSynthesizer:
    """Author conjunction hypothesis tools from existing SCM variables."""

    def __init__(self, *, scm: Any, tool_registry: Any) -> None:
        self._scm = scm
        self._tool_registry = tool_registry

    def attempt_one(self) -> Any | None:
        """Pick the next uncovered (a, b) pair and synthesize ``hyp_a_AND_b``.

        Returns the persisted :class:`NativeTool` on success, or ``None`` when
        the SCM has fewer than two endogenous variables or every binary pair is
        already covered by a synthesized tool.
        """

        endogenous = self._binary_endogenous()

        if len(endogenous) < 2:
            return None

        for a, b in combinations(endogenous, 2):
            name = self._conjunction_name(a, b)

            if self._tool_registry.get(name) is not None:
                continue

            return self._synthesize_conjunction(a, b, name)

        return None

    def _binary_endogenous(self) -> list[str]:
        """Endogenous variables whose declared domain is exactly ``(0, 1)``.

        The conjunction template assumes binary parents; non-binary endogenous
        variables (e.g. multi-valued classification outputs) are skipped here
        rather than coerced, so the framework's "no silent fallback" rule is
        preserved — a richer multi-domain synthesizer is the obvious extension
        and lives outside this class.
        """

        out: list[str] = []

        for name in self._scm.endogenous_names:
            domain = self._scm.domains.get(name, ())

            if tuple(domain) == (0, 1):
                out.append(name)

        return sorted(out)

    @staticmethod
    def _conjunction_name(a: str, b: str) -> str:
        lo, hi = sorted((a, b))

        return f"hyp_{lo}_AND_{hi}"

    def _synthesize_conjunction(self, a: str, b: str, name: str) -> Any:
        lo, hi = sorted((a, b))
        # NativeToolRegistry.verify / SCM callables use ``fn(values: dict)`` —
        # a single mapping argument — not positional parents.
        source = textwrap.dedent(
            f'''
            def {name}(values):
                v = dict(values)
                return 1 if (int(v[{repr(lo)}]) == 1 and int(v[{repr(hi)}]) == 1) else 0
            '''
        ).strip()
        sample_inputs: Sequence[dict] = (
            {lo: 0, hi: 0},
            {lo: 0, hi: 1},
            {lo: 1, hi: 0},
            {lo: 1, hi: 1},
        )

        tool = self._tool_registry.synthesize(
            name=name,
            source=source,
            parents=(lo, hi),
            domain=(0, 1),
            sample_inputs=sample_inputs,
            description=f"Conjunction hypothesis: {lo} AND {hi}",
        )

        logger.info(
            "HypothesisSynthesizer.synthesize: name=%s parents=(%s, %s) tool_id=%s",
            name,
            lo,
            hi,
            getattr(tool, "id", None),
        )

        return tool
