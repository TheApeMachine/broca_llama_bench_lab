"""SCMTargetPicker — discover ``(treatment, outcome)`` from a SCM's endogenous labels.

The router runs the SCM's ``do(·)`` operator at every turn to score the
``causal_effect`` candidate. The picker decides which two endogenous
variables to feed the operator: it prefers binary variables whose labels
match common keyword sets (``"treatment"``/``"intervention"``,
``"outcome"``/``"response"``), and falls back to the first two endogenous
binaries otherwise.
"""

from __future__ import annotations

from typing import Any, Mapping


class SCMTargetPicker:
    """Stateless target-pair discovery for the causal_effect router branch."""

    _KEYS_T = frozenset(("t", "treatment", "intervention"))
    _KEYS_Y = frozenset(("y", "outcome", "response"))

    @classmethod
    def pick(cls, scm: Any, labels: Mapping[str, Any]) -> tuple[str, str]:
        """Return ``(treatment, outcome)`` endogenous variable names."""

        endo = list(scm.endogenous_names)
        binaries = [n for n in endo if n in scm.domains and len(scm.domains[n]) == 2]

        def matches(var: str, keyset: frozenset[str]) -> bool:
            if var.strip().lower() in keyset:
                return True
            lab = labels.get(var)
            return lab is not None and str(lab).strip().lower() in keyset

        t_name: str | None = next((b for b in binaries if matches(b, cls._KEYS_T)), None)
        if t_name is None and binaries:
            t_name = binaries[0]
        elif t_name is None and endo:
            t_name = endo[0]
        else:
            t_name = "T"

        y_name: str | None = next(
            (b for b in binaries if b != t_name and matches(b, cls._KEYS_Y)), None
        )
        if y_name is None:
            for n in endo:
                if n != t_name:
                    y_name = n
                    break
        if y_name is None:
            y_name = t_name
        return t_name, y_name
