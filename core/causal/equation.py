from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict


@dataclass(frozen=True)
class EndogenousEquation:
    """Structural equation for an endogenous variable in a finite SCM.

    ``name`` is the variable being defined. ``parents`` lists upstream names whose
    values are read from a valuation dict. ``fn`` maps that parent dict to the
    variable's deterministic value.
    """

    name: str
    parents: tuple[str, ...]
    fn: Callable[[Dict[str, Any]], Any]
