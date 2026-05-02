from __future__ import annotations

from dataclasses import dataclass
from typing import Callable


@dataclass
class EndogenousEquation:
    name: str
    parents: tuple[str, ...]
    fn: Callable[[dict], object]
