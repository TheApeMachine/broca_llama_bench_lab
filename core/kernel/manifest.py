"""Runtime manifests for explicit Mosaic wiring and ablation profiles."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Literal

from .readiness import Readiness

FacultyMode = Literal["required", "disabled", "stub"]


@dataclass(frozen=True)
class FacultySpec:
    """One declared runtime faculty and its dependency contract."""

    key: str
    label: str
    mode: FacultyMode = "required"
    readiness: Readiness = Readiness.PROTOTYPE
    provides: tuple[str, ...] = ()
    requires: tuple[str, ...] = ()
    reason: str = ""

    @property
    def active(self) -> bool:
        return self.mode == "required"

    def disabled(self, reason: str) -> "FacultySpec":
        return replace(self, mode="disabled", reason=reason)

    def stubbed(self, reason: str) -> "FacultySpec":
        return replace(self, mode="stub", reason=reason, readiness=Readiness.TOY)

    def as_dict(self) -> dict[str, object]:
        return {
            "key": self.key,
            "label": self.label,
            "mode": self.mode,
            "readiness": self.readiness.value,
            "provides": list(self.provides),
            "requires": list(self.requires),
            "reason": self.reason,
        }


@dataclass(frozen=True)
class RuntimeManifest:
    """A complete declared runtime profile.

    This is the single source of truth for whether a subsystem is required,
    intentionally disabled for ablation, or replaced by an explicit test stub.
    """

    name: str
    description: str
    faculties: tuple[FacultySpec, ...]

    def __post_init__(self) -> None:
        keys = [f.key for f in self.faculties]
        dupes = sorted({k for k in keys if keys.count(k) > 1})
        if dupes:
            raise ValueError(f"RuntimeManifest {self.name!r} has duplicate faculty keys: {dupes}")

    def get(self, key: str) -> FacultySpec:
        for faculty in self.faculties:
            if faculty.key == key:
                return faculty
        raise KeyError(key)

    def with_faculty(self, updated: FacultySpec) -> "RuntimeManifest":
        return replace(
            self,
            faculties=tuple(updated if f.key == updated.key else f for f in self.faculties),
        )

    def disable(self, key: str, *, reason: str) -> "RuntimeManifest":
        return self.with_faculty(self.get(key).disabled(reason))

    def stub(self, key: str, *, reason: str) -> "RuntimeManifest":
        return self.with_faculty(self.get(key).stubbed(reason))

    @property
    def active_faculties(self) -> tuple[FacultySpec, ...]:
        return tuple(f for f in self.faculties if f.mode == "required")

    @property
    def disabled_faculties(self) -> tuple[FacultySpec, ...]:
        return tuple(f for f in self.faculties if f.mode == "disabled")

    @property
    def stubbed_faculties(self) -> tuple[FacultySpec, ...]:
        return tuple(f for f in self.faculties if f.mode == "stub")

    def as_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "description": self.description,
            "faculties": [f.as_dict() for f in self.faculties],
        }

    def graph_lines(self) -> list[str]:
        lines = [f"manifest:{self.name}"]
        for faculty in self.faculties:
            status = faculty.mode
            ready = faculty.readiness.value
            lines.append(f"  {faculty.key} [{status}, {ready}]")
            for req in faculty.requires:
                lines.append(f"    requires -> {req}")
            for provided in faculty.provides:
                lines.append(f"    provides -> {provided}")
        return lines
