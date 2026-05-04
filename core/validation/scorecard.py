"""Static implementation-readiness scorecards for declared faculties.

The manifest says what is wired; the scorecard says what still has to be true
before a faculty should be treated as a validated implementation rather than a
prototype, toy model, or experiment.  It is intentionally explicit and static so
project owners can see the gap without building models or reading the source.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Iterable

from ..kernel.manifest import RuntimeManifest
from ..kernel.profiles import manifest_for_profile
from ..kernel.readiness import Readiness


@dataclass(frozen=True)
class ImplementationGap:
    """One missing ingredient for a faculty to become more real."""

    faculty: str
    kind: str
    message: str
    severity: str = "warn"

    def as_dict(self) -> dict[str, str]:
        return {
            "faculty": self.faculty,
            "kind": self.kind,
            "message": self.message,
            "severity": self.severity,
        }


@dataclass(frozen=True)
class FacultyScore:
    """Readiness summary for one manifest faculty."""

    key: str
    label: str
    mode: str
    readiness: str
    gaps: tuple[ImplementationGap, ...] = field(default_factory=tuple)

    @property
    def status(self) -> str:
        if self.mode != "required":
            return "declared_" + self.mode
        if any(g.severity == "error" for g in self.gaps):
            return "blocked"
        if self.gaps:
            return "incomplete"
        return "ready"

    def as_dict(self) -> dict[str, object]:
        return {
            "key": self.key,
            "label": self.label,
            "mode": self.mode,
            "readiness": self.readiness,
            "status": self.status,
            "gaps": [gap.as_dict() for gap in self.gaps],
        }


@dataclass(frozen=True)
class ImplementationScorecard:
    """Project-level implementation-readiness report."""

    manifest_name: str
    scores: tuple[FacultyScore, ...]

    @property
    def status(self) -> str:
        active = [score for score in self.scores if score.mode == "required"]
        if any(score.status == "blocked" for score in active):
            return "blocked"
        if any(score.status == "incomplete" for score in active):
            return "incomplete"
        return "ready"

    def as_dict(self) -> dict[str, object]:
        return {
            "manifest": self.manifest_name,
            "status": self.status,
            "scores": [score.as_dict() for score in self.scores],
        }

    def to_json(self, *, indent: int = 2) -> str:
        return json.dumps(self.as_dict(), indent=indent, sort_keys=True)

    def table_lines(self) -> list[str]:
        lines = [f"Implementation scorecard: {self.manifest_name} ({self.status})"]
        for score in self.scores:
            lines.append(
                f"  {score.key:<32} {score.mode:<8} {score.readiness:<12} {score.status:<16} {score.label}"
            )
            for gap in score.gaps:
                lines.append(f"    - {gap.kind}: {gap.message}")
        return lines


class ImplementationAuditor:
    """Produces readiness gaps from the current manifest declaration."""

    _COMMON_PROTOTYPE_GAPS = (
        ("metric", "needs an empirical metric and a recorded baseline comparison"),
        ("ablation", "needs a manifest-level ablation proving this faculty changes behavior"),
    )

    _FACULTY_GAPS: dict[str, tuple[tuple[str, str], ...]] = {
        "reasoning.active_inference": (
            ("domain", "default POMDPs are tiny categorical demos; define real substrate state/action/observation builders"),
            ("policy_search", "policy enumeration needs scalable search or explicit horizon/budget contracts"),
            ("learning", "likelihoods should be fit from real interaction traces, not only hand-authored tables"),
        ),
        "reasoning.causal_scm": (
            ("assumptions", "SCM queries need user-visible assumptions, adjustment sets, and identifiability status"),
            ("sensitivity", "causal conclusions need sensitivity/stability checks before influencing answers"),
        ),
        "calibration.conformal": (
            ("calibration", "each channel needs calibration/evaluation splits and empirical coverage reporting"),
            ("drift", "online calibration needs exchangeability/drift policy that can freeze or reset channels"),
        ),
        "temporal.hawkes": (
            ("target", "define what Hawkes predicts and compare log likelihood against simple recency baselines"),
        ),
        "memory.vsa_hopfield": (
            ("capacity", "needs retrieval/collision curves under realistic memory loads"),
            ("grounding", "needs entity/synonym grounding so bound vectors represent durable concepts, not raw strings"),
        ),
        "control.grafts": (
            ("alignment", "graft projections need trained or validated alignment, strength bounds, and plan-adherence metrics"),
            ("safety", "untrained trainable grafts must be disabled or explicitly marked cold"),
        ),
        "control.recursion": (
            ("effect", "needs traces and task deltas showing recursion improves outputs rather than adding latency/noise"),
        ),
        "dmn.background": (
            ("phase_metrics", "each DMN phase needs a metric proving it improves memory, routing, or latency"),
            ("concurrency", "background writes need transaction boundaries and failure recovery contracts"),
        ),
        "native_tools": (
            ("sandbox", "untrusted generated tools should run only in isolated subprocess/container mode"),
            ("spec", "tool synthesis needs a formal spec/test/review lifecycle before execution"),
        ),
        "dynamic_grafts": (
            ("training", "activation-mode memory needs train/validation objectives and stale-mode eviction"),
        ),
        "swarm": (
            ("auth", "requires signed peer identity, replay protection, topic allow-lists, and rate limits"),
        ),
    }

    def audit(self, manifest: RuntimeManifest | str | None = None) -> ImplementationScorecard:
        resolved = manifest_for_profile(manifest) if isinstance(manifest, str) or manifest is None else manifest
        scores: list[FacultyScore] = []
        for faculty in resolved.faculties:
            gaps = tuple(self._gaps_for(faculty.key, faculty.readiness)) if faculty.mode == "required" else ()
            scores.append(
                FacultyScore(
                    key=faculty.key,
                    label=faculty.label,
                    mode=faculty.mode,
                    readiness=faculty.readiness.value,
                    gaps=gaps,
                )
            )
        return ImplementationScorecard(resolved.name, tuple(scores))

    def _gaps_for(self, key: str, readiness: Readiness) -> Iterable[ImplementationGap]:
        if readiness in {Readiness.TOY, Readiness.EXPERIMENTAL}:
            yield ImplementationGap(key, "readiness", f"declared as {readiness.value}; not validated for broad claims")
        if readiness in {Readiness.TOY, Readiness.PROTOTYPE, Readiness.EXPERIMENTAL}:
            for kind, message in self._COMMON_PROTOTYPE_GAPS:
                yield ImplementationGap(key, kind, message)
        for kind, message in self._FACULTY_GAPS.get(key, ()):  # faculty-specific gaps
            yield ImplementationGap(key, kind, message)
