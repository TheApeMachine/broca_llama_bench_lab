"""Runtime capability reports for explicit wiring visibility."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from .manifest import FacultySpec, RuntimeManifest
from .profiles import manifest_for_profile


@dataclass(frozen=True)
class CapabilityRecord:
    """Observed status of one declared faculty."""

    key: str
    label: str
    mode: str
    readiness: str
    present: bool
    health: str
    reason: str = ""
    provides: tuple[str, ...] = ()
    requires: tuple[str, ...] = ()
    details: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "label": self.label,
            "mode": self.mode,
            "readiness": self.readiness,
            "present": self.present,
            "health": self.health,
            "reason": self.reason,
            "provides": list(self.provides),
            "requires": list(self.requires),
            "details": dict(self.details),
        }


@dataclass(frozen=True)
class CapabilityReport:
    """Complete runtime capability report for one manifest."""

    manifest_name: str
    records: tuple[CapabilityRecord, ...]
    static_only: bool = False

    @property
    def failed(self) -> bool:
        return any(record.health == "fail" for record in self.records)

    @property
    def warned(self) -> bool:
        return any(record.health == "warn" for record in self.records)

    @property
    def status(self) -> str:
        if self.failed:
            return "fail"
        if self.warned:
            return "warn"
        return "pass"

    def as_dict(self) -> dict[str, Any]:
        return {
            "manifest": self.manifest_name,
            "status": self.status,
            "static_only": self.static_only,
            "records": [record.as_dict() for record in self.records],
        }

    def to_json(self, *, indent: int = 2) -> str:
        return json.dumps(self.as_dict(), indent=indent, sort_keys=True, default=str)

    def table_lines(self) -> list[str]:
        lines = [f"Capability report: {self.manifest_name} ({self.status})"]
        if self.static_only:
            lines.append("  static manifest only: runtime objects were not constructed")
        for record in self.records:
            present = "present" if record.present else "missing"
            lines.append(
                f"  {record.key:<32} {record.mode:<8} {record.health:<5} {present:<8} {record.readiness}"
            )
            if record.reason:
                lines.append(f"    reason: {record.reason}")
        return lines

    @classmethod
    def from_manifest(cls, manifest: RuntimeManifest, *, static_only: bool = True) -> "CapabilityReport":
        return cls(
            manifest.name,
            tuple(_static_record(faculty) for faculty in manifest.faculties),
            static_only=static_only,
        )

    @classmethod
    def from_controller(
        cls,
        controller: Any,
        manifest: RuntimeManifest | None = None,
    ) -> "CapabilityReport":
        manifest = manifest or manifest_for_profile("full")
        records = tuple(_record_from_controller(controller, faculty) for faculty in manifest.faculties)
        return cls(manifest.name, records, static_only=False)


def _static_record(faculty: FacultySpec) -> CapabilityRecord:
    health = "pass" if faculty.mode == "disabled" else "warn"
    reason = faculty.reason or ("runtime not constructed" if faculty.mode != "disabled" else "explicitly disabled")
    return CapabilityRecord(
        key=faculty.key,
        label=faculty.label,
        mode=faculty.mode,
        readiness=faculty.readiness.value,
        present=False,
        health=health,
        reason=reason,
        provides=faculty.provides,
        requires=faculty.requires,
    )


def _record_from_controller(controller: Any, faculty: FacultySpec) -> CapabilityRecord:
    present, details = _presence_for_key(controller, faculty.key)
    if faculty.mode == "disabled":
        health = "warn" if present else "pass"
        reason = faculty.reason or "explicitly disabled"
        if present:
            reason = f"{reason}; object is still present in the constructed legacy runtime"
    elif faculty.mode == "stub":
        health = "warn" if present else "fail"
        reason = faculty.reason or "explicit test stub"
    else:
        health = "pass" if present else "fail"
        reason = faculty.reason
    return CapabilityRecord(
        key=faculty.key,
        label=faculty.label,
        mode=faculty.mode,
        readiness=faculty.readiness.value,
        present=present,
        health=health,
        reason=reason,
        provides=faculty.provides,
        requires=faculty.requires,
        details=details,
    )


def _presence_for_key(controller: Any, key: str) -> tuple[bool, dict[str, Any]]:
    checks: dict[str, tuple[str, ...]] = {
        "host.llama": ("host", "tokenizer"),
        "memory.semantic": ("memory",),
        "memory.episodic": ("journal", "episode_graph"),
        "encoder.extraction": ("extraction_encoder",),
        "encoder.classification": ("classification_encoder",),
        "encoder.affect": ("affect_encoder",),
        "comprehension.intent_gate": ("intent_gate",),
        "comprehension.router": ("router",),
        "reasoning.active_inference": ("pomdp", "active_agent"),
        "reasoning.causal_scm": ("scm", "causal_pomdp", "causal_agent"),
        "calibration.conformal": ("relation_conformal", "native_tool_conformal", "conformal_calibration"),
        "temporal.hawkes": ("hawkes", "hawkes_persistence"),
        "memory.vsa_hopfield": ("vsa", "hopfield_memory"),
        "control.grafts": ("lexical_graft", "feature_graft", "concept_graft", "kv_memory_graft"),
        "control.swm": ("swm", "swm_publisher", "prediction_errors"),
        "control.recursion": ("recursion_controller", "recursion_halt"),
        "dmn.background": ("session",),
        "native_tools": ("tool_registry", "tool_foraging"),
        "dynamic_grafts": ("activation_memory", "dynamic_graft_synth"),
        "swarm": ("swarm",),
    }
    attrs = checks.get(key, (key.replace(".", "_"),))
    missing = [attr for attr in attrs if not hasattr(controller, attr)]
    details: dict[str, Any] = {"expected_attributes": list(attrs), "missing_attributes": missing}
    if key == "host.llama" and hasattr(controller, "host"):
        details["host_type"] = type(getattr(controller, "host")).__name__
        details["model_id"] = getattr(controller, "llama_model_id", None)
    if key == "dmn.background" and hasattr(controller, "session"):
        worker = getattr(getattr(controller, "session"), "background_worker", None)
        details["worker_constructed"] = worker is not None
        details["worker_running"] = bool(getattr(worker, "running", False)) if worker is not None else False
    if key == "control.grafts" and hasattr(controller, "host"):
        grafts = getattr(getattr(controller, "host"), "grafts", {})
        details["host_slots"] = sorted(str(slot) for slot in getattr(grafts, "keys", lambda: [])())
    return len(missing) == 0, details
