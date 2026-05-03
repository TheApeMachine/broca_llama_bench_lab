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


_FULL_FACULTIES: tuple[FacultySpec, ...] = (
    FacultySpec(
        "host.llama",
        "Frozen language host",
        readiness=Readiness.PROTOTYPE,
        provides=("host", "tokenizer", "embedding_matrix"),
        requires=("device",),
    ),
    FacultySpec(
        "memory.semantic",
        "SQLite semantic memory",
        readiness=Readiness.PROTOTYPE,
        provides=("memory", "claims"),
        requires=("database",),
    ),
    FacultySpec(
        "memory.episodic",
        "Workspace journal and episode graph",
        readiness=Readiness.PROTOTYPE,
        provides=("journal", "episode_graph"),
        requires=("database", "memory"),
    ),
    FacultySpec(
        "encoder.extraction",
        "GLiNER2 relation extraction encoder",
        readiness=Readiness.PROTOTYPE,
        provides=("relation_extractor", "gliner_hidden"),
        requires=("device",),
    ),
    FacultySpec(
        "encoder.classification",
        "GLiClass semantic classification encoder",
        readiness=Readiness.PROTOTYPE,
        provides=("intent_scores", "gliclass_hidden"),
        requires=("device",),
    ),
    FacultySpec(
        "encoder.affect",
        "Affect and emotion encoder",
        readiness=Readiness.PROTOTYPE,
        provides=("affect_state",),
        requires=("device",),
    ),
    FacultySpec(
        "comprehension.intent_gate",
        "Semantic intent gate",
        readiness=Readiness.PROTOTYPE,
        provides=("utterance_intent",),
        requires=("intent_scores",),
    ),
    FacultySpec(
        "comprehension.router",
        "Faculty router and frame selector",
        readiness=Readiness.PROTOTYPE,
        provides=("cognitive_frame",),
        requires=("memory", "utterance_intent"),
    ),
    FacultySpec(
        "reasoning.active_inference",
        "Finite categorical active-inference POMDPs",
        readiness=Readiness.TOY,
        provides=("pomdp", "active_agent"),
        requires=("events",),
        reason="Current default domain is a small Tiger/tool-foraging style categorical model.",
    ),
    FacultySpec(
        "reasoning.causal_scm",
        "Finite structural causal model",
        readiness=Readiness.PROTOTYPE,
        provides=("scm", "causal_agent"),
        requires=("pomdp",),
    ),
    FacultySpec(
        "calibration.conformal",
        "Conformal calibration and uncertainty sets",
        readiness=Readiness.PROTOTYPE,
        provides=("conformal_relation", "conformal_native_tool"),
        requires=("database",),
    ),
    FacultySpec(
        "temporal.hawkes",
        "Hawkes temporal excitation",
        readiness=Readiness.TOY,
        provides=("temporal_excitation",),
        requires=("database",),
    ),
    FacultySpec(
        "memory.vsa_hopfield",
        "VSA and Hopfield associative memory",
        readiness=Readiness.PROTOTYPE,
        provides=("vsa", "hopfield_memory"),
        requires=("host",),
    ),
    FacultySpec(
        "control.grafts",
        "Host graft stack",
        readiness=Readiness.PROTOTYPE,
        provides=("grafts", "graft_plan"),
        requires=("host", "cognitive_frame"),
    ),
    FacultySpec(
        "control.swm",
        "Substrate working memory and encoder publisher",
        readiness=Readiness.PROTOTYPE,
        provides=("swm", "prediction_errors"),
        requires=("vsa",),
    ),
    FacultySpec(
        "control.recursion",
        "Recursive SWM ↔ host latent loop",
        readiness=Readiness.EXPERIMENTAL,
        provides=("recursive_thought",),
        requires=("swm", "host", "grafts"),
    ),
    FacultySpec(
        "dmn.background",
        "Default-mode background worker",
        readiness=Readiness.EXPERIMENTAL,
        provides=("background_consolidation",),
        requires=("memory", "journal", "scm"),
    ),
    FacultySpec(
        "native_tools",
        "Native tool registry and synthesis",
        readiness=Readiness.EXPERIMENTAL,
        provides=("native_tool_registry", "tool_foraging"),
        requires=("database", "conformal_native_tool"),
    ),
    FacultySpec(
        "dynamic_grafts",
        "Persistent activation-mode graft memory",
        readiness=Readiness.EXPERIMENTAL,
        provides=("activation_memory", "dynamic_grafts"),
        requires=("host", "database", "grafts"),
    ),
    FacultySpec(
        "swarm",
        "UDP swarm propagation",
        mode="disabled",
        readiness=Readiness.TOY,
        provides=("swarm_events",),
        requires=("events",),
        reason="Disabled until authenticated peer identity and replay protection exist.",
    ),
)


def full_manifest() -> RuntimeManifest:
    return RuntimeManifest(
        name="full",
        description="Full declared Mosaic runtime. Swarm remains explicitly disabled by default.",
        faculties=_FULL_FACULTIES,
    )


def llm_only_manifest() -> RuntimeManifest:
    manifest = full_manifest()
    for key in [f.key for f in manifest.faculties if f.key != "host.llama"]:
        if key != "swarm":
            manifest = manifest.disable(key, reason="ablation: frozen language host only")
    return replace(manifest, name="llm_only", description="Ablation profile: host only.")


def no_recursion_manifest() -> RuntimeManifest:
    return replace(
        full_manifest().disable("control.recursion", reason="ablation: recursive latent loop disabled"),
        name="no_recursion",
        description="Ablation profile: full stack without recursive SWM-host loop.",
    )


def no_grafts_manifest() -> RuntimeManifest:
    manifest = full_manifest().disable("control.grafts", reason="ablation: host graft stack disabled")
    manifest = manifest.disable("control.recursion", reason="ablation: recursion requires grafts")
    return replace(manifest, name="no_grafts", description="Ablation profile: full stack without graft actuation.")


def no_memory_manifest() -> RuntimeManifest:
    manifest = full_manifest().disable("memory.semantic", reason="ablation: semantic memory disabled")
    manifest = manifest.disable("memory.episodic", reason="ablation: episodic journal disabled")
    return replace(manifest, name="no_memory", description="Ablation profile: memory disabled.")


def test_stub_manifest() -> RuntimeManifest:
    manifest = full_manifest()
    for key in ("host.llama", "encoder.extraction", "encoder.classification", "encoder.affect"):
        manifest = manifest.stub(key, reason="test profile: explicit stub replaces heavy model")
    return replace(manifest, name="test_stub", description="Unit-test profile with explicit heavy-model stubs.")


PROFILE_BUILDERS = {
    "full": full_manifest,
    "llm_only": llm_only_manifest,
    "no_recursion": no_recursion_manifest,
    "no_grafts": no_grafts_manifest,
    "no_memory": no_memory_manifest,
    "test_stub": test_stub_manifest,
}


def manifest_for_profile(profile: str | None) -> RuntimeManifest:
    name = (profile or "full").strip() or "full"
    try:
        return PROFILE_BUILDERS[name]()
    except KeyError as exc:
        raise ValueError(
            f"Unknown Mosaic runtime profile {name!r}; choose one of {sorted(PROFILE_BUILDERS)}"
        ) from exc
