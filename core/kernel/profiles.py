"""Declared Mosaic runtime profiles and ablation manifests."""

from __future__ import annotations

from dataclasses import replace

from .manifest import FacultySpec, RuntimeManifest
from .readiness import Readiness

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
