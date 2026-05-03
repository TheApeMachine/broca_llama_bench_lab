"""Legacy-runtime ablation adapters.

These are explicit physical adapters for the few ablations that can be enforced
without rewriting the old ``SubstrateController`` builder.  Anything not handled
here fails loudly instead of pretending to be disabled.
"""

from __future__ import annotations

from typing import Any

from ..substrate.recursion_controller import RecursionTrace
from .manifest import RuntimeManifest


class NoOpRecursionController:
    """A recursion controller that records an explicit zero-round ablation."""

    def run(self, **_kwargs: Any) -> RecursionTrace:
        return RecursionTrace(
            rounds=0,
            halts=[],
            thought_slots=[],
            llama_slots=[],
            final_thought_slot="",
            final_llama_slot="",
        )


class LegacyAblationApplier:
    """Apply supported manifest ablations to the legacy controller."""

    SUPPORTED_DISABLED = frozenset({"swarm", "control.recursion", "control.grafts"})

    def apply(self, controller: Any, manifest: RuntimeManifest) -> None:
        unsupported = [
            f.key
            for f in manifest.disabled_faculties
            if f.key not in self.SUPPORTED_DISABLED
        ]
        stubbed = [f.key for f in manifest.stubbed_faculties]
        if unsupported or stubbed:
            raise NotImplementedError(
                "This manifest profile declares ablations/stubs that the legacy "
                "runtime cannot yet physically enforce. Unsupported: "
                f"disabled={unsupported}, stubbed={stubbed}."
            )
        if manifest.get("control.grafts").mode == "disabled":
            self._disable_grafts(controller)
        if manifest.get("control.recursion").mode == "disabled":
            self._disable_recursion(controller)

    @staticmethod
    def _disable_recursion(controller: Any) -> None:
        controller.recursion_controller = NoOpRecursionController()
        if hasattr(controller, "runtime") and hasattr(controller.runtime, "chat"):
            # ChatOrchestrator reads mind.recursion_controller directly; keeping
            # the controller attribute current is sufficient.  Runtime is left
            # intact for compatibility with other facades.
            pass

    @staticmethod
    def _disable_grafts(controller: Any) -> None:
        host = getattr(controller, "host", None)
        clear_all = getattr(host, "clear_all_grafts", None)
        if callable(clear_all):
            removed = clear_all()
            controller._ablation_removed_grafts = [(slot, type(module).__name__) for slot, module in removed]
        for attr in ("lexical_graft", "feature_graft", "concept_graft", "kv_memory_graft", "swm_residual_graft"):
            if hasattr(controller, attr):
                getattr(controller, attr).enabled = False
