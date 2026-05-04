"""Kernel builder: the canonical path from manifest to runtime."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .capabilities import CapabilityReport
from .ablations import LegacyAblationApplier
from .health import SystemHealth
from .manifest import RuntimeManifest
from .profiles import manifest_for_profile


@dataclass(frozen=True)
class KernelBuildResult:
    """Objects produced by a manifest-aware build."""

    controller: Any
    manifest: RuntimeManifest
    capabilities: CapabilityReport
    health: SystemHealth


class KernelBuilder:
    """Build the current legacy substrate through an explicit manifest boundary.

    This first implementation deliberately keeps the existing controller as the
    composed runtime, but all callers now get a manifest, capability report, and
    invariant health report.  Unsupported ablation profiles are visible rather
    than silently pretending to remove faculties.
    """

    def build(
        self,
        *,
        profile: str | None = None,
        manifest: RuntimeManifest | None = None,
        **controller_kwargs: Any,
    ) -> KernelBuildResult:
        manifest = manifest or manifest_for_profile(profile)
        from ..cli import SubstrateControllerFactory

        controller = SubstrateControllerFactory().build(**controller_kwargs)
        LegacyAblationApplier().apply(controller, manifest)
        capabilities = CapabilityReport.from_controller(controller, manifest)
        health = SystemHealth.from_controller(controller, manifest=manifest)
        return KernelBuildResult(
            controller=controller,
            manifest=manifest,
            capabilities=capabilities,
            health=health,
        )
