"""CLI helpers for manifest, graph, and health inspection."""

from __future__ import annotations

import argparse
import sys
from typing import Any

from .capabilities import CapabilityReport
from .builder import KernelBuilder
from .manifest import PROFILE_BUILDERS, manifest_for_profile


def _profile_arg(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--profile",
        default="full",
        choices=sorted(PROFILE_BUILDERS),
        help="Runtime manifest profile to inspect or build.",
    )


def run_manifest_cli(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Print a Mosaic runtime manifest.")
    _profile_arg(parser)
    parser.add_argument("--json", action="store_true", help="Emit JSON instead of text.")
    args = parser.parse_args(argv or [])
    manifest = manifest_for_profile(args.profile)
    if args.json:
        import json

        print(json.dumps(manifest.as_dict(), indent=2, sort_keys=True), flush=True)
        return
    print(f"Manifest: {manifest.name}", flush=True)
    print(manifest.description, flush=True)
    for faculty in manifest.faculties:
        print(
            f"  {faculty.key:<32} {faculty.mode:<8} {faculty.readiness.value:<12} {faculty.label}",
            flush=True,
        )
        if faculty.reason:
            print(f"    reason: {faculty.reason}", flush=True)


def run_graph_cli(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Print the declared Mosaic dependency graph.")
    _profile_arg(parser)
    args = parser.parse_args(argv or [])
    print("\n".join(manifest_for_profile(args.profile).graph_lines()), flush=True)


def run_health_cli(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Print Mosaic runtime health.")
    _profile_arg(parser)
    parser.add_argument(
        "--static",
        action="store_true",
        help="Only inspect the manifest; do not construct models or the controller.",
    )
    parser.add_argument("--json", action="store_true", help="Emit JSON instead of text.")
    args = parser.parse_args(argv or [])
    manifest = manifest_for_profile(args.profile)
    if args.static:
        report = CapabilityReport.from_manifest(manifest, static_only=True)
        if args.json:
            print(report.to_json(), flush=True)
        else:
            print("\n".join(report.table_lines()), flush=True)
        return
    try:
        result = KernelBuilder().build(manifest=manifest)
    except Exception as exc:
        if args.json:
            import json

            print(
                json.dumps(
                    {
                        "status": "fail",
                        "manifest": manifest.name,
                        "error": repr(exc),
                    },
                    indent=2,
                    sort_keys=True,
                ),
                flush=True,
            )
        else:
            print(f"System health: fail\n  build_error: {exc!r}", flush=True)
        raise SystemExit(1) from exc
    if args.json:
        print(result.health.to_json(), flush=True)
    else:
        print("\n".join(result.health.table_lines()), flush=True)
    if result.health.status == "fail":
        raise SystemExit(1)


__all__ = ["run_graph_cli", "run_health_cli", "run_manifest_cli"]
