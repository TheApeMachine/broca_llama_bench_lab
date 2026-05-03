"""SubstrateInspector — JSON-friendly snapshot of substrate state for live UIs.

The TUI polls the substrate at ~5 Hz to refresh side panels and the activity
feed. Each subsystem is wrapped so a partial failure cannot break the UI;
the returned dict is a fresh copy.
"""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING, Any


if TYPE_CHECKING:
    from .controller import SubstrateController


logger = logging.getLogger(__name__)


class SubstrateInspector:
    """Read-only snapshot façade over the controller's internal state."""

    def __init__(self, mind: "SubstrateController") -> None:
        self._mind = mind

    def snapshot(self) -> dict[str, Any]:
        snap: dict[str, Any] = {"ts": time.time()}
        self._add_model(snap)
        self._add_memory(snap)
        self._add_journal(snap)
        self._add_workspace(snap)
        self._add_workers(snap)
        self._add_substrate(snap)
        self._add_misc(snap)
        return snap

    def _add_model(self, snap: dict[str, Any]) -> None:
        mind = self._mind
        try:
            device = next(mind.host.parameters()).device
            device_str = str(device)
        except (StopIteration, AttributeError):
            device_str = "unknown"
        snap["model"] = {
            "id": mind._llama_model_id,
            "device": device_str,
            "namespace": mind._namespace,
            "db_path": str(mind._db_path),
        }

    def _add_memory(self, snap: dict[str, Any]) -> None:
        mind = self._mind
        try:
            recent_claims = mind.memory.claims()[-8:]
            mean_conf = mind.memory.mean_confidence()
            snap["memory"] = {
                "count": int(mind.memory.count()),
                "subjects": len(mind.memory.subjects()),
                "mean_confidence": (float(mean_conf) if mean_conf is not None else None),
                "recent_claims": [
                    {
                        "subject": c.get("subject"),
                        "predicate": c.get("predicate"),
                        "object": c.get("object"),
                        "confidence": float(c.get("confidence", 0.0)),
                        "status": c.get("status"),
                    }
                    for c in recent_claims
                ],
            }
        except Exception:
            logger.exception("snapshot.memory failed")
            snap["memory"] = {"error": True}

    def _add_journal(self, snap: dict[str, Any]) -> None:
        mind = self._mind
        try:
            recent_journal = mind.journal.recent(8)
            snap["journal"] = {
                "count": int(mind.journal.count()),
                "recent": [
                    {
                        "id": int(r.get("id", 0)),
                        "intent": r.get("intent"),
                        "subject": r.get("subject"),
                        "answer": r.get("answer"),
                        "confidence": float(r.get("confidence", 0.0)),
                        "utterance": (r.get("utterance") or "")[:200],
                    }
                    for r in recent_journal
                ],
            }
        except Exception:
            logger.exception("snapshot.journal failed")
            snap["journal"] = {"error": True}

    def _add_workspace(self, snap: dict[str, Any]) -> None:
        mind = self._mind
        try:
            latest = mind.workspace.latest
            snap["workspace"] = {
                "frames_total": len(mind.workspace.frames),
                "working_window": len(mind.workspace.working),
                "intrinsic_cues": [
                    {
                        "urgency": float(c.urgency),
                        "faculty": c.faculty,
                        "source": c.source,
                        "evidence": dict(c.evidence) if isinstance(c.evidence, dict) else {},
                    }
                    for c in mind.workspace.intrinsic_cues
                ],
                "latest_frame": (
                    {
                        "intent": latest.intent,
                        "subject": latest.subject,
                        "answer": latest.answer,
                        "confidence": float(latest.confidence),
                    }
                    if latest is not None
                    else None
                ),
            }
        except Exception:
            logger.exception("snapshot.workspace failed")
            snap["workspace"] = {"error": True}

    def _add_workers(self, snap: dict[str, Any]) -> None:
        mind = self._mind
        try:
            bg = mind.session.background_worker
            snap["background"] = (
                bg.state_snapshot() if bg is not None else {"running": False}
            )
        except Exception:
            logger.exception("snapshot.background failed")
            snap["background"] = {"error": True}

        try:
            sw = mind.session.self_improve_worker
            if sw is None:
                snap["self_improve"] = {"running": False, "enabled": False}
            else:
                snap["self_improve"] = {
                    "running": bool(sw.running),
                    "enabled": bool(getattr(sw.config, "enabled", False)),
                    "iterations": sw.get_iterations(),
                    "interval_s": float(getattr(sw.config, "interval_s", 0.0)),
                    "last_summary": sw.last_summary,
                    "last_error": sw.last_error,
                }
        except Exception:
            logger.exception("snapshot.self_improve failed")
            snap["self_improve"] = {"error": True}

    def _add_substrate(self, snap: dict[str, Any]) -> None:
        mind = self._mind
        try:
            snap["substrate"] = {
                "vsa_atoms": len(mind.vsa),
                "hopfield_stored": len(mind.hopfield_memory),
                "hopfield_max_items": int(mind.hopfield_memory.max_items),
                "hawkes_channels": len(mind.hawkes.channels),
                "hawkes_intensity": dict(mind.hawkes.intensity_vector()),
                "tools": int(mind.tool_registry.count()),
                "macros": int(mind.macro_registry.count()),
                "deferred_relation_ingest_pending": mind.deferred_relation_ingest_count(),
                "ontology_axes": len(mind.ontology),
                "discovered_scm": mind.discovered_scm is not None,
            }
        except Exception:
            logger.exception("snapshot.substrate failed")
            snap["substrate"] = {"error": True}

    def _add_misc(self, snap: dict[str, Any]) -> None:
        mind = self._mind
        try:
            snap["encoders"] = mind.multimodal_perception.stats()
        except Exception:
            logger.exception("snapshot.encoders failed")
            snap["encoders"] = {"error": True}

        try:
            snap["affect"] = mind.affect_trace.summary()
        except Exception:
            logger.exception("snapshot.affect failed")
            snap["affect"] = {"error": True}

        try:
            snap["preferences"] = {
                "spatial_C": [float(x) for x in mind.spatial_preference.expected_C()],
                "causal_C": [float(x) for x in mind.causal_preference.expected_C()],
            }
        except Exception:
            logger.exception("snapshot.preferences failed")
            snap["preferences"] = {"error": True}

        try:
            snap["last_chat"] = (
                dict(mind.session.last_chat_meta) if mind.session.last_chat_meta else None
            )
        except Exception:
            snap["last_chat"] = None
