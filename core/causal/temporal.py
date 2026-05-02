"""Temporal observation rows for dynamic causal discovery."""

from __future__ import annotations

import math
import re
from typing import Mapping, Sequence


class TemporalCausalTraceBuilder:
    """Convert journal episodes with Hawkes traces into lagged PC rows."""

    _SAFE_RE = re.compile(r"[^A-Za-z0-9_]+")

    def __init__(self, rows: Sequence[Mapping[str, object]]) -> None:
        self.rows = sorted(
            [dict(row) for row in rows],
            key=lambda row: (float(row.get("ts", 0.0)), int(row.get("id", 0) or 0)),
        )
        self._positive_gaps = self._derive_positive_gaps()
        self._trace_channels = self._derive_trace_channels()

    def build_rows(self) -> list[dict[str, object]]:
        out: list[dict[str, object]] = []
        for idx, row in enumerate(self.rows):
            built = self._current_row(row)
            if idx > 0:
                built.update(self._lagged_row(self.rows[idx - 1], row))
            built.update(self._hawkes_residual_row(row))
            if len(built) >= 2:
                out.append(built)
        return out

    def _current_row(self, row: Mapping[str, object]) -> dict[str, object]:
        built: dict[str, object] = {"intent_t": self._value(row.get("intent", "unknown")) or "unknown"}
        subject = self._value(row.get("subject", ""))
        answer = self._value(row.get("answer", ""))
        if subject:
            built["subject_t"] = subject
        if answer and answer != "unknown":
            built["answer_t"] = answer
        return built

    def _lagged_row(
        self,
        prior: Mapping[str, object],
        current: Mapping[str, object],
    ) -> dict[str, object]:
        gap = max(0.0, float(current.get("ts", 0.0)) - float(prior.get("ts", 0.0)))
        built: dict[str, object] = {
            "intent_t_minus_1": self._value(prior.get("intent", "unknown")) or "unknown",
            "temporal_gap": self._gap_label(gap),
        }
        subject = self._value(prior.get("subject", ""))
        answer = self._value(prior.get("answer", ""))
        if subject:
            built["subject_t_minus_1"] = subject
        if answer and answer != "unknown":
            built["answer_t_minus_1"] = answer
        return built

    def _hawkes_residual_row(self, row: Mapping[str, object]) -> dict[str, object]:
        trace = self._trace(row)
        excitation = trace.get("excitation", {})
        if not isinstance(excitation, Mapping):
            raise TypeError("hawkes_trace.excitation must be a mapping")
        built: dict[str, object] = {}
        for channel in self._trace_channels:
            value = float(excitation.get(channel, 0.0) or 0.0)
            name = f"hawkes_residual__{self._safe_name(channel)}"
            built[name] = "present" if value > 0.0 else "absent"
        return built

    def _derive_positive_gaps(self) -> list[float]:
        gaps: list[float] = []
        for prior, current in zip(self.rows, self.rows[1:]):
            gap = float(current.get("ts", 0.0)) - float(prior.get("ts", 0.0))
            if math.isfinite(gap) and gap > 0.0:
                gaps.append(gap)
        return sorted(gaps)

    def _derive_trace_channels(self) -> tuple[str, ...]:
        channels: set[str] = set()
        for row in self.rows:
            trace = self._trace(row)
            excitation = trace.get("excitation", {})
            if not isinstance(excitation, Mapping):
                continue
            channels.update(str(ch) for ch in excitation.keys())
        return tuple(sorted(channels))

    def _gap_label(self, gap: float) -> str:
        if not self._positive_gaps:
            return "gap_zero"
        median = self._positive_gaps[len(self._positive_gaps) // 2]
        return "gap_le_median" if float(gap) <= median else "gap_gt_median"

    @staticmethod
    def _trace(row: Mapping[str, object]) -> Mapping[str, object]:
        evidence = row.get("evidence", {})
        if not isinstance(evidence, Mapping):
            return {}
        trace = evidence.get("hawkes_trace", {})
        if trace is None:
            return {}
        if not isinstance(trace, Mapping):
            raise TypeError("hawkes_trace must be a mapping")
        return trace

    @classmethod
    def _safe_name(cls, value: object) -> str:
        text = str(value).strip().lower()
        safe = cls._SAFE_RE.sub("_", text).strip("_")
        return safe or "unknown"

    @classmethod
    def _value(cls, value: object) -> str:
        if not str(value).strip():
            return ""
        return cls._safe_name(value)
