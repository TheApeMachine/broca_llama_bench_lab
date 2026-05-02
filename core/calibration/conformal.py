"""Conformal prediction with mathematical coverage guarantees.

Implements split-conformal classification (Vovk, Gammerman & Shafer; Lei et al.)
plus the *adaptive prediction set* (APS) variant of Romano et al. so the
substrate gets both an unbiased Vovk-style coverage guarantee and APS's
size-adaptivity (sets shrink for easy examples and grow for hard ones).

Given a calibration set of (score, label) pairs with marginal coverage target
``1 - α``, the construction guarantees

    P[ y_test ∈ C(x_test) ] ≥ 1 - α

with probability over the calibration draw, *for any* underlying scoring model
— including a frozen LLM. The substrate uses the size of ``C`` as a Fristonian
ambiguity signal: |C| = 1 collapses to a confident answer; |C| > 1 raises an
intrinsic cue and pushes the LLM toward a clarifying question.

Storage is a list of nonconformity scores (no labels needed at query time), so
the calibration is inexpensive to ship and update online.
"""

from __future__ import annotations

import logging
import math
import sqlite3
import threading
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

logger = logging.getLogger(__name__)


def _split_threshold(scores: Sequence[float], alpha: float) -> float:
    """Finite-sample-corrected (1 - α)(n+1)/n quantile (Lei et al., 2018)."""

    n = len(scores)
    if n == 0:
        return float("inf")
    a = max(1e-6, min(1.0 - 1e-6, float(alpha)))
    q_level = min(1.0, math.ceil((n + 1) * (1.0 - a)) / n)
    sorted_scores = sorted(scores)
    rank = int(math.ceil(q_level * n)) - 1
    rank = max(0, min(n - 1, rank))
    return float(sorted_scores[rank])


@dataclass
class ConformalSet:
    """Result of a conformal query."""

    labels: list[str]
    label_probs: dict[str, float]
    set_size: int
    threshold: float
    alpha: float
    method: str

    @property
    def p_values(self) -> dict[str, float]:
        """Deprecated alias for :attr:`label_probs`.

        This field holds model-estimated probabilities, not conformal *p*-values.

        Prefer :attr:`label_probs`.
        """

        warnings.warn(
            "Conformal.p_values is deprecated; use Conformal.label_probs instead",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.label_probs

    @property
    def confident(self) -> bool:
        return self.set_size == 1

    @property
    def ambiguity(self) -> float:
        """Log-cardinality of the prediction set (nats)."""

        return math.log(max(1, self.set_size))


class ConformalPredictor:
    """Split / adaptive conformal predictor for categorical outputs.

    Two methods are supported:

    * ``"lac"`` — Least-Ambiguous-set Classifier. Score is ``1 - p̂(y)``; the
      threshold is the Vovk-corrected quantile of calibration scores. This is
      the default and gives marginal coverage with the smallest expected sets.
    * ``"aps"`` — Adaptive Prediction Sets. Score is the cumulative softmax
      mass up to and including the true class (Romano et al. 2020). Sets adapt
      their size to local hardness; useful when classes have heterogeneous
      difficulty.
    """

    def __init__(
        self, *, alpha: float = 0.1, method: str = "lac", min_calibration: int = 8
    ):
        if method not in {"lac", "aps"}:
            raise ValueError(f"unknown conformal method: {method!r}")
        self.alpha = float(alpha)
        self.method = method
        self.min_calibration = int(min_calibration)
        self._scores: list[float] = []

    def load_scores(self, scores: Sequence[float]) -> None:
        """Replace the in-memory calibration score list (e.g. after loading from disk)."""

        self._scores = [float(s) for s in scores]

    @property
    def scores(self) -> list[float]:
        """Copy of stored nonconformity scores."""

        return list(self._scores)

    def get_scores(self) -> list[float]:
        warnings.warn(
            "ConformalPredictor.get_scores() is deprecated; use ConformalPredictor.scores instead",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.scores

    def __len__(self) -> int:
        return len(self._scores)

    def calibrate(
        self,
        p_label: float | None = None,
        p_distribution: Mapping[str, float] | None = None,
        *,
        true_label: str | None = None,
    ) -> None:
        """Append one calibration nonconformity score.

        For ``"lac"``, pass ``p_label`` = model probability of the true class.

        For ``"aps"``, pass the full ``p_distribution`` and ``true_label``; the
        score is cumulative predicted mass in descending-probability order up
        to and including the true label (Romano et al., 2020).
        """

        if self.method == "lac":
            if p_label is None:
                raise ValueError(
                    "LAC calibration requires p_label (estimated probability of the true label)"
                )
            score = max(0.0, min(1.0, 1.0 - float(p_label)))
        else:
            if p_distribution is None:
                raise ValueError("APS calibration requires p_distribution")
            if true_label is None:
                raise ValueError("APS calibration requires true_label")
            if true_label not in p_distribution:
                raise ValueError(
                    f"APS calibration: true_label {true_label!r} not found in p_distribution"
                )
            ranked = sorted(p_distribution.items(), key=lambda kv: -float(kv[1]))
            cumulative = 0.0
            score = 1.0
            for lab, p in ranked:
                cumulative += float(p)
                if lab == true_label:
                    score = float(cumulative)
                    break
        self._scores.append(score)
        logger.debug(
            "ConformalPredictor.calibrate: method=%s score=%.4f n=%d",
            self.method,
            score,
            len(self._scores),
        )

    def threshold(self) -> float:
        return _split_threshold(self._scores, self.alpha)

    def predict_set(self, distribution: Mapping[str, float]) -> ConformalSet:
        """Build the conformal prediction set for a label distribution."""

        if not distribution:
            return ConformalSet([], {}, 0, float("inf"), self.alpha, self.method)
        threshold = (
            self.threshold()
            if len(self._scores) >= self.min_calibration
            else float("inf")
        )
        label_probs: dict[str, float] = {}
        labels: list[str] = []
        if self.method == "lac":
            for label, p in distribution.items():
                score = max(0.0, min(1.0, 1.0 - float(p)))
                label_probs[label] = float(p)
                if score <= threshold or not math.isfinite(threshold):
                    labels.append(label)
        else:
            label_probs = {str(lab): float(p) for lab, p in distribution.items()}
            ranked = sorted(distribution.items(), key=lambda kv: -float(kv[1]))
            cumulative = 0.0
            for label, p in ranked:
                cumulative += float(p)
                labels.append(label)
                if cumulative >= threshold or not math.isfinite(threshold):
                    break
        if not labels:
            # Coverage guarantee requires a non-empty set; fall back to top-1.
            top = max(distribution.items(), key=lambda kv: float(kv[1]))[0]
            labels = [top]
        result = ConformalSet(
            labels=labels,
            label_probs=label_probs,
            set_size=len(labels),
            threshold=threshold,
            alpha=self.alpha,
            method=self.method,
        )
        logger.debug(
            "ConformalPredictor.predict_set: method=%s alpha=%.3f threshold=%.4f n_calib=%d set_size=%d top=%r",
            self.method,
            self.alpha,
            threshold,
            len(self._scores),
            result.set_size,
            labels[0] if labels else None,
        )
        return result


@dataclass
class OnlineConformalMartingale:
    """Power-martingale drift monitor over conformal nonconformity scores."""

    calibration_scores: list[float]
    alpha: float
    martingale: float = 1.0
    updates: int = 0

    def __post_init__(self) -> None:
        self.calibration_scores = [float(s) for s in self.calibration_scores]
        self.alpha = max(1e-6, min(1.0 - 1e-6, float(self.alpha)))

    def p_value(self, score: float) -> float:
        """Conformal p-value for one nonconformity score against calibration."""

        if not self.calibration_scores:
            raise ValueError("OnlineConformalMartingale requires calibration scores")
        s = float(score)
        ge = sum(1 for c in self.calibration_scores if float(c) >= s)
        return float((ge + 1.0) / (len(self.calibration_scores) + 1.0))

    def update(self, score: float) -> dict[str, float | bool | int]:
        """Advance the martingale and report whether exchangeability drifted."""

        p = max(1e-12, min(1.0, self.p_value(float(score))))
        eps = self.alpha
        self.martingale *= eps * (p ** (eps - 1.0))
        self.updates += 1
        drifted = bool(p <= self.alpha or self.martingale >= (1.0 / self.alpha))
        return {
            "p_value": float(p),
            "martingale": float(self.martingale),
            "updates": int(self.updates),
            "drifted": drifted,
            "alpha": float(self.alpha),
        }


class PersistentConformalCalibration:
    """SQLite-backed nonconformity-score store.

    Persists the calibration set across runs. Tightly scoped: one logical
    "channel" per ``(namespace, channel)`` pair so different LLM-driven tasks
    (relation extraction vs. action choice) don't pool into one threshold.
    """

    def __init__(self, path: str | Path, *, namespace: str = "main"):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.namespace = namespace
        self._conn: sqlite3.Connection | None = None
        self._lock = threading.RLock()
        self._init_schema()

    def _ensure_conn_locked(self) -> sqlite3.Connection:
        if self._conn is None:
            self._conn = sqlite3.connect(str(self.path), check_same_thread=False)
            self._conn.execute("PRAGMA journal_mode=WAL")
        return self._conn

    def _init_schema(self) -> None:
        with self._lock:
            con = self._ensure_conn_locked()
            con.execute(
                """
                CREATE TABLE IF NOT EXISTS conformal_scores (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    namespace TEXT NOT NULL,
                    channel TEXT NOT NULL,
                    method TEXT NOT NULL,
                    score REAL NOT NULL,
                    label TEXT NOT NULL,
                    created_at REAL NOT NULL
                )
                """
            )
            con.execute(
                "CREATE INDEX IF NOT EXISTS idx_conformal_lookup ON conformal_scores(namespace, channel, method)"
            )

    def close(self) -> None:
        with self._lock:
            if self._conn is not None:
                try:
                    self._conn.close()
                finally:
                    self._conn = None

    def __enter__(self) -> PersistentConformalCalibration:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    def add(self, channel: str, method: str, score: float, label: str = "") -> int:
        with self._lock:
            con = self._ensure_conn_locked()
            cur = con.execute(
                "INSERT INTO conformal_scores(namespace, channel, method, score, label, created_at) VALUES (?,?,?,?,?,?)",
                (
                    self.namespace,
                    channel,
                    method,
                    float(score),
                    str(label),
                    time.time(),
                ),
            )
            con.commit()
            return int(cur.lastrowid)

    def scores(self, channel: str, method: str) -> list[float]:
        with self._lock:
            con = self._ensure_conn_locked()
            rows = con.execute(
                "SELECT score FROM conformal_scores WHERE namespace=? AND channel=? AND method=? ORDER BY id",
                (self.namespace, channel, method),
            ).fetchall()
        return [float(r[0]) for r in rows]

    def hydrate(
        self, predictor: ConformalPredictor, channel: str
    ) -> ConformalPredictor:
        """Reload a predictor's calibration set from disk."""

        predictor.load_scores(self.scores(channel, predictor.method))
        return predictor

    def persist(
        self, predictor: ConformalPredictor, channel: str, *, label: str = ""
    ) -> None:
        """Append any new in-memory scores to the on-disk store.

        Only the *new* tail of in-memory scores is written when the stored
        sequence is a prefix of the predictor's scores. If the overlapping
        prefix disagrees with the database, stored rows for this channel/method
        are replaced with the predictor's full list.
        """

        existing = self.scores(channel, predictor.method)
        mem = predictor.scores
        n_pre = min(len(existing), len(mem))
        for i in range(n_pre):
            if not math.isclose(
                float(existing[i]), float(mem[i]), rel_tol=0.0, abs_tol=1e-12
            ):
                logger.warning(
                    "PersistentConformalCalibration.persist: score prefix mismatch at index %d for channel=%r method=%r; rewriting stored scores",
                    i,
                    channel,
                    predictor.method,
                )
                with self._lock:
                    con = self._ensure_conn_locked()
                    con.execute("BEGIN IMMEDIATE")
                    try:
                        con.execute(
                            "DELETE FROM conformal_scores WHERE namespace=? AND channel=? AND method=?",
                            (self.namespace, channel, predictor.method),
                        )
                        ts = time.time()
                        for s in mem:
                            con.execute(
                                "INSERT INTO conformal_scores(namespace, channel, method, score, label, created_at) VALUES (?,?,?,?,?,?)",
                                (
                                    self.namespace,
                                    channel,
                                    predictor.method,
                                    float(s),
                                    str(label),
                                    ts,
                                ),
                            )
                        con.commit()
                    except Exception:
                        con.rollback()
                        raise
                return
        new_tail = mem[len(existing) :]
        if not new_tail:
            return
        with self._lock:
            con = self._ensure_conn_locked()
            con.execute("BEGIN IMMEDIATE")
            try:
                ts = time.time()
                for s in new_tail:
                    con.execute(
                        "INSERT INTO conformal_scores(namespace, channel, method, score, label, created_at) VALUES (?,?,?,?,?,?)",
                        (
                            self.namespace,
                            channel,
                            predictor.method,
                            float(s),
                            str(label),
                            ts,
                        ),
                    )
                con.commit()
            except Exception:
                con.rollback()
                raise


def empirical_coverage(
    predictor: ConformalPredictor,
    calibration_set: Iterable[tuple[Mapping[str, float], str]],
) -> float:
    """Sanity check: empirical coverage on a held-out calibration sequence."""

    n = 0
    hits = 0
    for distribution, true_label in calibration_set:
        n += 1
        result = predictor.predict_set(distribution)
        if true_label in result.labels:
            hits += 1
    return hits / max(1, n)
