"""Online preference learning for the active-inference C matrix.

Friston's expected-free-energy minimization is steered by ``C`` — the prior
preference distribution over observations. A static ``C`` is the substrate's
"hardcoded personality"; making it Dirichlet-conjugate lets the architecture
update its preferences from user feedback in the principled Bayesian way:

* Each observation is treated as a draw from a multinomial whose parameters
  have a Dirichlet prior.
* User feedback (positive or negative) increments the prior's concentration
  vector for the relevant observation.
* The expected ``C`` distribution at any time is just the normalized
  concentration vector — one division to compute, instantly available to the
  POMDP.

Negative feedback (e.g. "stop asking me clarification questions") is modeled
as *evidence against* an observation: the concentration on that index
multiplies by a sub-unit factor so the substrate learns to avoid it without
ever going negative.
"""

from __future__ import annotations

import json
import logging
import math
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

logger = logging.getLogger(__name__)


@dataclass
class PreferenceEvent:
    observation_index: int
    polarity: float
    weight: float
    reason: str


class DirichletPreference:
    """Dirichlet-conjugate prior over ``C`` for a categorical POMDP.

    Concentration ``α_i`` keeps a running pseudocount of how often observation
    ``i`` was preferred. Mean preference is ``α_i / Σα``. Variance is
    ``α_i (Σα - α_i) / (Σα)² (Σα + 1)`` — small when the substrate has many
    observations, large when it has few, which is exactly the right behavior
    for online preference learning.
    """

    def __init__(self, n_observations: int, *, prior_strength: float = 1.0, initial_C: Sequence[float] | None = None):
        if n_observations <= 0:
            raise ValueError("n_observations must be positive")
        self.n_observations = int(n_observations)
        self.prior_strength = float(prior_strength)
        if initial_C is None:
            self.alpha = [self.prior_strength] * self.n_observations
        else:
            base = [max(1e-6, float(x)) for x in initial_C]
            if len(base) != self.n_observations:
                raise ValueError("initial_C length disagrees with n_observations")
            total = sum(base)
            self.alpha = [a * self.prior_strength * self.n_observations / total for a in base]
        self.history: list[PreferenceEvent] = []

    @property
    def mean(self) -> list[float]:
        total = sum(self.alpha)
        if total <= 0:
            return [1.0 / self.n_observations] * self.n_observations
        return [a / total for a in self.alpha]

    def expected_C(self) -> list[float]:
        return self.mean

    def variance(self) -> list[float]:
        total = sum(self.alpha)
        if total <= 0:
            return [0.0] * self.n_observations
        out = []
        denom = total * total * (total + 1.0)
        for a in self.alpha:
            out.append(float(a * (total - a) / max(denom, 1e-12)))
        return out

    def update(self, observation_index: int, *, polarity: float = 1.0, weight: float = 1.0, reason: str = "") -> None:
        """Update the Dirichlet given one labeled observation.

        ``polarity > 0`` increases the pseudocount on ``observation_index``;
        ``polarity < 0`` shrinks it (multiplicatively, via ``exp(polarity *
        weight)``) so the value stays strictly positive — the conjugate prior
        is only valid on the open simplex.
        """

        i = int(observation_index)
        if not (0 <= i < self.n_observations):
            raise IndexError(f"observation_index {i} out of range")
        w = float(max(0.0, weight))
        if polarity >= 0:
            self.alpha[i] += float(polarity) * w
        else:
            shrink = math.exp(float(polarity) * w)
            self.alpha[i] = max(1e-6, self.alpha[i] * shrink)
        self.history.append(PreferenceEvent(observation_index=i, polarity=float(polarity), weight=w, reason=str(reason)))
        logger.info(
            "DirichletPreference.update: idx=%d polarity=%+.3f weight=%.3f alpha[i]=%.4f mean=%s reason=%s",
            i,
            float(polarity),
            w,
            self.alpha[i],
            [round(m, 4) for m in self.mean],
            reason,
        )

    def kl_to_uniform(self) -> float:
        """KL divergence from the current expected C to the uniform distribution.

        Convenient summary of how strongly the substrate has formed a
        preference at all — 0 means no preference yet; growing values mean a
        sharper personality.
        """

        p = self.mean
        u = 1.0 / self.n_observations
        return float(sum(pi * math.log(pi / u) for pi in p if pi > 0))


class PersistentPreference:
    """Disk-backed Dirichlet store keyed by ``(namespace, faculty)``."""

    def __init__(self, path: str | Path, *, namespace: str = "main"):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.namespace = namespace
        self._init_schema()

    def _connect(self) -> sqlite3.Connection:
        con = sqlite3.connect(self.path)
        con.execute("PRAGMA journal_mode=WAL")
        return con

    def _init_schema(self) -> None:
        with self._connect() as con:
            con.execute(
                """
                CREATE TABLE IF NOT EXISTS preference_state (
                    namespace TEXT NOT NULL,
                    faculty TEXT NOT NULL,
                    n_observations INTEGER NOT NULL,
                    alpha_json TEXT NOT NULL,
                    history_json TEXT NOT NULL,
                    updated_at REAL NOT NULL,
                    PRIMARY KEY(namespace, faculty)
                )
                """
            )

    def save(self, faculty: str, prior: DirichletPreference) -> None:
        with self._connect() as con:
            con.execute(
                """
                INSERT INTO preference_state(namespace, faculty, n_observations, alpha_json, history_json, updated_at)
                VALUES (?,?,?,?,?,?)
                ON CONFLICT(namespace, faculty) DO UPDATE SET
                    alpha_json=excluded.alpha_json,
                    history_json=excluded.history_json,
                    updated_at=excluded.updated_at
                """,
                (
                    self.namespace,
                    faculty,
                    int(prior.n_observations),
                    json.dumps(list(prior.alpha)),
                    json.dumps([{
                        "observation_index": int(h.observation_index),
                        "polarity": float(h.polarity),
                        "weight": float(h.weight),
                        "reason": h.reason,
                    } for h in prior.history[-128:]]),
                    time.time(),
                ),
            )

    def load(self, faculty: str) -> DirichletPreference | None:
        with self._connect() as con:
            row = con.execute(
                "SELECT n_observations, alpha_json, history_json FROM preference_state WHERE namespace=? AND faculty=?",
                (self.namespace, faculty),
            ).fetchone()
        if row is None:
            return None
        prior = DirichletPreference(int(row[0]))
        prior.alpha = list(json.loads(row[1]))
        prior.history = [PreferenceEvent(**e) for e in json.loads(row[2])]
        return prior


def feedback_polarity_from_text(text: str) -> tuple[float, float]:
    """Cheap deterministic sentiment lookup as a fallback.

    Returns ``(polarity, weight)``. Designed to be replaced by an LLM-driven
    sentiment classifier in production; here it just gives the architecture a
    working bootstrap so unit tests can exercise the loop.
    """

    s = text.lower()
    weight = min(1.0, 0.2 + 0.05 * len(s.split()))
    positives = ("thanks", "great", "perfect", "good", "concise", "love", "helpful")
    negatives = ("stop", "no ", "worse", "bad", "wrong", "annoying", "too many")
    if any(p in s for p in positives) and not any(n in s for n in negatives):
        return 1.0, float(weight)
    if any(n in s for n in negatives):
        return -1.0, float(weight)
    return 0.0, float(weight) * 0.1
