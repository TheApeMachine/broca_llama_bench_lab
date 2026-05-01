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
import re
import sqlite3
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

logger = logging.getLogger(__name__)

_HISTORY_MAXLEN = 128


@dataclass
class PreferenceEvent:
    observation_index: int
    polarity: float
    weight: float
    reason: str
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


def _preference_event_from_dict(d: dict) -> PreferenceEvent:
    ts_raw = d.get("timestamp")
    if isinstance(ts_raw, str) and ts_raw.strip():
        ts = datetime.fromisoformat(ts_raw.replace("Z", "+00:00"))
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
    else:
        ts = datetime.fromtimestamp(0, tz=timezone.utc)
    return PreferenceEvent(
        observation_index=int(d["observation_index"]),
        polarity=float(d["polarity"]),
        weight=float(d["weight"]),
        reason=str(d.get("reason", "")),
        timestamp=ts,
    )


class DirichletPreference:
    """Dirichlet-conjugate prior over ``C`` for a categorical POMDP.

    Concentration ``α_i`` keeps a running pseudocount of how often observation
    ``i`` was preferred. Mean preference is ``α_i / Σα``. Variance is
    ``α_i (Σα - α_i) / (Σα)² (Σα + 1)`` — small when the substrate has many
    observations, large when it has few, which is exactly the right behavior
    for online preference learning.
    """

    def __init__(
        self,
        n_observations: int,
        *,
        prior_strength: float = 1.0,
        initial_C: Sequence[float] | None = None,
    ):
        if n_observations <= 0:
            raise ValueError("n_observations must be positive")
        self.n_observations = int(n_observations)
        self.prior_strength = float(prior_strength)
        if initial_C is None:
            self.alpha = [self.prior_strength] * self.n_observations
        else:
            parsed: list[float] = []
            for i, x in enumerate(initial_C):
                try:
                    v = float(x)
                except (TypeError, ValueError) as exc:
                    raise ValueError(f"initial_C[{i}]={x!r} is not numeric") from exc
                if v < 0:
                    raise ValueError(
                        f"initial_C[{i}]={x!r} (value {v}) must be non-negative"
                    )
                parsed.append(v)
            if len(parsed) != self.n_observations:
                raise ValueError("initial_C length disagrees with n_observations")
            base = [max(1e-6, v) for v in parsed]
            total = sum(base)
            self.alpha = [
                a * self.prior_strength * self.n_observations / total for a in base
            ]
        self.history: deque[PreferenceEvent] = deque(maxlen=_HISTORY_MAXLEN)

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
        safe_total = max(total, max(1e-6 * self.n_observations, 1e-3))
        denom = safe_total * safe_total * (safe_total + 1.0)
        out = []
        for a in self.alpha:
            out.append(float(a * (safe_total - a) / denom))
        return out

    def update(
        self,
        observation_index: int,
        *,
        polarity: float = 1.0,
        weight: float = 1.0,
        reason: str = "",
        epistemic_alpha_floor: float | None = None,
    ) -> None:
        """Update the Dirichlet given one labeled observation.

        ``polarity > 0`` increases the pseudocount on ``observation_index``;
        ``polarity < 0`` shrinks it (multiplicatively, via ``exp(polarity *
        weight)``) so the value stays strictly positive — the conjugate prior
        is only valid on the open simplex.

        ``epistemic_alpha_floor`` clamps the target concentration after a
        negative update so listening / information-seeking observations retain
        probability mass when external ambiguity signals demand it.
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
            if epistemic_alpha_floor is not None:
                self.alpha[i] = max(float(epistemic_alpha_floor), self.alpha[i])
        self.history.append(
            PreferenceEvent(
                observation_index=i,
                polarity=float(polarity),
                weight=w,
                reason=str(reason),
            )
        )
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


_NEGATIVE_SENTIMENT = re.compile(
    r"\b(?:stop|worse|bad|wrong|annoying)\b|\btoo many\b|\bno\s+(?:thanks?|thank you)\b",
)
_POSITIVE_SENTIMENT = re.compile(
    r"\b(?:thanks|great|perfect|good|concise|love|helpful)\b",
    re.I,
)


class PersistentPreference:
    """Disk-backed Dirichlet store keyed by ``(namespace, faculty)``."""

    def __init__(self, path: str | Path, *, namespace: str = "main"):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.namespace = namespace
        self._conn: sqlite3.Connection | None = None
        self._conn_lock = threading.Lock()
        self._schema_migrated: bool = False
        self._init_schema()

    def _conn_get(self) -> sqlite3.Connection:
        if self._conn is None:
            self._conn = sqlite3.connect(str(self.path), timeout=30.0, check_same_thread=False)
            self._conn.execute("PRAGMA journal_mode=WAL")
        return self._conn

    def close(self) -> None:
        with self._conn_lock:
            if self._conn is not None:
                self._conn.close()
                self._conn = None
            self._schema_migrated = False

    def __del__(self) -> None:  # pragma: no cover - best-effort cleanup
        try:
            self.close()
        except Exception:
            pass

    def _maybe_migrate_schema(self, con: sqlite3.Connection) -> None:
        if self._schema_migrated:
            return
        self._migrate_schema(con)
        self._schema_migrated = True

    def _migrate_schema(self, con: sqlite3.Connection) -> None:
        cols = {
            row[1]
            for row in con.execute("PRAGMA table_info(preference_state)").fetchall()
        }
        if "prior_strength" not in cols:
            con.execute(
                "ALTER TABLE preference_state ADD COLUMN prior_strength REAL NOT NULL DEFAULT 1.0"
            )

    def _init_schema(self) -> None:
        with self._conn_lock:
            con = self._conn_get()
            with con:
                con.execute(
                    """
                    CREATE TABLE IF NOT EXISTS preference_state (
                        namespace TEXT NOT NULL,
                        faculty TEXT NOT NULL,
                        n_observations INTEGER NOT NULL,
                        prior_strength REAL NOT NULL DEFAULT 1.0,
                        alpha_json TEXT NOT NULL,
                        history_json TEXT NOT NULL,
                        updated_at REAL NOT NULL,
                        PRIMARY KEY(namespace, faculty)
                    )
                    """
                )
                self._maybe_migrate_schema(con)

    def save(self, faculty: str, prior: DirichletPreference) -> None:
        with self._conn_lock:
            con = self._conn_get()
            with con:
                self._maybe_migrate_schema(con)
                con.execute(
                    """
                    INSERT INTO preference_state(
                        namespace, faculty, n_observations, prior_strength,
                        alpha_json, history_json, updated_at
                    )
                    VALUES (?,?,?,?,?,?,?)
                    ON CONFLICT(namespace, faculty) DO UPDATE SET
                        n_observations=excluded.n_observations,
                        prior_strength=excluded.prior_strength,
                        alpha_json=excluded.alpha_json,
                        history_json=excluded.history_json,
                        updated_at=excluded.updated_at
                    """,
                    (
                        self.namespace,
                        faculty,
                        int(prior.n_observations),
                        float(prior.prior_strength),
                        json.dumps(list(prior.alpha)),
                        json.dumps(
                            [
                                {
                                    "observation_index": int(h.observation_index),
                                    "polarity": float(h.polarity),
                                    "weight": float(h.weight),
                                    "reason": h.reason,
                                    "timestamp": h.timestamp.isoformat(),
                                }
                                for h in prior.history
                            ]
                        ),
                        time.time(),
                    ),
                )

    def load(self, faculty: str) -> DirichletPreference | None:
        with self._conn_lock:
            con = self._conn_get()
            with con:
                self._maybe_migrate_schema(con)
                row = con.execute(
                    "SELECT n_observations, prior_strength, alpha_json, history_json "
                    "FROM preference_state WHERE namespace=? AND faculty=?",
                    (self.namespace, faculty),
                ).fetchone()
        if row is None:
            return None
        n_obs, prior_strength, alpha_js, hist_js = row
        n_exp = int(n_obs)
        ps = float(prior_strength) if prior_strength is not None else 1.0
        try:
            raw_alpha = json.loads(alpha_js)
        except json.JSONDecodeError as exc:
            raise ValueError(f"PreferenceStore.load({faculty!r}): invalid alpha_json") from exc
        if not isinstance(raw_alpha, list):
            raise ValueError(
                f"PreferenceStore.load({faculty!r}): alpha must be a JSON list, got {type(raw_alpha).__name__}",
            )
        if len(raw_alpha) != n_exp:
            raise ValueError(
                f"PreferenceStore.load({faculty!r}): alpha length {len(raw_alpha)} != n_observations {n_exp}",
            )
        parsed_alpha: list[float] = []
        for i, x in enumerate(raw_alpha):
            try:
                v = float(x)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"PreferenceStore.load({faculty!r}): alpha[{i}]={x!r} is not numeric",
                ) from exc
            if v < 0:
                raise ValueError(
                    f"PreferenceStore.load({faculty!r}): alpha[{i}]={v!r} must be non-negative",
                )
            parsed_alpha.append(v)
        prior = DirichletPreference(n_exp, prior_strength=ps)
        prior.alpha = parsed_alpha
        prior.history = deque(
            (_preference_event_from_dict(e) for e in json.loads(hist_js)),
            maxlen=_HISTORY_MAXLEN,
        )
        return prior


def feedback_polarity_from_text(text: str) -> tuple[float, float]:
    """Cheap deterministic sentiment lookup as a fallback.

    Returns ``(polarity, weight)``. Designed to be replaced by an LLM-driven
    sentiment classifier in production; here it just gives the architecture a
    working bootstrap so unit tests can exercise the loop.
    """

    s = text.lower()
    weight = min(1.0, 0.2 + 0.05 * len(s.split()))
    negative_hit = bool(_NEGATIVE_SENTIMENT.search(s))
    positive_hit = bool(_POSITIVE_SENTIMENT.search(s))
    if positive_hit and not negative_hit:
        return 1.0, float(weight)
    if negative_hit:
        return -1.0, float(weight)
    return 0.0, float(weight) * 0.1
