"""DMNConfig — thresholds for the Default Mode Network's idle phases.

These knobs are deliberately exposed: the DMN's behavior is a function of
how aggressively the user wants the substrate to forget, disambiguate, and
speculate during idle cycles. All have safe defaults; the chat CLI can
override them via environment variables or constructor arguments.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class DMNConfig:
    """Tunable thresholds for the DMN's consolidation, separation, discovery,
    chunking, and REM phases."""

    # Phase 1 — consolidation
    decay_gamma: float = 0.99
    decay_prune_below: float = 0.01
    centrality_iterations: int = 20
    centrality_min_weight: float = 0.0
    centrality_boost_floor: float = 0.05  # only boost facts whose central episode beats this PageRank mass
    centrality_boost_factor: float = 1.05
    centrality_boost_cap: float = 0.999

    # Phase 2 — separation
    overlap_min_shared: int = 2
    overlap_ratio_floor: float = 0.66
    overlap_max_cues: int = 4

    # Phase 3 — latent discovery
    dream_attempts_per_tick: int = 3
    dream_ate_insight_threshold: float = 0.4
    transitive_min_pair_weight: float = 0.5
    transitive_cosine_threshold: float = 0.55
    transitive_max_new_edges: int = 4

    # REM sleep — trigger when the user has been quiet this long.
    sleep_idle_seconds: float = 600.0
    sleep_max_replay: int = 32
    sleep_min_observations_for_pc: int = 24
    sleep_pc_alpha: float = 0.05
    sleep_pc_max_variables: int = 20
    sleep_pc_max_conditioning_size: int = 3
    sleep_hawkes_min_events: int = 6
