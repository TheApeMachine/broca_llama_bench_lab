"""CognitiveBackgroundWorker — the substrate's DMN daemon thread.

Runs three thermodynamic phases on every tick — even when the user has been
silent — so the substrate physically reorganizes itself between turns:

1. **Consolidation.** Episodic association edges decay multiplicatively and
   weak ones are pruned. A PageRank pass over the survivors finds central
   episodes; semantic facts whose provenance cites a central episode get a
   confidence boost so frequently-used knowledge becomes harder to forget.

2. **Separation.** Subjects sharing suspiciously many ``(predicate, object)``
   pairs are scored for Fristonian ambiguity (binary entropy of the
   disambiguation distribution) and emit an intrinsic cue, biasing the next
   reply toward a clarifying question rather than a coin-flip identity.

3. **Latent discovery.** Two stochastic subroutines: the SCM is "dreamt" —
   random ``(treatment, outcome)`` pairs are selected, ``do(·)`` interventions
   are run, and large average treatment effects are persisted as
   ``latent_causal_insight`` reflections; the episode graph is walked for
   transitive closure: a strongly-connected ``(A, B)`` and ``(B, C)`` trigger
   a cosine-similarity test on A and C's frames, creating a new ``(A, C)``
   edge if the test fires.

Each phase emits one or more reflection-shaped dicts so the chat CLI and
debugger can replay what the DMN did between turns.
"""

from __future__ import annotations

import logging
import math
import random
import threading
import time
from dataclasses import asdict
from typing import Any, Mapping, Sequence

import torch
import torch.nn.functional as F

from ..causal.causal_discovery import (
    build_scm_from_skeleton,
    local_predicate_cluster,
    orient_temporal_edges,
    pc_algorithm,
    project_rows_to_variables,
)
from ..causal.temporal import TemporalCausalTraceBuilder
from ..frame import CognitiveFrame, FacultyCandidate, FrameDimensions, SubwordProjector
from ..memory import ClaimTrust, SymbolicMemory
from ..symbolic.vsa import VSACodebook, bundle, cosine as vsa_cosine
from ..temporal.hawkes import fit_excitation_em
from ..workspace import IntrinsicCue
from .config import DMNConfig


logger = logging.getLogger(__name__)

_SUBWORD = SubwordProjector()


def stable_sketch(text: str, *, dim: int | None = None) -> torch.Tensor:
    """Local lexical sketch helper used by the DMN's consolidation phases.

    Defers to :class:`SubwordProjector` (the canonical lexical encoder); kept
    here only because the DMN body refers to ``stable_sketch`` directly.
    """

    if dim is None or dim == FrameDimensions.SKETCH_DIM:
        return _SUBWORD.encode(text)
    return SubwordProjector(dim=dim).encode(text)


class CognitiveBackgroundWorker:
    """Default Mode Network for the cognitive substrate.

    Runs three thermodynamic phases on every tick — even when the user has
    been silent — so the substrate physically reorganizes itself between
    turns:

      1. **Consolidation.** Episodic association edges decay multiplicatively
         and weak ones are pruned. A PageRank pass over the survivors finds
         the central episodes; the confidence of any semantic fact whose
         provenance cites a central episode is boosted, so frequently-used
         knowledge becomes harder to forget.
      2. **Separation.** Subjects that share suspiciously many predicate /
         object pairs are scored for Fristonian ambiguity (binary entropy of
         the disambiguation distribution) and emit an intrinsic cue so the
         next reply tends toward a clarifying question rather than committing
         to a coin-flip identity.
      3. **Latent discovery.** Two stochastic subroutines: the SCM is
         "dreamt" — random treatment / outcome pairs are selected, do(·)
         interventions are run, and large average treatment effects are
         persisted as ``latent_causal_insight`` reflections; and the episode
         graph is walked for transitive closure: a strongly-connected (A, B)
         and (B, C) trigger a cosine-similarity test on A and C's frames,
         creating a new (A, C) edge if the test fires.

    Each phase emits one or more reflection-shaped dicts so the chat CLI and
    debugger can replay what the DMN did between turns.
    """

    def __init__(
        self,
        mind: "SubstrateController",
        *,
        interval_s: float = 5.0,
        config: DMNConfig | None = None,
        rng: random.Random | None = None,
        motor_trainer: Any | None = None,
    ):
        self.mind = mind
        self.interval_s = max(0.1, float(interval_s))
        self._stop = threading.Event()
        self._wake = threading.Event()
        self._thread: threading.Thread | None = None
        self.iterations = 0
        self.last_error: str | None = None
        self.config = config if config is not None else DMNConfig()
        self._rng = rng or random.Random(0xB0CA1)
        self.last_phase_summary: dict[str, dict[str, Any]] = {}
        self.last_user_activity_at: float = time.time()
        self.motor_trainer = motor_trainer
        self.last_rem_summary: dict[str, Any] = {}
        self._snapshot_lock = threading.Lock()

    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def start(self) -> None:
        if self.running:
            return
        self._stop.clear()
        self._wake.clear()
        self._thread = threading.Thread(target=self._loop, name="broca-dmn", daemon=True)
        self._thread.start()
        logger.info("CognitiveBackgroundWorker.start: interval=%.3fs config=%s", self.interval_s, asdict(self.config))

    def stop(self, timeout: float = 2.0) -> None:
        self._stop.set()
        self._wake.set()
        if self._thread is not None:
            self._thread.join(timeout=max(0.0, float(timeout)))
        logger.info("CognitiveBackgroundWorker.stop: iterations=%d last_error=%s", self.iterations, self.last_error)

    def notify_work(self) -> None:
        self._wake.set()

    def run_once(self) -> list[dict]:
        """Run one DMN tick: queued ingest, idle phases, then consolidation."""

        tick_started = time.time()
        reflections: list[dict] = []
        phase_summary: dict[str, dict[str, Any]] = {}

        for name, fn in (
            ("relation_ingest", self._phase0_relation_ingest),
            ("consolidation", self._phase1_consolidation),
            ("separation", self._phase2_separation),
            ("latent_discovery", self._phase3_latent_discovery),
            ("chunk_compilation", self._phase4_chunk_compilation),
            ("tool_foraging", self._phase5_tool_foraging),
        ):
            phase_started = time.time()
            try:
                phase_reflections, summary = fn()
            except Exception:
                logger.exception("DMN phase %s failed", name)
                phase_summary[name] = {"error": True}
                continue
            reflections.extend(phase_reflections)
            summary["duration_ms"] = int(round((time.time() - phase_started) * 1000))
            summary["reflections"] = len(phase_reflections)
            phase_summary[name] = summary
            logger.info("DMN.phase=%s %s", name, summary)

        # Reactive belief consolidation runs after the autonomous phases so
        # any DMN-promoted edges or boosted confidences are visible to it.
        try:
            with self.mind.session.cognitive_state_lock:
                claim_reflections = self.mind.consolidate_once()
        except Exception:
            logger.exception("DMN claim consolidation failed")
            claim_reflections = []
        reflections.extend(claim_reflections)
        phase_summary["claim_consolidation"] = {"reflections": len(claim_reflections)}

        # REM sleep — only if the user has been quiet long enough.
        idle = max(0.0, time.time() - self.last_user_activity_at)
        rem_summary_update: dict[str, Any] | None = None
        if idle >= self.config.sleep_idle_seconds:
            phase_started = time.time()
            try:
                rem_reflections, rem_summary = self._rem_sleep()
            except Exception:
                logger.exception("DMN REM sleep failed")
                rem_summary = {"error": True}
                rem_reflections = []
            rem_summary["duration_ms"] = int(round((time.time() - phase_started) * 1000))
            rem_summary["idle_seconds"] = float(idle)
            rem_summary["reflections"] = len(rem_reflections)
            phase_summary["rem_sleep"] = rem_summary
            rem_summary_update = rem_summary
            reflections.extend(rem_reflections)
            logger.info("DMN.phase=rem_sleep %s", rem_summary)

        with self._snapshot_lock:
            if rem_summary_update is not None:
                self.last_rem_summary = rem_summary_update
            self.iterations += 1
            self.last_error = None
            self.last_phase_summary = phase_summary
        duration_ms = int(round((time.time() - tick_started) * 1000))
        logger.debug(
            "CognitiveBackgroundWorker.run_once: iteration=%d total_reflections=%d duration_ms=%d idle=%.1fs",
            self.iterations,
            len(reflections),
            duration_ms,
            idle,
        )
        try:
            with self._snapshot_lock:
                iteration = int(self.iterations)
            self.mind.event_bus.publish(
                "dmn.tick",
                {
                    "iteration": iteration,
                    "duration_ms": duration_ms,
                    "reflections": len(reflections),
                    "idle_seconds": float(idle),
                    "phase_summary": dict(phase_summary),
                },
            )
        except Exception:
            logger.exception("DMN tick: event publish failed")
        return reflections

    def state_snapshot(self) -> dict[str, Any]:
        """Return a consistent view of worker fields for live UIs (thread-safe)."""

        with self._snapshot_lock:
            last_tau = float(self.last_user_activity_at)
            return {
                "running": bool(self.running),
                "iterations": int(self.iterations),
                "interval_s": float(self.interval_s),
                "last_phase_summary": dict(self.last_phase_summary),
                "last_rem_summary": dict(self.last_rem_summary),
                "last_error": self.last_error,
                "idle_seconds": float(max(0.0, time.time() - last_tau)),
                "deferred_relation_ingest_pending": self.mind.deferred_relation_ingest_count(),
            }

    def mark_user_active(self) -> None:
        """Reset the idle clock when the user types something."""

        with self._snapshot_lock:
            self.last_user_activity_at = time.time()

    def _phase0_relation_ingest(self) -> tuple[list[dict], dict[str, Any]]:
        reflections = self.mind.process_deferred_relation_ingest()
        return reflections, {
            "processed": len(reflections),
            "pending": self.mind.deferred_relation_ingest_count(),
        }

    # ------------------------------------------------------------------ Phase 1

    def _phase1_consolidation(self) -> tuple[list[dict], dict[str, Any]]:
        with self.mind.session.cognitive_state_lock:
            cfg = self.config
            graph = self.mind.episode_graph
            memory = self.mind.memory

            decayed, pruned = graph.decay_all(gamma=cfg.decay_gamma, prune_below=cfg.decay_prune_below)
            centrality = graph.centrality(
                damping=0.85,
                iterations=cfg.centrality_iterations,
                min_weight=cfg.centrality_min_weight,
            )

            boosts: list[dict[str, Any]] = []
            if centrality:
                top_episodes = {ep for ep, score in centrality.items() if score >= cfg.centrality_boost_floor}
                if top_episodes:
                    # Boost any fact whose recorded episode_ids intersect the central set.
                    for subj, pred, _obj, _conf, evidence in memory.all_facts():
                        eps = evidence.get("episode_ids") if isinstance(evidence, dict) else None
                        if not isinstance(eps, list):
                            continue
                        parsed: list[int] = []
                        for e in eps:
                            try:
                                ei = int(e)
                            except (TypeError, ValueError):
                                continue
                            parsed.append(ei)
                        intersection = [ei for ei in parsed if ei in top_episodes]
                        if not intersection:
                            continue
                        # Scale the boost by the maximum centrality mass that touched this fact.
                        mass = max(centrality.get(ei, 0.0) for ei in intersection)
                        factor = 1.0 + (cfg.centrality_boost_factor - 1.0) * min(1.0, mass / max(cfg.centrality_boost_floor, 1e-6))
                        result = memory.boost_confidence(
                            subj,
                            pred,
                            factor=factor,
                            cap=cfg.centrality_boost_cap,
                            reason=f"pagerank_central(mass={mass:.4f})",
                        )
                        if result is None:
                            continue
                        obj, old_conf, new_conf = result
                        boosts.append({
                            "subject": subj,
                            "predicate": pred,
                            "object": obj,
                            "old_confidence": old_conf,
                            "new_confidence": new_conf,
                            "factor": float(factor),
                            "central_episode_mass": float(mass),
                            "central_episodes": intersection,
                        })
                        logger.debug(
                            "DMN.phase1.boost: subj=%r pred=%r %.4f -> %.4f factor=%.4f mass=%.4f episodes=%s",
                            subj,
                            pred,
                            old_conf,
                            new_conf,
                            float(factor),
                            float(mass),
                            intersection,
                        )

            reflections: list[dict] = []
            if boosts:
                reflections.append({
                    "kind": "consolidation_boost",
                    "boosts": boosts,
                })
            summary = {
                "decayed_edges": int(decayed),
                "pruned_edges": int(pruned),
                "central_nodes": len(centrality),
                "boosted_facts": len(boosts),
            }
            return reflections, summary

    # ------------------------------------------------------------------ Phase 2

    def _phase2_separation(self) -> tuple[list[dict], dict[str, Any]]:
        cfg = self.config
        memory = self.mind.memory
        ws = self.mind.workspace
        pairs = memory.overlapping_subject_pairs(min_shared=cfg.overlap_min_shared)
        emitted: list[dict[str, Any]] = []
        new_cues: list[IntrinsicCue] = []
        for pair in pairs[: max(0, cfg.overlap_max_cues)]:
            ratio = float(pair["overlap_ratio"])
            if ratio < cfg.overlap_ratio_floor:
                continue
            # Fristonian ambiguity ≈ binary entropy of the maximum-entropy
            # disambiguation distribution under the observed overlap. Total
            # overlap → 50/50 hypothesis posterior → ambiguity = log(2).
            p = 0.5 + 0.5 * (1.0 - ratio)  # slide from 0.5 (full overlap) toward 1.0 (none)
            p = max(1e-6, min(1 - 1e-6, p))
            ambiguity = -(p * math.log(p) + (1 - p) * math.log(1 - p))
            urgency = float(min(1.0, ambiguity / math.log(2)))
            cue_evidence = {
                "subject_a": pair["subject_a"],
                "subject_b": pair["subject_b"],
                "shared_count": pair["shared_count"],
                "overlap_ratio": ratio,
                "ambiguity_nats": float(ambiguity),
                "shared_predicates": [list(t) for t in pair["shared"]],
            }
            new_cues.append(
                IntrinsicCue(urgency=urgency, faculty="entity_ambiguity", evidence=cue_evidence, source="dmn")
            )
            emitted.append(cue_evidence | {"urgency": urgency})
            logger.info(
                "DMN.phase2.cue: %r ↔ %r ratio=%.3f ambiguity=%.4f nats urgency=%.3f",
                pair["subject_a"],
                pair["subject_b"],
                ratio,
                ambiguity,
                urgency,
            )

        with self.mind.session.cognitive_state_lock:
            ws.intrinsic_cues = [
                c for c in ws.intrinsic_cues if not (c.faculty == "entity_ambiguity" and getattr(c, "source", None) == "dmn")
            ]
            ws.intrinsic_cues.extend(new_cues)

        reflections: list[dict] = []
        if emitted:
            reflections.append({"kind": "separation_cue", "cues": emitted})

        summary = {
            "candidate_pairs": len(pairs),
            "cues_emitted": len(emitted),
        }
        return reflections, summary

    # ------------------------------------------------------------------ Phase 3

    def _phase3_latent_discovery(self) -> tuple[list[dict], dict[str, Any]]:
        reflections: list[dict] = []
        causal = self._causal_dreaming()
        reflections.extend(causal["reflections"])
        transitive = self._transitive_episode_closure()
        reflections.extend(transitive["reflections"])
        summary = {
            "causal_attempts": causal["attempts"],
            "causal_insights": causal["insights"],
            "transitive_pairs_examined": transitive["pairs_examined"],
            "transitive_edges_added": transitive["edges_added"],
        }
        return reflections, summary

    def _causal_dreaming(self) -> dict[str, Any]:
        with self.mind.session.cognitive_state_lock:
            cfg = self.config
            scm = getattr(self.mind, "scm", None)
            if scm is None:
                return {"reflections": [], "attempts": 0, "insights": 0}
            endogenous = list(scm.endogenous_names)
            if len(endogenous) < 2:
                return {"reflections": [], "attempts": 0, "insights": 0}

            attempts = 0
            insights: list[dict[str, Any]] = []
            for _ in range(max(0, int(cfg.dream_attempts_per_tick))):
                attempts += 1
                treatment, outcome = self._rng.sample(endogenous, 2)
                try:
                    t_dom = scm.domains.get(treatment)
                    o_dom = scm.domains.get(outcome)
                    if not t_dom or not o_dom or len(t_dom) < 2 or len(o_dom) < 2:
                        continue
                    t_pos, t_neg = t_dom[0], t_dom[1]
                    outcome_value = o_dom[0]
                    p_pos = scm.probability({outcome: outcome_value}, given={}, interventions={treatment: t_pos})
                    p_neg = scm.probability({outcome: outcome_value}, given={}, interventions={treatment: t_neg})
                except (KeyError, ValueError, RuntimeError):
                    logger.debug("DMN.phase3.dream: failed treatment=%s outcome=%s", treatment, outcome, exc_info=True)
                    continue
                ate = float(p_pos - p_neg)
                logger.debug(
                    "DMN.phase3.dream: do(%s=%s)→P(%s=%s)=%.4f vs do(%s=%s)→%.4f ate=%.4f",
                    treatment,
                    t_pos,
                    outcome,
                    outcome_value,
                    p_pos,
                    treatment,
                    t_neg,
                    p_neg,
                    ate,
                )
                if abs(ate) < cfg.dream_ate_insight_threshold:
                    continue
                relation_label = scm.labels.get("positive_effect" if ate >= 0 else "negative_effect")
                relation = relation_label or ("causes_increase" if ate >= 0 else "causes_decrease")
                evidence = {
                    "treatment": treatment,
                    "outcome": outcome,
                    "outcome_value": outcome_value,
                    "treatment_values": [t_pos, t_neg],
                    "p_do_positive": float(p_pos),
                    "p_do_negative": float(p_neg),
                    "ate": ate,
                    "instrument": "dmn_causal_dream",
                }
                dedupe = f"latent_causal_insight:{treatment}->{outcome}:{relation}"
                reflection_id = self.mind.memory.record_reflection(
                    "latent_causal_insight",
                    treatment,
                    relation,
                    f"dreamt that intervening on {treatment} {relation} {outcome} (ATE={ate:+.2f})",
                    evidence,
                    dedupe_key=dedupe,
                )
                if reflection_id is None:
                    continue
                insights.append({"id": reflection_id, "kind": "latent_causal_insight", **evidence})
                logger.info(
                    "DMN.phase3.dream.insight: id=%d %s %s %s ate=%+.3f",
                    reflection_id,
                    treatment,
                    relation,
                    outcome,
                    ate,
                )

            return {"reflections": insights, "attempts": attempts, "insights": len(insights)}

    def _transitive_episode_closure(self) -> dict[str, Any]:
        cfg = self.config
        graph = self.mind.episode_graph
        edges = graph.edges(min_weight=cfg.transitive_min_pair_weight)
        if not edges:
            return {"reflections": [], "pairs_examined": 0, "edges_added": 0}

        # Index neighbors by node so we can spot A–B–C chains efficiently.
        neighbors: dict[int, list[tuple[int, float]]] = {}
        for lo, hi, w in edges:
            neighbors.setdefault(lo, []).append((hi, w))
            neighbors.setdefault(hi, []).append((lo, w))

        pairs_examined = 0
        added: list[dict[str, Any]] = []
        text_encoder = getattr(self.mind, "text_encoder", None)
        for hub, hub_edges in neighbors.items():
            if len(hub_edges) < 2 or len(added) >= cfg.transitive_max_new_edges:
                continue
            # Sample a pair of distinct neighbors; the random thermal kick is
            # what lets the system jump between local minima rather than
            # rediscovering the same closures every tick.
            sampled = self._rng.sample(hub_edges, k=min(len(hub_edges), 4))
            for i in range(len(sampled)):
                for j in range(i + 1, len(sampled)):
                    a, _w_a = sampled[i]
                    c, _w_c = sampled[j]
                    if a == c:
                        continue
                    pairs_examined += 1
                    if graph.weight(a, c) > 0.0:
                        continue
                    cosine = self._episode_frame_similarity(a, c, text_encoder=text_encoder)
                    logger.debug(
                        "DMN.phase3.transitive: hub=%d a=%d c=%d cosine=%s",
                        hub,
                        a,
                        c,
                        ("%.4f" % cosine) if cosine is not None else "n/a",
                    )
                    if cosine is None or cosine < cfg.transitive_cosine_threshold:
                        continue
                    delta = float(cosine)
                    graph.bump(a, c, delta=delta)
                    added.append({
                        "lo": min(a, c),
                        "hi": max(a, c),
                        "via_hub": hub,
                        "cosine": float(cosine),
                        "weight_added": delta,
                    })
                    logger.info("DMN.phase3.transitive.edge: %d↔%d via %d cosine=%.4f", a, c, hub, cosine)
                    if len(added) >= cfg.transitive_max_new_edges:
                        break
                if len(added) >= cfg.transitive_max_new_edges:
                    break

        reflections: list[dict] = []
        if added:
            reflections.append({"kind": "transitive_episode_closure", "edges": added})
        return {"reflections": reflections, "pairs_examined": pairs_examined, "edges_added": len(added)}

    def _episode_frame_similarity(self, a: int, b: int, *, text_encoder) -> float | None:
        row_a = self.mind.journal.fetch(a)
        row_b = self.mind.journal.fetch(b)
        if row_a is None or row_b is None:
            return None
        frame_a = CognitiveFrame.from_episode_row(row_a)
        frame_b = CognitiveFrame.from_episode_row(row_b)
        text_a = " ".join(_frame_descriptor_tokens(frame_a))
        text_b = " ".join(_frame_descriptor_tokens(frame_b))
        if not text_a.strip() or not text_b.strip():
            return None
        try:
            return float(_cosine(_text_vector(text_a, text_encoder), _text_vector(text_b, text_encoder)))
        except (RuntimeError, ValueError):
            logger.debug("DMN.phase3.transitive.similarity_failed a=%d b=%d", a, b, exc_info=True)
            return None

    # ------------------------------------------------------------------ Phase 4

    def _phase4_chunk_compilation(self) -> tuple[list[dict], dict[str, Any]]:
        """Detect repeated motifs in the workspace journal and compile them into macros.

        Implements the proceduralization side of the System-2 → System-1
        transition: every repeated reasoning trajectory becomes a single
        ``CompiledMacro`` whose mean feature vector is what the
        :class:`TrainableFeatureGraft` injects when the substrate next sees the
        macro's prefix.
        """

        compiler = getattr(self.mind, "chunking_compiler", None)
        if compiler is None:
            return [], {"compiled": 0, "candidates": 0, "scanned": 0}
        result = compiler.run_once()
        return list(result.get("reflections") or []), {
            "compiled": int(result.get("compiled", 0)),
            "candidates": int(result.get("candidates", 0)),
            "scanned": int(result.get("scanned", 0)),
        }

    # ------------------------------------------------------------------ Phase 5

    def _phase5_tool_foraging(self) -> tuple[list[dict], dict[str, Any]]:
        """Decide whether the substrate should synthesize a new native tool.

        We do **not** synthesize the tool here — that requires an LLM call to
        produce candidate Python source and is therefore an external,
        user-or-agent-driven step.  What we *do* run during DMN time is the
        active-inference math itself: when the unified faculty's posterior is
        confused (high entropy) and there are few existing tools, the EFE of
        ``synthesize_tool`` collapses below the alternatives, and we emit a
        ``tool_synthesis_recommended`` reflection so a downstream agent
        knows to act.
        """

        slot = self.mind.tool_foraging
        agent = slot.agent
        unified = getattr(self.mind, "unified_agent", None)
        registry = getattr(self.mind, "tool_registry", None)
        if agent is None or unified is None or registry is None:
            return [], {"ran": False}

        try:
            coupled = unified.decide()
        except Exception:
            logger.exception("DMN.phase5.tool_foraging: unified_agent.decide failed")
            return [], {"ran": False, "error": True}

        if coupled.faculty == "spatial":
            posterior = list(coupled.spatial_decision.posterior_over_policies)
        else:
            posterior = list(coupled.causal_decision.posterior_over_policies)

        n = len(posterior)
        if n < 2:
            insufficient_prior = 0.5
        else:
            h = belief_entropy(posterior)
            h_max = math.log(n)
            insufficient_prior = max(1e-6, min(1 - 1e-6, h / max(h_max, 1e-9)))

        agent.update_belief(insufficient_prior=float(insufficient_prior))
        decision = agent.decide()
        recommended = decision.action_name == "synthesize_tool"

        reflections: list[dict] = []
        if recommended:
            evidence = {
                "action": decision.action_name,
                "insufficient_prior": float(insufficient_prior),
                "n_existing_tools": int(registry.count()),
                "policy_efe": [
                    {
                        "policy": list(int(a) for a in p.policy),
                        "expected_free_energy": float(p.expected_free_energy),
                    }
                    for p in decision.policies
                ],
                "instrument": "dmn_tool_foraging",
                "coupled_faculty": coupled.faculty,
            }
            reflection_id = self.mind.memory.record_reflection(
                "tool_synthesis_recommended",
                "tool_foraging",
                "synthesize_tool",
                f"EFE math recommends synthesizing a new tool (insufficient_prior={insufficient_prior:.3f})",
                evidence,
                dedupe_key=f"tool_synthesis_recommended:{int(registry.count())}",
            )
            if reflection_id is not None:
                reflections.append({"id": reflection_id, "kind": "tool_synthesis_recommended", **evidence})
                logger.info(
                    "DMN.phase5.tool_foraging.recommend: id=%d insufficient_prior=%.3f n_tools=%d",
                    reflection_id,
                    insufficient_prior,
                    registry.count(),
                )

        summary = {
            "ran": True,
            "recommended": recommended,
            "insufficient_prior": float(insufficient_prior),
            "n_existing_tools": int(registry.count()),
            "chosen_action": decision.action_name,
        }
        return reflections, summary

    # ------------------------------------------------------------------ REM sleep

    def _rem_sleep(self) -> tuple[list[dict], dict[str, Any]]:
        """REM-style consolidation: motor learning + causal discovery + Hawkes refit.

        Runs only when the user has been idle long enough that a multi-second
        compute spike won't affect interactive latency. Each subroutine is
        wrapped so a failure in one doesn't block the others.
        """

        cfg = self.config
        summary: dict[str, Any] = {}
        reflections: list[dict] = []

        # 1. Motor learning — re-train the Broca grafts on recent journals.
        motor = {"ran": False}
        if self.motor_trainer is None:
            summary["motor"] = motor
        else:
            replay_buf = self.mind.motor_replay
            lock = self.mind.session.cognitive_state_lock
            with lock:
                replay = list(replay_buf)[-cfg.sleep_max_replay :]
            try:
                step = self.motor_trainer.step(replay)
            except Exception:
                logger.exception("REM.motor: step failed")
                step = {"skipped": True, "reason": "exception"}
            motor.update(step)
            motor["ran"] = True
            if step.get("skipped") is False:
                reflections.append({"kind": "rem_motor_learning", **step})
            summary["motor"] = motor

        # 2. Hawkes refit — relearn excitation matrix from recent journal events.
        hawkes_summary: dict[str, Any] = {"ran": False}
        try:
            recent = self.mind.journal.recent(limit=128)
        except Exception:
            logger.exception("REM.hawkes: journal recent failed")
            recent = []
        events: list[tuple[str, float]] = []
        for row in recent:
            channel = str(row.get("intent", "") or "unknown")
            ts = float(row.get("ts", 0.0))
            events.append((channel, ts))
        if len(events) >= cfg.sleep_hawkes_min_events:
            channels = sorted({c for c, _ in events})
            try:
                mu, alpha = fit_excitation_em(events, channels, beta=self.mind.hawkes.beta)
            except Exception:
                logger.exception("REM.hawkes: EM fit failed")
                mu, alpha = None, None
            if mu is not None and alpha is not None:
                with self.mind.session.cognitive_state_lock:
                    self.mind.hawkes.refit(channels, mu, alpha)
                    try:
                        self.mind.hawkes_persistence.save(self.mind.hawkes)
                    except Exception:
                        logger.exception("REM.hawkes: persistence save failed")
                hawkes_summary = {
                    "ran": True,
                    "channels": channels,
                    "events": len(events),
                    "mu_max": float(max(mu)) if mu else 0.0,
                    "alpha_norm": float(sum(sum(abs(x) for x in row) for row in alpha)),
                }
                reflections.append({"kind": "rem_hawkes_refit", **hawkes_summary})
        summary["hawkes"] = hawkes_summary

        # 3. Causal discovery — local PC on a small predicate cluster, then rebuild SCM.
        cd_summary: dict[str, Any] = {"ran": False}
        try:
            full_rows = self._collect_observations_for_pc()
        except Exception:
            logger.exception("REM.causal_discovery: observation collection failed")
            full_rows = []
        observations: list[dict[str, object]] = []
        pc_variables: list[str] | None = None
        if len(full_rows) >= cfg.sleep_min_observations_for_pc:
            all_vars = sorted({str(k) for row in full_rows for k in row})
            if len(all_vars) > int(cfg.sleep_pc_max_variables):
                cluster = local_predicate_cluster(
                    full_rows,
                    max_variables=int(cfg.sleep_pc_max_variables),
                    rng=self._rng,
                )
                observations = project_rows_to_variables(full_rows, cluster)
                pc_variables = cluster
            else:
                observations = full_rows
                pc_variables = None
        if len(observations) >= cfg.sleep_min_observations_for_pc:
            try:
                graph = pc_algorithm(
                    observations,
                    pc_variables,
                    alpha=cfg.sleep_pc_alpha,
                    max_conditioning_size=int(cfg.sleep_pc_max_conditioning_size),
                )
                graph = orient_temporal_edges(graph)
                if graph.directed_edges or graph.undirected_edges:
                    new_scm = build_scm_from_skeleton(graph, observations)
                    self.mind.discovered_scm = new_scm
                    cd_summary = {
                        "ran": True,
                        "n_observations": len(observations),
                        "n_predicate_columns": len(graph.variables),
                        "local_pc": pc_variables is not None,
                        "directed_edges": [list(e) for e in sorted(graph.directed_edges)],
                        "undirected_edges": [sorted(list(e)) for e in graph.undirected_edges],
                        "variables": list(graph.variables),
                    }
                    reflections.append({"kind": "rem_causal_discovery", **cd_summary})
            except Exception:
                logger.exception("REM.causal_discovery: PC algorithm failed")
        summary["causal_discovery"] = cd_summary

        # 4. Persist preference + ontology to disk.
        try:
            self.mind.preference_persistence.save("spatial", self.mind.spatial_preference)
            self.mind.preference_persistence.save("causal", self.mind.causal_preference)
            self.mind.ontology_persistence.save(self.mind.ontology)
            self.mind.conformal_calibration.persist(self.mind.relation_conformal, "relation_extraction")
        except Exception:
            logger.exception("REM.persist: save failed")

        return reflections, summary

    def _collect_observations_for_pc(self) -> list[dict[str, object]]:
        """Build a row-per-subject observation table for PC discovery.

        Each row is one subject; columns are predicates; cells are the stored
        objects. Rows missing a column are dropped per-pair by the CI test, so
        sparse coverage isn't fatal.
        """

        rows: list[dict[str, object]] = []
        for subject in self.mind.memory.subjects():
            record = {pred: obj for pred, obj, _conf, _ev in self.mind.memory.records_for_subject(subject)}
            if record:
                rows.append(record)
        journal_rows = self.mind.journal.recent(limit=int(self.config.sleep_max_replay))
        rows.extend(TemporalCausalTraceBuilder(journal_rows).build_rows())
        return rows

    def _loop(self) -> None:
        while not self._stop.is_set():
            self._wake.wait(self.interval_s)
            self._wake.clear()
            if self._stop.is_set():
                return
            try:
                self.run_once()
            except Exception as exc:  # pragma: no cover - background safety net
                logger.exception("Broca background DMN loop failed")
                self.last_error = repr(exc)
