"""Small measurable checks for the finite active-inference implementation."""

from __future__ import annotations

from dataclasses import dataclass

from ..agent.active_inference import (
    ActiveInferenceAgent,
    TigerDoorEnv,
    build_tiger_pomdp,
    random_episode,
    run_episode,
)
from ..agent.invariants import POMDPInvariants


@dataclass(frozen=True)
class TigerValidationReport:
    """Active-vs-random smoke benchmark for the Tiger POMDP."""

    episodes: int
    active_success: float
    random_success: float
    active_reward: float
    random_reward: float
    invariant_status: str

    @property
    def reward_delta(self) -> float:
        return float(self.active_reward - self.random_reward)

    @property
    def status(self) -> str:
        if self.invariant_status != "pass":
            return "invalid_model"
        return "pass" if self.reward_delta >= 0.0 else "regressed"

    def as_dict(self) -> dict[str, float | int | str]:
        return {
            "episodes": self.episodes,
            "active_success": self.active_success,
            "random_success": self.random_success,
            "active_reward": self.active_reward,
            "random_reward": self.random_reward,
            "reward_delta": self.reward_delta,
            "invariant_status": self.invariant_status,
            "status": self.status,
        }


class ActiveInferenceValidator:
    """Runs a deterministic Tiger-domain validation without model downloads."""

    def tiger_smoke(self, *, seed: int = 0, episodes: int = 32) -> TigerValidationReport:
        pomdp = build_tiger_pomdp()
        invariant_status = POMDPInvariants().validate(pomdp, name="tiger_pomdp").status
        active = ActiveInferenceAgent(pomdp, horizon=1, learn=True)
        active_env = TigerDoorEnv(seed=seed + 101)
        random_env = TigerDoorEnv(seed=seed + 101)
        active_success = 0
        random_success = 0
        active_reward = 0.0
        random_reward = 0.0
        for _ in range(max(1, int(episodes))):
            ok, reward, _trace = run_episode(active, active_env, max_steps=3)
            rok, rreward = random_episode(random_env, max_steps=3)
            active_success += int(ok)
            random_success += int(rok)
            active_reward += float(reward)
            random_reward += float(rreward)
        n = max(1, int(episodes))
        return TigerValidationReport(
            episodes=n,
            active_success=active_success / n,
            random_success=random_success / n,
            active_reward=active_reward / n,
            random_reward=random_reward / n,
            invariant_status=invariant_status,
        )
