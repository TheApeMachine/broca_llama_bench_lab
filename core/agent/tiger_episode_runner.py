"""Episode runner for the tiger-door active-inference environment."""

from __future__ import annotations

import logging

from .active_agent import ActiveInferenceAgent
from .tiger_door_env import TigerDoorEnv

logger = logging.getLogger(__name__)


class TigerEpisodeRunner:
    """Run active or random episodes inside ``TigerDoorEnv``."""

    def run(
        self,
        agent: ActiveInferenceAgent,
        env: TigerDoorEnv,
        *,
        max_steps: int = 3,
    ) -> tuple[bool, float, list[dict]]:
        pomdp = agent.pomdp
        env.reset()
        agent.reset_belief()
        trace = []
        total = 0.0
        success = False

        for _ in range(max_steps):
            decision = agent.decide()

            if decision.action is None:
                raise ValueError(
                    "run_episode: agent.decide() returned no action (empty policy); "
                    "use horizon >= 1 for TigerDoorEnv episodes."
                )

            obs_name, reward, done = env.step(decision.action_name)

            if obs_name not in pomdp.observation_names:
                raise ValueError(
                    f"run_episode: unexpected observation name {obs_name!r}; "
                    f"allowed {list(pomdp.observation_names)}"
                )

            observation = pomdp.observation_names.index(obs_name)
            posterior = agent.update(decision.action, observation)
            logger.debug(
                "run_episode: action=%s -> obs=%s reward=%.2f done=%s",
                decision.action_name,
                obs_name,
                reward,
                done,
            )
            total += reward
            success = success or obs_name == "reward"
            trace.append(
                {
                    "action": decision.action_name,
                    "observation": obs_name,
                    "reward": reward,
                    "posterior": {
                        name: round(float(probability), 3)
                        for name, probability in zip(pomdp.state_names, posterior)
                    },
                }
            )

            if done:
                break

        return success, total, trace

    def random(self, env: TigerDoorEnv, *, max_steps: int = 3) -> tuple[bool, float]:
        env.reset()
        total = 0.0
        success = False

        for _ in range(max_steps):
            action = env.rng.choice(["listen", "open_left", "open_right"])
            observation, reward, done = env.step(action)
            total += reward
            success = success or observation == "reward"

            if done:
                break

        return success, total
