"""Tiger-door environment for active-inference experiments."""

from __future__ import annotations

import random

from .pomdp_builder import POMDPBuilder


class TigerDoorEnv:
    """Noisy listen/open environment for active-inference experiments."""

    def __init__(self, seed: int = 0) -> None:
        self.reliability = POMDPBuilder().listen_channel_reliability(n_hidden_states=2)
        self.rng = random.Random(seed)
        self.hidden_state = 0

    def reset(self) -> int:
        self.hidden_state = 0 if self.rng.random() < 0.5 else 1

        return self.hidden_state

    def step(self, action_name: str) -> tuple[str, float, bool]:
        state = self.hidden_state

        if action_name == "listen":
            return self._listen_observation(state)

        if action_name == "open_left":
            return ("reward", 1.0, True) if state == 0 else ("punish", -2.0, True)

        if action_name == "open_right":
            return ("reward", 1.0, True) if state == 1 else ("punish", -2.0, True)

        raise KeyError(action_name)

    def _listen_observation(self, state: int) -> tuple[str, float, bool]:
        if self.rng.random() < self.reliability:
            observation = "hear_left" if state == 0 else "hear_right"
        else:
            observation = "hear_right" if state == 0 else "hear_left"

        return observation, -0.01, False
