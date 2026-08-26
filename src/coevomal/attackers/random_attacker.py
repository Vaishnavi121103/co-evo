"""Random-search attacker (lower-bound baseline).

Applies random functionality-preserving mutations with no learning. It is
the weakest attacker and calibrates how much the RL agents actually gain
over blind mutation, mirroring the random-action baseline reported in the
gym-malware line of work.
"""

from __future__ import annotations

import numpy as np

from coevomal.attackers.base import Attacker
from coevomal.environment.malware_env import MalwareEvasionEnv


class RandomAttacker(Attacker):
    def __init__(self, n_actions: int, seed: int = 0) -> None:
        self.n_actions = int(n_actions)
        self.rng = np.random.default_rng(seed)

    def train(self, env: MalwareEvasionEnv, episodes: int) -> None:  # no-op
        return None

    def act(self, obs: np.ndarray, greedy: bool = True) -> int:
        return int(self.rng.integers(0, self.n_actions))
