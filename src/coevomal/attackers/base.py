"""Attacker interface and shared evaluation logic.

An attacker is an RL agent that learns, against a *snapshot* of the current
defender, a policy of functionality-preserving mutations that drive a
malicious sample below the defender's decision threshold within a query
budget.

Every concrete attacker implements :meth:`train` (learn against an env) and
:meth:`act` (greedy action for evaluation). The shared
:meth:`evaluate_evasion` runs the learned greedy policy over a pool of
malicious samples and reports the evasion rate, the mean queries spent, and
the actual evasive feature vectors discovered -- the latter feed the
defender's adversarial retraining.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np

from coevomal.environment.malware_env import MalwareEvasionEnv


@dataclass
class EvasionResult:
    evasion_rate: float
    mean_queries: float
    evasive_samples: np.ndarray = field(default_factory=lambda: np.empty((0,)))
    n_evaluated: int = 0


class Attacker(ABC):
    """Abstract RL evasion agent."""

    @abstractmethod
    def train(self, env: MalwareEvasionEnv, episodes: int) -> None:
        """Learn a policy against ``env`` (which wraps a frozen defender)."""

    @abstractmethod
    def act(self, obs: np.ndarray, greedy: bool = True) -> int:
        """Return an action for a single observation."""

    def reset_policy(self) -> None:
        """Optionally re-initialise the policy at the start of a round.

        Default is a no-op: agents keep (warm-start) their policy across
        rounds, which is realistic for a persistent adversary. Override to
        study cold-start attackers.
        """

    # ---- shared evaluation --------------------------------------------------
    def evaluate_evasion(
        self, env: MalwareEvasionEnv, malicious_pool: np.ndarray
    ) -> EvasionResult:
        """Greedily roll out the learned policy over ``malicious_pool``."""
        evaded = 0
        queries: list[int] = []
        found: list[np.ndarray] = []
        for i in range(malicious_pool.shape[0]):
            q_before = env.total_queries
            obs = env.reset(malicious_pool[i])
            done = False
            info: dict = {}
            while not done:
                action = self.act(obs, greedy=True)
                obs, _reward, done, info = env.step(action)
            queries.append(env.total_queries - q_before)
            if info.get("evaded", False):
                evaded += 1
                found.append(env.current_sample())
        n = malicious_pool.shape[0]
        samples = np.vstack(found) if found else np.empty((0, malicious_pool.shape[1]),
                                                          dtype=np.float32)
        return EvasionResult(
            evasion_rate=evaded / max(1, n),
            mean_queries=float(np.mean(queries)) if queries else 0.0,
            evasive_samples=samples,
            n_evaluated=n,
        )
