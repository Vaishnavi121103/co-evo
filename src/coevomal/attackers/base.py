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
    """Outcome of a greedy evaluation rollout over a malicious pool.

    ``evasion_rate`` is the headline number: the fraction of the pool the
    defender ends up calling benign. It has two distinct sources, which this
    harness keeps separate because conflating them overstates the attacker:

    * ``pre_evasive_rate`` -- samples the defender *already* misclassified
      before any mutation (its baseline false-negative rate).
    * ``attack_success_rate`` -- of the samples the defender initially caught,
      the fraction the attacker actually flipped. This is the true measure of
      attacker strength and the one to read when judging a retraining policy.
    """

    evasion_rate: float
    mean_queries: float
    evasive_samples: np.ndarray = field(default_factory=lambda: np.empty((0,)))
    n_evaluated: int = 0
    pre_evasive_rate: float = 0.0
    attack_success_rate: float = 0.0


class Attacker(ABC):
    """Abstract RL evasion agent."""

    @abstractmethod
    def train(self, env: MalwareEvasionEnv, episodes: int) -> None:
        """Learn a policy against ``env`` (which wraps a frozen defender)."""

    @abstractmethod
    def act(self, obs: np.ndarray, greedy: bool = True) -> int:
        """Return an action for a single observation."""

    def act_batch(self, obs: np.ndarray, greedy: bool = True) -> np.ndarray:
        """Actions for a *batch* of observations.

        Default implementation loops :meth:`act`; network-based agents override
        this with a single batched forward pass, which is what makes the
        evaluation fast path fast. Must be equivalent to per-row :meth:`act`.
        """
        return np.array([self.act(o, greedy=greedy) for o in obs], dtype=np.int64)

    def reset_policy(self) -> None:
        """Optionally re-initialise the policy at the start of a round.

        Default is a no-op: agents keep (warm-start) their policy across
        rounds, which is realistic for a persistent adversary. Override to
        study cold-start attackers.
        """

    # ---- shared evaluation --------------------------------------------------
    def evaluate_evasion(
        self, env: MalwareEvasionEnv, malicious_pool: np.ndarray,
        greedy: bool = True,
    ) -> EvasionResult:
        """Greedily roll out the learned policy over ``malicious_pool``.

        Delegates to the environment's batched lockstep rollout, which is
        semantically identical to a per-sample reset/step loop but issues
        batched defender queries.
        """
        evaded, queries, final_x, initial_evaded = env.batch_rollout(
            self, malicious_pool, greedy=greedy
        )
        n = malicious_pool.shape[0]
        samples = (
            final_x[evaded]
            if evaded.any()
            else np.empty((0, malicious_pool.shape[1]), dtype=np.float32)
        )
        # Attack success is measured only over samples the defender caught.
        caught = ~initial_evaded
        n_caught = int(caught.sum())
        flipped = int((evaded & caught).sum())
        return EvasionResult(
            evasion_rate=float(evaded.sum()) / max(1, n),
            mean_queries=float(np.mean(queries)) if n else 0.0,
            evasive_samples=samples,
            n_evaluated=n,
            pre_evasive_rate=float(initial_evaded.sum()) / max(1, n),
            attack_success_rate=flipped / n_caught if n_caught else 0.0,
        )
