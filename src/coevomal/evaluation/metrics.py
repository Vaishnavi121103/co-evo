"""Evaluation harness: standardized co-evolution metrics.

This module is the actual novel deliverable of the project. Every policy
combination is scored with *identical* metrics so results are directly
comparable -- something the single-mechanism papers in the related-work
cluster do not provide.

Per-round we log:

* ``evasion_rate``   -- fraction of the held-out malicious pool the defender
  ends up calling benign (primary outcome).
* ``attack_success_rate`` -- of the samples the defender initially *caught*,
  the fraction the attacker flipped. This isolates attacker strength from the
  defender's baseline false negatives and is the cleaner signal when comparing
  retraining policies.
* ``pre_evasive_rate`` -- the defender's baseline false-negative rate on the
  pool, before any mutation.
* ``mean_queries``   -- attacker query complexity: mean defender queries per
  sample; rising values mean evasion is getting *harder*.
* ``retrained``      -- whether the defender retrained this round.
* ``retrain_seconds``/``train_samples`` -- retraining cost.
* ``clean_accuracy`` -- defender accuracy on held-out clean data, to catch
  robustness bought at the price of clean performance.

Across rounds we summarise:

* ``oscillation_index`` -- std of the evasion rate over a trailing window
  (the key novel stability metric): low => converged, high => the defender
  is chasing the attacker.
* ``rounds_to_convergence`` -- first round after which the trailing-window
  std stays below ``tol`` (``None`` => never converged / divergent).
* ``mean_evasion_tail`` -- mean evasion rate over the final window (the
  robustness the policy actually settles at).
* ``total_retrain_seconds`` / ``retrain_count`` -- total defence cost, the
  denominator of the robustness-per-cost frontier.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any

import numpy as np


@dataclass
class RoundLog:
    round: int
    evasion_rate: float
    attack_success_rate: float
    pre_evasive_rate: float
    mean_queries: float
    retrained: bool
    retrain_seconds: float
    train_samples: int
    clean_accuracy: float
    buffer_size: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def oscillation_index(evasion_rates: list[float], window: int) -> float:
    """Std of the evasion rate over the trailing ``window`` rounds.

    High values mean the evasion rate keeps swinging round-to-round -- the
    defender is oscillating rather than settling on a stable boundary.
    """
    if len(evasion_rates) < 2:
        return 0.0
    tail = evasion_rates[-window:]
    return float(np.std(tail))


def rounds_to_convergence(
    evasion_rates: list[float], window: int, tol: float
) -> int | None:
    """First round index at which the trailing-window std stays <= ``tol``.

    Returns ``None`` if convergence is never reached within the run (treated
    as divergence / perpetual oscillation).
    """
    for end in range(window, len(evasion_rates) + 1):
        if float(np.std(evasion_rates[end - window:end])) <= tol:
            return end - 1  # 0-based index of the round that completes it
    return None


@dataclass
class ExperimentResult:
    """Container for a full co-evolution run + derived summary metrics."""

    config: dict[str, Any]
    rounds: list[RoundLog] = field(default_factory=list)

    # ---- per-round accessors ------------------------------------------------
    @property
    def evasion_rates(self) -> list[float]:
        return [r.evasion_rate for r in self.rounds]

    @property
    def mean_queries(self) -> list[float]:
        return [r.mean_queries for r in self.rounds]

    # ---- summary ------------------------------------------------------------
    def summary(self, window: int = 4, tol: float = 0.03) -> dict[str, Any]:
        rates = self.evasion_rates
        conv = rounds_to_convergence(rates, window, tol)
        retrain_secs = sum(r.retrain_seconds for r in self.rounds)
        retrain_count = sum(1 for r in self.rounds if r.retrained)
        tail = rates[-window:] if rates else [0.0]
        asr = [r.attack_success_rate for r in self.rounds]
        asr_tail = asr[-window:] if asr else [0.0]
        return {
            "mean_attack_success_tail": float(np.mean(asr_tail)),
            "final_attack_success_rate": asr[-1] if asr else float("nan"),
            "oscillation_index": oscillation_index(rates, window),
            "rounds_to_convergence": conv,
            "converged": conv is not None,
            "mean_evasion_tail": float(np.mean(tail)),
            "final_evasion_rate": rates[-1] if rates else float("nan"),
            "max_evasion_rate": max(rates) if rates else float("nan"),
            "total_retrain_seconds": retrain_secs,
            "retrain_count": retrain_count,
            # robustness-per-cost: lower tail evasion per retrain-second is
            # better. Guard against divide-by-zero for no-retrain baselines.
            "robustness_per_cost": (
                (1.0 - float(np.mean(tail))) / retrain_secs
                if retrain_secs > 0
                else float("inf")
            ),
        }

    def to_dict(self) -> dict[str, Any]:
        # Use the run's configured convergence window/tol so the persisted
        # summary matches what the run reported (falling back to the summary
        # defaults if the config does not specify them).
        window = int(self.config.get("convergence_window", 4))
        tol = float(self.config.get("convergence_tol", 0.03))
        return {
            "config": self.config,
            "rounds": [r.to_dict() for r in self.rounds],
            "summary": self.summary(window=window, tol=tol),
        }
