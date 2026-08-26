"""Retraining policies -- the central independent variable of the study.

A retraining policy answers three orthogonal questions each co-evolution
round:

1. **Cadence** -- *should* we retrain this round?
   ``every_round`` | ``every_n`` | ``threshold`` (retrain only when the
   round's evasion rate exceeds a trigger).

2. **Data selection** -- *on what* do we retrain?
   ``full_replay`` (base data + every evasive sample ever found) |
   ``hard_mining`` (base data + the hardest / most-confidently-evasive
   samples, capped) | ``bounded_buffer`` (base data + a FIFO window of the
   most recent evasive samples).

3. **Mode** -- fit ``scratch`` or ``finetune`` (warm-start) the defender.

The factorial study crosses cadence x data-selection (x mode) so that the
harness can attribute convergence vs. oscillation to a *single* policy
variable rather than to a bundled defense mechanism. That controlled
attribution is the contribution the surrounding literature does not offer.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from coevomal.config import RetrainPolicyConfig


@dataclass
class ReplayBuffer:
    """Accumulates discovered evasive samples according to a selection rule."""

    strategy: str = "full_replay"     # {"full_replay", "hard_mining", "bounded_buffer"}
    capacity: int = 2000
    _samples: list[np.ndarray] = field(default_factory=list)
    _scores: list[float] = field(default_factory=list)  # "hardness" (defender prob)

    def add(self, samples: np.ndarray, scores: np.ndarray | None = None) -> None:
        if samples.size == 0:
            return
        if scores is None:
            scores = np.zeros(samples.shape[0], dtype=np.float32)
        for row, sc in zip(samples, scores):
            self._samples.append(np.asarray(row, dtype=np.float32))
            self._scores.append(float(sc))
        self._enforce()

    def _enforce(self) -> None:
        if self.strategy == "full_replay":
            return  # unbounded: keep everything ever found
        if len(self._samples) <= self.capacity:
            return
        if self.strategy == "bounded_buffer":
            # FIFO: keep the most recent `capacity` samples.
            self._samples = self._samples[-self.capacity:]
            self._scores = self._scores[-self.capacity:]
        elif self.strategy == "hard_mining":
            # Keep the `capacity` "hardest" samples -- the ones the defender
            # was most confident were benign (lowest malicious prob), i.e.
            # the deepest evasions.
            order = np.argsort(self._scores)[: self.capacity]
            self._samples = [self._samples[i] for i in order]
            self._scores = [self._scores[i] for i in order]

    def rescore(self, defender) -> None:
        """Re-evaluate every buffered sample under the current defender.

        Only meaningful for ``hard_mining``, whose selection is a ranking over
        these scores; re-applies the capacity rule afterwards.
        """
        if not self._samples:
            return
        X = np.vstack(self._samples)
        self._scores = [float(v) for v in defender.predict_proba(X)]
        self._enforce()

    def as_array(self) -> np.ndarray:
        if not self._samples:
            return np.empty((0,), dtype=np.float32)
        return np.vstack(self._samples)

    def __len__(self) -> int:
        return len(self._samples)


class RetrainingPolicy:
    """Bundles cadence + data-selection + mode behind a small decision API."""

    def __init__(self, cfg: RetrainPolicyConfig) -> None:
        self.cfg = cfg
        self.buffer = ReplayBuffer(
            strategy=cfg.data_selection, capacity=cfg.buffer_size
        )

    # ---- cadence ------------------------------------------------------------
    def should_retrain(self, round_idx: int, evasion_rate: float) -> bool:
        """Decide whether the defender retrains this round."""
        cadence = self.cfg.cadence
        if cadence == "every_round":
            return True
        if cadence == "every_n":
            # round_idx is 0-based; retrain on rounds 0, n, 2n, ...
            return (round_idx % max(1, self.cfg.every_n)) == 0
        if cadence == "threshold":
            return evasion_rate >= self.cfg.trigger_threshold
        raise ValueError(f"unknown cadence '{cadence}'")

    # ---- data selection -----------------------------------------------------
    def record(self, evasive_samples: np.ndarray, defender) -> None:
        """Add newly discovered evasive samples to the replay buffer.

        For ``hard_mining`` the whole buffer is re-scored against the *current*
        defender, not just the new arrivals. Scores recorded in earlier rounds
        were produced by a defender that has since been retrained, so ranking
        a mixed buffer on them compares numbers from different models and the
        "hardest" selection silently drifts toward whatever was inserted when
        the defender happened to be weakest.
        """
        # Refresh the existing buffer *before* inserting, so the capacity rule
        # below ranks old and new samples on scores from the same (current)
        # model. Rescoring afterwards would be too late: the trim would already
        # have run against stale numbers.
        if self.cfg.data_selection == "hard_mining":
            self.buffer.rescore(defender)
        if evasive_samples.size == 0:
            return
        scores = defender.predict_proba(evasive_samples)
        self.buffer.add(evasive_samples, scores)

    def build_training_set(
        self, base_X: np.ndarray, base_y: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Combine the base data with buffered evasive samples (label malicious)."""
        adv = self.buffer.as_array()
        if adv.size == 0:
            return base_X, base_y
        adv_y = np.ones(adv.shape[0], dtype=np.int64)  # evasions are still malware
        X = np.vstack([base_X, adv]).astype(np.float32)
        y = np.concatenate([base_y, adv_y]).astype(np.int64)
        return X, y

    # ---- mode ---------------------------------------------------------------
    @property
    def warm_start(self) -> bool:
        return self.cfg.mode == "finetune"

    def describe(self) -> str:
        c = self.cfg
        return f"{c.cadence}/{c.data_selection}/{c.mode}"
