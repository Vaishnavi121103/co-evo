"""Defender interface.

A defender is the malware classifier under attack. The harness only relies
on a small surface: fit on a labelled feature set, expose ``predict_proba``
for the attacker's queries, and produce an independent, frozen *snapshot*
that the attacker trains against for the duration of a round (so the
attacker never sees mid-round retraining).
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class Defender(ABC):
    """Abstract malware classifier."""

    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray, warm_start: bool = False) -> "Defender":
        """Train (or continue training when ``warm_start``) on ``(X, y)``."""

    @abstractmethod
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return P(malicious) for each row of ``X`` as a 1-D array."""

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        return (self.predict_proba(X) >= threshold).astype(np.int64)

    @abstractmethod
    def snapshot(self) -> "Defender":
        """Return a frozen, independent copy for the attacker to train against."""

    @property
    @abstractmethod
    def is_fitted(self) -> bool:
        ...
