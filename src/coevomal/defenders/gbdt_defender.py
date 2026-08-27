"""Gradient-boosted-tree defender (EMBER/LightGBM stand-in).

The EMBER reference model is a LightGBM classifier. LightGBM is an
optional dependency here, so by default we use scikit-learn's
``HistGradientBoostingClassifier`` -- a histogram-based GBDT that is
algorithmically very close to LightGBM and ships with scikit-learn. If
LightGBM is installed the code path is trivial to switch; the harness
only needs ``predict_proba``.

Two retraining *modes* are supported, matching the policy study:

* ``scratch``   -- fit a brand-new model on the selected data.
* ``finetune``  -- continue boosting (``warm_start``) from the current
  model by adding more trees, which is cheaper but can lock in earlier
  decision boundaries.
"""

from __future__ import annotations

import copy

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

from coevomal.defenders.base import Defender


class GBDTDefender(Defender):
    def __init__(
        self,
        max_iter: int = 150,
        learning_rate: float = 0.08,
        max_depth: int | None = None,
        finetune_iter: int | None = None,
        max_total_iter: int = 2000,
        seed: int = 0,
    ) -> None:
        self.max_iter = int(max_iter)
        self.learning_rate = float(learning_rate)
        self.max_depth = max_depth
        # Trees added per warm-start round. Kept separate from `max_iter` so
        # the *initial* model is identical across arms: setting max_iter would
        # also change the from-scratch fit and make the arms incomparable.
        # `max_iter` alone gives the cost-matched arm (same incremental fit
        # cost as a scratch refit); a larger value gives the capacity-matched
        # arm, which the audit shows is what warm-starting needs to overturn an
        # entrenched boundary.
        self.finetune_iter = int(finetune_iter if finetune_iter else max_iter)
        # Ceiling on total boosting rounds under repeated warm-start fits. It
        # must sit ABOVE what a full run can reach, otherwise the arm silently
        # falls back to a scratch refit in its final rounds and contaminates
        # the tail average that settled evasion is computed from.
        self.max_total_iter = int(max_total_iter)
        self.seed = int(seed)
        self._model: HistGradientBoostingClassifier | None = None

    def _new_model(self, warm_start: bool = False) -> HistGradientBoostingClassifier:
        return HistGradientBoostingClassifier(
            max_iter=self.max_iter,
            learning_rate=self.learning_rate,
            max_depth=self.max_depth,
            warm_start=warm_start,
            random_state=self.seed,
        )

    def fit(self, X: np.ndarray, y: np.ndarray, warm_start: bool = False) -> "GBDTDefender":
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.int64)
        if warm_start and self._model is not None:
            # Continue boosting: add another `max_iter` trees on the new data.
            target = min(
                self._model.max_iter + self.finetune_iter, self.max_total_iter
            )
            if target > self._model.max_iter:
                self._model.set_params(warm_start=True, max_iter=target)
                self._model.fit(X, y)
            else:
                # Ceiling reached. Refitting from scratch here would silently
                # convert a fine-tune round into a scratch round, so this path
                # is a misconfiguration rather than a fallback: size
                # max_total_iter above rounds * finetune_iter.
                raise RuntimeError(
                    f"warm-start ceiling reached (max_total_iter="
                    f"{self.max_total_iter}); raise it above "
                    f"rounds*finetune_iter so the fine-tune arm is not "
                    f"silently converted into a scratch refit"
                )
        else:
            self._model = self._new_model(warm_start=False)
            self._model.fit(X, y)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        assert self._model is not None, "defender is not fitted"
        X = np.asarray(X, dtype=np.float32)
        proba = self._model.predict_proba(X)
        # Column index of the malicious class (label == 1).
        classes = list(self._model.classes_)
        mal_col = classes.index(1) if 1 in classes else proba.shape[1] - 1
        return proba[:, mal_col]

    def snapshot(self) -> "GBDTDefender":
        clone = GBDTDefender(
            max_iter=self.max_iter,
            learning_rate=self.learning_rate,
            max_depth=self.max_depth,
            finetune_iter=self.finetune_iter,
            max_total_iter=self.max_total_iter,
            seed=self.seed,
        )
        clone._model = copy.deepcopy(self._model)
        return clone

    @property
    def is_fitted(self) -> bool:
        return self._model is not None
