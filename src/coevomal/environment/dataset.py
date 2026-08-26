"""Datasets for the defender side of the co-evolution loop.

The defender is trained purely on static feature vectors, so it needs no
live malware handling. By default we generate a **synthetic EMBER-like**
feature space that is fast, fully reproducible, and requires no external
downloads -- ideal for exercising the harness and for the factorial
policy study. A thin :func:`load_ember` hook is provided so the real
EMBER feature vectors can be swapped in without touching the rest of the
pipeline.

A :class:`FeatureSpace` records which feature dimensions the attacker is
allowed to perturb, and in which direction. In real gym-malware the
mutations (append bytes, add a section, add an import) are
*functionality preserving* and only ever *add* structure; we mirror that
by marking mutable features as "additive only" so the synthetic attack
stays a faithful proxy.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

MALICIOUS = 1
BENIGN = 0


@dataclass
class FeatureSpace:
    """Describes the feature vector layout and what the attacker may change."""

    n_features: int
    mutable_idx: np.ndarray          # indices the attacker may perturb
    additive_only: np.ndarray        # bool mask (over mutable_idx) of add-only dims
    low: np.ndarray                  # per-feature clip lower bound
    high: np.ndarray                 # per-feature clip upper bound

    @property
    def n_mutable(self) -> int:
        return int(self.mutable_idx.size)


@dataclass
class Dataset:
    """A simple in-memory feature-vector dataset."""

    X: np.ndarray
    y: np.ndarray
    feature_space: FeatureSpace

    def malicious(self) -> np.ndarray:
        return self.X[self.y == MALICIOUS]

    def benign(self) -> np.ndarray:
        return self.X[self.y == BENIGN]


def make_synthetic(
    n_features: int = 32,
    n_train: int = 4000,
    n_test_malicious: int = 400,
    class_separation: float = 1.4,
    mutable_fraction: float = 0.6,
    seed: int = 0,
) -> tuple[Dataset, np.ndarray]:
    """Build a synthetic EMBER-like dataset.

    Returns ``(train_dataset, test_malicious)`` where ``test_malicious`` is a
    held-out pool of malicious feature vectors the attacker will try to
    perturb into the benign region.

    Benign and malicious classes are two Gaussian blobs separated along a
    random direction. ``class_separation`` controls the baseline difficulty:
    larger values make the frozen classifier stronger and the attacker's job
    harder.
    """
    rng = np.random.default_rng(seed)

    # Random but fixed class-mean offset direction.
    direction = rng.standard_normal(n_features)
    direction /= np.linalg.norm(direction)
    offset = direction * class_separation

    def sample(n: int, label: int) -> np.ndarray:
        base = rng.standard_normal((n, n_features))
        if label == MALICIOUS:
            base = base + offset
        return base

    n_half = n_train // 2
    X_ben = sample(n_half, BENIGN)
    X_mal = sample(n_train - n_half, MALICIOUS)
    X = np.vstack([X_ben, X_mal]).astype(np.float32)
    y = np.concatenate([
        np.full(n_half, BENIGN),
        np.full(n_train - n_half, MALICIOUS),
    ]).astype(np.int64)

    perm = rng.permutation(X.shape[0])
    X, y = X[perm], y[perm]

    test_mal = sample(n_test_malicious, MALICIOUS).astype(np.float32)

    n_mutable = max(1, int(round(mutable_fraction * n_features)))
    mutable_idx = np.sort(rng.choice(n_features, size=n_mutable, replace=False))
    # Half of the mutable features are "add-only" (append-style) mutations.
    additive_only = rng.random(n_mutable) < 0.5

    lo = np.full(n_features, -6.0, dtype=np.float32)
    hi = np.full(n_features, 6.0, dtype=np.float32)
    fs = FeatureSpace(
        n_features=n_features,
        mutable_idx=mutable_idx,
        additive_only=additive_only,
        low=lo,
        high=hi,
    )
    return Dataset(X=X, y=y, feature_space=fs), test_mal


def load_ember(ember_path: str, mutable_fraction: float = 0.6, seed: int = 0):
    """Load real EMBER feature vectors (extension point).

    EMBER ships as vectorized ``X_train`` / ``y_train`` arrays. Wire them in
    here and the orchestrator, attacker, defender, and metrics all work
    unchanged. Left as a documented stub because EMBER is an external ~10GB
    download and is not required for the synthetic policy study.
    """
    raise NotImplementedError(
        "Real EMBER loading is an intentional extension point. Populate "
        "`X, y` from the EMBER vectorized features at "
        f"'{ember_path}', build a FeatureSpace over the byte/section/import "
        "features that gym-malware can additively mutate, and return a "
        "(Dataset, test_malicious) tuple mirroring make_synthetic()."
    )
