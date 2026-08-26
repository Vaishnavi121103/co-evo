"""Tests for the dataset and evasion environment."""

import numpy as np

from coevomal.environment import MalwareEvasionEnv, make_synthetic
from coevomal.environment.dataset import MALICIOUS


def test_synthetic_shapes_and_labels():
    ds, test_mal = make_synthetic(n_features=16, n_train=200, n_test_malicious=30, seed=1)
    assert ds.X.shape == (200, 16)
    assert set(np.unique(ds.y).tolist()) == {0, 1}
    assert test_mal.shape == (30, 16)
    assert ds.feature_space.n_mutable >= 1


class _StubDefender:
    """Deterministic defender: malicious prob is a function of feature sum."""

    def predict_proba(self, X):
        s = np.asarray(X).reshape(X.shape[0], -1).sum(axis=1)
        # higher sum => more "malicious"; squashed to (0,1)
        return 1.0 / (1.0 + np.exp(-s))


def _make_env(seed=0):
    ds, _ = make_synthetic(n_features=16, n_train=200, n_test_malicious=10, seed=seed)
    return MalwareEvasionEnv(
        defender=_StubDefender(),
        malicious_pool=ds.malicious(),
        feature_space=ds.feature_space,
        n_actions=8,
        max_steps=10,
        seed=seed,
    )


def test_env_step_counts_queries_and_terminates():
    env = _make_env()
    obs = env.reset()
    assert obs.shape == (16,)
    q0 = env.total_queries
    done = False
    steps = 0
    while not done:
        _obs, reward, done, info = env.step(env.rng.integers(0, env.n_actions))
        steps += 1
        assert isinstance(reward, float)
    assert steps <= env.max_steps
    assert env.total_queries > q0  # each step queried the defender


def test_env_additive_only_features_never_decrease():
    env = _make_env(seed=3)
    fs = env.fs
    add_idx = fs.mutable_idx[fs.additive_only]
    if add_idx.size == 0:
        return
    x0 = env.reset().copy()
    for _ in range(env.max_steps):
        _obs, _r, done, _i = env.step(int(env.rng.integers(0, env.n_actions)))
        if done:
            break
    x1 = env.current_sample()
    # Add-only dims should not have gone below their starting value
    # (mutations on them are non-negative; clipping only bounds above).
    assert np.all(x1[add_idx] >= x0[add_idx] - 1e-5)
