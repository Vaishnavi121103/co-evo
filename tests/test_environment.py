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


def test_mimicry_directions_move_toward_benign():
    """Mimicry actions must point from the malicious mean toward the benign one.

    This is what makes the action space meaningful at high dimensionality:
    random directions in ~10^3 dims are nearly orthogonal to any useful
    descent direction, so the attack cannot work without semantic actions.
    """
    from coevomal.environment import build_mimicry_directions

    ds, _ = make_synthetic(n_features=40, n_train=600, n_test_malicious=10, seed=5)
    D = build_mimicry_directions(
        ds.X, ds.y, ds.feature_space, n_actions=10, n_random=2, seed=0
    )
    assert D.shape == (10, 40)
    # unit-norm directions
    assert np.allclose(np.linalg.norm(D, axis=1), 1.0, atol=1e-5)

    ben = ds.X[ds.y == 0].mean(axis=0)
    mal = ds.X[ds.y == 1].mean(axis=0)
    delta = ben - mal
    # The block-wise mimicry actions (all but the trailing random ones) should
    # have positive alignment with the benign-minus-malicious direction.
    aligned = [float(D[i] @ delta) for i in range(D.shape[0] - 2)]
    assert all(a > 0 for a in aligned), aligned


def test_mimicry_respects_additive_only():
    """Add-only features must never be pushed downward by a mimicry action."""
    from coevomal.environment import build_mimicry_directions

    ds, _ = make_synthetic(n_features=30, n_train=400, n_test_malicious=5, seed=7)
    fs = ds.feature_space
    D = build_mimicry_directions(ds.X, ds.y, fs, n_actions=8, seed=1)
    add_idx = fs.mutable_idx[fs.additive_only]
    if add_idx.size:
        assert (D[:, add_idx] >= -1e-6).all()


def test_env_seed_changes_exemplar_targets_between_rounds():
    """Each co-evolution round must explore new benign targets.

    With one fixed seed the attacker redraws the same exemplars every round and
    keeps re-attacking a region the defender was already retrained on, so the
    defender appears to win by memorising a few points rather than by
    generalising.
    """
    ds, _ = make_synthetic(n_features=20, n_train=400, n_test_malicious=10, seed=1)

    def targets(seed):
        env = MalwareEvasionEnv(
            _StubDefender(), ds.malicious(), ds.feature_space, n_actions=6,
            max_steps=3, seed=seed, benign_pool=ds.benign(),
        )
        return env._sample_targets(5)

    assert np.array_equal(targets(7), targets(7))          # reproducible
    assert not np.array_equal(targets(7), targets(8))      # round-to-round variety


def test_orchestrator_uses_distinct_seeds_per_round_and_purpose():
    from coevomal.config import ExperimentConfig
    from coevomal.orchestrator import CoEvolutionOrchestrator

    cfg = ExperimentConfig(name="seedcheck", rounds=1)
    cfg.dataset.n_features, cfg.dataset.n_train = 12, 200
    cfg.dataset.n_test_malicious, cfg.env.n_actions, cfg.env.max_steps = 10, 6, 4
    cfg.defender.max_iter = 10
    orch = CoEvolutionOrchestrator(cfg, verbose=False)
    orch.defender.fit(orch.X_train, orch.y_train)
    snap = orch.defender.snapshot()
    seeds = {
        (r, p): orch._make_env(snap, round_idx=r, purpose=p).rng.integers(0, 10**9)
        for r in (0, 1, 2) for p in ("train", "eval")
    }
    assert len(set(seeds.values())) == len(seeds), "env seeds collide"
