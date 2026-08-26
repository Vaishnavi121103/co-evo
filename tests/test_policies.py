"""Tests for retraining policies: cadence decisions and data-selection buffers."""

import numpy as np

from coevomal.config import RetrainPolicyConfig
from coevomal.policies import ReplayBuffer, RetrainingPolicy


def test_cadence_every_round():
    p = RetrainingPolicy(RetrainPolicyConfig(cadence="every_round"))
    assert all(p.should_retrain(r, 0.5) for r in range(5))


def test_cadence_every_n():
    p = RetrainingPolicy(RetrainPolicyConfig(cadence="every_n", every_n=3))
    decisions = [p.should_retrain(r, 0.5) for r in range(6)]
    assert decisions == [True, False, False, True, False, False]


def test_cadence_threshold():
    p = RetrainingPolicy(RetrainPolicyConfig(cadence="threshold", trigger_threshold=0.3))
    assert p.should_retrain(0, 0.4) is True
    assert p.should_retrain(0, 0.2) is False


def test_bounded_buffer_is_fifo_capped():
    buf = ReplayBuffer(strategy="bounded_buffer", capacity=3)
    for i in range(5):
        buf.add(np.full((1, 4), i, dtype=np.float32))
    arr = buf.as_array()
    assert arr.shape[0] == 3
    # keeps the 3 most recent (values 2,3,4)
    assert set(np.unique(arr).tolist()) == {2.0, 3.0, 4.0}


def test_hard_mining_keeps_lowest_scores():
    buf = ReplayBuffer(strategy="hard_mining", capacity=2)
    X = np.arange(4, dtype=np.float32).reshape(4, 1)
    scores = np.array([0.9, 0.1, 0.5, 0.05], dtype=np.float32)
    buf.add(X, scores)
    kept = set(buf.as_array().reshape(-1).tolist())
    # lowest malicious prob == deepest evasions == rows 3 (0.05) and 1 (0.1)
    assert kept == {3.0, 1.0}


def test_full_replay_is_unbounded():
    buf = ReplayBuffer(strategy="full_replay", capacity=2)
    for i in range(10):
        buf.add(np.full((1, 2), i, dtype=np.float32))
    assert len(buf) == 10


def test_build_training_set_labels_evasions_malicious():
    p = RetrainingPolicy(RetrainPolicyConfig(data_selection="full_replay"))

    class _Def:
        def predict_proba(self, X):
            return np.full(X.shape[0], 0.2, dtype=np.float32)

    evasive = np.ones((3, 4), dtype=np.float32)
    p.record(evasive, _Def())
    base_X = np.zeros((5, 4), dtype=np.float32)
    base_y = np.zeros(5, dtype=np.int64)
    X, y = p.build_training_set(base_X, base_y)
    assert X.shape[0] == 8
    assert y.sum() == 3  # the 3 evasive rows are labelled malicious


def test_hard_mining_rescores_against_current_defender():
    """Buffered scores must be refreshed, not left stale from earlier rounds.

    Scores recorded in earlier rounds came from a defender that has since been
    retrained; ranking a mixed buffer on them compares numbers produced by
    different models.
    """
    from coevomal.config import RetrainPolicyConfig
    from coevomal.policies import RetrainingPolicy

    p = RetrainingPolicy(RetrainPolicyConfig(data_selection="hard_mining", buffer_size=2))

    class _Weak:                      # early defender: everything looks benign
        def predict_proba(self, X):
            return np.full(X.shape[0], 0.01, dtype=np.float32)

    class _Strong:                    # retrained defender: row 0 now caught
        def predict_proba(self, X):
            out = np.full(X.shape[0], 0.02, dtype=np.float32)
            out[X[:, 0] == 0.0] = 0.99
            return out

    p.record(np.array([[0.0, 0.0], [1.0, 1.0]], dtype=np.float32), _Weak())
    p.record(np.array([[2.0, 2.0]], dtype=np.float32), _Strong())
    kept = p.buffer.as_array()
    assert kept.shape[0] == 2
    # Under the current defender row [0,0] is no longer a deep evasion, so the
    # capacity-2 hard-mining buffer must have dropped it.
    assert not any(np.allclose(r, [0.0, 0.0]) for r in kept), kept


def test_robustness_per_cost_is_nan_without_retraining():
    """A zero-cost baseline must not report infinite efficiency."""
    from coevomal.evaluation.metrics import ExperimentResult, RoundLog

    r = ExperimentResult(config={})
    r.rounds.append(RoundLog(round=0, evasion_rate=0.5, attack_success_rate=0.5,
                             pre_evasive_rate=0.0, mean_queries=1.0, retrained=False,
                             retrain_seconds=0.0, train_samples=0,
                             clean_accuracy=0.9, buffer_size=0))
    assert np.isnan(r.summary()["robustness_per_cost"])
