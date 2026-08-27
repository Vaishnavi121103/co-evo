"""Tests for retraining policies: cadence decisions and data-selection buffers."""

import numpy as np
import pytest

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
                             retrain_seconds=0.0, trees_fitted=0, train_samples=0,
                             clean_accuracy=0.9, buffer_size=0))
    assert np.isnan(r.summary()["robustness_per_cost"])


def test_never_cadence_is_the_frozen_baseline():
    """`never` must never retrain -- the frozen-classifier lower bound."""
    from coevomal.config import RetrainPolicyConfig
    from coevomal.policies import RetrainingPolicy

    p = RetrainingPolicy(RetrainPolicyConfig(cadence="never"))
    assert not any(p.should_retrain(r, ev) for r in range(10) for ev in (0.0, 0.5, 1.0))


def test_unknown_cadence_raises_helpfully():
    from coevomal.config import RetrainPolicyConfig
    from coevomal.policies import RetrainingPolicy

    p = RetrainingPolicy(RetrainPolicyConfig(cadence="weekly"))
    with pytest.raises(ValueError, match="every_round"):
        p.should_retrain(0, 0.5)


def test_policy_key_is_backwards_compatible():
    """`scratch` stays out of the key so completed cells still resume.

    The mode axis was added after the main sweep had already been run; if the
    key changed shape, every finished cell would silently be recomputed.
    """
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))
    from run_multiseed import _policy_key

    assert _policy_key("random", "every_round", "full_replay", "scratch") == \
        "random__every_round__full_replay"
    assert _policy_key("random", "every_round", "full_replay", "finetune") == \
        "random__every_round__full_replay__finetune"
    assert _policy_key("dqn", "never", "full_replay", "scratch") == \
        "dqn__never__full_replay"


def test_minimax_cadence_alternates_turns():
    """Minimax alternation: the defender retrains on every other round."""
    from coevomal.config import RetrainPolicyConfig
    from coevomal.policies import RetrainingPolicy

    p = RetrainingPolicy(RetrainPolicyConfig(cadence="minimax"))
    assert [p.should_retrain(r, 0.5) for r in range(6)] == \
        [True, False, True, False, True, False]


def test_minimax_escalates_only_on_defender_rest_rounds():
    """The attacker's budget grows on the rounds the defender sits out.

    Escalation is what makes this an alternating *game* rather than just a
    slower cadence: each side strengthens on its own turn.
    """
    from coevomal.config import ExperimentConfig
    from coevomal.orchestrator import CoEvolutionOrchestrator

    cfg = ExperimentConfig(name="mmx", rounds=4, early_stop=False)
    cfg.dataset.n_features, cfg.dataset.n_train = 12, 240
    cfg.dataset.n_test_malicious, cfg.env.n_actions = 12, 6
    cfg.env.max_steps, cfg.defender.max_iter = 20, 12
    cfg.attacker.name = "random"
    cfg.retrain.cadence = "minimax"
    o = CoEvolutionOrchestrator(cfg, verbose=False)

    base = o._attacker_budget()
    assert base == 20
    o._attacker_turns = 1
    assert o._attacker_budget() > base
    o._attacker_turns = 99                       # ceiling applies
    assert o._attacker_budget() == int(round(20 * cfg.env.minimax_max_escalation))

    cfg.retrain.cadence = "every_round"          # other cadences stay fixed
    o2 = CoEvolutionOrchestrator(cfg, verbose=False)
    o2._attacker_turns = 5
    assert o2._attacker_budget() == 20


def test_finetune_ceiling_raises_instead_of_silently_refitting():
    """Hitting the warm-start ceiling must fail loudly, not become a scratch fit.

    The old behaviour refit from scratch at the ceiling, which silently turned
    the final fine-tune rounds into scratch rounds and contaminated the tail
    average that settled evasion is computed from.
    """
    from coevomal.defenders import GBDTDefender

    rng = np.random.default_rng(0)
    X = rng.standard_normal((200, 8)).astype(np.float32)
    y = (rng.random(200) < 0.5).astype(int)
    d = GBDTDefender(max_iter=20, max_total_iter=60, seed=0).fit(X, y)
    d.fit(X, y, warm_start=True)          # 20 -> 40
    d.fit(X, y, warm_start=True)          # 40 -> 60 (at ceiling)
    with pytest.raises(RuntimeError, match="max_total_iter"):
        d.fit(X, y, warm_start=True)      # would silently refit before


def test_finetune_iter_is_independent_of_initial_size():
    """The capacity-matched arm must not change the initial model.

    Raising `max_iter` would also enlarge the from-scratch fit, making the two
    arms start from different defenders and confounding the comparison.
    """
    from coevomal.defenders import GBDTDefender

    rng = np.random.default_rng(1)
    X = rng.standard_normal((200, 8)).astype(np.float32)
    y = (rng.random(200) < 0.5).astype(int)
    cost = GBDTDefender(max_iter=20, seed=0).fit(X, y)
    cap = GBDTDefender(max_iter=20, finetune_iter=80, max_total_iter=500, seed=0).fit(X, y)
    assert cost._model.n_iter_ == cap._model.n_iter_ == 20     # identical start
    cost.fit(X, y, warm_start=True)
    cap.fit(X, y, warm_start=True)
    assert cost._model.n_iter_ == 40                            # +20 cost-matched
    assert cap._model.n_iter_ == 100                            # +80 capacity-matched


def test_defender_early_stopping_is_pinned_off():
    """Training-set size must not silently change the fitting regime.

    scikit-learn enables early stopping above 10,000 samples by default. The
    training set here is clean data plus retained adversarial samples, so that
    threshold is policy-dependent: an unbounded-replay policy crosses it while
    a capped one never does, and the model would then fit far fewer trees.
    """
    from coevomal.defenders import GBDTDefender

    rng = np.random.default_rng(0)
    n = 12000                                   # comfortably past the threshold
    X = rng.standard_normal((n, 6)).astype(np.float32)
    y = (rng.random(n) < 0.5).astype(int)
    d = GBDTDefender(max_iter=40, seed=0).fit(X, y)
    assert d._model.do_early_stopping_ is False
    assert d._model.n_iter_ == 40               # full budget, not a truncated fit
    assert d.trees_last_fit == 40
