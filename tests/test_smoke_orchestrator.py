"""End-to-end smoke test: a tiny co-evolution run must complete and log."""

from coevomal.config import ExperimentConfig
from coevomal.orchestrator import CoEvolutionOrchestrator


def _tiny_config() -> ExperimentConfig:
    cfg = ExperimentConfig(name="pytest_smoke", rounds=2, convergence_window=2)
    cfg.dataset.n_features = 12
    cfg.dataset.n_train = 300
    cfg.dataset.n_test_malicious = 20
    cfg.env.n_actions = 6
    cfg.env.max_steps = 8
    cfg.defender.max_iter = 40
    cfg.attacker.name = "dqn"
    cfg.attacker.train_episodes = 15
    cfg.attacker.device = "cpu"
    return cfg


def test_orchestrator_runs_and_logs():
    cfg = _tiny_config()
    orch = CoEvolutionOrchestrator(cfg, verbose=False)
    result = orch.run()
    assert 1 <= len(result.rounds) <= cfg.rounds
    for log in result.rounds:
        assert 0.0 <= log.evasion_rate <= 1.0
        assert 0.0 <= log.clean_accuracy <= 1.0
    summ = result.summary(cfg.convergence_window, cfg.convergence_tol)
    assert "oscillation_index" in summ
    assert "robustness_per_cost" in summ


def test_ppo_attacker_runs():
    cfg = _tiny_config()
    cfg.attacker.name = "ppo"
    cfg.attacker.rollout = 64
    orch = CoEvolutionOrchestrator(cfg, verbose=False)
    result = orch.run()
    assert len(result.rounds) >= 1
