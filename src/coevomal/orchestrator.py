"""Co-evolution orchestrator: the round loop.

Each round:

1. **Snapshot** the current defender and build an evasion environment around
   that frozen snapshot (so the attacker never sees mid-round retraining).
2. **Train** the attacker against the snapshot for ``attacker.train_episodes``.
3. **Evaluate** evasion on a held-out malicious pool -> evasion rate,
   attacker query complexity, and the concrete evasive feature vectors.
4. **Record** the new evasive samples into the retraining policy's buffer.
5. **Retrain** the defender *iff* the policy's cadence says so, on the data
   the policy's selection strategy chooses, in scratch or finetune mode.
6. **Log** every metric identically for later cross-policy comparison.

The loop stops at ``rounds`` or early once the trailing-window oscillation
index drops below ``convergence_tol`` (configurable off).
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split

from coevomal.attackers import make_attacker
from coevomal.config import ExperimentConfig
from coevomal.defenders import make_defender
from coevomal.environment import (
    MalwareEvasionEnv,
    build_mimicry_directions,
    load_ember,
    make_synthetic,
)
from coevomal.evaluation.metrics import ExperimentResult, RoundLog, oscillation_index
from coevomal.policies import RetrainingPolicy
from coevomal.utils import seed_everything


class CoEvolutionOrchestrator:
    def __init__(self, cfg: ExperimentConfig, verbose: bool = True) -> None:
        self.cfg = cfg
        self.verbose = verbose
        seed_everything(cfg.seed)

        # ---- data -----------------------------------------------------------
        d = cfg.dataset
        if d.name == "synthetic":
            self.dataset, self.test_malicious = make_synthetic(
                n_features=d.n_features,
                n_train=d.n_train,
                n_test_malicious=d.n_test_malicious,
                class_separation=d.class_separation,
                mutable_fraction=d.mutable_fraction,
                seed=d.seed,
            )
        elif d.name == "ember":
            self.dataset, self.test_malicious = load_ember(
                d.ember_path,
                n_train=d.n_train,
                n_test_malicious=d.n_test_malicious,
                mutable_fraction=d.mutable_fraction,
                seed=d.seed,
            )
        else:
            raise ValueError(f"unknown dataset '{d.name}'")

        # Clean train/holdout split for clean-accuracy tracking.
        self.X_train, self.X_clean, self.y_train, self.y_clean = train_test_split(
            self.dataset.X, self.dataset.y,
            test_size=0.2, random_state=cfg.seed, stratify=self.dataset.y,
        )
        self.feature_space = self.dataset.feature_space

        # Semantic (mimicry) action set, derived once from the clean training
        # data so every round and every policy shares the same action space.
        self.directions = (
            build_mimicry_directions(
                self.X_train, self.y_train, self.feature_space,
                n_actions=cfg.env.n_actions, seed=cfg.seed,
            )
            if cfg.env.action_space == "mimicry"
            else None
        )

        # ---- components -----------------------------------------------------
        self.defender = make_defender(cfg.defender)
        self.policy = RetrainingPolicy(cfg.retrain)
        self.attacker = make_attacker(
            cfg.attacker, obs_dim=self.feature_space.n_features, n_actions=cfg.env.n_actions
        )

        self.result = ExperimentResult(config=cfg.to_dict())

    # ---- helpers ------------------------------------------------------------
    def _make_env(self, defender_snapshot) -> MalwareEvasionEnv:
        e = self.cfg.env
        return MalwareEvasionEnv(
            defender=defender_snapshot,
            malicious_pool=self.dataset.malicious(),
            feature_space=self.feature_space,
            n_actions=e.n_actions,
            max_steps=e.max_steps,
            step_scale=e.step_scale,
            evade_threshold=e.evade_threshold,
            reward_evade_bonus=e.reward_evade_bonus,
            reward_step_penalty=e.reward_step_penalty,
            reward_mode=e.reward_mode,
            seed=self.cfg.seed,
            directions=self.directions,
        )

    def _clean_accuracy(self) -> float:
        preds = self.defender.predict(self.X_clean, threshold=self.cfg.env.evade_threshold)
        return float(np.mean(preds == self.y_clean))

    def _log(self, msg: str) -> None:
        if self.verbose:
            print(msg, flush=True)

    # ---- main loop ----------------------------------------------------------
    def run(self) -> ExperimentResult:
        # Round 0 setup: train the initial defender on clean data.
        self.defender.fit(self.X_train, self.y_train)
        self._log(
            f"[init] defender={self.cfg.defender.name} "
            f"attacker={self.cfg.attacker.name} "
            f"policy={self.policy.describe()} "
            f"clean_acc={self._clean_accuracy():.3f}"
        )

        for rnd in range(self.cfg.rounds):
            # 1-2. attacker trains against a frozen snapshot
            snapshot = self.defender.snapshot()
            env = self._make_env(snapshot)
            self.attacker.reset_policy()
            self.attacker.train(env, episodes=self.cfg.attacker.train_episodes)

            # 3. evaluate evasion on held-out malicious pool
            eval_env = self._make_env(snapshot)
            res = self.attacker.evaluate_evasion(eval_env, self.test_malicious)

            # 4. record discovered evasions
            self.policy.record(res.evasive_samples, self.defender)

            # 5. retrain defender per policy
            retrain_seconds = 0.0
            train_samples = 0
            retrained = self.policy.should_retrain(rnd, res.evasion_rate)
            if retrained:
                X, y = self.policy.build_training_set(self.X_train, self.y_train)
                train_samples = int(X.shape[0])
                t0 = time.perf_counter()
                self.defender.fit(X, y, warm_start=self.policy.warm_start)
                retrain_seconds = time.perf_counter() - t0

            # 6. log
            log = RoundLog(
                round=rnd,
                evasion_rate=res.evasion_rate,
                attack_success_rate=res.attack_success_rate,
                pre_evasive_rate=res.pre_evasive_rate,
                mean_queries=res.mean_queries,
                retrained=retrained,
                retrain_seconds=retrain_seconds,
                train_samples=train_samples,
                clean_accuracy=self._clean_accuracy(),
                buffer_size=len(self.policy.buffer),
            )
            self.result.rounds.append(log)
            self._log(
                f"[round {rnd:02d}] evasion={res.evasion_rate:.3f} "
                f"attack_succ={res.attack_success_rate:.3f} "
                f"pre_evasive={res.pre_evasive_rate:.3f} "
                f"queries={res.mean_queries:.1f} "
                f"retrained={'Y' if retrained else '-'} "
                f"clean_acc={log.clean_accuracy:.3f} "
                f"buffer={log.buffer_size}"
            )

            # early stop on convergence
            osc = oscillation_index(self.result.evasion_rates, self.cfg.convergence_window)
            if (
                rnd + 1 >= self.cfg.convergence_window
                and osc <= self.cfg.convergence_tol
            ):
                self._log(f"[converged] oscillation_index={osc:.4f} <= tol at round {rnd}")
                break

        return self.result

    # ---- persistence --------------------------------------------------------
    def save(self, out_dir: str | Path | None = None) -> Path:
        import json

        out_dir = Path(out_dir or self.cfg.output_dir) / self.cfg.name
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "result.json").write_text(
            json.dumps(self.result.to_dict(), indent=2), encoding="utf-8"
        )
        self.cfg.to_yaml(out_dir / "config.yaml")
        try:
            from coevomal.evaluation.plots import plot_run

            plot_run(self.result, out_dir / "dynamics.png", title=self.cfg.name)
        except Exception as exc:  # plotting is best-effort
            self._log(f"[warn] plot failed: {exc}")
        return out_dir
