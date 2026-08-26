"""Experiment configuration objects and YAML (de)serialization.

Everything an experiment needs is captured in :class:`ExperimentConfig`
so that a run is fully reproducible from a single config file. The
factorial study varies a handful of these fields (retraining cadence,
data-selection strategy, attacker algorithm) while holding the rest
fixed -- see ``experiments/run_factorial.py``.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

import yaml


@dataclass
class DatasetConfig:
    """Configuration for the (synthetic, by default) feature-space dataset."""

    name: str = "synthetic"          # {"synthetic", "ember"}
    n_features: int = 32
    n_train: int = 4000
    n_test_malicious: int = 400      # held-out malicious pool the attacker probes
    class_separation: float = 1.4    # higher => easier baseline classification
    mutable_fraction: float = 0.6    # fraction of features the attacker may perturb
    ember_path: str | None = None    # set when name == "ember"
    seed: int = 0


@dataclass
class EnvConfig:
    """Configuration for the malware evasion environment."""

    action_space: str = "mimicry"    # {"mimicry", "random"} -- see build_mimicry_directions
    n_actions: int = 12              # discrete functionality-preserving mutations
    max_steps: int = 30              # attacker budget per sample (query budget)
    step_scale: float = 0.25         # magnitude of a single mutation
    evade_threshold: float = 0.5     # defender prob below which a sample is "benign"
    reward_evade_bonus: float = 10.0
    reward_step_penalty: float = 0.02
    reward_mode: str = "logit"       # {"logit", "prob"} -- logit avoids saturation


@dataclass
class DefenderConfig:
    """Configuration for the classifier under attack."""

    name: str = "gbdt"               # {"gbdt", "mlp"}
    # GBDT (sklearn HistGradientBoosting -- LightGBM stand-in) params:
    max_iter: int = 150
    learning_rate: float = 0.08
    max_depth: int | None = None
    # MLP (torch) params:
    hidden: int = 128
    epochs: int = 40
    seed: int = 0


@dataclass
class AttackerConfig:
    """Configuration for the RL evasion agent."""

    name: str = "dqn"                # {"dqn", "ppo", "random"}
    train_episodes: int = 400        # episodes of training per co-evolution round
    gamma: float = 0.95
    lr: float = 1e-3
    hidden: int = 128
    # DQN:
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay: float = 0.995
    buffer_size: int = 10000
    batch_size: int = 64
    target_sync: int = 200
    train_every: int = 4             # gradient step every N env steps (standard DQN)
    # PPO:
    ppo_epochs: int = 4
    clip: float = 0.2
    rollout: int = 512
    device: str = "auto"             # {"auto", "cpu", "cuda"}
    seed: int = 0


@dataclass
class RetrainPolicyConfig:
    """The pluggable retraining policy -- the central independent variable."""

    cadence: str = "every_round"     # {"every_round", "every_n", "threshold"}
    every_n: int = 3                 # used when cadence == "every_n"
    trigger_threshold: float = 0.3   # evasion-rate trigger when cadence == "threshold"
    data_selection: str = "full_replay"  # {"full_replay", "hard_mining", "bounded_buffer"}
    buffer_size: int = 2000          # cap for bounded_buffer / hard_mining
    mode: str = "scratch"            # {"scratch", "finetune"}


@dataclass
class ExperimentConfig:
    """Top-level experiment description."""

    name: str = "default"
    rounds: int = 15
    convergence_window: int = 4      # trailing window for convergence/oscillation
    convergence_tol: float = 0.03    # evasion-rate std below which we call convergence
    seed: int = 0
    output_dir: str = "results"

    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    env: EnvConfig = field(default_factory=EnvConfig)
    defender: DefenderConfig = field(default_factory=DefenderConfig)
    attacker: AttackerConfig = field(default_factory=AttackerConfig)
    retrain: RetrainPolicyConfig = field(default_factory=RetrainPolicyConfig)

    # ---- (de)serialization -------------------------------------------------
    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_yaml(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as fh:
            yaml.safe_dump(self.to_dict(), fh, sort_keys=False)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ExperimentConfig":
        data = dict(data or {})
        nested = {
            "dataset": DatasetConfig,
            "env": EnvConfig,
            "defender": DefenderConfig,
            "attacker": AttackerConfig,
            "retrain": RetrainPolicyConfig,
        }
        kwargs: dict[str, Any] = {}
        for key, value in data.items():
            if key in nested and isinstance(value, dict):
                kwargs[key] = nested[key](**value)
            else:
                kwargs[key] = value
        return cls(**kwargs)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "ExperimentConfig":
        with Path(path).open("r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh) or {}
        return cls.from_dict(data)

    def replace(self, **overrides: Any) -> "ExperimentConfig":
        """Return a deep-ish copy with top-level or ``section.field`` overrides.

        Nested fields use dotted keys, e.g. ``replace(**{"retrain.cadence": "every_n"})``.
        """
        clone = ExperimentConfig.from_dict(self.to_dict())
        for key, value in overrides.items():
            if "." in key:
                section, sub = key.split(".", 1)
                setattr(getattr(clone, section), sub, value)
            else:
                setattr(clone, key, value)
        return clone


def dataclass_fields(obj: Any) -> list[str]:
    """Helper: list field names of a dataclass instance/type."""
    return [f.name for f in dataclasses.fields(obj)]
