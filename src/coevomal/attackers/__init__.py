"""Attacker package + factory."""

from __future__ import annotations

from coevomal.attackers.base import Attacker, EvasionResult
from coevomal.attackers.dqn import DQNAttacker
from coevomal.attackers.ppo import PPOAttacker
from coevomal.attackers.random_attacker import RandomAttacker
from coevomal.config import AttackerConfig


def make_attacker(cfg: AttackerConfig, obs_dim: int, n_actions: int) -> Attacker:
    if cfg.name == "dqn":
        return DQNAttacker(
            obs_dim=obs_dim,
            n_actions=n_actions,
            hidden=cfg.hidden,
            gamma=cfg.gamma,
            lr=cfg.lr,
            epsilon_start=cfg.epsilon_start,
            epsilon_end=cfg.epsilon_end,
            epsilon_decay=cfg.epsilon_decay,
            buffer_size=cfg.buffer_size,
            batch_size=cfg.batch_size,
            target_sync=cfg.target_sync,
            device=cfg.device,
            seed=cfg.seed,
        )
    if cfg.name == "ppo":
        return PPOAttacker(
            obs_dim=obs_dim,
            n_actions=n_actions,
            hidden=cfg.hidden,
            gamma=cfg.gamma,
            lr=cfg.lr,
            clip=cfg.clip,
            ppo_epochs=cfg.ppo_epochs,
            rollout=cfg.rollout,
            batch_size=cfg.batch_size,
            device=cfg.device,
            seed=cfg.seed,
        )
    if cfg.name == "random":
        return RandomAttacker(n_actions=n_actions, seed=cfg.seed)
    raise ValueError(f"unknown attacker '{cfg.name}' (expected 'dqn', 'ppo' or 'random')")


__all__ = [
    "Attacker",
    "EvasionResult",
    "DQNAttacker",
    "PPOAttacker",
    "RandomAttacker",
    "make_attacker",
]
