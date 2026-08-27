"""Defender package + factory."""

from __future__ import annotations

from coevomal.config import DefenderConfig
from coevomal.defenders.base import Defender
from coevomal.defenders.gbdt_defender import GBDTDefender
from coevomal.defenders.mlp_defender import MLPDefender


def make_defender(cfg: DefenderConfig) -> Defender:
    if cfg.name == "gbdt":
        return GBDTDefender(
            max_iter=cfg.max_iter,
            learning_rate=cfg.learning_rate,
            max_depth=cfg.max_depth,
            finetune_iter=cfg.finetune_iter,
            max_total_iter=cfg.max_total_iter,
            seed=cfg.seed,
        )
    if cfg.name == "mlp":
        return MLPDefender(hidden=cfg.hidden, epochs=cfg.epochs, seed=cfg.seed)
    raise ValueError(f"unknown defender '{cfg.name}' (expected 'gbdt' or 'mlp')")


__all__ = ["Defender", "GBDTDefender", "MLPDefender", "make_defender"]
