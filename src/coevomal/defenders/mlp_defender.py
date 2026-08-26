"""Torch MLP defender (deep-learning variant, MalConv stand-in).

A small feed-forward network over the static feature vectors. It plays the
role of the "swap in a deep model if time allows" defender from the
roadmap: the true MalConv consumes raw bytes, but at the feature-vector
interface used throughout this harness an MLP is the natural analogue and
lets the policy study include a neural defender without a raw-PE pipeline.

``finetune`` continues training from the current weights; ``scratch``
re-initialises the network.
"""

from __future__ import annotations

import copy

import numpy as np
import torch
from torch import nn

from coevomal.defenders.base import Defender
from coevomal.utils import resolve_device


class _Net(nn.Module):
    def __init__(self, in_dim: int, hidden: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


class MLPDefender(Defender):
    def __init__(
        self,
        hidden: int = 128,
        epochs: int = 40,
        lr: float = 1e-3,
        batch_size: int = 256,
        device: str = "auto",
        seed: int = 0,
    ) -> None:
        self.hidden = int(hidden)
        self.epochs = int(epochs)
        self.lr = float(lr)
        self.batch_size = int(batch_size)
        self.device = resolve_device(device)
        self.seed = int(seed)
        self._net: _Net | None = None
        self._in_dim: int | None = None

    def _ensure_net(self, in_dim: int, reset: bool) -> None:
        if self._net is None or reset or self._in_dim != in_dim:
            torch.manual_seed(self.seed)
            self._net = _Net(in_dim, self.hidden).to(self.device)
            self._in_dim = in_dim

    def fit(self, X: np.ndarray, y: np.ndarray, warm_start: bool = False) -> "MLPDefender":
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32)
        self._ensure_net(X.shape[1], reset=not warm_start)
        assert self._net is not None

        xb_all = torch.from_numpy(X).to(self.device)
        yb_all = torch.from_numpy(y).to(self.device)
        opt = torch.optim.Adam(self._net.parameters(), lr=self.lr)
        loss_fn = nn.BCEWithLogitsLoss()
        n = X.shape[0]
        self._net.train()
        for _ in range(self.epochs):
            perm = torch.randperm(n, device=self.device)
            for start in range(0, n, self.batch_size):
                idx = perm[start:start + self.batch_size]
                opt.zero_grad()
                logits = self._net(xb_all[idx])
                loss = loss_fn(logits, yb_all[idx])
                loss.backward()
                opt.step()
        return self

    @torch.no_grad()
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        assert self._net is not None, "defender is not fitted"
        self._net.eval()
        xb = torch.from_numpy(np.asarray(X, dtype=np.float32)).to(self.device)
        probs = torch.sigmoid(self._net(xb))
        return probs.detach().cpu().numpy().reshape(-1)

    def snapshot(self) -> "MLPDefender":
        clone = MLPDefender(
            hidden=self.hidden,
            epochs=self.epochs,
            lr=self.lr,
            batch_size=self.batch_size,
            device=str(self.device),
            seed=self.seed,
        )
        clone._in_dim = self._in_dim
        if self._net is not None:
            clone._net = copy.deepcopy(self._net).to(self.device)
        return clone

    @property
    def is_fitted(self) -> bool:
        return self._net is not None
