"""Deep Q-Network evasion attacker (torch).

A compact DQN with an experience-replay buffer, a target network, and
epsilon-greedy exploration -- the standard recipe used by the gym-malware /
DQEAF line of work. It plays discrete functionality-preserving mutation
actions to minimise the defender's malicious score within the query budget.

Kept intentionally small (two hidden layers) so a full co-evolution run of
many retraining rounds is tractable on CPU while still benefiting from GPU
when available.
"""

from __future__ import annotations

import random
from collections import deque

import numpy as np
import torch
from torch import nn

from coevomal.attackers.base import Attacker
from coevomal.environment.malware_env import MalwareEvasionEnv
from coevomal.utils import resolve_device


class _QNet(nn.Module):
    def __init__(self, obs_dim: int, n_actions: int, hidden: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, n_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DQNAttacker(Attacker):
    def __init__(
        self,
        obs_dim: int,
        n_actions: int,
        hidden: int = 128,
        gamma: float = 0.95,
        lr: float = 1e-3,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.05,
        epsilon_decay: float = 0.995,
        buffer_size: int = 10000,
        batch_size: int = 64,
        target_sync: int = 200,
        device: str = "auto",
        seed: int = 0,
    ) -> None:
        self.obs_dim = int(obs_dim)
        self.n_actions = int(n_actions)
        self.gamma = float(gamma)
        self.batch_size = int(batch_size)
        self.target_sync = int(target_sync)
        self.epsilon = float(epsilon_start)
        self.epsilon_end = float(epsilon_end)
        self.epsilon_decay = float(epsilon_decay)
        self.device = resolve_device(device)

        torch.manual_seed(seed)
        random.seed(seed)
        self.rng = np.random.default_rng(seed)

        self.q = _QNet(obs_dim, n_actions, hidden).to(self.device)
        self.target = _QNet(obs_dim, n_actions, hidden).to(self.device)
        self.target.load_state_dict(self.q.state_dict())
        self.opt = torch.optim.Adam(self.q.parameters(), lr=lr)
        self.buffer: deque = deque(maxlen=buffer_size)
        self._learn_steps = 0

    # ---- acting -------------------------------------------------------------
    def act(self, obs: np.ndarray, greedy: bool = True) -> int:
        if not greedy and self.rng.random() < self.epsilon:
            return int(self.rng.integers(0, self.n_actions))
        with torch.no_grad():
            t = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
            q = self.q(t.unsqueeze(0))
            return int(torch.argmax(q, dim=1).item())

    # ---- learning -----------------------------------------------------------
    def _optimise(self) -> None:
        if len(self.buffer) < self.batch_size:
            return
        batch = random.sample(self.buffer, self.batch_size)
        obs, act, rew, nxt, done = zip(*batch)
        obs = torch.as_tensor(np.array(obs), dtype=torch.float32, device=self.device)
        act = torch.as_tensor(act, dtype=torch.int64, device=self.device).unsqueeze(1)
        rew = torch.as_tensor(rew, dtype=torch.float32, device=self.device)
        nxt = torch.as_tensor(np.array(nxt), dtype=torch.float32, device=self.device)
        done = torch.as_tensor(done, dtype=torch.float32, device=self.device)

        q_sa = self.q(obs).gather(1, act).squeeze(1)
        with torch.no_grad():
            q_next = self.target(nxt).max(dim=1).values
            target = rew + self.gamma * q_next * (1.0 - done)
        loss = nn.functional.smooth_l1_loss(q_sa, target)
        self.opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.q.parameters(), 10.0)
        self.opt.step()

        self._learn_steps += 1
        if self._learn_steps % self.target_sync == 0:
            self.target.load_state_dict(self.q.state_dict())

    def train(self, env: MalwareEvasionEnv, episodes: int) -> None:
        for _ in range(episodes):
            obs = env.reset()
            done = False
            while not done:
                action = self.act(obs, greedy=False)
                nxt, reward, done, _info = env.step(action)
                self.buffer.append((obs, action, reward, nxt, float(done)))
                obs = nxt
                self._optimise()
            self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
