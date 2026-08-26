"""Proximal Policy Optimization evasion attacker (torch).

A clipped-objective actor-critic PPO agent. PPO is the roadmap's secondary
attacker axis and the standard fallback when DQN training is unstable
(a known failure mode in this literature where DQN can collapse to a
near-random policy). Sharing the :class:`Attacker` interface with the DQN
agent means the co-evolution loop and every metric are computed identically
across both, which is the whole point of the shared harness.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical

from coevomal.attackers.base import Attacker
from coevomal.environment.malware_env import MalwareEvasionEnv
from coevomal.utils import resolve_device


class _ActorCritic(nn.Module):
    def __init__(self, obs_dim: int, n_actions: int, hidden: int) -> None:
        super().__init__()
        self.body = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
        )
        self.pi = nn.Linear(hidden, n_actions)
        self.v = nn.Linear(hidden, 1)

    def forward(self, x: torch.Tensor):
        h = self.body(x)
        return self.pi(h), self.v(h).squeeze(-1)


class PPOAttacker(Attacker):
    def __init__(
        self,
        obs_dim: int,
        n_actions: int,
        hidden: int = 128,
        gamma: float = 0.95,
        lr: float = 1e-3,
        clip: float = 0.2,
        ppo_epochs: int = 4,
        rollout: int = 512,
        batch_size: int = 64,
        device: str = "auto",
        seed: int = 0,
    ) -> None:
        self.obs_dim = int(obs_dim)
        self.n_actions = int(n_actions)
        self.gamma = float(gamma)
        self.clip = float(clip)
        self.ppo_epochs = int(ppo_epochs)
        self.rollout = int(rollout)
        self.batch_size = int(batch_size)
        self.device = resolve_device(device)

        torch.manual_seed(seed)
        self.rng = np.random.default_rng(seed)
        self.net = _ActorCritic(obs_dim, n_actions, hidden).to(self.device)
        self.opt = torch.optim.Adam(self.net.parameters(), lr=lr)

    # ---- acting -------------------------------------------------------------
    def act(self, obs: np.ndarray, greedy: bool = True) -> int:
        with torch.no_grad():
            t = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
            logits, _ = self.net(t)
            if greedy:
                return int(torch.argmax(logits, dim=1).item())
            return int(Categorical(logits=logits).sample().item())

    def act_batch(self, obs: np.ndarray, greedy: bool = True) -> np.ndarray:
        """Batched action selection (single forward pass)."""
        with torch.no_grad():
            t = torch.as_tensor(np.asarray(obs), dtype=torch.float32, device=self.device)
            logits, _ = self.net(t)
            if greedy:
                return torch.argmax(logits, dim=1).cpu().numpy().astype(np.int64)
            return Categorical(logits=logits).sample().cpu().numpy().astype(np.int64)

    def _act_train(self, obs: np.ndarray):
        t = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        logits, value = self.net(t)
        dist = Categorical(logits=logits)
        action = dist.sample()
        return int(action.item()), float(dist.log_prob(action).item()), float(value.item())

    # ---- learning -----------------------------------------------------------
    def _returns(self, rewards, dones, last_value):
        out = np.zeros(len(rewards), dtype=np.float32)
        running = last_value
        for t in reversed(range(len(rewards))):
            running = rewards[t] + self.gamma * running * (1.0 - dones[t])
            out[t] = running
        return out

    def _update(self, traj) -> None:
        obs = torch.as_tensor(np.array(traj["obs"]), dtype=torch.float32, device=self.device)
        act = torch.as_tensor(traj["act"], dtype=torch.int64, device=self.device)
        old_logp = torch.as_tensor(traj["logp"], dtype=torch.float32, device=self.device)
        ret = torch.as_tensor(traj["ret"], dtype=torch.float32, device=self.device)
        with torch.no_grad():
            _, values = self.net(obs)
        adv = ret - values
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        n = obs.shape[0]
        idx = np.arange(n)
        for _ in range(self.ppo_epochs):
            self.rng.shuffle(idx)
            for start in range(0, n, self.batch_size):
                b = idx[start:start + self.batch_size]
                logits, value = self.net(obs[b])
                dist = Categorical(logits=logits)
                logp = dist.log_prob(act[b])
                ratio = torch.exp(logp - old_logp[b])
                s1 = ratio * adv[b]
                s2 = torch.clamp(ratio, 1 - self.clip, 1 + self.clip) * adv[b]
                policy_loss = -torch.min(s1, s2).mean()
                value_loss = nn.functional.mse_loss(value, ret[b])
                entropy = dist.entropy().mean()
                loss = policy_loss + 0.5 * value_loss - 0.01 * entropy
                self.opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.net.parameters(), 10.0)
                self.opt.step()

    def train(self, env: MalwareEvasionEnv, episodes: int) -> None:
        # Interpret `episodes` as a total-episode budget; collect fixed-size
        # rollouts and run a PPO update after each.
        traj = {"obs": [], "act": [], "logp": [], "rew": [], "done": [], "ret": []}
        obs = env.reset()
        episodes_done = 0
        steps = 0
        while episodes_done < episodes:
            action, logp, _value = self._act_train(obs)
            nxt, reward, done, _info = env.step(action)
            traj["obs"].append(obs)
            traj["act"].append(action)
            traj["logp"].append(logp)
            traj["rew"].append(reward)
            traj["done"].append(float(done))
            obs = nxt
            steps += 1
            if done:
                obs = env.reset()
                episodes_done += 1
            if steps >= self.rollout:
                with torch.no_grad():
                    t = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
                    _, last_v = self.net(t.unsqueeze(0))
                traj["ret"] = list(
                    self._returns(traj["rew"], traj["done"], float(last_v.item()))
                )
                self._update(traj)
                traj = {k: [] for k in traj}
                steps = 0
        if traj["obs"]:  # flush the tail
            traj["ret"] = list(self._returns(traj["rew"], traj["done"], 0.0))
            self._update(traj)
