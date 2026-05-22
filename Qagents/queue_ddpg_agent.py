import numpy as np
import random
from collections import deque

import torch
import torch.nn as nn
import torch.optim as optim


# ─────────────────────────────────────────
# Actor
# ─────────────────────────────────────────

class Actor(nn.Module):
    """
    Maps state → continuous action v ∈ [-1, 1] via Tanh.
    The train script rescales to [0.1, 1.0] via env.action_space clipping.

    state_dim is now 5 (includes queue).
    """
    def __init__(self, state_dim, action_dim, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden),    nn.ReLU(),
            nn.Linear(hidden, action_dim),nn.Tanh(),
        )

    def forward(self, x):
        return self.net(x)


# ─────────────────────────────────────────
# Critic
# ─────────────────────────────────────────

class Critic(nn.Module):
    """
    Maps (state, action) → Q-value scalar.
    State and action are concatenated before the first linear layer.
    """
    def __init__(self, state_dim, action_dim, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden),                 nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, state, action):
        return self.net(torch.cat([state, action], dim=1))


# ─────────────────────────────────────────
# DDPG Agent
# ─────────────────────────────────────────

class DDPGAgent:
    """
    Deep Deterministic Policy Gradient (DDPG).

    Key design decisions
    --------------------
    - Separate actor and critic networks, each with a target copy.
    - Target networks updated via soft (Polyak) averaging every train step:
          θ_target ← τ·θ + (1-τ)·θ_target
      This stabilises the Q-value targets compared to hard periodic syncs.
    - Experience replay buffer (50k) breaks temporal correlation, same as DQN.
    - Gaussian exploration noise added during training; removed at evaluation
      via act(state, noise=False).
    - train() skips every other step (train_step % 2) — halves gradient
      overhead on long episodes without meaningfully slowing learning.
    - Critic targets clamped at ±1e5 to prevent Q-value explosion on high-
      variance DR workloads early in training.
    - Gradient clipping on critic (1.0) — actor gradients flow through the
      critic so instability there propagates to the actor.
    - state_dim=5 includes the queue feature introduced in the continuous env.
    """

    def __init__(
        self,
        state_dim,
        action_dim,
        actor_lr   = 1e-4,
        critic_lr  = 1e-4,
        gamma      = 0.99,
        tau        = 0.005,
        buffer_size= 50_000,
        batch_size = 64,
        noise_std  = 0.1,
    ):
        self.gamma      = gamma
        self.tau        = tau
        self.batch_size = batch_size
        self.noise_std  = noise_std
        self.train_step = 0

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # actor — online + target
        self.actor        = Actor(state_dim, action_dim).to(self.device)
        self.actor_target = Actor(state_dim, action_dim).to(self.device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_target.eval()

        # critic — online + target
        self.critic        = Critic(state_dim, action_dim).to(self.device)
        self.critic_target = Critic(state_dim, action_dim).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_target.eval()

        self.actor_opt  = optim.Adam(self.actor.parameters(),  lr=actor_lr)
        self.critic_opt = optim.Adam(self.critic.parameters(), lr=critic_lr)

        self.memory = deque(maxlen=buffer_size)

        print(
            f"DDPGAgent | device={self.device} | "
            f"state_dim={state_dim} | action_dim={action_dim}"
        )

    # ── helpers ───────────────────────────────────────────────────────────────

    def _to_tensor(self, x, dtype=torch.float32):
        return torch.from_numpy(np.array(x, dtype=np.float32)).to(self.device)

    # ── action selection ──────────────────────────────────────────────────────

    def act(self, state, noise=True):
        """
        Deterministic action from the actor network.
        noise=True  — adds Gaussian exploration noise (training).
        noise=False — pure deterministic policy (evaluation).
        Output clipped to [-1, 1]; env.step clips to [0.1, 1.0].
        """
        s = self._to_tensor(state).unsqueeze(0)
        with torch.no_grad():
            action = self.actor(s).cpu().numpy()[0]

        if noise:
            action = action + np.random.normal(0, self.noise_std, size=action.shape)

        return np.clip(action, -1.0, 1.0)

    # ── buffer ────────────────────────────────────────────────────────────────

    def store(self, transition):
        """transition = (state, action, reward, next_state, done)"""
        self.memory.append(transition)

    # ── learning ──────────────────────────────────────────────────────────────

    def train(self):
        """
        One DDPG gradient step (critic then actor then soft update).
        Called every env step; skips every other step to reduce overhead.
        Returns early if buffer not warm.
        """
        if len(self.memory) < self.batch_size:
            return

        self.train_step += 1
        if self.train_step % 2 != 0:
            return

        batch                                    = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        s  = self._to_tensor(np.array(states))
        a  = self._to_tensor(np.array(actions))
        r  = self._to_tensor(np.array(rewards)).unsqueeze(1)
        ns = self._to_tensor(np.array(next_states))
        d  = self._to_tensor(np.array(dones, dtype=np.float32)).unsqueeze(1)

        # ── critic update ─────────────────────────────────────────────────────
        with torch.no_grad():
            next_a   = self.actor_target(ns)
            target_q = self.critic_target(ns, next_a)
            target   = r + (1.0 - d) * self.gamma * target_q
            target   = torch.clamp(target, -1e5, 1e5)   # prevent explosion

        current_q   = self.critic(s, a)
        critic_loss = nn.MSELoss()(current_q, target)

        self.critic_opt.zero_grad()
        critic_loss.backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)
        self.critic_opt.step()

        # ── actor update ──────────────────────────────────────────────────────
        actor_loss = -self.critic(s, self.actor(s)).mean()

        self.actor_opt.zero_grad()
        actor_loss.backward()
        self.actor_opt.step()

        # ── soft target update (Polyak averaging) ─────────────────────────────
        for tp, p in zip(self.actor_target.parameters(), self.actor.parameters()):
            tp.data.copy_(self.tau * p.data + (1.0 - self.tau) * tp.data)

        for tp, p in zip(self.critic_target.parameters(), self.critic.parameters()):
            tp.data.copy_(self.tau * p.data + (1.0 - self.tau) * tp.data)

    # ── persistence ───────────────────────────────────────────────────────────

    def save(self, path_actor, path_critic=None):
        torch.save(self.actor.state_dict(), path_actor)
        print(f"Actor saved → {path_actor}")
        if path_critic:
            torch.save(self.critic.state_dict(), path_critic)
            print(f"Critic saved → {path_critic}")

    def load(self, path_actor, path_critic=None):
        self.actor.load_state_dict(torch.load(path_actor, map_location=self.device))
        self.actor_target.load_state_dict(self.actor.state_dict())
        print(f"Actor loaded ← {path_actor}")
        if path_critic:
            self.critic.load_state_dict(torch.load(path_critic, map_location=self.device))
            self.critic_target.load_state_dict(self.critic.state_dict())
            print(f"Critic loaded ← {path_critic}")