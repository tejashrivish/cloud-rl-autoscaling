import numpy as np
import random
from collections import deque

import torch
import torch.nn as nn
import torch.optim as optim


# ─────────────────────────────────────────
# Q-Network
# ─────────────────────────────────────────

class DQN(nn.Module):
    """
    Maps state → Q-values for each action.
    Three hidden layers with ReLU activations.
    """
    def __init__(self, state_dim, action_dim, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden),    nn.ReLU(),
            nn.Linear(hidden, hidden),    nn.ReLU(),
            nn.Linear(hidden, action_dim),
        )

    def forward(self, x):
        return self.net(x)


# ─────────────────────────────────────────
# DQN Agent
# ─────────────────────────────────────────

class DQNAgent:
    """
    Double DQN with:
      - experience replay    (breaks temporal correlation)
      - target network       (stabilises Q-value targets)
      - epsilon-greedy       (exploration vs exploitation)
      - gradient clipping    (handles high-variance episodes)
      - HuberLoss            (robust to large reward errors)

    Key design decisions
    --------------------
    - epsilon decays once per EPISODE (end_episode), not per step
    - target network syncs every `target_update` EPISODES, not steps
    - isolated _rng so epsilon-greedy never disturbs global numpy state
    """

    def __init__(
        self,
        state_dim,
        action_dim,
        lr            = 1e-3,
        gamma         = 0.99,
        epsilon       = 1.0,
        epsilon_min   = 0.05,
        epsilon_decay = 0.995,
        batch_size    = 64,
        buffer_size   = 50_000,
        target_update = 10,       # episodes between target network syncs
    ):
        self.action_dim    = action_dim
        self.gamma         = gamma
        self.epsilon       = epsilon
        self.epsilon_min   = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.batch_size    = batch_size
        self.target_update = target_update
        self.ep_count      = 0

        # isolated RNG — never interferes with np.random used elsewhere
        self._rng = np.random.default_rng(seed=0)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # online network  — updated every train() call
        self.q_net      = DQN(state_dim, action_dim).to(self.device)
        # target network  — synced every `target_update` episodes
        self.target_net = DQN(state_dim, action_dim).to(self.device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr)
        self.memory    = deque(maxlen=buffer_size)

        print(f"DQNAgent | device={self.device} | state_dim={state_dim} | action_dim={action_dim}")

    # ── action selection ──────────────────

    def act(self, state):
        """Epsilon-greedy: explore randomly or exploit the online network."""
        if self._rng.random() < self.epsilon:
            return int(self._rng.integers(self.action_dim))
        s = torch.from_numpy(np.array(state)).float().unsqueeze(0).to(self.device)
        with torch.no_grad():
            return int(self.q_net(s).argmax(dim=1).item())

    def get_action(self, state):
        """Greedy action for evaluation — no exploration."""
        s = torch.from_numpy(np.array(state)).float().unsqueeze(0).to(self.device)
        with torch.no_grad():
            return int(self.q_net(s).argmax(dim=1).item())

    # ── memory ───────────────────────────

    def push(self, state, action, reward, next_state, done):
        """Store a transition. done must be float (0.0 or 1.0)."""
        self.memory.append((
            np.array(state,      dtype=np.float32),
            action,
            reward,
            np.array(next_state, dtype=np.float32),
            float(done),
        ))

    # ── learning ─────────────────────────

    def train(self):
        """
        One gradient step on a random mini-batch (Double DQN).

        Online network  — selects the next action
        Target network  — evaluates its Q-value

        Returns loss value, or None if buffer not warm yet.
        """
        if len(self.memory) < self.batch_size:
            return None

        batch                                         = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones  = zip(*batch)

        s  = torch.from_numpy(np.array(states)).float().to(self.device)
        a  = torch.from_numpy(np.array(actions)).long().unsqueeze(1).to(self.device)
        r  = torch.from_numpy(np.array(rewards)).float().unsqueeze(1).to(self.device)
        ns = torch.from_numpy(np.array(next_states)).float().to(self.device)
        d  = torch.from_numpy(np.array(dones)).float().unsqueeze(1).to(self.device)

        # Q(s, a) — value of the action actually taken
        q_values = self.q_net(s).gather(1, a)

        # Double DQN target:
        # online net picks the next action, target net evaluates it
        with torch.no_grad():
            next_a  = self.q_net(ns).argmax(1, keepdim=True)
            next_q  = self.target_net(ns).gather(1, next_a)
            targets = r + (1 - d) * self.gamma * next_q

        # HuberLoss — less sensitive to large errors than MSE
        loss = nn.HuberLoss()(q_values, targets)

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.q_net.parameters(), 1.0)
        self.optimizer.step()

        return loss.item()

    # ── end-of-episode bookkeeping ────────

    def end_episode(self):
        """
        Call ONCE at the end of every episode — NOT per step.
          - decays epsilon by one step
          - syncs target network every `target_update` episodes
        """
        self.epsilon   = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        self.ep_count += 1
        if self.ep_count % self.target_update == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())

    # ── persistence ───────────────────────

    def save(self, path):
        torch.save(self.q_net.state_dict(), path)
        print(f"Model saved → {path}")

    def load(self, path):
        self.q_net.load_state_dict(torch.load(path, map_location=self.device))
        self.target_net.load_state_dict(self.q_net.state_dict())
        print(f"Model loaded ← {path}")