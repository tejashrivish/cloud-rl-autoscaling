import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical


# ─────────────────────────────────────────
# Actor-Critic Network
# ─────────────────────────────────────────

class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden=128):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(state_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden),    nn.ReLU(),
        )
        self.actor  = nn.Linear(hidden, action_dim)
        self.critic = nn.Linear(hidden, 1)

    def forward(self, x):
        features = self.trunk(x)
        return self.actor(features), self.critic(features)


# ─────────────────────────────────────────
# A2C Agent
# ─────────────────────────────────────────

class A2CAgent:

    def __init__(
        self,
        state_dim,
        action_dim,
        lr           = 1e-4,
        gamma        = 0.99,
        value_coef   = 0.5,
        entropy_coef = 0.05,
    ):
        self.gamma        = gamma
        self.value_coef   = value_coef
        self.entropy_coef = entropy_coef

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model     = ActorCritic(state_dim, action_dim).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)

        print(f"A2CAgent | device={self.device} | state_dim={state_dim} | action_dim={action_dim}")

    # ── helpers ───────────────────────────────────────────────────────────────

    def _to_tensor(self, state):
        return torch.from_numpy(
            np.array(state, dtype=np.float32)
        ).unsqueeze(0).to(self.device)

    # ── action selection ──────────────────────────────────────────────────────

    def select_action(self, state):
        logits, value = self.model(self._to_tensor(state))
        dist          = Categorical(logits=logits)
        action        = dist.sample()
        return action.item(), dist.log_prob(action), value, dist

    def get_action(self, state):
        with torch.no_grad():
            logits, _ = self.model(self._to_tensor(state))
            return int(logits.argmax(dim=1).item())

    # ── single-step update ────────────────────────────────────────────────────

    def update(self, log_prob, value, reward, next_value, done, dist):
        r    = torch.tensor([[reward]],     dtype=torch.float32).to(self.device)
        mask = torch.tensor([[1.0 - done]], dtype=torch.float32).to(self.device)

        advantage   = r + self.gamma * next_value.detach() * mask - value
        actor_loss  = -(log_prob * advantage.detach())
        critic_loss = advantage.pow(2)
        entropy     = dist.entropy()

        loss = actor_loss + self.value_coef * critic_loss - self.entropy_coef * entropy

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.model.parameters(), 0.3)
        self.optimizer.step()

    # ── batched rollout update ────────────────────────────────────────────────

    def update_batch(self, rewards, log_probs, values, next_value, dones):
        """
        Full-episode rollout update.

        Advantage normalisation guarded against zero std — occurs when
        all rewards in a rollout are identical (e.g. all clipped to same
        value), which would produce NaN without the guard.
        Gradient clipping tightened to 0.3 to prevent explosion under
        high-variance DR workloads.
        """
        returns = []
        R       = next_value

        for reward, done in zip(reversed(rewards), reversed(dones)):
            R = reward + self.gamma * R * (1.0 - float(done))
            returns.insert(0, R)

        returns     = torch.tensor(returns, dtype=torch.float32).to(self.device)
        log_probs_t = torch.stack(log_probs).squeeze()
        values_t    = torch.stack(values).squeeze()

        advantages = returns - values_t.detach()

        # guard against zero std — happens when all rewards are clipped
        # to the same value, which would produce NaN without this check
        if advantages.numel() > 1 and advantages.std() > 1e-8:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        actor_loss  = -(log_probs_t * advantages).mean()
        critic_loss = (returns - values_t).pow(2).mean()
        entropy     = -(log_probs_t.exp() * log_probs_t).mean()

        loss = actor_loss + self.value_coef * critic_loss - self.entropy_coef * entropy

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.model.parameters(), 0.3)
        self.optimizer.step()

    # ── persistence ───────────────────────────────────────────────────────────

    def save(self, path):
        torch.save(self.model.state_dict(), path)
        print(f"Model saved → {path}")

    def load(self, path):
        self.model.load_state_dict(torch.load(path, map_location=self.device))
        print(f"Model loaded ← {path}")