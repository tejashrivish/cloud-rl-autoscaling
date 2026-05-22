import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical


# ─────────────────────────────────────────
# Actor-Critic Network
# ─────────────────────────────────────────

class ActorCritic(nn.Module):
    """
    Shared trunk → two heads.

    Actor head  : logits → Categorical distribution over actions (policy π)
    Critic head : scalar state-value estimate V(s)

    Sharing the trunk lets low-level feature extraction be learned jointly
    while the two heads specialise independently.
    """
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
    """
    Advantage Actor-Critic (A2C).

    Two update modes — same network, same optimizer, different callers
    ------------------------------------------------------------------
    update()       single-step TD update, called inside the step loop.
                   Used by: train_a2c.py, train_a2c_dr.py

    update_batch() full-episode rollout update, called once per episode
                   after collecting the full lists of rewards / log_probs
                   / values / dones.
                   Used by: train_a2c_curriculum.py

    Key design decisions
    --------------------
    - select_action() returns (action, log_prob, value, dist) so the
      caller has everything needed for either update mode without a
      second forward pass.
    - get_action() is a separate greedy path for evaluation — no graph,
      no sampling, plain int.
    - Entropy computed from dist.entropy() in update() (exact closed form).
      update_batch() uses single-action approximation since dist objects
      are not stored across the rollout — see note in update_batch().
    - Gradient clipping at 0.5 prevents large policy steps on high-
      variance rollouts (critical with DR workloads and queue dynamics).
    - update_batch() normalises advantages per rollout — keeps gradient
      magnitudes comparable across episodes with different workload scales.
    """

    def __init__(
        self,
        state_dim,
        action_dim,
        lr           = 3e-4,
        gamma        = 0.99,
        value_coef   = 0.5,
        entropy_coef = 0.01,
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
        """
        Sample from π(·|s).
        Returns (action, log_prob, value, dist).
        dist is passed to update() for exact entropy computation.
        """
        logits, value = self.model(self._to_tensor(state))
        dist          = Categorical(logits=logits)
        action        = dist.sample()
        return action.item(), dist.log_prob(action), value, dist

    def get_action(self, state):
        """Greedy argmax of π for evaluation. No graph, plain int."""
        with torch.no_grad():
            logits, _ = self.model(self._to_tensor(state))
            return int(logits.argmax(dim=1).item())

    # ── single-step update ────────────────────────────────────────────────────

    def update(self, log_prob, value, reward, next_value, done, dist):
        """
        One TD gradient step — call inside the step loop.

        Advantage = r + γ·V(s')·(1-done) - V(s)

        L = -log π(a|s) · A.detach()       (actor)
          + value_coef · A²                (critic)
          - entropy_coef · H(π)            (entropy bonus, exact)
        """
        r    = torch.tensor([[reward]],     dtype=torch.float32).to(self.device)
        mask = torch.tensor([[1.0 - done]], dtype=torch.float32).to(self.device)

        advantage   = r + self.gamma * next_value.detach() * mask - value

        actor_loss  = -(log_prob * advantage.detach())
        critic_loss = advantage.pow(2)
        entropy     = dist.entropy()   # exact H(π) — possible because dist is in scope

        loss = actor_loss + self.value_coef * critic_loss - self.entropy_coef * entropy

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
        self.optimizer.step()

    # ── batched rollout update ────────────────────────────────────────────────

    def update_batch(self, rewards, log_probs, values, next_value, dones):
        """
        Full-episode rollout update — call once per episode.

        Discounted returns computed backwards from bootstrap value:
            R_t = r_t + γ · R_{t+1} · (1 - done_t)

        Advantages normalised to zero-mean unit-variance within each
        rollout so gradient magnitudes stay comparable across episodes
        with different workload scales (essential with DR + queue).

        Entropy uses single-action approximation -(p · log p) since
        dist objects are not stored across the rollout. To get exact
        entropy, collect dist.entropy().detach() per step and average.

        Parameters
        ----------
        rewards    : list[float]  — rewards collected during rollout
        log_probs  : list[tensor] — log π(a_t|s_t) from select_action()
        values     : list[tensor] — V(s_t) from select_action()
        next_value : float        — bootstrap V(s_T); 0.0 if terminal
        dones      : list[bool]   — termination flags per step
        """
        # ── discounted returns ────────────────────────────────────────────────
        returns = []
        R       = next_value

        for reward, done in zip(reversed(rewards), reversed(dones)):
            R = reward + self.gamma * R * (1.0 - float(done))
            returns.insert(0, R)

        returns = torch.tensor(returns, dtype=torch.float32).to(self.device)  # (T,)

        # ── stack per-step tensors ────────────────────────────────────────────
        # squeeze() → (T,) regardless of whether value came out (1,1) or (1,)
        log_probs_t = torch.stack(log_probs).squeeze()   # (T,)
        values_t    = torch.stack(values).squeeze()      # (T,)

        # ── advantages ────────────────────────────────────────────────────────
        advantages = returns - values_t.detach()

        if advantages.numel() > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # ── losses ────────────────────────────────────────────────────────────
        actor_loss  = -(log_probs_t * advantages).mean()
        critic_loss = (returns - values_t).pow(2).mean()
        entropy     = -(log_probs_t.exp() * log_probs_t).mean()

        loss = actor_loss + self.value_coef * critic_loss - self.entropy_coef * entropy

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
        self.optimizer.step()

    # ── persistence ───────────────────────────────────────────────────────────

    def save(self, path):
        torch.save(self.model.state_dict(), path)
        print(f"Model saved → {path}")

    def load(self, path):
        self.model.load_state_dict(torch.load(path, map_location=self.device))
        print(f"Model loaded ← {path}")