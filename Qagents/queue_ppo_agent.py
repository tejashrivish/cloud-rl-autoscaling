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
# PPO Agent
# ─────────────────────────────────────────

class PPOAgent:
    """
    Proximal Policy Optimization (PPO) — clip variant.

    Key design decisions
    --------------------
    - Rollout buffer collects one full episode, then update() runs
      `ppo_epochs` passes over the buffer in random mini-batches.
      This is the standard PPO loop: collect → update → clear.
    - Clipped surrogate objective prevents large destructive policy
      updates: L = min(r·A, clip(r, 1-ε, 1+ε)·A).
    - Advantage normalised per update call — keeps gradient magnitudes
      comparable across episodes with different workload scales (DR).
    - GAE (Generalised Advantage Estimation) combines TD and MC returns:
      λ=1.0 → pure MC (low bias, high variance)
      λ=0.0 → pure TD (high bias, low variance)
      λ=0.95 → standard PPO default.
    - Entropy bonus prevents premature policy collapse.
    - act() returns (action, log_prob, value) — everything needed to
      fill the rollout buffer in one forward pass.
    - get_action() is a separate greedy path for evaluation.
    """

    def __init__(
        self,
        state_dim,
        action_dim,
        lr           = 3e-4,
        gamma        = 0.99,
        gae_lambda   = 0.95,
        clip_eps     = 0.2,    # PPO clip range
        ppo_epochs   = 4,      # gradient passes over each rollout
        batch_size   = 64,
        value_coef   = 0.5,
        entropy_coef = 0.01,
    ):
        self.gamma        = gamma
        self.gae_lambda   = gae_lambda
        self.clip_eps     = clip_eps
        self.ppo_epochs   = ppo_epochs
        self.batch_size   = batch_size
        self.value_coef   = value_coef
        self.entropy_coef = entropy_coef

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model     = ActorCritic(state_dim, action_dim).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)

        # rollout buffer — cleared after every update()
        self.buffer = []

        print(f"PPOAgent | device={self.device} | state_dim={state_dim} | action_dim={action_dim}")

    # ── helpers ───────────────────────────────────────────────────────────────

    def _to_tensor(self, x, dtype=torch.float32):
        return torch.tensor(np.array(x), dtype=dtype).to(self.device)

    # ── action selection ──────────────────────────────────────────────────────

    def act(self, state):
        """
        Sample from π(·|s).
        Returns (action, log_prob, value) for the rollout buffer.
        """
        with torch.no_grad():
            logits, value = self.model(
                self._to_tensor(state).unsqueeze(0)
            )
            dist     = Categorical(logits=logits)
            action   = dist.sample()
            log_prob = dist.log_prob(action)

        return action.item(), log_prob.item(), value.item()

    def get_action(self, state):
        """Greedy argmax of π for evaluation. No graph, plain int."""
        with torch.no_grad():
            logits, _ = self.model(self._to_tensor(state).unsqueeze(0))
            return int(logits.argmax(dim=1).item())

    # ── buffer ────────────────────────────────────────────────────────────────

    def store(self, transition):
        """
        Store one transition.
        transition = (state, action, log_prob, reward, done, value)
        """
        self.buffer.append(transition)

    def _clear_buffer(self):
        self.buffer = []

    # ── GAE advantage computation ─────────────────────────────────────────────

    def _compute_gae(self, rewards, values, dones, next_value):
        """
        Generalised Advantage Estimation.

        A_t = δ_t + (γλ)·δ_{t+1} + (γλ)²·δ_{t+2} + ...
        δ_t = r_t + γ·V(s_{t+1})·(1-done_t) - V(s_t)

        Returns advantages (T,) and discounted returns (T,).
        """
        T          = len(rewards)
        advantages = np.zeros(T, dtype=np.float32)
        gae        = 0.0

        for t in reversed(range(T)):
            mask       = 1.0 - float(dones[t])
            next_val   = next_value if t == T - 1 else values[t + 1]
            delta      = rewards[t] + self.gamma * next_val * mask - values[t]
            gae        = delta + self.gamma * self.gae_lambda * mask * gae
            advantages[t] = gae

        returns = advantages + np.array(values, dtype=np.float32)
        return advantages, returns

    # ── PPO update ────────────────────────────────────────────────────────────

    def update(self):
        """
        Run ppo_epochs passes of clipped PPO over the rollout buffer.

        Each pass shuffles the buffer into mini-batches and computes:
            L_clip   = clipped surrogate objective
            L_critic = value function MSE
            L_entropy= entropy bonus
            L        = -L_clip + value_coef·L_critic - entropy_coef·L_entropy
        """
        if not self.buffer:
            return

        states, actions, old_log_probs, rewards, dones, values = zip(*self.buffer)

        states       = np.array(states,       dtype=np.float32)
        actions      = np.array(actions,      dtype=np.int64)
        old_log_probs= np.array(old_log_probs,dtype=np.float32)
        rewards      = list(rewards)
        dones        = list(dones)
        values       = list(values)

        # bootstrap from last value if episode didn't terminate naturally
        next_value = 0.0 if dones[-1] else values[-1]

        advantages, returns = self._compute_gae(rewards, values, dones, next_value)

        # normalise advantages over the whole rollout
        if len(advantages) > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # convert to tensors
        s_t   = self._to_tensor(states)
        a_t   = self._to_tensor(actions,       torch.long)
        lp_t  = self._to_tensor(old_log_probs)
        adv_t = self._to_tensor(advantages)
        ret_t = self._to_tensor(returns)

        T = len(states)

        for _ in range(self.ppo_epochs):
            # shuffle indices each epoch
            idx = torch.randperm(T)

            for start in range(0, T, self.batch_size):
                mb = idx[start:start + self.batch_size]

                logits, values_pred = self.model(s_t[mb])
                dist        = Categorical(logits=logits)
                new_log_prob= dist.log_prob(a_t[mb])
                entropy     = dist.entropy().mean()

                # importance ratio
                ratio = (new_log_prob - lp_t[mb]).exp()

                # clipped surrogate
                adv_mb      = adv_t[mb]
                surr1       = ratio * adv_mb
                surr2       = ratio.clamp(1 - self.clip_eps, 1 + self.clip_eps) * adv_mb
                actor_loss  = -torch.min(surr1, surr2).mean()

                critic_loss = (ret_t[mb] - values_pred.squeeze()).pow(2).mean()

                loss = actor_loss + self.value_coef * critic_loss - self.entropy_coef * entropy

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
                self.optimizer.step()

        self._clear_buffer()

    # ── persistence ───────────────────────────────────────────────────────────

    def save(self, path):
        torch.save(self.model.state_dict(), path)
        print(f"Model saved → {path}")

    def load(self, path):
        self.model.load_state_dict(torch.load(path, map_location=self.device))
        print(f"Model loaded ← {path}")