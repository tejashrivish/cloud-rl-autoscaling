import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class PPOAgent:

    def __init__(self, state_dim, action_dim, lr=3e-4, gamma=0.99, eps_clip=0.2):

        self.gamma = gamma
        self.eps_clip = eps_clip

        # =====================================================
        # CUDA SETUP
        # =====================================================
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"🚀 Using device: {self.device}")

        self.model = ActorCritic(state_dim, action_dim).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

        # rollout buffer (CPU storage → GPU only during update)
        self.states = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.dones = []
        self.values = []

    # =========================================================
    # ACT (CUDA inference)
    # =========================================================

    def act(self, state):

        state = torch.FloatTensor(state).unsqueeze(0).to(self.device, non_blocking=True)

        with torch.no_grad():
            probs, value = self.model(state)

        dist = torch.distributions.Categorical(probs)

        action = dist.sample()
        log_prob = dist.log_prob(action)

        return (
            action.item(),
            log_prob.detach().cpu(),
            value.detach().cpu()
        )

    # =========================================================
    # STORE (CPU ONLY — prevents GPU memory explosion)
    # =========================================================

    def store(self, transition):

        state, action, log_prob, reward, done, value = transition

        self.states.append(np.array(state, dtype=np.float32))
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.rewards.append(float(reward))
        self.dones.append(float(done))
        self.values.append(value)

    # =========================================================
    # RETURNS (GPU COMPUTE)
    # =========================================================

    def compute_returns(self):

        returns = []
        R = 0

        for r, d in zip(reversed(self.rewards), reversed(self.dones)):

            if d:
                R = 0

            R = r + self.gamma * R
            returns.insert(0, R)

        return torch.tensor(returns, dtype=torch.float32, device=self.device)

    # =========================================================
    # UPDATE (CUDA BATCH TRAINING)
    # =========================================================

    def update(self):

        # ---------------------------
        # CPU → GPU transfer once
        # ---------------------------
        states = torch.tensor(np.array(self.states), dtype=torch.float32).to(self.device, non_blocking=True)
        actions = torch.tensor(self.actions, dtype=torch.long).to(self.device)
        old_log_probs = torch.stack(self.log_probs).to(self.device)
        values = torch.stack(self.values).squeeze().to(self.device)

        returns = self.compute_returns()

        # advantages
        advantages = returns - values.detach()
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        n = len(states)
        batch_size = 64

        # =====================================================
        # PPO EPOCHS
        # =====================================================

        for _ in range(4):

            idx = torch.randperm(n, device=self.device)

            for start in range(0, n, batch_size):

                batch_idx = idx[start:start + batch_size]

                b_states = states[batch_idx]
                b_actions = actions[batch_idx]
                b_old_log = old_log_probs[batch_idx]
                b_adv = advantages[batch_idx]
                b_returns = returns[batch_idx]

                probs, state_values = self.model(b_states)

                dist = torch.distributions.Categorical(probs)

                log_probs = dist.log_prob(b_actions)

                ratio = torch.exp(log_probs - b_old_log)

                surr1 = ratio * b_adv
                surr2 = torch.clamp(
                    ratio,
                    1 - self.eps_clip,
                    1 + self.eps_clip
                ) * b_adv

                actor_loss = -torch.min(surr1, surr2).mean()
                critic_loss = F.mse_loss(state_values.squeeze(), b_returns)

                loss = actor_loss + 0.5 * critic_loss

                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()

                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)

                self.optimizer.step()

        # =====================================================
        # CLEANUP (VERY IMPORTANT FOR CUDA)
        # =====================================================

        self.states.clear()
        self.actions.clear()
        self.log_probs.clear()
        self.rewards.clear()
        self.dones.clear()
        self.values.clear()

        torch.cuda.empty_cache()


# =========================================================
# ACTOR-CRITIC NETWORK (CUDA READY)
# =========================================================

class ActorCritic(nn.Module):

    def __init__(self, state_dim, action_dim):

        super().__init__()

        self.shared = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU()
        )

        self.actor = nn.Linear(128, action_dim)
        self.critic = nn.Linear(128, 1)

    def forward(self, x):

        x = self.shared(x)

        logits = self.actor(x)
        value = self.critic(x)

        probs = torch.softmax(logits, dim=-1)

        return probs, value