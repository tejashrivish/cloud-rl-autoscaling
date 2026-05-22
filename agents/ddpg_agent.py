import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque


# ========================
# Actor
# ========================
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim),
            nn.Tanh()
        )

    def forward(self, x):
        return self.net(x)


# ========================
# Critic
# ========================
class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, state, action):
        x = torch.cat([state, action], dim=1)
        return self.net(x)


# ========================
# DDPG Agent
# ========================
class DDPGAgent:
    def __init__(self, state_dim, action_dim):

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.actor = Actor(state_dim, action_dim).to(self.device)
        self.actor_target = Actor(state_dim, action_dim).to(self.device)

        self.critic = Critic(state_dim, action_dim).to(self.device)
        self.critic_target = Critic(state_dim, action_dim).to(self.device)

        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_target.load_state_dict(self.critic.state_dict())

        # ✅ Lower LR for stability (no reward scaling)
        self.actor_opt = optim.Adam(self.actor.parameters(), lr=1e-4)
        self.critic_opt = optim.Adam(self.critic.parameters(), lr=1e-4)

        self.memory = deque(maxlen=50000)

        self.batch_size = 64
        self.gamma = 0.99
        self.tau = 0.005

        self.noise_std = 0.1
        self.train_step = 0

    # ========================
    def act(self, state, noise=True):

        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)

        with torch.no_grad():
               action = self.actor(state).cpu().numpy()[0]

    # -------------------------------------------------
    # only add noise during training
    # -------------------------------------------------
        if noise:
               noise_val = np.random.normal(0, self.noise_std, size=action.shape)
               action = action + noise_val

        return np.clip(action, -1, 1)

    # ========================
    def store(self, transition):
        self.memory.append(transition)

    # ========================
    def train(self):
        if len(self.memory) < self.batch_size:
            return

        # train every 2 steps (faster)
        self.train_step += 1
        if self.train_step % 2 != 0:
            return

        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        # ✅ FAST conversion (fixed warning)
        states = torch.from_numpy(np.array(states, dtype=np.float32)).to(self.device)
        actions = torch.from_numpy(np.array(actions, dtype=np.float32)).to(self.device)
        rewards = torch.from_numpy(np.array(rewards, dtype=np.float32)).unsqueeze(1).to(self.device)
        next_states = torch.from_numpy(np.array(next_states, dtype=np.float32)).to(self.device)
        dones = torch.from_numpy(np.array(dones, dtype=np.float32)).unsqueeze(1).to(self.device)

        # ========================
        # Critic update
        # ========================
        with torch.no_grad():
            next_actions = self.actor_target(next_states)
            target_q = self.critic_target(next_states, next_actions)

            target = rewards + (1 - dones) * self.gamma * target_q

            # ✅ clamp to prevent explosion
            target = torch.clamp(target, -1e5, 1e5)

        current_q = self.critic(states, actions)
        critic_loss = nn.MSELoss()(current_q, target)

        self.critic_opt.zero_grad()
        critic_loss.backward()

        # ✅ gradient clipping
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)

        self.critic_opt.step()

        # ========================
        # Actor update
        # ========================
        actor_loss = -self.critic(states, self.actor(states)).mean()

        self.actor_opt.zero_grad()
        actor_loss.backward()
        self.actor_opt.step()

        # ========================
        # Soft update
        # ========================
        for target_param, param in zip(self.actor_target.parameters(), self.actor.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)