import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque


# ======================
# Actor Network
# ======================
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim),
            nn.Tanh()   # output in [-1, 1]
        )

    def forward(self, x):
        return self.net(x)


# ======================
# Critic Network
# ======================
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


# ======================
# DDPG Agent
# ======================
class DDPGAgent:
    def __init__(self, state_dim, action_dim):

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.actor = Actor(state_dim, action_dim).to(self.device)
        self.actor_target = Actor(state_dim, action_dim).to(self.device)

        self.critic = Critic(state_dim, action_dim).to(self.device)
        self.critic_target = Critic(state_dim, action_dim).to(self.device)

        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.actor_opt = optim.Adam(self.actor.parameters(), lr=1e-4)
        self.critic_opt = optim.Adam(self.critic.parameters(), lr=1e-3)

        self.memory = deque(maxlen=50000)

        self.batch_size = 64
        self.gamma = 0.99
        self.tau = 0.005

        # ✅ exploration noise
        self.noise_std = 0.2
        self.noise_decay = 0.995
        self.noise_min = 0.05

        self.action_dim = action_dim

    # ======================
    # Select Action
    # ======================
    def select_action(self, state, explore=True):

        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)

        with torch.no_grad():
            action = self.actor(state).cpu().numpy()[0]

        if explore:
            noise = np.random.normal(0, self.noise_std, size=self.action_dim)
            action = action + noise

        return np.clip(action, -1, 1)

    # ======================
    # Store
    # ======================
    def store(self, transition):
        self.memory.append(transition)

    # ======================
    # Train
    # ======================
    def train(self):

        if len(self.memory) < self.batch_size:
            return

        batch = random.sample(self.memory, self.batch_size)

        states, actions, rewards, next_states, dones = zip(*batch)

        # ✅ FAST conversion (fix warning)
        states = torch.FloatTensor(np.array(states)).to(self.device)
        actions = torch.FloatTensor(np.array(actions)).to(self.device)
        rewards = torch.FloatTensor(np.array(rewards)).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(np.array(next_states)).to(self.device)
        dones = torch.FloatTensor(np.array(dones)).unsqueeze(1).to(self.device)

        # ======================
        # Critic update
        # ======================
        with torch.no_grad():

            next_actions = self.actor_target(next_states)

            # ✅ target policy smoothing (TD3 trick)
            noise = torch.clamp(torch.randn_like(next_actions) * 0.2, -0.5, 0.5)
            next_actions = torch.clamp(next_actions + noise, -1, 1)

            target_q = self.critic_target(next_states, next_actions)
            target = rewards + (1 - dones) * self.gamma * target_q

        current_q = self.critic(states, actions)

        critic_loss = nn.MSELoss()(current_q, target)

        self.critic_opt.zero_grad()
        critic_loss.backward()
        self.critic_opt.step()

        # ======================
        # Actor update (delayed style)
        # ======================
        actor_loss = -self.critic(states, self.actor(states)).mean()

        self.actor_opt.zero_grad()
        actor_loss.backward()
        self.actor_opt.step()

        # ======================
        # Soft update
        # ======================
        for target_param, param in zip(self.actor_target.parameters(), self.actor.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        # ✅ decay noise
        self.noise_std = max(self.noise_min, self.noise_std * self.noise_decay)