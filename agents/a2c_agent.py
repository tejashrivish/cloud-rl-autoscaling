import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np


class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(ActorCritic, self).__init__()

        self.shared = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU()
        )

        # Actor (policy)
        self.actor = nn.Sequential(
            nn.Linear(64, action_dim),
            nn.Softmax(dim=-1)
        )

        # Critic (value)
        self.critic = nn.Linear(64, 1)

    def forward(self, x):
        x = self.shared(x)
        return self.actor(x), self.critic(x)


class A2CAgent:
    def __init__(self, state_dim, action_dim):

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = ActorCritic(state_dim, action_dim).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-3)

        self.gamma = 0.99

    def select_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)

        probs, value = self.model(state)

        dist = torch.distributions.Categorical(probs)
        action = dist.sample()

        return action.item(), dist.log_prob(action), value

    def update(self, log_prob, value, reward, next_value, done):

        reward = torch.tensor([reward], dtype=torch.float32).to(self.device)

        # TD target
        target = reward + (1 - done) * self.gamma * next_value

        # Advantage
        advantage = target - value

        # Actor loss
        actor_loss = -log_prob * advantage.detach()

        # Critic loss
        critic_loss = advantage.pow(2)

        loss = actor_loss + critic_loss

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()