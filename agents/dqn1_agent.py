import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque


class DQN(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(DQN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim)
        )

    def forward(self, x):
        return self.net(x)


class DQNAgent:
    def __init__(self, state_dim, action_dim):

        self.state_dim = state_dim
        self.action_dim = action_dim

        self.gamma = 0.99
        self.lr = 1e-3
        self.batch_size = 64

        # standard epsilon
        self.epsilon = 1.0
        self.epsilon_min = 0.1
        self.epsilon_decay = 0.995

        self.memory = deque(maxlen=5000)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.q_net = DQN(state_dim, action_dim).to(self.device)
        self.target_net = DQN(state_dim, action_dim).to(self.device)
        self.target_net.load_state_dict(self.q_net.state_dict())

        self.optimizer = optim.Adam(self.q_net.parameters(), lr=self.lr)
        self.loss_fn = nn.MSELoss()

        self.target_update_freq = 100

        self.step_count = 0

    def act(self, state):
        if np.random.rand() < self.epsilon:
            return random.randrange(self.action_dim)

        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)

        with torch.no_grad():
            q_values = self.q_net(state)

        return torch.argmax(q_values).item()

    def store(self, transition):
        self.memory.append(transition)

    def train(self):
        if len(self.memory) < self.batch_size:
            return

        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        # states = torch.FloatTensor(states).to(self.device)
        # actions = torch.LongTensor(actions).unsqueeze(1).to(self.device)
        # rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        # next_states = torch.FloatTensor(next_states).to(self.device)
        # dones = torch.FloatTensor(dones).unsqueeze(1).to(self.device)
        states = torch.from_numpy(np.array(states)).float().to(self.device)
        actions = torch.from_numpy(np.array(actions)).long().unsqueeze(1).to(self.device)
        rewards = torch.from_numpy(np.array(rewards)).float().unsqueeze(1).to(self.device)
        next_states = torch.from_numpy(np.array(next_states)).float().to(self.device)
        dones = torch.from_numpy(np.array(dones)).float().unsqueeze(1).to(self.device)

        # Q(s,a)
        q_values = self.q_net(states).gather(1, actions)

        # standard DQN target
        with torch.no_grad():
            max_next_q = self.target_net(next_states).max(1, keepdim=True)[0]
            target = rewards + (1 - dones) * self.gamma * max_next_q

        loss = self.loss_fn(q_values, target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # epsilon decay
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        # update target network
        self.step_count += 1
        if self.step_count % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())