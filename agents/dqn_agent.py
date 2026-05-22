import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque


# Q-Network
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

        self.epsilon = 1.0
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.9995  # ✅ slower decay

        self.memory = deque(maxlen=10000)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.q_net = DQN(state_dim, action_dim).to(self.device)
        self.target_net = DQN(state_dim, action_dim).to(self.device)
        self.target_net.load_state_dict(self.q_net.state_dict())

        self.optimizer = optim.Adam(self.q_net.parameters(), lr=self.lr)

        self.update_target_every = 100
        self.step_count = 0

    def act(self, state):
        if np.random.rand() < self.epsilon:
            return random.randrange(self.action_dim)

        state = torch.from_numpy(np.array(state)).float().unsqueeze(0).to(self.device)

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

        # ✅ Fast tensor conversion
        states = torch.from_numpy(np.array(states)).float().to(self.device)
        actions = torch.from_numpy(np.array(actions)).long().unsqueeze(1).to(self.device)
        rewards = torch.from_numpy(np.array(rewards)).float().unsqueeze(1).to(self.device)
        next_states = torch.from_numpy(np.array(next_states)).float().to(self.device)
        dones = torch.from_numpy(np.array(dones)).float().unsqueeze(1).to(self.device)

        # states = torch.tensor(states, dtype=torch.float32).to(self.device)
        # actions = torch.tensor(actions, dtype=torch.long).unsqueeze(1).to(self.device)
        # rewards = torch.tensor(rewards, dtype=torch.float32).unsqueeze(1).to(self.device)
        # next_states = torch.tensor(next_states, dtype=torch.float32).to(self.device)
        # dones = torch.tensor(dones, dtype=torch.float32).unsqueeze(1).to(self.device)
        
        


        # Q(s,a)
        q_values = self.q_net(states).gather(1, actions)

        # ✅ Double DQN target
        with torch.no_grad():
            next_actions = self.q_net(next_states).argmax(1, keepdim=True)
            next_q = self.target_net(next_states).gather(1, next_actions)
            target = rewards + (1 - dones) * self.gamma * next_q

        # Loss
        loss = nn.MSELoss()(q_values, target)

        self.optimizer.zero_grad()
        loss.backward()

        # ✅ Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), 1.0)

        self.optimizer.step()

        # ✅ Stable epsilon decay
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

        # Update target network
        self.step_count += 1
        if self.step_count % self.update_target_every == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())