import numpy as np
from collections import defaultdict


import numpy as np
import random


class QLearningAgent:
    def __init__(self, state_bins, action_dim):

        self.action_dim = action_dim
        self.q_table = np.zeros(state_bins + [action_dim])

        self.lr = 0.1
        self.gamma = 0.99

        self.epsilon = 1.0
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.995

        self.state_bins = state_bins

    def discretize(self, state):
        return tuple(
            min(int(s * b), b - 1)
            for s, b in zip(state, self.state_bins)
        )

    def act(self, state):
        s = self.discretize(state)

        if random.random() < self.epsilon:
            return random.randrange(self.action_dim)

        return np.argmax(self.q_table[s])
    def get_action(self, state):
          s = self.discretize(state)
          return np.argmax(self.q_table[s])
    def safe_act(self, state, env):

         if np.random.rand() < self.epsilon:
            action = np.random.randint(self.action_dim)
         else:
            action = self.get_action(state)

    # 🔥 pass through safety filter
         w = env.workload[env.t]
         action = env.safe_action(action, w)

         return action

    def update(self, state, action, reward, next_state):
        s = self.discretize(state)
        ns = self.discretize(next_state)

        best_next = np.max(self.q_table[ns])

        self.q_table[s][action] += self.lr * (
            reward + self.gamma * best_next - self.q_table[s][action]
        )

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay