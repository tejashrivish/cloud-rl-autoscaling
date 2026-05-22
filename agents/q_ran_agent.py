import numpy as np
import random


class QLearningAgent:
    def __init__(self, state_bins, action_dim):

        self.state_bins = state_bins
        self.action_dim = action_dim

        # 🔥 FIXED Q-table init
        self.q_table = np.zeros(tuple(state_bins) + (action_dim,))

        # hyperparameters
        self.lr = 0.1
        self.gamma = 0.99

        self.epsilon = 1.0
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.995

    # -----------------------------
    # Discretization
    # -----------------------------
    def discretize(self, state):

        indices = []

        for s, b in zip(state, self.state_bins):
            idx = int(s * (b - 1))   # 🔥 FIXED scaling
            idx = np.clip(idx, 0, b - 1)
            indices.append(idx)

        return tuple(indices)

    # -----------------------------
    # Training action (ε-greedy)
    # -----------------------------
    def act(self, state):

        s = self.discretize(state)

        if random.random() < self.epsilon:
            return random.randrange(self.action_dim)

        return np.argmax(self.q_table[s])

    # -----------------------------
    # Testing action (greedy)
    # -----------------------------
    def get_action(self, state):

        s = self.discretize(state)
        return np.argmax(self.q_table[s])

    # -----------------------------
    # Q-learning update
    # -----------------------------
    def update(self, state, action, reward, next_state, done):

        s = self.discretize(state)
        ns = self.discretize(next_state)

        best_next = np.max(self.q_table[ns])

        target = reward + (1 - done) * self.gamma * best_next

        self.q_table[s][action] += self.lr * (
            target - self.q_table[s][action]
        )

    # -----------------------------
    # Epsilon decay
    # -----------------------------
    def decay_epsilon(self):

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay