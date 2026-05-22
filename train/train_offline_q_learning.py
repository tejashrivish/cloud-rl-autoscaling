import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from agents.q_learning import QLearningAgent
from environment.cloud_env import CloudAutoScalingEnv
from data.offline_dataset import create_offline_dataset
import config

# -----------------------------
# Load workload
# -----------------------------
df = pd.read_csv(config.DATA_PATH)
workload = df["requests"].values.astype(np.float32)

env = CloudAutoScalingEnv(workload)

# -----------------------------
# Create offline dataset
# -----------------------------
dataset = create_offline_dataset(config.DATA_PATH)

# -----------------------------
# Agent
# -----------------------------
state_bins = [10, 10, 10, 10]   # 4D state
action_dim = 3

agent = QLearningAgent(state_bins, action_dim)

# -----------------------------
# Evaluation function
# -----------------------------
def evaluate_policy(env, agent):

    state, _ = env.reset()
    total_reward = 0
    done = False

    while not done:
        action = agent.act(state)   # greedy (no exploration)
        state, reward, done, _, _ = env.step(action)
        total_reward += reward

    return total_reward


# -----------------------------
# Training
# -----------------------------
episodes = 50

train_rewards = []
eval_rewards = []

for ep in range(episodes):

    total_train_reward = 0

    # -------- OFFLINE TRAINING --------
    for (state, action, reward, next_state, done) in dataset:
        agent.update(state, action, reward, next_state)
        total_train_reward += reward

    train_rewards.append(total_train_reward)

    # -------- POLICY EVALUATION --------
    eval_reward = evaluate_policy(env, agent)
    eval_rewards.append(eval_reward)

    print(f"Episode {ep+1}: Train Reward = {total_train_reward:.2f}, Eval Reward = {eval_reward:.2f}")


# -----------------------------
# Save results
# -----------------------------
os.makedirs("results", exist_ok=True)
os.makedirs("models", exist_ok=True)

# Plot rewards
plt.figure()
plt.plot(eval_rewards, label="Evaluation Reward")
plt.xlabel("Episode")
plt.ylabel("Reward")
plt.title("Offline Q-Learning Performance")
plt.legend()
plt.savefig("results/offline_q_learning_rewards.png")
plt.close()

np.save(config.MODELS_DIR + "q_table.npy", agent.q_table)

print("✅ Offline training complete. Results saved in results/ and models/")