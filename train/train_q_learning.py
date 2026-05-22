import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from environment.cloud_env import CloudAutoScalingEnv

from agents.q_learning import QLearningAgent
import config


df = pd.read_csv(config.DATA_PATH)
workload = df["requests"].values.astype(np.float32)

split = int(config.TRAIN_SPLIT * len(workload))
train_w = workload[:split]
split = int(0.8 * len(workload))


test_workload = workload[split:]

env = CloudAutoScalingEnv(train_w)

agent = QLearningAgent(state_bins=[10, 10, 10, 10], action_dim=3)

def evaluate(env, agent):

    state, _ = env.reset()
    total_reward = 0
    done = False

    while not done:
        action = agent.get_action(state)   # greedy
        state, reward, done, _, _ = env.step(action)
        total_reward += reward

    return total_reward


rewards = []
test_rewards = []

for ep in range(config.EPISODES):

    state, _ = env.reset()
    total_reward = 0

    done = False

    while not done:
        action = agent.act(state)

        next_state, reward, done, _, _ = env.step(action)

        agent.update(state, action, reward, next_state)

        state = next_state
        total_reward += reward

    rewards.append(total_reward)
    test_env = CloudAutoScalingEnv(test_workload)
    test_reward = evaluate(test_env, agent)
    test_rewards.append(test_reward)

    print(f"Ep {ep+1}: Train = {total_reward:.2f}, Test = {test_reward:.2f}")

   

# # create folders
# os.makedirs(config.RESULTS_DIR + "q_learning/", exist_ok=True)
# os.makedirs(config.MODELS_DIR, exist_ok=True)

# # save plot
# plt.plot(rewards)
# plt.savefig(config.RESULTS_DIR + "q_learning/rewards.png")
# plt.close()

# # save Q-table
# np.save(config.MODELS_DIR + "q_table.npy", agent.q_table)


# -----------------------------
# Save results
# -----------------------------
os.makedirs("results", exist_ok=True)
os.makedirs("models", exist_ok=True)

# Plot
plt.figure()
plt.plot(rewards, label="Train")
plt.plot(test_rewards, label="Test (unseen real)")
plt.xlabel("Episode")
plt.ylabel("Reward")
plt.title("Q-Learning")
plt.legend()
plt.savefig("results/q_learning.png")
plt.close()

# Save Q-table
np.save("models/q_table.npy", agent.q_table)

