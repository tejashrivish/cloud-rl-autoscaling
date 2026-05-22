import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch
import config

from environment.cloud_env import CloudAutoScalingEnv
from agents.a2c_agent import A2CAgent


# Load workload


df = pd.read_csv(config.DATA_PATH)
workload = df["requests"].values.astype(np.float32)

split = int(config.TRAIN_SPLIT * len(workload))
train_w = workload[:split]
split = int(0.8 * len(workload))


test_workload = workload[split:]

env = CloudAutoScalingEnv(train_w)

state_dim = 4
action_dim = 3

agent = A2CAgent(state_dim, action_dim)
def evaluate(env, agent):

    state, _ = env.reset()

    total_reward = 0
    done = False

    while not done:

        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(agent.device)

        with torch.no_grad():
            probs, _ = agent.model(state_tensor)

            action = torch.multinomial(probs, 1).item()

        state, reward, done, _, _ = env.step(action)

        total_reward += reward

    return total_reward

episodes = 200
rewards = []
test_rewards = []

for ep in range(episodes):
    state, _ = env.reset()
    total_reward = 0
    done = False

    while not done:
        action, log_prob, value = agent.select_action(state)

        next_state, reward, done, _, _ = env.step(action)

        next_state_tensor = torch.FloatTensor(next_state).unsqueeze(0).to(agent.device)
        _, next_value = agent.model(next_state_tensor)

        agent.update(log_prob, value, reward, next_value, done)

        state = next_state
        total_reward += reward

    rewards.append(total_reward)
    test_env = CloudAutoScalingEnv(test_workload)
    test_reward = evaluate(test_env, agent)
    test_rewards.append(test_reward)

    print(f"Ep {ep+1}: Train = {total_reward:.2f}, Test = {test_reward:.2f}")


# Save results
os.makedirs("results", exist_ok=True)

# Plot
plt.plot(rewards,label="Train")
plt.plot(test_rewards, label="Test (unseen real)")
plt.xlabel("Episode")
plt.ylabel("Reward")
plt.title("A2C Training")
plt.savefig("results/a2c_rewards.png")
plt.close()

# Save model
torch.save(agent.model.state_dict(), "results/a2c_model.pth")

print("✅ A2C Training complete")