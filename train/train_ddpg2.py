import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch
from environment.continuous_action_env import CloudAutoScalingEnvContinuous
from agents.ddpg2_agent import DDPGAgent
import config


# ======================
# Load workload
# ======================
df = pd.read_csv(config.DATA_PATH)
workload = df["requests"].values.astype(np.float32)


# ======================
# Env
# ======================
env = CloudAutoScalingEnvContinuous(workload)

state_dim = env.observation_space.shape[0]

# ✅ IMPORTANT: use >1 for better learning
action_dim = 1   # keep 1 if env expects scalar


agent = DDPGAgent(state_dim, action_dim)

episodes = 200
rewards = []


# ======================
# Training loop
# ======================
for ep in range(episodes):

    state, _ = env.reset()
    total_reward = 0

    done = False

    while not done:

        action = agent.select_action(state)

        next_state, reward, done, _, _ = env.step(action)

        agent.store((
            np.array(state, dtype=np.float32),
            np.array(action, dtype=np.float32),
            reward,
            np.array(next_state, dtype=np.float32),
            done
        ))

        agent.train()

        state = next_state
        total_reward += reward

    rewards.append(total_reward)

    print(f"Episode {ep+1}: Reward = {total_reward:.2f}, Noise = {agent.noise_std:.3f}")


# ======================
# Save results
# ======================
os.makedirs("results", exist_ok=True)

smoothed = np.convolve(rewards, np.ones(10)/10, mode='valid')

plt.figure()
plt.plot(rewards, alpha=0.3, label="Raw")
plt.plot(smoothed, label="Smoothed")
plt.xlabel("Episode")
plt.ylabel("Reward")
plt.title("DDPG Training")
plt.legend()
plt.savefig("results/ddpg_rewards.png")
plt.close()

torch.save(agent.actor.state_dict(), "results/ddpg_actor.pth")

print("✅ DDPG training complete")