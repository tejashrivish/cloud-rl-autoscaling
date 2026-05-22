import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch

from environment.continuous_action_env import CloudAutoScalingEnvContinuous
from agents.ddpg_agent import DDPGAgent
import config


# =========================================================
# CURRICULUM RANDOMIZATION (same style as PPO/A2C)
# =========================================================

def randomize_workload(base_workload, ep=0, total_eps=200):

    rng = np.random.default_rng(ep)

    progress = ep / max(total_eps - 1, 1)

    scale_half = 0.05 + 0.05 * progress  # 5% → 10%

    scale = rng.uniform(1 - scale_half, 1 + scale_half)

    noise_std = 0.01 + 0.01 * progress
    noise = rng.normal(0, noise_std * np.std(base_workload), size=len(base_workload))

    w = base_workload * scale + noise

    if progress > 0.3:
        w = np.roll(w, rng.integers(0, len(base_workload)))

    floor = 0.05 * np.mean(base_workload)

    return np.clip(w, floor, None).astype(np.float32)


# =========================================================
# LOAD DATA
# =========================================================

df = pd.read_csv(config.DATA_PATH)
base_workload = df["requests"].values.astype(np.float32)


# =========================================================
# ENV + AGENT
# =========================================================

env = CloudAutoScalingEnvContinuous(base_workload)

state_dim = env.observation_space.shape[0]
action_dim = 1

agent = DDPGAgent(state_dim, action_dim)


# =========================================================
# EVALUATION FUNCTION (IMPORTANT FIX)
# =========================================================

def evaluate(env, agent, runs=5):

    rewards = []

    for _ in range(runs):

        state, _ = env.reset()
        total_reward = 0
        done = False

        while not done:

            action = agent.act(state, noise=False)  # deterministic policy

            state, reward, done, _, _ = env.step(action)
            total_reward += reward

        rewards.append(total_reward)

    return np.mean(rewards)


# =========================================================
# TRAINING
# =========================================================

episodes = 200

train_rewards = []
test_rewards = []


for ep in range(episodes):

    # 🔥 curriculum randomization
    workload = randomize_workload(base_workload, ep=ep, total_eps=episodes)

    env = CloudAutoScalingEnvContinuous(workload)

    state, _ = env.reset()
    total_reward = 0
    done = False

    while not done:

        action = agent.act(state)

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

    train_rewards.append(total_reward)

    # =====================================================
    # TEST PHASE (clean environment, no noise)
    # =====================================================

    test_env = CloudAutoScalingEnvContinuous(base_workload)
    test_reward = evaluate(test_env, agent, runs=5)

    test_rewards.append(test_reward)

    print(f"Ep {ep+1}: Train = {total_reward:.2f}, Test = {test_reward:.2f}")


# =========================================================
# PLOT RESULTS
# =========================================================

os.makedirs("results", exist_ok=True)

plt.figure()

plt.plot(train_rewards, alpha=0.3, label="Train")
plt.plot(np.convolve(train_rewards, np.ones(10)/10, mode='valid'), label="Train Smooth")
plt.plot(test_rewards, label="Test (avg 5 runs)")

plt.xlabel("Episode")
plt.ylabel("Reward")
plt.title("DDPG + Curriculum Randomization")

plt.legend()
plt.savefig("results/ddpg_fixed.png")
plt.close()


# =========================================================
# SAVE MODEL
# =========================================================

torch.save(agent.actor.state_dict(), "results/ddpg_actor.pth")

print("✅ DDPG training complete (fixed + test added)")
