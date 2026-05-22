import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch

from environment.cloud_env import CloudAutoScalingEnv
from agents.ppo_agent import PPOAgent
import config


# =========================================================
# LOAD WORKLOAD
# =========================================================

df = pd.read_csv(config.DATA_PATH)
base_workload = df["requests"].values.astype(np.float32)


# =========================================================
# CURRICULUM RANDOMIZATION (INTEGRATED)
# =========================================================

def randomize_workload(base_workload, seed=None, ep=0, total_eps=300):
    """
    Curriculum domain randomization — stable RL version.
    """

    rng = np.random.default_rng(seed)

    progress = ep / max(total_eps - 1, 1)

    # 1. amplitude scaling
    scale_half = 0.1 + 0.1 * progress
    scale = rng.uniform(1 - scale_half, 1 + scale_half)

    # 2. proportional noise (stable)
    noise_std = 0.03 + 0.02 * progress
    noise = rng.normal(0, noise_std, size=len(base_workload))
    noise = np.clip(noise, -0.15, 0.15)

    w = base_workload * scale + (base_workload * noise)

    # 3. temporal shift
    if progress > 0.3:
        shift = rng.integers(0, len(base_workload))
        w = np.roll(w, shift)

    # 4. floor clamp
    floor = 0.05 * np.mean(base_workload)
    return np.clip(w, floor, None).astype(np.float32)


# =========================================================
# ENV + AGENT
# =========================================================

env = CloudAutoScalingEnv(randomize_workload(base_workload))

state_dim = env.observation_space.shape[0]
action_dim = 3

agent = PPOAgent(state_dim, action_dim)


# =========================================================
# EVALUATION
# =========================================================

def evaluate(env, agent, runs=5):

    rewards = []

    for _ in range(runs):

        state, _ = env.reset()
        total_reward = 0
        done = False

        while not done:

            with torch.no_grad():
                action, _, _ = agent.act(state)

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

test_env = CloudAutoScalingEnv(randomize_workload(base_workload))

for ep in range(episodes):

    # 🔥 CURRICULUM RANDOMIZATION APPLIED HERE
    train_w = randomize_workload(
        base_workload,
        seed=ep,
        ep=ep,
        total_eps=episodes
    )

    env = CloudAutoScalingEnv(train_w)

    state, _ = env.reset()

    total_reward = 0
    done = False

    while not done:

        action, log_prob, value = agent.act(state)

        next_state, reward, done, _, _ = env.step(action)

        agent.store((state, action, log_prob, reward, done, value))

        state = next_state
        total_reward += reward

    agent.update()

    # =====================================================
    # TEST EVALUATION
    # =====================================================

    test_reward = evaluate(test_env, agent, runs=5)

    train_rewards.append(total_reward)
    test_rewards.append(test_reward)

    print(f"Ep {ep+1}: Train = {total_reward:.2f}, Test = {test_reward:.2f}")


# =========================================================
# PLOT RESULTS
# =========================================================

os.makedirs("results", exist_ok=True)

plt.figure()

plt.plot(train_rewards, alpha=0.3, label="Train")
plt.plot(np.convolve(train_rewards, np.ones(10)/10, mode='valid'), label="Train Smooth")
plt.plot(test_rewards, label="Test")

plt.title("PPO + Curriculum Domain Randomization")
plt.legend()

plt.savefig("results/ppo_curriculum_randomized.png")
plt.close()


# =========================================================
# SAVE MODEL
# =========================================================

torch.save(agent.model.state_dict(), "results/ppo_curriculum.pth")

print("✅ PPO training with curriculum randomization complete")