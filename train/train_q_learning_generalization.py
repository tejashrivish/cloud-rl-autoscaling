import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from environment.cloud_env import CloudAutoScalingEnv
from agents.q_ran_agent import QLearningAgent
import config


# -----------------------------
# Load dataset
# -----------------------------
df = pd.read_csv(config.DATA_PATH)
workload = df["requests"].values.astype(np.float32)

# -----------------------------
# Train/Test split
# -----------------------------
split = int(0.8 * len(workload))

train_workload = workload[:split]
test_workload = workload[split:]


# -----------------------------
# Domain Randomization
# -----------------------------
# def randomize_workload(base_workload):

#     noise = np.random.normal(0, 0.1, size=len(base_workload))
#     scale = np.random.uniform(0.8, 1.2)

#     new_workload = base_workload * scale + noise * base_workload

#     return np.clip(new_workload, 0, None)
def randomize_workload(
    base_workload,
    scale_range=(0.8, 1.2),
    noise_std=0.05,          # fraction of local value — tighter than before
    shift=True,              # temporal roll
    spike_prob=0.02,         # probability per timestep of injecting a spike
    spike_magnitude=(1.5, 3.0),
    shuffle_segments=False,  # shuffle chunks of length `segment_len`
    segment_len=24,
    min_floor=0.05,          # minimum fraction of mean workload
):
    w = base_workload.copy().astype(np.float64)
    n = len(w)

    # 1. Amplitude scaling
    scale = np.random.uniform(*scale_range)
    w = w * scale

    # 2. Proportional noise (bounded: avoids zero-collapse)
    noise = np.random.normal(0, noise_std, size=n)
    noise = np.clip(noise, -0.15, 0.15)   # cap at ±15% per step
    w = w * (1 + noise)

    # 3. Spike injection
    spike_mask = np.random.rand(n) < spike_prob
    spike_factors = np.random.uniform(*spike_magnitude, size=n)
    w = np.where(spike_mask, w * spike_factors, w)

    # 4. Temporal shift (roll breaks phase memorisation)
    if shift:
        roll_by = np.random.randint(0, n)
        w = np.roll(w, roll_by)

    # 5. Segment shuffle (preserves local autocorrelation)
    if shuffle_segments:
        n_segs = n // segment_len
        segs = [w[i*segment_len:(i+1)*segment_len] for i in range(n_segs)]
        remainder = w[n_segs*segment_len:]
        np.random.shuffle(segs)
        w = np.concatenate(segs + [remainder])

    # 6. Floor clamp — no zero deserts
    floor = min_floor * np.mean(base_workload)
    w = np.clip(w, floor, None)

    return w.astype(np.float32)

# -----------------------------
# Agent
# -----------------------------
state_bins = [10, 10, 10, 10]
action_dim = 3

agent = QLearningAgent(state_bins, action_dim)


# -----------------------------
# Evaluation function
# -----------------------------
def evaluate(env, agent):

    state, _ = env.reset()
    total_reward = 0
    done = False

    while not done:
        action = agent.get_action(state)   # greedy
        state, reward, done, _, _ = env.step(action)
        total_reward += reward

    return total_reward


# -----------------------------
# Training
# -----------------------------
episodes = 200

train_rewards = []
test_rewards = []

for ep in range(episodes):

    # 🔥 Randomized environment every episode
    randomized_workload = randomize_workload(train_workload)
    env = CloudAutoScalingEnv(randomized_workload)

    state, _ = env.reset()
    total_reward = 0
    done = False

    while not done:

        action = agent.act(state)

        next_state, reward, done, _, _ = env.step(action)

        agent.update(state, action, reward, next_state,done)

        state = next_state
        total_reward += reward

    agent.decay_epsilon()

    train_rewards.append(total_reward)

    # -----------------------------
    # Evaluate on REAL unseen data
    # -----------------------------
    test_env = CloudAutoScalingEnv(test_workload)
    test_reward = evaluate(test_env, agent)
    test_rewards.append(test_reward)

    print(f"Ep {ep+1}: Train = {total_reward:.2f}, Test = {test_reward:.2f}")


# -----------------------------
# Save results
# -----------------------------
os.makedirs("results", exist_ok=True)
os.makedirs("models", exist_ok=True)

# Plot
plt.figure()
plt.plot(train_rewards, label="Train (randomized)")
plt.plot(test_rewards, label="Test (unseen real)")
plt.xlabel("Episode")
plt.ylabel("Reward")
plt.title("Q-Learning with Domain Randomization")
plt.legend()
plt.savefig("results/q_learning_generalization.png")
plt.close()

# Save Q-table
np.save("models/q_table_generalized.npy", agent.q_table)

print("✅ Generalization training complete.")