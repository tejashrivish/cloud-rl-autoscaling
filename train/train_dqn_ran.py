import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch

from environment.cloud_env import CloudAutoScalingEnv
from agents.dqn_agent import DQNAgent
import config


# =========================================
# Domain Randomization
# =========================================
def randomize_workload(
    base_workload,
    scale_range=(0.8, 1.2),
    noise_std=0.05,
    shift=True,
    spike_prob=0.02,
    spike_magnitude=(1.5, 3.0),
    shuffle_segments=False,
    segment_len=24,
    min_floor=0.05,
):

    w = base_workload.copy().astype(np.float64)

    n = len(w)

    # ---------------------------------
    # 1. Amplitude Scaling
    # ---------------------------------
    scale = np.random.uniform(*scale_range)
    w = w * scale

    # ---------------------------------
    # 2. Proportional Noise
    # ---------------------------------
    noise = np.random.normal(0, noise_std, size=n)

    noise = np.clip(noise, -0.15, 0.15)

    w = w * (1 + noise)

    # ---------------------------------
    # 3. Spike Injection
    # ---------------------------------
    spike_mask = np.random.rand(n) < spike_prob

    spike_factors = np.random.uniform(
        *spike_magnitude,
        size=n
    )

    w = np.where(
        spike_mask,
        w * spike_factors,
        w
    )

    # ---------------------------------
    # 4. Temporal Shift
    # ---------------------------------
    if shift:

        roll_by = np.random.randint(0, n)

        w = np.roll(w, roll_by)

    # ---------------------------------
    # 5. Segment Shuffle
    # ---------------------------------
    if shuffle_segments:

        n_segs = n // segment_len

        segs = [
            w[i * segment_len:(i + 1) * segment_len]
            for i in range(n_segs)
        ]

        remainder = w[n_segs * segment_len:]

        np.random.shuffle(segs)

        w = np.concatenate(segs + [remainder])

    # ---------------------------------
    # 6. Floor Clamp
    # ---------------------------------
    floor = min_floor * np.mean(base_workload)

    w = np.clip(w, floor, None)

    return w.astype(np.float32)


# =========================================
# Load Dataset
# =========================================
df = pd.read_csv(config.DATA_PATH)

workload = df["requests"].values.astype(np.float32)


# =========================================
# Train/Test Split
# =========================================
split = int(config.TRAIN_SPLIT * len(workload))

train_workload = workload[:split]

test_workload = workload[split:]


# =========================================
# Environment Setup
# =========================================
dummy_env = CloudAutoScalingEnv(train_workload)

state_dim = dummy_env.observation_space.shape[0]

action_dim = dummy_env.action_space.n


# =========================================
# Agent
# =========================================
agent = DQNAgent(state_dim, action_dim)


# =========================================
# Evaluation Function
# =========================================
def evaluate(env, agent):

    state, _ = env.reset()

    total_reward = 0

    done = False

    while not done:

        # Move tensor to correct device
        state_tensor = (
            torch.FloatTensor(state)
            .unsqueeze(0)
            .to(agent.device)
        )

        with torch.no_grad():

            q_values = agent.q_net(state_tensor)

        action = torch.argmax(q_values).item()

        next_state, reward, done, _, _ = env.step(action)

        state = next_state

        total_reward += reward

    return total_reward


# =========================================
# Training
# =========================================
episodes = 200

train_rewards = []

test_rewards = []


for ep in range(episodes):

    # ---------------------------------
    # Randomize workload every episode
    # ---------------------------------
    randomized_workload = randomize_workload(
        train_workload
    )

    train_env = CloudAutoScalingEnv(
        randomized_workload
    )

    state, _ = train_env.reset()

    total_reward = 0

    done = False

    while not done:

        # ε-greedy action
        action = agent.act(state)

        next_state, reward, done, _, _ = train_env.step(action)

        # Store transition
        agent.store((
            np.array(state, dtype=np.float32),
            action,
            reward,
            np.array(next_state, dtype=np.float32),
            done
        ))

        # Train DQN
        agent.train()

        state = next_state

        total_reward += reward

    # ---------------------------------
    # Testing on unseen real workload
    # ---------------------------------
    test_env = CloudAutoScalingEnv(
        test_workload
    )

    test_reward = evaluate(
        test_env,
        agent
    )

    train_rewards.append(total_reward)

    test_rewards.append(test_reward)

    print(
        f"Episode {ep+1}: "
        f"Train = {total_reward:.2f}, "
        f"Test = {test_reward:.2f}, "
        f"Epsilon = {agent.epsilon:.3f}"
    )


# =========================================
# Create Folders
# =========================================
os.makedirs("results", exist_ok=True)

os.makedirs("models", exist_ok=True)


# =========================================
# Plot Rewards
# =========================================
plt.figure(figsize=(10, 5))

# Raw curves
plt.plot(
    train_rewards,
    alpha=0.4,
    label="Train (Randomized)"
)

plt.plot(
    test_rewards,
    alpha=0.8,
    label="Test (Real)"
)

# Smoothed train curve
if len(train_rewards) >= 10:

    smooth_train = np.convolve(
        train_rewards,
        np.ones(10) / 10,
        mode='valid'
    )

    plt.plot(
        range(9, len(train_rewards)),
        smooth_train,
        linewidth=2,
        label="Smoothed Train"
    )

# Smoothed test curve
if len(test_rewards) >= 10:

    smooth_test = np.convolve(
        test_rewards,
        np.ones(10) / 10,
        mode='valid'
    )

    plt.plot(
        range(9, len(test_rewards)),
        smooth_test,
        linewidth=2,
        label="Smoothed Test"
    )

plt.xlabel("Episode")

plt.ylabel("Reward")

plt.title("DQN with Domain Randomization")

plt.legend()

plt.grid(True)

plt.savefig(
    "results/dqn_generalization.png"
)

plt.close()


# =========================================
# Save Model
# =========================================
torch.save(
    agent.q_net.state_dict(),
    "models/dqn_generalized.pth"
)


print("\n✅ DQN Generalization Training Complete")

print("📈 Plot saved: results/dqn_generalization.png")

print("💾 Model saved: models/dqn_generalized.pth")