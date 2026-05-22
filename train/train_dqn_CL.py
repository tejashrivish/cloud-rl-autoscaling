import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch

from environment.cloud_env import CloudAutoScalingEnv
from agents.dqn_CL_agent import DQNAgent
import config


# ─────────────────────────────────────────
# Load dataset
# ─────────────────────────────────────────

df       = pd.read_csv(config.DATA_PATH)
workload = df["requests"].values.astype(np.float32)

split       = int(config.TRAIN_SPLIT * len(workload))
train_w     = workload[:split]
test_w      = workload[split:]


# ─────────────────────────────────────────
# Environments
# ─────────────────────────────────────────

train_env = CloudAutoScalingEnv(train_w)
test_env  = CloudAutoScalingEnv(test_w)   # created once, reset each episode

# use env properties — not observation_space / action_space
state_dim  =  4
action_dim =  3


# ─────────────────────────────────────────
# Agent
# ─────────────────────────────────────────

agent = DQNAgent(
    state_dim     = state_dim,
    action_dim    = action_dim,
    lr            = 1e-3,
    gamma         = 0.99,
    epsilon       = 1.0,
    epsilon_min   = 0.05,
    epsilon_decay = 0.995,
    batch_size    = 64,
    buffer_size   = 50_000,
    target_update = 10,
)


# ─────────────────────────────────────────
# Evaluation
# ─────────────────────────────────────────

def evaluate(env, agent):
    """One greedy episode on the given environment."""
    state, _ = env.reset()
    total    = 0.0
    done     = False
    while not done:
        action             = agent.get_action(state)   # greedy, no epsilon
        state, reward, done, _, _ = env.step(action)
        total += reward
    return total


# ─────────────────────────────────────────
# Training loop
# ─────────────────────────────────────────

EPISODES      = 200
train_rewards = []
test_rewards  = []
losses        = []

for ep in range(EPISODES):

    state, _ = train_env.reset()
    total     = 0.0
    ep_losses = []
    done      = False

    while not done:
        action                         = agent.act(state)
        next_state, reward, done, _, _ = train_env.step(action)

        # push — explicit args, done cast to float
        agent.push(state, action, reward, next_state, float(done))

        loss = agent.train()
        if loss is not None:
            ep_losses.append(loss)

        state  = next_state
        total += reward

    # ── end of episode ────────────────────
    # epsilon decay + target sync — called once per episode, not per step
    agent.end_episode()

    avg_loss = np.mean(ep_losses) if ep_losses else 0.0
    train_rewards.append(total)
    test_rewards.append(evaluate(test_env, agent))
    losses.append(avg_loss)

    print(
        f"Ep {ep+1:3d} | "
        f"train={total:9.1f} | "
        f"test={test_rewards[-1]:9.1f} | "
        f"loss={avg_loss:.4f} | "
        f"ε={agent.epsilon:.3f}"
    )


# ─────────────────────────────────────────
# Save model
# ─────────────────────────────────────────

os.makedirs("results", exist_ok=True)
agent.save("results/dqn_model_Cl.pth")


# ─────────────────────────────────────────
# Plot
# ─────────────────────────────────────────

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# rewards
axes[0].plot(train_rewards, alpha=0.4, label="Train")
axes[0].plot(test_rewards,  alpha=0.8, label="Test")

if len(train_rewards) >= 10:
    smooth = np.convolve(train_rewards, np.ones(10) / 10, mode="valid")
    axes[0].plot(range(9, len(train_rewards)), smooth, linewidth=2, label="Train smoothed")

if len(test_rewards) >= 10:
    smooth = np.convolve(test_rewards, np.ones(10) / 10, mode="valid")
    axes[0].plot(range(9, len(test_rewards)), smooth, linewidth=2, label="Test smoothed")

axes[0].set_xlabel("Episode")
axes[0].set_ylabel("Total reward")
axes[0].set_title("DQN  CL— Train vs Test rewards")
axes[0].legend()

# loss
axes[1].plot(losses, color="orange", alpha=0.8)
axes[1].set_xlabel("Episode")
axes[1].set_ylabel("HuberLoss")
axes[1].set_title("DQN — Training loss")

plt.tight_layout()
plt.savefig("results/dqn_train_test_rewards_CL.png", dpi=150)
plt.close()

print("\n✅ Training complete")
print("📈 Plot  → results/dqn_train_test_rewards.png")
print("💾 Model → results/dqn_model_CL.pth")


# ─────────────────────────────────────────
# Summary
# ─────────────────────────────────────────

last_n    = 20
avg_train = np.mean(train_rewards[-last_n:])
avg_test  = np.mean(test_rewards[-last_n:])
gap       = avg_train - avg_test

print(f"\n{'─'*50}")
print(f"Final {last_n}-episode average")
print(f"  Train : {avg_train:>10.1f}")
print(f"  Test  : {avg_test:>10.1f}")
print(f"  Gap   : {gap:>10.1f}  (lower = better generalisation)")
print(f"{'─'*50}")