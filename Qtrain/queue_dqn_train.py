import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from QEnv.queueEnv import CloudAutoScalingEnv
from Qagents.queue_dqn_agent import DQNAgent
import config


# ─────────────────────────────────────────
# Load dataset
# ─────────────────────────────────────────

df       = pd.read_csv(config.DATA_PATH)
workload = df["requests"].values.astype(np.float32)

split   = int(config.TRAIN_SPLIT * len(workload))
train_w = workload[:split]
test_w  = workload[split:]


# ─────────────────────────────────────────
# Domain Randomization
# ─────────────────────────────────────────

def randomize_workload(base_workload, seed=None, ep=0, total_eps=300):
    """
    Curriculum randomization — conservative early, aggressive late.

    seed=ep  →  same run always produces the same episode sequence
                (reproducible), but each episode sees a different
                workload (diverse).

    Transforms
    ----------
    1. Amplitude scaling  — ±10% early, grows to ±20% by final episode
    2. Proportional noise — 3% std early, 5% late  (hard cap ±15%)
    3. Temporal shift     — applied only after 30% of training to avoid
                            disrupting early learning
    4. Floor clamp        — minimum 5% of mean, prevents zero-demand
                            regions that encourage always-scale-down policy
    """
    rng = np.random.default_rng(seed)

    progress   = ep / max(total_eps - 1, 1)    # 0.0 → 1.0
    scale_half = 0.10 + 0.10 * progress        # ±10% → ±20%
    noise_std  = 0.03 + 0.02 * progress        #   3% →   5%

    scale = rng.uniform(1 - scale_half, 1 + scale_half)

    noise = rng.normal(0, noise_std, size=len(base_workload))
    noise = np.clip(noise, -0.15, 0.15)

    w = base_workload * scale * (1 + noise)

    if progress > 0.3:
        w = np.roll(w, rng.integers(0, len(base_workload)))

    floor = 0.05 * np.mean(base_workload)
    return np.clip(w, floor, None).astype(np.float32)


# ─────────────────────────────────────────
# Dimensions — derived from env, not hardcoded
# ─────────────────────────────────────────

_tmp_env   = CloudAutoScalingEnv(train_w)
state_dim  =5 
action_dim =3
del _tmp_env


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
    epsilon_decay = 0.998,      # slower decay — DR + queue needs more exploration
    batch_size    = 64,
    buffer_size   = 50_000,
    target_update = 10,
)


# ─────────────────────────────────────────
# Evaluation
# ─────────────────────────────────────────

def evaluate(agent, test_workload):
    """
    One greedy episode on the real unseen test workload.
    No randomization — this is the true generalisation measure.
    Queue resets to zero at env.reset(), matching training conditions.
    """
    env      = CloudAutoScalingEnv(test_workload)
    state, _ = env.reset()
    total    = 0.0
    done     = False
    while not done:
        action                         = agent.get_action(state)
        state, reward, done, _, _      = env.step(action)
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

    w        = randomize_workload(train_w, seed=ep, ep=ep, total_eps=EPISODES)
    env      = CloudAutoScalingEnv(w)
    state, _ = env.reset()

    total     = 0.0
    ep_losses = []
    done      = False

    while not done:
        action                         = agent.act(state)
        next_state, reward, done, _, _ = env.step(action)

        agent.push(state, action, reward, next_state, float(done))

        loss = agent.train()
        if loss is not None:
            ep_losses.append(loss)

        state  = next_state
        total += reward

    # epsilon decay + target sync — once per episode, not per step
    agent.end_episode()

    avg_loss = float(np.mean(ep_losses)) if ep_losses else 0.0
    train_rewards.append(total)
    test_rewards.append(evaluate(agent, test_w))
    losses.append(avg_loss)

    print(
        f"[DQN-DR] Ep {ep+1:3d} | "
        f"train={total:9.1f} | "
        f"test={test_rewards[-1]:9.1f} | "
        f"loss={avg_loss:.4f} | "
        f"ε={agent.epsilon:.3f}"
    )


# ─────────────────────────────────────────
# Save model
# ─────────────────────────────────────────

os.makedirs("Queue_results", exist_ok=True)
agent.save("Queue_results/dqn_dr_model.pth")


# ─────────────────────────────────────────
# Plot
# ─────────────────────────────────────────

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

axes[0].plot(train_rewards, alpha=0.4, label="Train (randomized)")
axes[0].plot(test_rewards,  alpha=0.8, label="Test (unseen real)")

if len(train_rewards) >= 10:
    smooth = np.convolve(train_rewards, np.ones(10) / 10, mode="valid")
    axes[0].plot(range(9, len(train_rewards)), smooth, linewidth=2, label="Train smoothed")

if len(test_rewards) >= 10:
    smooth = np.convolve(test_rewards, np.ones(10) / 10, mode="valid")
    axes[0].plot(range(9, len(test_rewards)), smooth, linewidth=2, label="Test smoothed")

axes[0].set_xlabel("Episode")
axes[0].set_ylabel("Total reward")
axes[0].set_title("DQN + Domain Randomization — rewards")
axes[0].legend()

axes[1].plot(losses, color="orange", alpha=0.8)
axes[1].set_xlabel("Episode")
axes[1].set_ylabel("HuberLoss")
axes[1].set_title("DQN + Domain Randomization — loss")

plt.tight_layout()
plt.savefig("Queue_results/dqn_dr_rewards.png", dpi=150)
plt.close()

print("\nDQN + Domain Randomization training complete")
print("Plot  → results/dqn_dr_rewards.png")
print("Model → results/dqn_dr_model.pth")


# ─────────────────────────────────────────
# Summary
# ─────────────────────────────────────────

last_n    = 20
avg_train = np.mean(train_rewards[-last_n:])
avg_test  = np.mean(test_rewards[-last_n:])
gap       = avg_train - avg_test

print(f"\n{'─'*55}")
print(f"Final {last_n}-episode average")
print(f"  Train : {avg_train:>10.1f}")
print(f"  Test  : {avg_test:>10.1f}")
print(f"  Gap   : {gap:>10.1f}  (lower = better generalisation)")
print(f"{'─'*55}")