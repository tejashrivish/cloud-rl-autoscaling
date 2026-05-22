import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch

from QEnv.queueEnv import CloudAutoScalingEnv
from Qagents.queue_ppo_agent import PPOAgent
import config


# =========================================================
# LOAD WORKLOAD
# =========================================================

df            = pd.read_csv(config.DATA_PATH)
base_workload = df["requests"].values.astype(np.float32)

split          = int(config.TRAIN_SPLIT * len(base_workload))
train_workload = base_workload[:split]
test_workload  = base_workload[split:]


# =========================================================
# DIMENSIONS — derived from env, not hardcoded
# =========================================================

_tmp_env   = CloudAutoScalingEnv(train_workload)
state_dim  = _tmp_env.observation_space.shape[0]   # 5: w, ma, trend, v, queue
action_dim = _tmp_env.action_space.n               # 3: scale down, no-op, scale up
del _tmp_env


# =========================================================
# AGENT
# =========================================================

agent = PPOAgent(
    state_dim    = state_dim,
    action_dim   = action_dim,
    lr           = 3e-4,
    gamma        = 0.99,
    gae_lambda   = 0.95,
    clip_eps     = 0.2,
    ppo_epochs   = 4,
    batch_size   = 64,
    value_coef   = 0.5,
    entropy_coef = 0.01,
)


# =========================================================
# CURRICULUM DOMAIN RANDOMIZATION
# =========================================================

def randomize_workload(base_workload, seed=None, ep=0, total_eps=200):
    """
    Curriculum randomization — conservative early, aggressive late.

    1. Amplitude scaling  → ±10% early, ±20% late
    2. Proportional noise → 3% std early, 5% late  (hard cap ±15%)
    3. Temporal shift     → only after 30% of training
    4. Floor clamp        → prevents zero-demand collapse

    seed=ep: reproducible across runs, diverse across episodes.
    """
    rng = np.random.default_rng(seed)

    progress   = ep / max(total_eps - 1, 1)
    scale_half = 0.10 + 0.10 * progress
    noise_std  = 0.03 + 0.02 * progress

    scale = rng.uniform(1 - scale_half, 1 + scale_half)
    noise = np.clip(rng.normal(0, noise_std, size=len(base_workload)), -0.15, 0.15)

    w = base_workload * scale * (1 + noise)

    if progress > 0.3:
        w = np.roll(w, rng.integers(0, len(base_workload)))

    floor = 0.05 * np.mean(base_workload)
    return np.clip(w, floor, None).astype(np.float32)


# =========================================================
# EVALUATION
# =========================================================

GREEDY_AFTER = 50   # switch stochastic → greedy eval after this episode

def evaluate(agent, test_workload, greedy=True):
    """
    One episode on the real unseen test workload.

    greedy=False (early): samples from π — shows real improvement before
      the argmax flips. Prevents flat test reward during early training.
    greedy=True  (later): argmax of π — stable measure of converged policy.

    Full 5-dim state (w, ma, trend, v, queue). No slicing.
    Queue resets to 0.0 at env.reset().
    """
    env      = CloudAutoScalingEnv(test_workload)
    state, _ = env.reset()
    total    = 0.0
    done     = False

    while not done:
        if greedy:
            action = agent.get_action(state)
        else:
            action, _, _ = agent.act(state)

        state, reward, done, _, _ = env.step(action)
        total += reward

    return total


# =========================================================
# TRAINING
# =========================================================

EPISODES      = 200
train_rewards = []
test_rewards  = []

for ep in range(EPISODES):

    # ── randomized workload for this episode ──────────────────────────────────
    w        = randomize_workload(train_workload, seed=ep, ep=ep, total_eps=EPISODES)
    env      = CloudAutoScalingEnv(w)
    state, _ = env.reset()       # shape (5,): w, ma, trend, v, queue

    total_reward = 0.0
    done         = False

    # ── episode rollout ───────────────────────────────────────────────────────
    while not done:
        action, log_prob, value = agent.act(state)

        next_state, reward, done, _, _ = env.step(action)

        # no reward clipping — clipping destroys the advantage signal and
        # causes flat test reward. GAE + advantage normalisation in
        # update() handles reward scale differences instead.
        agent.store((state, action, log_prob, float(reward), done, value))

        state         = next_state
        total_reward += reward

    # ── PPO update — runs ppo_epochs passes over the full rollout ─────────────
    agent.update()

    # ── evaluation ────────────────────────────────────────────────────────────
    greedy      = ep >= GREEDY_AFTER
    test_reward = evaluate(agent, test_workload, greedy=greedy)

    train_rewards.append(total_reward)
    test_rewards.append(test_reward)

    print(
        f"[PPO-DR] Ep {ep+1:3d} | "
        f"train={total_reward:9.2f} | "
        f"test={test_reward:9.2f} | "
        f"eval={'greedy' if greedy else 'stoch '}"
    )


# =========================================================
# SAVE
# =========================================================

os.makedirs("Queue_results", exist_ok=True)
agent.save("Queue_results/ppo_curriculum_model.pth")


# =========================================================
# PLOT
# =========================================================

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# ── reward curves ─────────────────────────────────────────
axes[0].plot(train_rewards, alpha=0.4, label="Train (randomized)")
axes[0].plot(test_rewards,  alpha=0.8, label="Test (unseen real)")

if len(train_rewards) >= 10:
    smooth = np.convolve(train_rewards, np.ones(10) / 10, mode="valid")
    axes[0].plot(range(9, len(train_rewards)), smooth, linewidth=2, label="Train smoothed")

if len(test_rewards) >= 10:
    smooth = np.convolve(test_rewards, np.ones(10) / 10, mode="valid")
    axes[0].plot(range(9, len(test_rewards)), smooth, linewidth=2, label="Test smoothed")

axes[0].axvline(GREEDY_AFTER, color="gray", linestyle="--", linewidth=1,
                label=f"Greedy eval starts (ep {GREEDY_AFTER})")
axes[0].set_xlabel("Episode")
axes[0].set_ylabel("Total reward")
axes[0].set_title("PPO + Curriculum DR — rewards")
axes[0].legend()

# ── test reward only ──────────────────────────────────────
axes[1].plot(test_rewards, color="darkorange", alpha=0.6, label="Test reward")
if len(test_rewards) >= 10:
    smooth = np.convolve(test_rewards, np.ones(10) / 10, mode="valid")
    axes[1].plot(range(9, len(test_rewards)), smooth, color="darkorange",
                 linewidth=2, label="Test smoothed")
axes[1].axvline(GREEDY_AFTER, color="gray", linestyle="--", linewidth=1)
axes[1].set_xlabel("Episode")
axes[1].set_ylabel("Test reward")
axes[1].set_title("Generalisation — test reward over training")
axes[1].legend()

plt.tight_layout()
plt.savefig("Queue_results/ppo_curriculum.png", dpi=150)
plt.close()

print("\nPPO + Curriculum DR training complete")
print("Plot  → results/ppo_curriculum.png")
print("Model → results/ppo_curriculum_model.pth")


# =========================================================
# SUMMARY
# =========================================================

last_n    = 20
avg_train = np.mean(train_rewards[-last_n:])
avg_test  = np.mean(test_rewards[-last_n:])
gap       = avg_train - avg_test

print(f"\n{'─'*55}")
print(f"Final {last_n}-episode average")
print(f"  Train : {avg_train:>10.2f}")
print(f"  Test  : {avg_test:>10.2f}")
print(f"  Gap   : {gap:>10.2f}  (lower = better generalisation)")
print(f"{'─'*55}")