import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch

from QEnv.queueEnvCont import CloudAutoScalingEnvContinuous
from Qagents.queue_ddpg_agent import DDPGAgent
import config


# =========================================================
# LOAD DATA
# =========================================================

df            = pd.read_csv(config.DATA_PATH)
base_workload = df["requests"].values.astype(np.float32)

split          = int(config.TRAIN_SPLIT * len(base_workload))
train_workload = base_workload[:split]
test_workload  = base_workload[split:]


# =========================================================
# DIMENSIONS — derived from env, not hardcoded
# =========================================================

_tmp_env   = CloudAutoScalingEnvContinuous(train_workload)
state_dim  = _tmp_env.observation_space.shape[0]   # 5: w, ma, trend, v, queue
action_dim = _tmp_env.action_space.shape[0]        # 1: continuous v
del _tmp_env


# =========================================================
# AGENT
# =========================================================

agent = DDPGAgent(
    state_dim  = state_dim,
    action_dim = action_dim,
    actor_lr   = 1e-4,
    critic_lr  = 1e-4,
    gamma      = 0.99,
    tau        = 0.005,
    buffer_size= 50_000,
    batch_size = 64,
    noise_std  = 0.1,
)


# =========================================================
# CURRICULUM RANDOMIZATION
# =========================================================

def randomize_workload(base_workload, ep=0, total_eps=200):
    """
    Curriculum randomization — conservative early, aggressive late.

    seed via default_rng(ep): reproducible across runs, diverse across
    episodes. Each episode always gets the same variant on re-run.

    1. Amplitude scaling  → ±5% early, ±10% late
    2. Absolute noise     → std scales with workload std (0.01 → 0.02×)
    3. Temporal shift     → only after 30% of training
    4. Floor clamp        → no zero-demand deserts
    """
    rng = np.random.default_rng(ep)

    progress   = ep / max(total_eps - 1, 1)
    scale_half = 0.05 + 0.05 * progress
    scale      = rng.uniform(1 - scale_half, 1 + scale_half)

    noise_std  = 0.01 + 0.01 * progress
    noise      = rng.normal(0, noise_std * np.std(base_workload), size=len(base_workload))

    w = base_workload * scale + noise

    if progress > 0.3:
        w = np.roll(w, rng.integers(0, len(base_workload)))

    floor = 0.05 * np.mean(base_workload)
    return np.clip(w, floor, None).astype(np.float32)


# =========================================================
# EVALUATION
# =========================================================

def evaluate(agent, test_workload, runs=3):
    """
    Deterministic evaluation on the real unseen test workload.
    act(noise=False) — no exploration, pure policy.
    Full 5-dim state including queue. Queue resets at env.reset().
    Averaged over `runs` episodes to reduce variance.
    """
    rewards = []

    for _ in range(runs):
        env      = CloudAutoScalingEnvContinuous(test_workload)
        state, _ = env.reset()
        total    = 0.0
        done     = False

        while not done:
            action                    = agent.act(state, noise=False)
            state, reward, done, _, _ = env.step(action)
            total += reward

        rewards.append(total)

    return float(np.mean(rewards))


# =========================================================
# TRAINING
# =========================================================

EPISODES      = 200
train_rewards = []
test_rewards  = []

for ep in range(EPISODES):

    # ── randomized workload for this episode ──────────────────────────────────
    w        = randomize_workload(train_workload, ep=ep, total_eps=EPISODES)
    env      = CloudAutoScalingEnvContinuous(w)
    state, _ = env.reset()       # shape (5,): w, ma, trend, v, queue

    total_reward = 0.0
    done         = False

    # ── episode rollout ───────────────────────────────────────────────────────
    while not done:
        action = agent.act(state, noise=True)

        next_state, reward, done, _, _ = env.step(action)

        agent.store((
            np.array(state,      dtype=np.float32),
            np.array(action,     dtype=np.float32),
            float(reward),
            np.array(next_state, dtype=np.float32),
            float(done),
        ))

        agent.train()

        state         = next_state
        total_reward += reward

    # ── evaluation on real unseen test workload ───────────────────────────────
    test_reward = evaluate(agent, test_workload, runs=3)

    train_rewards.append(total_reward)
    test_rewards.append(test_reward)

    print(
        f"[DDPG-DR] Ep {ep+1:3d} | "
        f"train={total_reward:9.2f} | "
        f"test={test_reward:9.2f}"
    )


# =========================================================
# SAVE
# =========================================================

os.makedirs("results", exist_ok=True)
agent.save(
    "results/ddpg_actor.pth",
    "results/ddpg_critic.pth",
)


# =========================================================
# PLOT
# =========================================================

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

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
axes[0].set_title("DDPG + Curriculum DR — rewards")
axes[0].legend()

axes[1].plot(test_rewards, color="darkorange", alpha=0.6, label="Test reward")
if len(test_rewards) >= 10:
    smooth = np.convolve(test_rewards, np.ones(10) / 10, mode="valid")
    axes[1].plot(range(9, len(test_rewards)), smooth, color="darkorange",
                 linewidth=2, label="Test smoothed")
axes[1].set_xlabel("Episode")
axes[1].set_ylabel("Test reward")
axes[1].set_title("Generalisation — test reward over training")
axes[1].legend()

plt.tight_layout()
plt.savefig("results/ddpg_curriculum.png", dpi=150)
plt.close()

print("\nDDPG + Curriculum DR training complete")
print("Plot   → results/ddpg_curriculum.png")
print("Actor  → results/ddpg_actor.pth")
print("Critic → results/ddpg_critic.pth")


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