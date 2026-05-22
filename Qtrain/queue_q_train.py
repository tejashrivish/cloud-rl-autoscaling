import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from QEnv.queueEnv import CloudAutoScalingEnv
from Qagents.queue_q_agent import QLearningAgent
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

assert state_dim == 5, f"Expected 5-dim state, got {state_dim}. Check CloudAutoScalingEnv."


# =========================================================
# AGENT
# =========================================================

state_bins = [10, 10, 10, 10, 10]

agent = QLearningAgent(
    state_bins    = state_bins,
    action_dim    = action_dim,
    lr            = 0.1,
    gamma         = 0.99,
    epsilon       = 1.0,
    epsilon_min   = 0.05,
    epsilon_decay = 0.995,
)


# =========================================================
# DOMAIN RANDOMIZATION
# =========================================================

def randomize_workload(base_workload, seed=None, ep=0, total_eps=200):
    """
    Curriculum randomization — conservative early, aggressive late.

    seed=ep: reproducible across runs — re-running the script produces
    the identical episode sequence, but each episode sees a distinct
    workload variant. Pass seed=None for true stochasticity.

    Transforms
    ----------
    1. Amplitude scaling  → ±5% early, ±10% late
    2. Absolute noise     → std scales with workload std (0.01 → 0.02×)
    3. Temporal shift     → only after 30% of training
    4. Floor clamp        → prevents zero-demand collapse
    """
    rng = np.random.default_rng(seed)

    progress   = ep / max(total_eps - 1, 1)

    scale_half = 0.05 + 0.05 * progress          # 5% → 10%
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

def evaluate(agent, test_workload):
    """
    One greedy episode on the real unseen test workload.
    get_action() — no epsilon, no exploration.
    Full 5-dim state including queue. No slicing.
    """
    env      = CloudAutoScalingEnv(test_workload)
    state, _ = env.reset()
    total    = 0.0
    done     = False

    while not done:
        action                    = agent.get_action(state)
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
    # seed=ep: episode 0 always gets workload-0, episode 1 gets workload-1, etc.
    # Re-run the script → identical sequence.
    # Change EPISODES → only new episodes differ.
    w        = randomize_workload(train_workload, seed=ep, ep=ep, total_eps=EPISODES)
    env      = CloudAutoScalingEnv(w)
    state, _ = env.reset()       # shape (5,): w, ma, trend, v, queue

    total_reward = 0.0
    done         = False

    # ── episode rollout ───────────────────────────────────────────────────────
    while not done:
        action                         = agent.act(state)
        next_state, reward, done, _, _ = env.step(action)

        agent.update(state, action, float(reward), next_state, done)

        state         = next_state
        total_reward += reward

    # epsilon decays once per episode — not per step
    agent.decay_epsilon()

    # ── evaluation on real unseen test workload ───────────────────────────────
    train_rewards.append(total_reward)
    test_rewards.append(evaluate(agent, test_workload))

    print(
        f"[Q-DR] Ep {ep+1:3d} | "
        f"train={total_reward:9.2f} | "
        f"test={test_rewards[-1]:9.2f} | "
        f"ε={agent.epsilon:.3f}"
    )


# =========================================================
# SAVE
# =========================================================

os.makedirs("Queue_results", exist_ok=True)
os.makedirs("models",        exist_ok=True)

agent.save("models/q_table_curriculum.npy")


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
axes[0].set_title("Q-Learning + Curriculum DR — rewards")
axes[0].legend()

epsilons = [max(0.05, 1.0 * (0.995 ** ep)) for ep in range(EPISODES)]
axes[1].plot(epsilons, color="purple", linewidth=2)
axes[1].set_xlabel("Episode")
axes[1].set_ylabel("Epsilon")
axes[1].set_title("Epsilon decay over training")

plt.tight_layout()
plt.savefig("Queue_results/q_curriculum.png", dpi=150)
plt.close()

print("\nQ-Learning + Curriculum DR training complete")
print("Plot    → Queue_results/q_curriculum.png")
print("Q-table → models/q_table_curriculum.npy")


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