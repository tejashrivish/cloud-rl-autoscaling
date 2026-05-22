import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

from QEnv.queueEnv import CloudAutoScalingEnv
from Qagents.queue_a2c_agent import A2CAgent
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

agent = A2CAgent(
    state_dim    = state_dim,
    action_dim   = action_dim,
    lr           = 1e-3,       # higher lr shifts logits faster — fixes flat test reward
    gamma        = 0.99,
    value_coef   = 0.5,
    entropy_coef = 0.02,
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

GREEDY_AFTER = 50   # switch from stochastic → greedy eval after this episode

def evaluate(agent, test_workload, greedy=True):
    """
    One episode on the real unseen test workload.

    greedy=False (early training): samples from π — reflects the actual
      learned distribution before any single action dominates the argmax.
      Shows real improvement from episode 1.

    greedy=True (later training): argmax of π — deterministic, stable
      measure of the converged policy.

    Queue resets to 0.0 at env.reset(). Full 5-dim state, no slicing.
    """
    env      = CloudAutoScalingEnv(test_workload)
    state, _ = env.reset()
    total    = 0.0
    done     = False

    while not done:
        if greedy:
            action = agent.get_action(state)
        else:
            action, _, _, _ = agent.select_action(state)

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

    log_probs    = []
    values       = []
    rewards      = []
    dones        = []
    total_reward = 0.0
    done         = False

    # ── episode rollout ───────────────────────────────────────────────────────
    while not done:
        action, log_prob, value, dist = agent.select_action(state)

        next_state, reward, done, _, _ = env.step(action)

        # no reward clipping — clipping flattens the advantage signal and
        # causes the policy to treat all actions as equally bad, which is
        # why the test reward was stuck. advantage normalisation in
        # update_batch() handles scale differences instead.
        reward = float(reward)

        log_probs.append(log_prob)
        values.append(value)
        rewards.append(reward)
        dones.append(done)

        total_reward += reward
        state         = next_state

    # ── bootstrap terminal value ──────────────────────────────────────────────
    if done:
        next_value = 0.0
    else:
        ns_t = torch.from_numpy(
            np.array(state, dtype=np.float32)
        ).unsqueeze(0).to(agent.device)
        with torch.no_grad():
            _, nv = agent.model(ns_t)
        next_value = nv.item()

    # ── batched A2C update ────────────────────────────────────────────────────
    agent.update_batch(rewards, log_probs, values, next_value, dones)

    # ── evaluation ────────────────────────────────────────────────────────────
    # stochastic eval early (shows real improvement immediately),
    # greedy eval later (stable measure of converged policy)
    greedy      = ep >= GREEDY_AFTER
    test_reward = evaluate(agent, test_workload, greedy=greedy)

    train_rewards.append(total_reward)
    test_rewards.append(test_reward)

    print(
        f"[A2C-DR] Ep {ep+1:3d} | "
        f"train={total_reward:9.2f} | "
        f"test={test_reward:9.2f} | "
        f"eval={'greedy' if greedy else 'stoch '}"
    )


# =========================================================
# SAVE
# =========================================================

os.makedirs("results", exist_ok=True)
agent.save("results/a2c_curriculum_model.pth")


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

# mark the switch from stochastic → greedy evaluation
axes[0].axvline(GREEDY_AFTER, color="gray", linestyle="--", linewidth=1, label=f"Greedy eval starts (ep {GREEDY_AFTER})")
axes[0].set_xlabel("Episode")
axes[0].set_ylabel("Total reward")
axes[0].set_title("A2C + Curriculum DR — rewards")
axes[0].legend()

# ── test reward only (cleaner view of generalisation) ─────
axes[1].plot(test_rewards, color="darkorange", alpha=0.6, label="Test reward")
if len(test_rewards) >= 10:
    smooth = np.convolve(test_rewards, np.ones(10) / 10, mode="valid")
    axes[1].plot(range(9, len(test_rewards)), smooth, color="darkorange", linewidth=2, label="Test smoothed")
axes[1].axvline(GREEDY_AFTER, color="gray", linestyle="--", linewidth=1)
axes[1].set_xlabel("Episode")
axes[1].set_ylabel("Test reward")
axes[1].set_title("Generalisation — test reward over training")
axes[1].legend()

plt.tight_layout()
plt.savefig("results/a2c_curriculum.png", dpi=150)
plt.close()

print("\nA2C + Curriculum DR training complete")
print("Plot  → results/a2c_curriculum.png")
print("Model → results/a2c_curriculum_model.pth")


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