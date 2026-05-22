# import sys, os
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# import numpy as np
# import pandas as pd
# import torch
# import matplotlib.pyplot as plt

# from QEnv.queueEnv import CloudAutoScalingEnv
# from Qagents.a2c_agent import A2CAgent
# import config


# # =========================================================
# # LOAD WORKLOAD
# # =========================================================

# df            = pd.read_csv(config.DATA_PATH)
# base_workload = df["requests"].values.astype(np.float32)

# split          = int(config.TRAIN_SPLIT * len(base_workload))
# train_workload = base_workload[:split]
# test_workload  = base_workload[split:]


# # =========================================================
# # DIMENSIONS — derived from env, not hardcoded
# # =========================================================

# _tmp_env   = CloudAutoScalingEnv(train_workload)
# state_dim  = _tmp_env.observation_space.shape[0]   # 5: w, ma, trend, v, queue
# action_dim = _tmp_env.action_space.n               # 3: scale down, no-op, scale up
# del _tmp_env


# # =========================================================
# # AGENT
# # =========================================================

# agent = A2CAgent(
#     state_dim    = state_dim,
#     action_dim   = action_dim,
#     lr           = 1e-4,       # reduced from 1e-3 — prevents gradient explosion
#     gamma        = 0.99,
#     value_coef   = 0.5,
#     entropy_coef = 0.05,       # increased — resists policy collapse after divergence
# )


# # =========================================================
# # CURRICULUM DOMAIN RANDOMIZATION
# # =========================================================

# def randomize_workload(base_workload, seed=None, ep=0, total_eps=200):
#     """
#     Curriculum randomization — conservative early, aggressive late.

#     1. Amplitude scaling  → ±10% early, ±20% late
#     2. Proportional noise → 3% std early, 5% late  (hard cap ±15%)
#     3. Temporal shift     → only after 30% of training
#     4. Floor clamp        → prevents zero-demand collapse

#     seed=ep: reproducible across runs, diverse across episodes.
#     """
#     rng = np.random.default_rng(seed)

#     progress   = ep / max(total_eps - 1, 1)
#     scale_half = 0.10 + 0.10 * progress
#     noise_std  = 0.03 + 0.02 * progress

#     scale = rng.uniform(1 - scale_half, 1 + scale_half)
#     noise = np.clip(rng.normal(0, noise_std, size=len(base_workload)), -0.15, 0.15)

#     w = base_workload * scale * (1 + noise)

#     if progress > 0.3:
#         w = np.roll(w, rng.integers(0, len(base_workload)))

#     floor = 0.05 * np.mean(base_workload)
#     return np.clip(w, floor, None).astype(np.float32)


# # =========================================================
# # EVALUATION
# # =========================================================

# GREEDY_AFTER = 50   # switch from stochastic → greedy eval after this episode

# def evaluate(agent, test_workload, greedy=True):
#     """
#     One episode on the real unseen test workload.

#     greedy=False (early): samples from π — shows real improvement before
#       argmax flips. Prevents flat test reward during early training.
#     greedy=True  (later): argmax of π — stable measure of converged policy.

#     Queue resets to 0.0 at env.reset(). Full 5-dim state, no slicing.
#     """
#     env      = CloudAutoScalingEnv(test_workload)
#     state, _ = env.reset()
#     total    = 0.0
#     done     = False

#     while not done:
#         if greedy:
#             action = agent.get_action(state)
#         else:
#             action, _, _, _ = agent.select_action(state)

#         state, reward, done, _, _ = env.step(action)
#         total += reward

#     return total


# # =========================================================
# # TRAINING
# # =========================================================

# EPISODES      = 200
# train_rewards = []
# test_rewards  = []

# # per-step reward clip bounds — matched to actual reward range
# # min reward per step ≈ -(1.0 + 10 * max_sla_penalty) ≈ -11.0
# REWARD_CLIP_MIN = -10.0
# REWARD_CLIP_MAX =  0.0

# for ep in range(EPISODES):

#     # ── randomized workload for this episode ──────────────────────────────────
#     w        = randomize_workload(train_workload, seed=ep, ep=ep, total_eps=EPISODES)
#     env      = CloudAutoScalingEnv(w)
#     state, _ = env.reset()       # shape (5,): w, ma, trend, v, queue

#     log_probs    = []
#     values       = []
#     rewards      = []
#     dones        = []
#     total_reward = 0.0
#     done         = False

#     # ── episode rollout ───────────────────────────────────────────────────────
#     while not done:
#         action, log_prob, value, dist = agent.select_action(state)

#         next_state, reward, done, _, _ = env.step(action)

#         # reward clipping — prevents catastrophic gradient explosion
#         # when DR workload causes queue to accumulate over thousands of steps.
#         # clip range matched to true per-step reward bounds [-10, 0].
#         # advantage normalisation in update_batch handles remaining scale.
#         clipped_reward = float(np.clip(reward, REWARD_CLIP_MIN, REWARD_CLIP_MAX))

#         log_probs.append(log_prob)
#         values.append(value)
#         rewards.append(clipped_reward)
#         dones.append(done)

#         total_reward += reward   # track true reward for logging
#         state         = next_state

#     # ── bootstrap terminal value ──────────────────────────────────────────────
#     if done:
#         next_value = 0.0
#     else:
#         ns_t = torch.from_numpy(
#             np.array(state, dtype=np.float32)
#         ).unsqueeze(0).to(agent.device)
#         with torch.no_grad():
#             _, nv = agent.model(ns_t)
#         next_value = nv.item()

#     # ── batched A2C update ────────────────────────────────────────────────────
#     agent.update_batch(rewards, log_probs, values, next_value, dones)

#     # ── evaluation ────────────────────────────────────────────────────────────
#     greedy      = ep >= GREEDY_AFTER
#     test_reward = evaluate(agent, test_workload, greedy=greedy)

#     train_rewards.append(total_reward)
#     test_rewards.append(test_reward)

#     print(
#         f"[A2C-DR] Ep {ep+1:3d} | "
#         f"train={total_reward:9.2f} | "
#         f"test={test_reward:9.2f} | "
#         f"eval={'greedy' if greedy else 'stoch '}"
#     )


# # =========================================================
# # SAVE
# # =========================================================

# os.makedirs("Queue_results", exist_ok=True)
# agent.save("Queue_results/a2c_curriculum_model.pth")


# # =========================================================
# # PLOT
# # =========================================================

# fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# axes[0].plot(train_rewards, alpha=0.4, label="Train (randomized)")
# axes[0].plot(test_rewards,  alpha=0.8, label="Test (unseen real)")

# if len(train_rewards) >= 10:
#     smooth = np.convolve(train_rewards, np.ones(10) / 10, mode="valid")
#     axes[0].plot(range(9, len(train_rewards)), smooth, linewidth=2, label="Train smoothed")

# if len(test_rewards) >= 10:
#     smooth = np.convolve(test_rewards, np.ones(10) / 10, mode="valid")
#     axes[0].plot(range(9, len(test_rewards)), smooth, linewidth=2, label="Test smoothed")

# axes[0].axvline(GREEDY_AFTER, color="gray", linestyle="--", linewidth=1,
#                 label=f"Greedy eval starts (ep {GREEDY_AFTER})")
# axes[0].set_xlabel("Episode")
# axes[0].set_ylabel("Total reward")
# axes[0].set_title("A2C + Curriculum DR — rewards")
# axes[0].legend()

# axes[1].plot(test_rewards, color="darkorange", alpha=0.6, label="Test reward")
# if len(test_rewards) >= 10:
#     smooth = np.convolve(test_rewards, np.ones(10) / 10, mode="valid")
#     axes[1].plot(range(9, len(test_rewards)), smooth, color="darkorange",
#                  linewidth=2, label="Test smoothed")
# axes[1].axvline(GREEDY_AFTER, color="gray", linestyle="--", linewidth=1)
# axes[1].set_xlabel("Episode")
# axes[1].set_ylabel("Test reward")
# axes[1].set_title("Generalisation — test reward over training")
# axes[1].legend()

# plt.tight_layout()
# plt.savefig("Queue_results/a2c_curriculum.png", dpi=150)
# plt.close()

# print("\nA2C + Curriculum DR training complete")
# print("Plot  → Queue_results/a2c_curriculum.png")
# print("Model → Queue_results/a2c_curriculum_model.pth")


# # =========================================================
# # SUMMARY
# # =========================================================

# last_n    = 20
# avg_train = np.mean(train_rewards[-last_n:])
# avg_test  = np.mean(test_rewards[-last_n:])
# gap       = avg_train - avg_test

# print(f"\n{'─'*55}")
# print(f"Final {last_n}-episode average")
# print(f"  Train : {avg_train:>10.2f}")
# print(f"  Test  : {avg_test:>10.2f}")
# print(f"  Gap   : {gap:>10.2f}  (lower = better generalisation)")
# print(f"{'─'*55}")



import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

from QEnv.queueEnv import CloudAutoScalingEnv
from Qagents.a2c_agent import A2CAgent
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
# DIMENSIONS
# =========================================================

_tmp_env   = CloudAutoScalingEnv(train_workload)
state_dim  = _tmp_env.observation_space.shape[0]
action_dim = _tmp_env.action_space.n
del _tmp_env


# =========================================================
# AGENT
# =========================================================

agent = A2CAgent(
    state_dim    = state_dim,
    action_dim   = action_dim,
    lr           = 3e-4,
    gamma        = 0.99,
    value_coef   = 0.5,
    entropy_coef = 0.05,
)


# =========================================================
# CURRICULUM DOMAIN RANDOMIZATION
# =========================================================

def randomize_workload(base_workload, seed=None, ep=0, total_eps=200):
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

GREEDY_AFTER = 50

def evaluate(agent, test_workload, greedy=True):
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
# TRAINING — single-step TD updates
# =========================================================
# update_batch() over a full ~3500-step rollout is unstable —
# a single queue-explosion episode produces returns in the
# billions even with per-step clipping, overwhelming the
# advantage normalisation. Single-step update() with per-step
# reward clipping is stable by construction: each gradient
# step sees at most one clipped reward.

EPISODES      = 200
REWARD_CLIP   = 10.0   # clip to [-10, 0] — matched to true per-step bounds
train_rewards = []
test_rewards  = []

for ep in range(EPISODES):

    w        = randomize_workload(train_workload, seed=ep, ep=ep, total_eps=EPISODES)
    env      = CloudAutoScalingEnv(w)
    state, _ = env.reset()

    total_reward = 0.0
    done         = False

    while not done:
        action, log_prob, value, dist = agent.select_action(state)

        next_state, reward, done, _, _ = env.step(action)

        # clip reward for gradient stability
        clipped = float(np.clip(reward, -REWARD_CLIP, 0.0))

        # get V(s') for TD target
        with torch.no_grad():
            ns_t       = torch.from_numpy(
                np.array(next_state, dtype=np.float32)
            ).unsqueeze(0).to(agent.device)
            _, next_val = agent.model(ns_t)

        # single-step TD update — stable regardless of episode length
        agent.update(log_prob, value, clipped, next_val, done, dist)

        total_reward += reward   # log true reward
        state         = next_state

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

os.makedirs("Queue_results", exist_ok=True)
agent.save("Queue_results/a2c_curriculum_model.pth")


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

axes[0].axvline(GREEDY_AFTER, color="gray", linestyle="--", linewidth=1,
                label=f"Greedy eval starts (ep {GREEDY_AFTER})")
axes[0].set_xlabel("Episode")
axes[0].set_ylabel("Total reward")
axes[0].set_title("A2C + Curriculum DR — rewards")
axes[0].legend()

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
plt.savefig("Queue_results/a2c_curriculum.png", dpi=150)
plt.close()

print("\nA2C + Curriculum DR training complete")
print("Plot  → Queue_results/a2c_curriculum.png")
print("Model → Queue_results/a2c_curriculum_model.pth")


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