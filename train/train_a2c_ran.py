# import sys, os
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# import numpy as np
# import pandas as pd
# import torch
# import matplotlib.pyplot as plt

# from environment.cloud_env import CloudAutoScalingEnv
# from agents.a2c_ran_agent import A2CAgent
# import config


# # ✅ Load base workload
# df = pd.read_csv(config.DATA_PATH)
# base_workload = df["requests"].values.astype(np.float32)

# state_dim = 4
# action_dim = 3

# agent = A2CAgent(state_dim, action_dim)

# episodes = 200
# max_steps = 500

# rewards_history = []


# def randomize_workload(workload):
#     scale = np.random.uniform(0.5, 1.5)
#     noise = np.random.normal(0, 0.05, size=workload.shape)

#     w = workload * scale + noise
#     return np.clip(w, 0, None)


# for ep in range(episodes):

#     # 🔥 DOMAIN RANDOMIZATION
#     train_workload = randomize_workload(base_workload)
#     env = CloudAutoScalingEnv(train_workload)

#     state, _ = env.reset()

#     log_probs = []
#     values = []
#     rewards = []
#     dones = []

#     total_reward = 0

#     for step in range(max_steps):

#         state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(agent.device)

#         probs, value = agent.model(state_tensor)
#         dist = torch.distributions.Categorical(probs)

#         action = dist.sample()
#         log_prob = dist.log_prob(action)

#         next_state, reward, done, _, _ = env.step(action.item())

#         # ✅ reward clipping (important)
#         reward = np.clip(reward, -1, 1)

#         log_probs.append(log_prob)
#         values.append(value)
#         rewards.append(reward)
#         dones.append(done)

#         state = next_state
#         total_reward += reward

#         if done:
#             break

#     # bootstrap value
#     next_state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(agent.device)
#     _, next_value = agent.model(next_state_tensor)
#     next_value = next_value.detach().item()

#     agent.update(rewards, log_probs, values, next_value, dones)

#     rewards_history.append(total_reward)

#     print(f"Ep {ep+1}: Reward = {total_reward:.2f}")


# # 📈 Plot
# os.makedirs("results", exist_ok=True)

# smoothed = np.convolve(rewards_history, np.ones(10)/10, mode='valid')

# plt.figure()
# plt.plot(rewards_history, alpha=0.3)
# plt.plot(smoothed)
# plt.title("A2C + Domain Randomization")
# plt.savefig("results/a2c_randomized.png")
# plt.close()

# print("✅ A2C training complete")







import sys, os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

from environment.cloud_env import CloudAutoScalingEnv
from agents.a2c_ran_agent import A2CAgent
import config


# =========================================================
# LOAD WORKLOAD
# =========================================================

df = pd.read_csv(config.DATA_PATH)
base_workload = df["requests"].values.astype(np.float32)

split = int(config.TRAIN_SPLIT * len(base_workload))

train_workload = base_workload[:split]
test_workload = base_workload[split:]


# =========================================================
# AGENT
# =========================================================

state_dim = 4   # NO queue state used
action_dim = 3

agent = A2CAgent(state_dim, action_dim)


# =========================================================
# CURRICULUM DOMAIN RANDOMIZATION
# =========================================================

def randomize_workload(base_workload, seed=None, ep=0, total_eps=300):
    """
    Curriculum randomization — conservative early, aggressive late.

    1. amplitude scaling  → ±10% → ±20%
    2. proportional noise → 3% → 5%
    3. temporal shift     → after 30% training
    4. floor clamp        → prevents zero-demand collapse
    """

    rng = np.random.default_rng(seed)

    progress = ep / max(total_eps - 1, 1)

    scale_half = 0.1 + 0.1 * progress
    noise_std = 0.03 + 0.02 * progress

    scale = rng.uniform(1 - scale_half, 1 + scale_half)

    noise = rng.normal(0, noise_std, size=len(base_workload))
    noise = np.clip(noise, -0.15, 0.15)

    w = base_workload * scale * (1 + noise)

    if progress > 0.3:
        w = np.roll(w, rng.integers(0, len(base_workload)))

    floor = 0.05 * np.mean(base_workload)
    return np.clip(w, floor, None).astype(np.float32)


# =========================================================
# EVALUATION
# =========================================================

def evaluate(env, agent):

    state, _ = env.reset()

    state = state[:4]

    total_reward = 0
    done = False

    while not done:

        state_tensor = (
            torch.tensor(state, dtype=torch.float32)
            .unsqueeze(0)
            .to(agent.device)
        )

        with torch.no_grad():
            probs, _ = agent.model(state_tensor)
            dist = torch.distributions.Categorical(probs)
            action = dist.sample().item()

        next_state, reward, done, _, _ = env.step(action)

        next_state = next_state[:4]

        total_reward += reward
        state = next_state

    return total_reward


# =========================================================
# TRAINING
# =========================================================

episodes = 200

train_rewards = []
test_rewards = []

for ep in range(episodes):

    # -----------------------------------------------------
    # CURRICULUM RANDOMIZED WORKLOAD
    # -----------------------------------------------------

    train_w = randomize_workload(
        train_workload,
        seed=ep,
        ep=ep,
        total_eps=episodes
    )

    env = CloudAutoScalingEnv(train_w)

    state, _ = env.reset()
    state = state[:4]

    log_probs = []
    values = []
    rewards = []
    dones = []

    total_reward = 0
    done = False

    # -----------------------------------------------------
    # EPISODE ROLLOUT
    # -----------------------------------------------------

    while not done:

        state_tensor = (
            torch.tensor(state, dtype=torch.float32)
            .unsqueeze(0)
            .to(agent.device)
        )

        probs, value = agent.model(state_tensor)

        dist = torch.distributions.Categorical(probs)

        action = dist.sample()
        log_prob = dist.log_prob(action)

        next_state, reward, done, _, _ = env.step(action.item())

        next_state = next_state[:4]

        reward = np.clip(reward, -1, 1)

        log_probs.append(log_prob)
        values.append(value)
        rewards.append(reward)
        dones.append(done)

        total_reward += reward
        state = next_state

    # -----------------------------------------------------
    # BOOTSTRAP VALUE
    # -----------------------------------------------------

    next_state_tensor = (
        torch.tensor(state, dtype=torch.float32)
        .unsqueeze(0)
        .to(agent.device)
    )

    with torch.no_grad():
        _, next_value = agent.model(next_state_tensor)
        next_value = next_value.item()

    # -----------------------------------------------------
    # A2C UPDATE
    # -----------------------------------------------------

    agent.update(
        rewards,
        log_probs,
        values,
        next_value,
        dones
    )

    # -----------------------------------------------------
    # EVALUATION
    # -----------------------------------------------------

    test_env = CloudAutoScalingEnv(test_workload)
    test_reward = evaluate(test_env, agent)

    train_rewards.append(total_reward)
    test_rewards.append(test_reward)

    print(
        f"Episode {ep+1}: "
        f"Train = {total_reward:.2f}, "
        f"Test = {test_reward:.2f}"
    )


# =========================================================
# PLOT RESULTS
# =========================================================

os.makedirs("results", exist_ok=True)

plt.figure(figsize=(10, 5))
plt.plot(train_rewards, label="Train")
plt.plot(test_rewards, label="Test")
plt.xlabel("Episode")
plt.ylabel("Reward")
plt.title("A2C + Curriculum Domain Randomization (No Queue State)")
plt.legend()
plt.savefig("results/a2c_curriculum.png")
plt.close()


# =========================================================
# SAVE MODEL
# =========================================================

torch.save(
    agent.model.state_dict(),
    "results/a2c_curriculum_model.pth"
)

print("✅ Training complete")
