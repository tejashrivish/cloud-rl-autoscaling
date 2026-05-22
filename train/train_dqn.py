# import sys
# import os
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# import numpy as np
# import matplotlib.pyplot as plt
# import pandas as pd
# import torch

# from environment.cloud_env import CloudAutoScalingEnv
# from agents.dqn_agent import DQNAgent
# import config

# # ✅ Load workload
# df = pd.read_csv(config.DATA_PATH)
# workload = df["requests"].values.astype(np.float32)

# # ✅ Create environment
# env = CloudAutoScalingEnv(workload)

# action_dim = 3
# state_dim = env.observation_space.shape[0]
# agent = DQNAgent(state_dim, action_dim)

# episodes = 200
# rewards = []

# for ep in range(episodes):
#     state, _ = env.reset()
#     total_reward = 0

#     done = False

#     while not done:
#         action = agent.act(state)

#         next_state, reward, done, _, _ = env.step(action)

#         agent.store((np.array(state, dtype=np.float32),action,reward,np.array(next_state, dtype=np.float32),done
# ))
#         agent.train()

#         state = next_state
#         total_reward += reward

#     rewards.append(total_reward)

#     print(f"Episode {ep+1}: Reward = {total_reward:.2f}, Epsilon = {agent.epsilon:.3f}")


# # ✅ Save results
# os.makedirs("results", exist_ok=True)

# # 📈 Smooth rewards
# smoothed = np.convolve(rewards, np.ones(10)/10, mode='valid')

# plt.figure()
# plt.plot(rewards, alpha=0.3, label="Raw")
# plt.plot(smoothed, label="Smoothed")
# plt.xlabel("Episode")
# plt.ylabel("Reward")
# plt.title("DQN Training")
# plt.legend()
# plt.savefig("results/dqn_rewards.png")
# plt.close()

# # ✅ Save model
# torch.save(agent.q_net.state_dict(), "results/dqn_model.pth")

# print("✅ Training complete. Results saved in results/")




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
# Load workload dataset
# =========================================
df = pd.read_csv(config.DATA_PATH)

workload = df["requests"].values.astype(np.float32)


# =========================================
# Train / Test Split
# =========================================
split = int(config.TRAIN_SPLIT * len(workload))

train_workload = workload[:split]
test_workload = workload[split:]


# =========================================
# Create environments
# =========================================
train_env = CloudAutoScalingEnv(train_workload)

state_dim = train_env.observation_space.shape[0]
action_dim = train_env.action_space.n

agent = DQNAgent(state_dim, action_dim)


# =========================================
# Evaluation Function
# =========================================
def evaluate(env, agent):

    state, _ = env.reset()

    total_reward = 0
    done = False

    while not done:

        # Greedy action during testing
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(agent.device)

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
    # Testing on unseen workload
    # ---------------------------------
    test_env = CloudAutoScalingEnv(test_workload)

    test_reward = evaluate(test_env, agent)

    train_rewards.append(total_reward)
    test_rewards.append(test_reward)

    print(
        f"Episode {ep+1}: "
        f"Train Reward = {total_reward:.2f}, "
        f"Test Reward = {test_reward:.2f}, "
        f"Epsilon = {agent.epsilon:.3f}"
    )


# =========================================
# Save Results
# =========================================
os.makedirs("results", exist_ok=True)


# =========================================
# Plot Rewards
# =========================================
plt.figure(figsize=(10, 5))

plt.plot(train_rewards, alpha=0.5, label="Train Reward")
plt.plot(test_rewards, alpha=0.8, label="Test Reward")

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
plt.title("DQN Training vs Testing")
plt.legend()

plt.savefig("results/dqn_train_test_rewards.png")
plt.close()


# =========================================
# Save Model
# =========================================
torch.save(
    agent.q_net.state_dict(),
    "results/dqn_model.pth"
)

print("\n✅ DQN Training + Testing Complete")
print("📈 Plot saved: results/dqn_train_test_rewards.png")
print("💾 Model saved: results/dqn_model.pth")