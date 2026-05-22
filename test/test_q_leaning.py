import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import pandas as pd

from environment.cloud_env import CloudAutoScalingEnv
from agents.q_learning import QLearningAgent
import config


df = pd.read_csv(config.DATA_PATH)
workload = df["requests"].values.astype(np.float32)

split = int(config.TRAIN_SPLIT * len(workload))
test_w = workload[split:]

env = CloudAutoScalingEnv(test_w)

agent = QLearningAgent(state_bins=[10,10,10,10], action_dim=3)
agent.q_table = np.load(config.MODELS_DIR + "q_table.npy")

agent.epsilon = 0.0

state, _ = env.reset()
total_reward = 0

done = False

while not done:
    action = agent.act(state)
    state, reward, done, _, _ = env.step(action)
    total_reward += reward

print("Test Reward:", total_reward)