import numpy as np
import pandas as pd

def create_offline_dataset(csv_path, max_mips=1000):
    df = pd.read_csv(csv_path)
    workload = df["requests"].values.astype(np.float32)

    dataset = []

    v = 0.5  # start allocation

    for t in range(len(workload) - 1):

        w = workload[t]
        w_next = workload[t + 1]

        # features
        ma = np.mean(workload[max(0, t-5):t+1])
        trend = w - workload[t-1] if t > 0 else 0

        state = np.array([
            w / workload.max(),
            ma / workload.max(),
            trend / workload.max(),
            v
        ], dtype=np.float32)

        # 🔒 SAFE HEURISTIC POLICY
        capacity = v * max_mips

        if w > capacity:         # under-provision → scale up
            action = 2
        elif w < 0.5 * capacity: # over-provision → scale down
            action = 0
        else:
            action = 1

        # apply action
        if action == 0:
            v = max(0.1, v - 0.1)
        elif action == 2:
            v = min(1.0, v + 0.1)

        # compute reward (same as env)
        capacity = v * max_mips + 1e-6
        rt = w / capacity if w <= capacity else (w / capacity) ** 2

        cost = v
        sla_penalty = max(0.0, rt - 1.0) ** 2

        reward = -(cost + 10 * sla_penalty)

        next_state = np.array([
            w_next / workload.max(),
            ma / workload.max(),
            trend / workload.max(),
            v
        ], dtype=np.float32)

        done = (t == len(workload) - 2)

        dataset.append((state, action, reward, next_state, done))

    return dataset