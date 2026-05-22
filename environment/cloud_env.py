import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd


class CloudAutoScalingEnv(gym.Env):
    def __init__(self, workload, sla=1.0, max_mips=1000):
        super().__init__()

        self.workload = workload.astype(np.float32)
        self.max_workload = np.max(workload)

        self.sla = sla
        self.max_mips = max_mips

        self.t = 0

        self.action_space = spaces.Discrete(3)

        # [current, moving_avg, trend, allocation]
        self.observation_space = spaces.Box(
            low=0.0, high=np.inf, shape=(4,), dtype=np.float32
        )

        self.v = 0.5

    def moving_avg(self, t, k=5):
        start = max(0, t - k)
        return np.mean(self.workload[start:t+1])

    def trend(self, t):
        if t == 0:
            return 0
        return self.workload[t] - self.workload[t - 1]

    def response_time(self, w, v):
        capacity = v * self.max_mips + 1e-6
        return w / capacity if w <= capacity else (w / capacity) ** 2

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.t = 0
        self.v = 0.5
        return self._get_state(), {}

    
    def step(self, action):

         w = self.workload[self.t]

    # 🔥 apply safety layer
         action = self.safe_action(action, w)

    # scaling logic
         if action == 0:
           self.v = max(0.1, self.v - 0.1)
         elif action == 2:
           self.v = min(1.0, self.v + 0.1)

         rt = self.response_time(w, self.v)

         cost = self.v
         sla_penalty = max(0.0, rt - self.sla) ** 2
         reward = -(cost + 10 * sla_penalty)

         self.t += 1

         terminated = self.t >= len(self.workload) - 1

         return self._get_state(), reward, terminated, False, {}

    def _get_state(self):
        w = self.workload[self.t]
        ma = self.moving_avg(self.t)
        tr = self.trend(self.t)

        return np.array([
            w / self.max_workload,
            ma / self.max_workload,
            tr / self.max_workload,
            self.v
        ], dtype=np.float32)

    def safe_action(self, action, w):

         capacity = self.v * self.max_mips

    # ❌ prevent scaling down if already overloaded
         if w > capacity and action == 0:
             return 1  # force no-op

         return action



