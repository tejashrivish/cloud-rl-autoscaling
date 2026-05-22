import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CloudAutoScalingEnvContinuous(gym.Env):
    def __init__(self, workload, sla=1.0, max_mips=1000):
        super().__init__()

        # -----------------------------
        # Workload
        # -----------------------------
        self.workload = workload.astype(np.float32)
        self.max_workload = np.max(self.workload)

        self.sla = sla
        self.max_mips = max_mips

        self.t = 0

        # -----------------------------
        # Continuous Action Space
        # v ∈ [0.1, 1.0]
        # -----------------------------
        self.action_space = spaces.Box(
            low=np.array([0.1], dtype=np.float32),
            high=np.array([1.0], dtype=np.float32),
            dtype=np.float32
        )

        # -----------------------------
        # State Space
        # [current, moving_avg, trend, allocation]
        # -----------------------------
        self.observation_space = spaces.Box(
            low=0.0,
            high=np.inf,
            shape=(4,),
            dtype=np.float32
        )

        self.v = 0.5
        self.prev_v = self.v

    # -----------------------------
    # Helpers
    # -----------------------------
    def moving_avg(self, t, k=5):
        start = max(0, t - k)
        return np.mean(self.workload[start:t+1])

    def trend(self, t):
        if t == 0:
            return 0.0
        return self.workload[t] - self.workload[t - 1]

    def response_time(self, w, v):
        capacity = v * self.max_mips + 1e-6
        if w <= capacity:
            return w / capacity
        else:
            return (w / capacity) ** 2

    # -----------------------------
    # Reset
    # -----------------------------
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.t = 0
        self.v = 0.5
        self.prev_v = self.v

        return self._get_state(), {}

    # -----------------------------
    # Step
    # -----------------------------
    def step(self, action):

        # 🔥 Ensure action is valid array
        action = np.array(action, dtype=np.float32)

        # 🔥 Clip to valid range
        v = float(np.clip(action[0], 0.1, 1.0))

        w = self.workload[self.t]

        # -----------------------------
        # Safety correction
        # -----------------------------
        capacity = v * self.max_mips
        if w > capacity:
            v = min(1.0, w / self.max_mips)

        self.v = v

        # -----------------------------
        # Compute response time
        # -----------------------------
        rt = self.response_time(w, self.v)

        # -----------------------------
        # Reward components
        # -----------------------------
        cost = self.v

        sla_penalty = max(0.0, rt - self.sla) ** 2

        # 🔥 instability penalty
        instability = abs(self.v - self.prev_v)

        # final reward
        reward = -(cost + 10 * sla_penalty + 0.5 * instability)

        self.prev_v = self.v

        # -----------------------------
        # Move time
        # -----------------------------
        self.t += 1
        terminated = self.t >= len(self.workload) - 1

        return self._get_state(), reward, terminated, False, {}

    # -----------------------------
    # State
    # -----------------------------
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