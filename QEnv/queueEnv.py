import gymnasium as gym
from gymnasium import spaces
import numpy as np


class CloudAutoScalingEnv(gym.Env):
    def __init__(self, workload, sla=1.0, max_mips=1000):
        super().__init__()

        self.workload     = workload.astype(np.float32)
        self.max_workload = np.max(workload)
        self.sla          = sla
        self.max_mips     = max_mips

        self.t     = 0
        self.v     = 0.5
        self.queue = 0.0  # unserved requests carried forward

        self.action_space = spaces.Discrete(3)

        # [current, moving_avg, trend, allocation, queue]
        self.observation_space = spaces.Box(
            low=0.0, high=np.inf, shape=(5,), dtype=np.float32
        )

    # ── internal helpers ──────────────────────────────────────────────────────

    def _moving_avg(self, t, k=5):
        start = max(0, t - k)
        return float(np.mean(self.workload[start:t + 1]))

    def _trend(self, t):
        if t == 0:
            return 0.0
        return float(self.workload[t] - self.workload[t - 1])

    # ── gym interface ─────────────────────────────────────────────────────────

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.t     = 0
        self.v     = 0.5
        self.queue = 0.0
        return self._get_state(), {}

    def step(self, action):
        w      = self.workload[self.t]
        action = self.safe_action(action, w)

        # apply scaling delta
        if action == 0:
            self.v = max(0.1, self.v - 0.1)
        elif action == 2:
            self.v = min(1.0, self.v + 0.1)

        capacity     = self.v * self.max_mips
        total_demand = w + self.queue           # current + backlog

        served     = min(total_demand, capacity)
        self.queue = max(0.0, total_demand - served)  # remainder carries forward

        # rt rises naturally as queue grows — no squared branch needed
        rt          = total_demand / (capacity + 1e-6)
        sla_penalty = max(0.0, rt - self.sla) ** 2
        reward      = -(self.v + 10.0 * sla_penalty)

        self.t    += 1
        terminated = self.t >= len(self.workload) - 1

        return self._get_state(), reward, terminated, False, {}

    def _get_state(self):
        t  = min(self.t, len(self.workload) - 1)
        w  = self.workload[t]
        ma = self._moving_avg(t)
        tr = self._trend(t)

        return np.array([
            w           / self.max_workload,
            ma          / self.max_workload,
            tr          / self.max_workload,
            self.v,
            self.queue  / self.max_workload,
        ], dtype=np.float32)

    def safe_action(self, action, w):
        """Block scale-down when total demand (including backlog) exceeds capacity."""
        capacity = self.v * self.max_mips
        if (w + self.queue) > capacity and action == 0:
            return 1  # force no-op
        return action