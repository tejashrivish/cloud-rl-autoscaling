import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CloudAutoScalingEnvContinuous(gym.Env):
    """
    Continuous-action cloud auto-scaling environment with queue carryover.

    Action space  : Box([0.1], [1.0]) — agent sets allocation ratio v directly
    Observation   : 5-dim vector [w, ma, trend, v, queue] (all normalised)

    Queue mechanics
    ---------------
    Unserved requests at timestep t carry forward to t+1:
        total_demand = w_t + queue_{t-1}
        served       = min(total_demand, capacity)
        queue_t      = total_demand - served

    This makes the environment temporally coupled — a bad action at t
    accumulates backlog that punishes future steps even if v recovers.

    Reward
    ------
        reward = -(cost + 10·sla_penalty + 0.5·instability)

        cost        = v                              (over-provisioning)
        sla_penalty = max(0, rt - sla)²             (under-provisioning)
        instability = |v - prev_v|                  (thrashing penalty)
        rt          = total_demand / capacity        (queue-aware)

    Safety correction
    -----------------
    If the agent sets v too low for the current total_demand, v is raised
    to the minimum feasible value before the queue is computed. This
    prevents the queue from exploding in early training when the policy
    is random, while still giving the agent a meaningful signal.
    """

    def __init__(self, workload, sla=1.0, max_mips=1000):
        super().__init__()

        self.workload     = workload.astype(np.float32)
        self.max_workload = float(np.max(self.workload))
        self.sla          = sla
        self.max_mips     = max_mips

        self.t      = 0
        self.v      = 0.5
        self.prev_v = 0.5
        self.queue  = 0.0   # unserved requests carried forward

        # continuous action: agent sets v ∈ [0.1, 1.0] directly
        self.action_space = spaces.Box(
            low  = np.array([0.1], dtype=np.float32),
            high = np.array([1.0], dtype=np.float32),
            dtype= np.float32,
        )

        # [w, moving_avg, trend, v, queue] — all normalised
        self.observation_space = spaces.Box(
            low  = -np.inf,
            high =  np.inf,
            shape= (5,),
            dtype= np.float32,
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
        self.t      = 0
        self.v      = 0.5
        self.prev_v = 0.5
        self.queue  = 0.0
        return self._get_state(), {}

    def step(self, action):
        # parse and clip action
        action = np.array(action, dtype=np.float32)
        v      = float(np.clip(action[0], 0.1, 1.0))

        w        = self.workload[self.t]
        capacity = v * self.max_mips

        # ── safety correction ──────────────────────────────────────────────────
        # if total demand (current + backlog) exceeds requested capacity,
        # raise v to the minimum that can serve at least the current workload.
        # This prevents the queue from exploding on random early-training actions.
        total_demand = w + self.queue
        if total_demand > capacity:
            v        = float(np.clip(total_demand / self.max_mips, 0.1, 1.0))
            capacity = v * self.max_mips

        self.v = v

        # ── queue mechanics ────────────────────────────────────────────────────
        served     = min(total_demand, capacity)
        self.queue = max(0.0, total_demand - served)

        # rt rises naturally as queue grows — no squared branch needed
        rt          = total_demand / (capacity + 1e-6)
        sla_penalty = max(0.0, rt - self.sla) ** 2

        # ── reward ─────────────────────────────────────────────────────────────
        cost        = self.v
        instability = abs(self.v - self.prev_v)
        reward      = -(cost + 10.0 * sla_penalty + 0.5 * instability)

        self.prev_v = self.v
        self.t     += 1
        terminated  = self.t >= len(self.workload) - 1

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
            self.queue  / (2.0 * self.max_workload),  # soft-capped: queue can exceed max_workload under DR
        ], dtype=np.float32)