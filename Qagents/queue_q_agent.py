import numpy as np


class QLearningAgent:
    """
    Tabular Q-Learning with discrete state bins.

    Q-learning is off-policy — it learns the optimal Q-value regardless
    of the exploration policy used to generate transitions. This makes it
    compatible with epsilon-greedy without the on-policy bias that would
    affect A2C or PPO if they replayed old transitions.

    State discretization
    --------------------
    The continuous 5-dim state (w, ma, trend, v, queue) is mapped to
    integer bin indices, then to a flat Q-table index. Each dimension
    gets its own number of bins specified in `state_bins`.

    5-dim state:
        0  w      — current workload   / max_workload    ∈ [0, 1]
        1  ma     — moving average     / max_workload    ∈ [0, 1]
        2  trend  — workload delta     / max_workload    ∈ [-1, 1] approx
        3  v      — allocation ratio                     ∈ [0.1, 1.0]
        4  queue  — backlog            / max_workload    ∈ [0, ∞) soft-capped

    Key design decisions
    --------------------
    - state_bins per dimension — allocate more bins to dimensions with
      finer structure. trend and queue benefit from more bins since they
      have wider dynamic ranges than w or v.
    - clip_ranges defines the expected min/max per dimension BEFORE
      binning. Values outside the range are clamped to the edge bin,
      so out-of-distribution states never cause index errors.
    - epsilon decays once per episode via decay_epsilon(), not per step.
    - Q-table initialized to zero — optimistic initialization would also
      work but zero is conservative and safe for negative-reward envs.
    - get_action() is greedy (no epsilon) — used for evaluation.
    - act() is epsilon-greedy — used during training.
    """

    # per-dimension [min, max] used for binning
    # trend can be negative; queue capped at 2.0 (normalised)
    CLIP_RANGES = [
        [0.0,  1.0],   # w      / max_workload
        [0.0,  1.0],   # ma     / max_workload
        [-0.5, 0.5],   # trend  / max_workload  (spike shifts can exceed ±0.5)
        [0.1,  1.0],   # v      allocation ratio
        [0.0,  2.0],   # queue  / max_workload  (soft cap — see env note)
    ]

    def __init__(
        self,
        state_bins,         # list of 5 ints, one per state dimension
        action_dim,
        lr            = 0.1,
        gamma         = 0.99,
        epsilon       = 1.0,
        epsilon_min   = 0.05,
        epsilon_decay = 0.995,
    ):
        assert len(state_bins) == 5, "state_bins must have exactly 5 entries (w, ma, trend, v, queue)"

        self.state_bins    = state_bins
        self.action_dim    = action_dim
        self.lr            = lr
        self.gamma         = gamma
        self.epsilon       = epsilon
        self.epsilon_min   = epsilon_min
        self.epsilon_decay = epsilon_decay

        # Q-table shape: (bins_0, bins_1, ..., bins_4, action_dim)
        self.q_table = np.zeros(state_bins + [action_dim], dtype=np.float32)

        print(
            f"QLearningAgent | state_bins={state_bins} | action_dim={action_dim} | "
            f"Q-table shape={self.q_table.shape} | "
            f"entries={self.q_table.size:,}"
        )

    # ── state discretization ──────────────────────────────────────────────────

    def _discretize(self, state):
        """
        Map continuous state vector (5,) → tuple of bin indices.
        Each dimension is clipped to its CLIP_RANGE then binned uniformly.
        """
        indices = []
        for i, (val, n_bins, (lo, hi)) in enumerate(
            zip(state, self.state_bins, self.CLIP_RANGES)
        ):
            val      = float(np.clip(val, lo, hi))
            bin_idx  = int((val - lo) / (hi - lo) * n_bins)
            bin_idx  = min(bin_idx, n_bins - 1)   # edge clamp
            indices.append(bin_idx)
        return tuple(indices)

    # ── action selection ──────────────────────────────────────────────────────

    def act(self, state):
        """Epsilon-greedy action for training."""
        if np.random.random() < self.epsilon:
            return np.random.randint(self.action_dim)
        return self.get_action(state)

    def get_action(self, state):
        """Greedy action for evaluation — no exploration."""
        idx = self._discretize(state)
        return int(np.argmax(self.q_table[idx]))

    # ── Q update ─────────────────────────────────────────────────────────────

    def update(self, state, action, reward, next_state, done):
        """
        Tabular Q-learning update (off-policy, TD(0)):

            Q(s,a) ← Q(s,a) + α · [r + γ·max_a' Q(s',a') · (1-done) - Q(s,a)]
        """
        idx      = self._discretize(state)
        next_idx = self._discretize(next_state)

        current_q = self.q_table[idx][action]
        max_next_q = 0.0 if done else float(np.max(self.q_table[next_idx]))
        target    = reward + self.gamma * max_next_q

        self.q_table[idx][action] += self.lr * (target - current_q)

    # ── epsilon decay ─────────────────────────────────────────────────────────

    def decay_epsilon(self):
        """Call once per episode — not per step."""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    # ── persistence ───────────────────────────────────────────────────────────

    def save(self, path):
        np.save(path, self.q_table)
        print(f"Q-table saved → {path}")

    def load(self, path):
        self.q_table = np.load(path)
        print(f"Q-table loaded ← {path}")