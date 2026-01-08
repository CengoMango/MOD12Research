from dataclasses import dataclass
from typing import List, Optional

import numpy as np
from numpy.typing import NDArray

from base_dir.hyper_parameters import AgentConfig
from base_dir.shared_files.goal_spaces import GoalSpace
from base_dir.shared_files.helpers.goal_policy_input_vector import GOAL_POLICY_INPUT_VECTOR_SIZE
from base_dir.shared_files.helpers.high_level_actions import HighLevelActions


@dataclass
class RolloutRecord:
    goal_space_id: int
    theta: NDArray[np.floating]  # policy parameters
    partner_goal_id: int
    fitness: float
    intrinsic_reward: float
    exploit: bool = False
    rollout_idx: int = 0


class KnowledgeBase:
    """
    In-memory buffer implemented with **fixed-width NumPy arrays** for
    lightning-fast filtering on `goal_space_id`, `exploit`, `fitness`, etc.

    ▸  Theta *must* have the same length (`param_dim`) for every record.
       If that is not true in your set-up, keep using the list version or
       store a flattened/ padded copy of theta.
    """

    # --------------------------------------------------------------------- #
    # construction / memory management
    # --------------------------------------------------------------------- #
    def __init__(self, goal_spaces_length, config: AgentConfig, init_capacity: int = 2048, ) -> None:
        num_tokens = len(HighLevelActions) + goal_spaces_length
        self.param_dim = (GOAL_POLICY_INPUT_VECTOR_SIZE * config.neuro_policy_hidden_dim  # W1
                          + config.neuro_policy_hidden_dim  # b1
                          + config.neuro_policy_hidden_dim * num_tokens  # W2
                          + num_tokens)  # b2
        self._capacity = init_capacity
        self._size = 0

        # allocate empty arrays up-front so that `add_record` is O(1) amortised
        self._goal_id = np.empty(init_capacity, dtype=np.int32)
        self._fitness = np.empty(init_capacity, dtype=np.float32)
        self._intr_reward = np.empty(init_capacity, dtype=np.float32)
        self._exploit = np.empty(init_capacity, dtype=bool)
        self._rollout_idx = np.empty(init_capacity, dtype=np.int32)
        self._theta = np.empty((init_capacity, self.param_dim),
                               dtype=np.float32)
        self._partner_goal = np.empty(init_capacity, dtype=np.int32)

    def _grow(self) -> None:
        """Double the storage capacity (called automatically)."""
        new_cap = self._capacity * 2
        self._goal_id = np.resize(self._goal_id, new_cap)
        self._fitness = np.resize(self._fitness, new_cap)
        self._intr_reward = np.resize(self._intr_reward, new_cap)
        self._exploit = np.resize(self._exploit, new_cap)
        self._rollout_idx = np.resize(self._rollout_idx, new_cap)
        self._theta = np.resize(self._theta,
                                (new_cap, self.param_dim))
        self._capacity = new_cap
        self._partner_goal = np.resize(self._partner_goal, new_cap)

    # --------------------------------------------------------------------- #
    # public API
    # --------------------------------------------------------------------- #
    def add_record(self, rec: RolloutRecord) -> None:
        """Append a new rollout to the buffer (O(1) amortised)."""
        if self._size == self._capacity:
            self._grow()

        i = self._size
        self._goal_id[i] = rec.goal_space_id
        self._fitness[i] = rec.fitness
        self._intr_reward[i] = rec.intrinsic_reward
        self._exploit[i] = rec.exploit
        self._rollout_idx[i] = rec.rollout_idx
        self._theta[i] = rec.theta
        self._partner_goal[i] = rec.partner_goal_id
        self._size += 1

    def nearest(
            self,
            k: int,
            exploit: Optional[bool] = None,
            goal: Optional[GoalSpace] = None,
    ) -> List[RolloutRecord]:
        """
        Return the **k most-recent** records matching the (optional) filters.
        Executed entirely in NumPy, then converted back to dataclass objects.
        """

        n = self._size
        if n == 0 or k <= 0:
            return []

        # build boolean mask: start with all True
        mask = np.ones(n, dtype=bool)
        if exploit is not None:
            mask &= (self._exploit[:n] == exploit)
        if goal is not None:
            mask &= (self._goal_id[:n] == goal.goal_id)

        # indices of rows that match, newest → oldest
        idx = np.flatnonzero(mask)[::-1]  # reverse chronological
        idx = idx[:k]  # keep at most k

        return [self._make_record(i) for i in idx]

    # ------------------------------------------------------------------ #
    # utilities
    # ------------------------------------------------------------------ #
    def __len__(self) -> int:
        return self._size

    def _make_record(self, i: int) -> RolloutRecord:
        """Internal helper to rebuild a RolloutRecord dataclass."""
        return RolloutRecord(
            goal_space_id=int(self._goal_id[i]),
            theta=self._theta[i].copy(),  # copy to keep immutable
            fitness=float(self._fitness[i]),
            intrinsic_reward=float(self._intr_reward[i]),
            exploit=bool(self._exploit[i]),
            rollout_idx=int(self._rollout_idx[i]),
            partner_goal_id=int(self._partner_goal[i]),
        )

    # ------------------------------------------------------------------ #
    # save & load
    # ------------------------------------------------------------------ #
    def save_buffer(self, path: str) -> None:
        """
        Saves the raw NumPy arrays – far smaller & faster than pickling
        a Python list of objects.
        """
        np.savez_compressed(
            path,
            param_dim=self.param_dim,
            size=self._size,
            goal_id=self._goal_id[:self._size],
            fitness=self._fitness[:self._size],
            intr_reward=self._intr_reward[:self._size],
            exploit=self._exploit[:self._size],
            rollout_idx=self._rollout_idx[:self._size],
            partner_goal=self._partner_goal[:self._size],
            theta=self._theta[:self._size],
        )

    @classmethod
    def load_buffer(cls, path: str, config: AgentConfig) -> "KnowledgeBase":
        data = np.load(path, allow_pickle=True)

        # 1️⃣  Build an *empty* KB with the **correct** param_dim, not “goal_spaces_length”
        kb = cls(goal_spaces_length=0, config=config)  # dummy
        kb.param_dim = int(data["param_dim"])  # ← overwrite

        # 2️⃣  Allocate arrays with that exact width
        kb._theta = np.empty((int(data["size"]), kb.param_dim), dtype=np.float32)

        # ---- copy payload ---------------------------------------------------
        kb._size = int(data["size"])
        kb._goal_id = data["goal_id"]
        kb._fitness = data["fitness"]
        kb._intr_reward = data["intr_reward"]
        kb._exploit = data["exploit"]
        kb._rollout_idx = data["rollout_idx"]
        kb._theta[:] = data["theta"]
        kb._capacity = kb._size
        kb._partner_goal = data["partner_goal"]
        return kb
