# neuro_policy.py
import random
from typing import List

import numpy as np

from base_dir.hyper_parameters import AgentConfig
from base_dir.proto3.knowledge_base import KnowledgeBase
from base_dir.shared_files.goal_spaces import GoalSpace
from base_dir.shared_files.helpers.goal_policy_input_vector import GOAL_POLICY_INPUT_VECTOR_SIZE
from base_dir.shared_files.helpers.high_level_actions import HighLevelActions


def he_init(fan_in: int, fan_out: int) -> np.ndarray:
    """Kaiming-He init for a linear layer (ReLU)."""
    std = np.sqrt(2.0 / fan_in)
    return np.random.randn(fan_in, fan_out).astype(np.float32) * std


class NeuroPolicy:
    def __init__(self, goal_spaces: List[GoalSpace], config: AgentConfig, theta: np.ndarray | None = None):

        self.inp_dim = GOAL_POLICY_INPUT_VECTOR_SIZE
        self.hidden_dim = config.neuro_policy_hidden_dim
        self.num_tokens = len(HighLevelActions) + len(goal_spaces)

        if theta is None:  # fresh initialization
            W1 = he_init(self.inp_dim, self.hidden_dim)
            b1 = np.zeros(self.hidden_dim, dtype=np.float32)
            W2 = he_init(self.hidden_dim, self.num_tokens)
            b2 = np.zeros(self.num_tokens, dtype=np.float32)
            self.theta = self._pack(W1, b1, W2, b2)
        else:  # copy provided weights
            self.theta = theta.astype(np.float32)

    # ──────────────────────────── public ──────────────────────────────
    def select_token(self,
                     obs_vec: np.ndarray,
                     greedy: bool) -> int:

        x = np.concatenate([obs_vec]).astype(np.float32)

        W1, b1, W2, b2 = self._unpack()
        h = np.maximum(0.0, x @ W1 + b1)  # ReLU
        z = h @ W2 + b2  # logits  (num_tokens,)

        if greedy:
            return int(np.argmax(z))
        else:
            p = np.exp(z - z.max(), dtype=np.float32)
            p /= p.sum()

            return int(np.random.choice(self.num_tokens, p=p))

    # ────────────────────────── helpers ───────────────────────────────
    def _pack(self, W1, b1, W2, b2) -> np.ndarray:
        return np.concatenate([W1.ravel(), b1, W2.ravel(), b2])

    def _unpack(self):
        """Recover weight matrices from flat θ."""
        D, H, A = self.inp_dim, self.hidden_dim, self.num_tokens
        i = 0
        W1 = self.theta[i:i + D * H].reshape(D, H)
        i += D * H
        b1 = self.theta[i:i + H]
        i += H
        W2 = self.theta[i:i + H * A].reshape(H, A)
        i += H * A
        b2 = self.theta[i:i + A]
        return W1, b1, W2, b2


class GoalSpacePolicy:
    """
    A NeuroPolicy that is specialized for a specific goal space.
    """

    def __init__(self, goal: GoalSpace, neuro_policy: NeuroPolicy, exploit: bool):
        self.goal = goal
        self.neuro_policy = neuro_policy
        self.exploit = exploit


def get_policy(selected_goal: GoalSpace, kb: KnowledgeBase, goal_spaces: List[GoalSpace], exploit: bool,
                     config: AgentConfig) -> NeuroPolicy:
    if not kb.nearest(goal=selected_goal, k=1):
        return NeuroPolicy(goal_spaces, config)  # no records for this goal, return a fresh policy
    if exploit:
        rec = kb.nearest(goal=selected_goal, k=config.parent_policy_recent)
        # exploit: use the best policy from the knowledge base
        best_idx = np.argmax([r.fitness for r in rec])
        return NeuroPolicy(theta=rec[best_idx].theta, goal_spaces=goal_spaces, config=config)

    rec = kb.nearest(goal=selected_goal, k=config.mutate_records)
    # explore: pick a random parent policy to mutate from
    parent_policy = random.choice(rec)
    adaptive_noise = config.adaptive_noise_std / (
            max(0, parent_policy.intrinsic_reward) + config.adaptive_noise_std)  # more noise when progress is low
    child_theta = parent_policy.theta + np.random.normal(0, adaptive_noise, parent_policy.theta.shape)
    return NeuroPolicy(theta=child_theta, goal_spaces=goal_spaces, config=config)
