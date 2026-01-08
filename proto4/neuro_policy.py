# neuro_policy.py
import random
from typing import List

import numpy as np

from base_dir.hyper_parameters import AgentConfig
from base_dir.proto4.knowledge_base import KnowledgeBase
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

        # Q-learning mode toggle (optional field on AgentConfig)
        # If config.use_q_learning is True, select_token will use epsilon-greedy and q_update is available.
        self.use_q_learning = getattr(config, "use_q_learning", False)
        # Q-learning hyperparams (optional)
        self.q_epsilon = getattr(config, "q_epsilon", 0.1)
        self.q_lr = getattr(config, "q_lr", 1e-2)
        self.q_gamma = getattr(config, "q_gamma", 0.99)

        if theta is None:  # fresh initialization
            W1 = he_init(self.inp_dim, self.hidden_dim)
            b1 = np.zeros(self.hidden_dim, dtype=np.float32)
            W2 = he_init(self.hidden_dim, self.num_tokens)
            b2 = np.zeros(self.num_tokens, dtype=np.float32)
            self.theta = self._pack(W1, b1, W2, b2)
        else:  # copy provided weights
            # ensure float32 type
            self.theta = theta.astype(np.float32)

    # ──────────────────────────── public ──────────────────────────────
    def select_token(self,
                     obs_vec: np.ndarray,
                     greedy: bool) -> int:

        x = np.concatenate([obs_vec]).astype(np.float32)

        # Forward through MLP
        W1, b1, W2, b2 = self._unpack()
        h = np.maximum(0.0, x @ W1 + b1)  # ReLU
        z = h @ W2 + b2  # logits / Q-values depending on mode  (num_tokens,)

        if self.use_q_learning:
            # epsilon-greedy behaviour for Q-learning
            eps = self.q_epsilon
            if greedy:
                return int(np.argmax(z))
            else:
                if random.random() < eps:
                    return int(random.randrange(self.num_tokens))
                else:
                    return int(np.argmax(z))
        else:
            # original behaviour: greedy argmax or softmax sampling
            if greedy:
                return int(np.argmax(z))
            else:
                p = np.exp(z - z.max(), dtype=np.float32)
                p /= p.sum()
                return int(np.random.choice(self.num_tokens, p=p))

    # Single-step Q-learning update on the MLP parameters (TD(0))
    def q_update(self,
                 obs: np.ndarray,
                 action: int,
                 reward: float,
                 next_obs: np.ndarray,
                 done: bool,
                 lr: float | None = None,
                 gamma: float | None = None) -> float:
        """
        Perform one TD(0) update of the network parameters to learn Q-values.
        Uses the same 2-layer MLP as function approximator.

        Returns the TD error (target - current) for monitoring.
        """
        if lr is None:
            lr = self.q_lr
        if gamma is None:
            gamma = self.q_gamma

        # Forward current
        x = np.concatenate([obs]).astype(np.float32)  # shape (D,)
        W1, b1, W2, b2 = self._unpack()
        h_pre = x @ W1 + b1  # pre-activation (H,)
        h = np.maximum(0.0, h_pre)  # ReLU activation (H,)
        z = h @ W2 + b2  # Q-values for current state (A,)

        q_curr = float(z[action])

        # Forward next
        x_next = np.concatenate([next_obs]).astype(np.float32)
        h_next_pre = x_next @ W1 + b1
        h_next = np.maximum(0.0, h_next_pre)
        z_next = h_next @ W2 + b2
        q_next_max = float(np.max(z_next)) if not done else 0.0

        # TD target and error
        target = reward + (0.0 if done else gamma * q_next_max)
        td_error = target - q_curr

        # Gradient of 0.5 * error^2 -> -td_error * grad(q_curr)
        # grad w.r.t outputs z: dL/dz[action] = -td_error, others 0
        dz = np.zeros_like(z, dtype=np.float32)
        dz[action] = -td_error  # shape (A,)

        # Gradients to W2, b2
        # dW2 = h[:, None] @ dz[None, :]
        dW2 = np.outer(h, dz)  # shape (H, A)
        db2 = dz  # shape (A,)

        # Backprop into h: dh = W2 @ dz
        dh = W2 @ dz  # shape (H,)

        # Backprop through ReLU: dh_pre = dh * (h_pre > 0)
        dh_pre = dh * (h_pre > 0).astype(np.float32)

        # Gradients to W1, b1
        dW1 = np.outer(x, dh_pre)  # shape (D, H)
        db1 = dh_pre  # shape (H,)

        # Update parameters (gradient descent)
        W1 = W1 - lr * dW1
        b1 = b1 - lr * db1
        W2 = W2 - lr * dW2
        b2 = b2 - lr * db2

        # Pack back into theta
        self.theta = self._pack(W1, b1, W2, b2)

        # Return the TD error (positive when target > current)
        return float(td_error)

    # ────────────────────────── helpers ───────────────────────────────
    def _pack(self, W1, b1, W2, b2) -> np.ndarray:
        return np.concatenate([W1.ravel(), b1, W2.ravel(), b2]).astype(np.float32)

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
