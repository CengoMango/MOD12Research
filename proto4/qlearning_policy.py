import random
from collections import deque
from typing import Deque, Tuple, Optional

import numpy as np

from base_dir.hyper_parameters import AgentConfig


# ─────────────────────────────────────────────────────────────
# Q-Network
# ─────────────────────────────────────────────────────────────
class QNetwork:
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        self.W1 = self._he_init(input_dim, hidden_dim)
        self.b1 = np.zeros(hidden_dim, dtype=np.float32)

        self.W2 = self._he_init(hidden_dim, output_dim)
        self.b2 = np.zeros(output_dim, dtype=np.float32)

    @staticmethod
    def _he_init(fan_in, fan_out):
        std = np.sqrt(2.0 / fan_in)
        return np.random.randn(fan_in, fan_out).astype(np.float32) * std

    def forward(self, x: np.ndarray) -> np.ndarray:
        h = np.maximum(0.0, x @ self.W1 + self.b1)
        return h @ self.W2 + self.b2

    def copy_from(self, other: "QNetwork"):
        self.W1[:] = other.W1
        self.b1[:] = other.b1
        self.W2[:] = other.W2
        self.b2[:] = other.b2


# ─────────────────────────────────────────────────────────────
# DQN Policy (Goal-aware wrapper)
# ─────────────────────────────────────────────────────────────
class QLearningPolicy:
    """
    Goal-specific Deep Q-Learning Policy:
    - Lazy network initialization
    - Experience replay with minibatch
    - Target network updates
    - Epsilon-greedy action selection with decay
    """

    def __init__(
        self,
        obs_dim: Optional[int],
        num_actions: int,
        config: AgentConfig,
        goal_id: Optional[int] = None,  # Optional goal awareness
    ):
        self.obs_dim = obs_dim
        self.num_actions = num_actions
        self.config = config
        self.goal_id = goal_id  # Can be used to tag this policy to a specific goal

        # Networks initialized lazily
        self.q_net: Optional[QNetwork] = None
        self.target_net: Optional[QNetwork] = None

        # Replay buffer
        self.replay_buffer: Deque[
            Tuple[np.ndarray, int, float, Optional[np.ndarray], bool]
        ] = deque(maxlen=config.q_learning_max_states)

        # Exploration parameters
        self.epsilon = config.q_learning_epsilon
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.995

        self.batch_size = 32
        self.train_steps = 0

    # ─────────────────────────────────────────────────────────
    def _maybe_init_networks(self, obs_vec: np.ndarray):
        """Initialize networks on first observation."""
        if self.q_net is not None:
            return

        self.obs_dim = obs_vec.shape[0]
        self.q_net = QNetwork(self.obs_dim, self.config.q_hidden_dim, self.num_actions)
        self.target_net = QNetwork(self.obs_dim, self.config.q_hidden_dim, self.num_actions)
        self.target_net.copy_from(self.q_net)

    # ─────────────────────────────────────────────────────────
    def select_action(self, obs_vec: np.ndarray, greedy: bool) -> int:
        obs_vec = obs_vec.astype(np.float32)
        self._maybe_init_networks(obs_vec)

        if not greedy and random.random() < self.epsilon:
            return random.randrange(self.num_actions)

        q_vals = self.q_net.forward(obs_vec)
        return int(np.argmax(q_vals))

    # ─────────────────────────────────────────────────────────
    def store_transition(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: Optional[np.ndarray],
        done: bool
    ):
        self.replay_buffer.append(
            (
                state.astype(np.float32),
                action,
                reward,
                None if next_state is None else next_state.astype(np.float32),
                done,
            )
        )

    # ─────────────────────────────────────────────────────────
    def train_step(self):
        if self.q_net is None or len(self.replay_buffer) < self.batch_size:
            return

        batch = random.sample(self.replay_buffer, self.batch_size)

        for state, action, reward, next_state, done in batch:
            q_vals = self.q_net.forward(state)
            q_sa = q_vals[action]

            # Compute TD target
            if done or next_state is None:
                target = reward
            else:
                target = reward + self.config.q_learning_gamma * np.max(
                    self.target_net.forward(next_state)
                )

            td_error = q_sa - target

            # Forward pass for gradient computation
            h = np.maximum(0.0, state @ self.q_net.W1 + self.q_net.b1)

            # Compute gradients
            grad_W2 = np.zeros_like(self.q_net.W2)
            grad_b2 = np.zeros_like(self.q_net.b2)
            grad_W2[:, action] = h * td_error
            grad_b2[action] = td_error

            dh = self.q_net.W2[:, action] * td_error
            dh[h <= 0] = 0.0

            grad_W1 = np.outer(state, dh)
            grad_b1 = dh

            # Gradient descent step
            lr = self.config.q_learning_lr
            self.q_net.W1 -= lr * grad_W1
            self.q_net.b1 -= lr * grad_b1
            self.q_net.W2 -= lr * grad_W2
            self.q_net.b2 -= lr * grad_b2

        self.train_steps += 1

        # Periodically sync target network
        if self.train_steps % 200 == 0:
            self.target_net.copy_from(self.q_net)

        # Decay epsilon
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
