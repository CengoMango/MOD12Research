import random
from typing import List, Optional

import numpy as np

from base_dir.hyper_parameters import AgentConfig, GREEDY
from base_dir.proto1.goal_policy import GoalSpaceSWM, select_goal, update_goal_space_swm
from base_dir.proto1.knowledge_base import KnowledgeBase, RolloutRecord
from base_dir.proto1.qlearning_policy import QLearningPolicy
from base_dir.shared_files.goal_spaces import GoalSpace, create_goal_spaces, reset_goal_spaces
from base_dir.shared_files.helpers.get_plan import get_plan
from base_dir.shared_files.helpers.goal_policy_input_vector import get_goal_policy_input_vector
from base_dir.shared_files.helpers.high_level_actions import HighLevelActions, get_motion_goals

from overcooked_ai_py.agents.agent import Agent
from overcooked_ai_py.mdp.actions import Action
from overcooked_ai_py.mdp.overcooked_env import OvercookedEnv
from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld, OvercookedState
from overcooked_ai_py.planning.planners import MotionPlanner


class IMGEPAgent(Agent):
    def __init__(
        self,
        env: OvercookedEnv,
        mdp: OvercookedGridworld,
        agent_id: int,
        config: AgentConfig,
    ):
        self.agent_id = agent_id
        self.env: OvercookedEnv = env
        self.mdp: OvercookedGridworld = mdp
        self.mp: MotionPlanner = env.mp
        self.mlam = env.mlam
        self.config = config

        # ---------------- Goal spaces & KB ----------------
        self.goal_spaces: List[GoalSpace] = create_goal_spaces()
        self.kb = KnowledgeBase(len(self.goal_spaces), config=config)
        self.goal_space_swms: List[GoalSpaceSWM] = []
        self.update_goal_space_swms()

        # ---------------- Single Q-learning policy ----------------
        self.q_policy = QLearningPolicy(
            obs_dim=None,
            num_actions=len(HighLevelActions) + len(self.goal_spaces),
            config=config,
        )

        # ---------------- Rollout bookkeeping ----------------
        self.previous_state: Optional[OvercookedState] = None
        self.previous_obs: Optional[np.ndarray] = None
        self.previous_action: Optional[int] = None

        self.t = 0
        self.goal_reach_time_step: Optional[int] = None
        self.rollout_fitness = 0.0
        self.rollout_intrinsic_reward = 0.0
        self.path = []
        self.main_goal: Optional[GoalSpace] = None
        self.total_rollouts = 0  # Track total rollouts for adaptive training

    # -----------------------------------------------------
    def update_goal_space_swms(self):
        self.goal_space_swms = update_goal_space_swm(
            self.kb, self.goal_spaces, self.config
        )

    # -----------------------------------------------------
    def reset(self, mdp: Optional[OvercookedGridworld] = None, exploit: bool = False):
        super().reset()
        self.mdp = mdp
        self.t = 0
        self.path = []
        self.goal_reach_time_step = None
        self.rollout_fitness = 0.0
        self.rollout_intrinsic_reward = 0.0
        self.previous_state = None
        self.previous_obs = None
        self.previous_action = None

        reset_goal_spaces(self.goal_spaces)
        exploit = random.random() < self.config.exploit_prob

        # Select main goal (like original Proto1)
        self.main_goal, _ = select_goal(
            self.goal_space_swms,
            goal_spaces=self.goal_spaces,
            exploit=exploit,
        )
        if self.main_goal is None:
            self.main_goal = random.choice(self.goal_spaces)

    # -----------------------------------------------------
    def action(self, state: OvercookedState) -> Action:
        if self.goal_reach_time_step is not None:
            return Action.STAY

        self.t += 1
        goal = self.main_goal

        # Goal success detection
        if (
            self.previous_state
            and goal.success(self.agent_id, state, self.previous_state, self.mdp)
            and self.goal_reach_time_step is None
        ):
            self.goal_reach_time_step = self.t

        # Fitness accumulation
        if self.previous_state:
            self.rollout_fitness += goal.fitness(
                pick_step=self.goal_reach_time_step,
                state=state,
                previous_state=self.previous_state,
                agent_id=self.agent_id,
                mdp=self.mdp,
            )

        # Follow existing motion plan
        if self.path:
            step = self.path.pop(0)
            return self._return_action(step, state)

        # Observation vector
        obs_vec = get_goal_policy_input_vector(state, self.mdp, self.mp, self.agent_id)

        # Select action token via Q-learning policy
        greedy = GREEDY
        action_token = self.q_policy.select_action(obs_vec, greedy=greedy)

        # Store transition from previous step
        if self.previous_obs is not None:
            self.q_policy.store_transition(
                self.previous_obs,
                self.previous_action,
                -0.01,  # small shaping reward
                obs_vec,
                False,
            )
            
            # Periodic training during rollout to speed up convergence
            # Train every 5 steps if we have enough samples
            if self.t % 5 == 0 and len(self.q_policy.replay_buffer) >= 32:
                for _ in range(2):  # Quick training steps during rollout
                    self.q_policy.train_step()

        self.previous_obs = obs_vec
        self.previous_action = action_token

        # Interpret action token
        if action_token < len(HighLevelActions):
            token = HighLevelActions(action_token)
            motion_goals = get_motion_goals(self.mlam, self.mdp, token, state)
            self.path = get_plan(
                state.players[self.agent_id].pos_and_or,
                motion_goals,
                self.mlam,
            )
        else:
            # Action token corresponds to a goal space (currently not fully implemented)
            # For now, fall through to random action if goal action selected
            # TODO: Implement goal selection via Q-learning policy
            pass

        # If no plan, take random legal action
        if not self.path:
            return self._return_action(
                random.choice(list(Action.MOTION_ACTIONS)), state
            )

        return self._return_action(self.path.pop(0), state)

    # -----------------------------------------------------
    def _return_action(self, action, state):
        self.previous_state = state
        return action

    # -----------------------------------------------------
    def finish_rollout(self, info):
        goal = self.main_goal

        # Intrinsic reward = delta fitness vs nearest prior experiment
        prev_record = self.kb.nearest(goal=goal, k=self.config.ir_avg_prev_records, exploit=True)
        prev_f = sum(r.fitness for r in prev_record) / len(prev_record) if prev_record else 0.0
        rollout_fitness = max(0.0, self.rollout_fitness)
        fitness_difference = rollout_fitness - prev_f
        self.rollout_intrinsic_reward = max(0.0, fitness_difference)

        # Store final transition with scaled intrinsic reward to provide learning signal
        # Scale down intrinsic reward to prevent destabilizing Q-value updates
        if self.previous_obs is not None and self.previous_action is not None:
            # Scale intrinsic reward by 0.1 to keep it in reasonable range relative to step rewards
            scaled_ir = self.rollout_intrinsic_reward * 0.1
            final_reward = -0.01 + scaled_ir  # Combine shaping + scaled intrinsic reward
            self.q_policy.store_transition(
                self.previous_obs,
                self.previous_action,
                final_reward,
                None,  # No next state (rollout terminated)
                True,  # done=True
            )

        # Get previous rollout index and increment
        prev_record_idx = self.kb.nearest(1)
        rollout_idx = prev_record_idx[0].rollout_idx + 1 if prev_record_idx else 0

        # Save rollout record in knowledge base
        self.kb.add_record(
            RolloutRecord(
                goal_space_id=goal.goal_id.value,
                theta=np.zeros(1, dtype=np.float32),  # Q-learning policy does not store theta
                fitness=rollout_fitness,
                intrinsic_reward=self.rollout_intrinsic_reward,
                exploit=True,
                rollout_idx=rollout_idx,
            )
        )

        # Train Q-learning policy with adaptive number of steps
        # More training early on to speed up initial convergence
        self.total_rollouts += 1
        if self.total_rollouts < 100:
            # Early rollouts: train more aggressively
            train_steps = 30
        elif self.total_rollouts < 500:
            # Mid rollouts: moderate training
            train_steps = 20
        else:
            # Later rollouts: standard training
            train_steps = 15
        
        for _ in range(train_steps):
            self.q_policy.train_step()

        self.update_goal_space_swms()
