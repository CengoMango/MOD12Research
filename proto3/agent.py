import random
from typing import List, Optional
import numpy as np

from base_dir.hyper_parameters import AgentConfig, GREEDY
from base_dir.proto3.goal_policy import GoalSpaceSWM, select_goal, update_goal_space_swm
from base_dir.proto3.knowledge_base import KnowledgeBase, RolloutRecord
from base_dir.proto3.qlearning_policy import QLearningPolicy

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
        config: AgentConfig
    ):
        self.agent_id = agent_id
        self.env = env
        self.mdp = mdp
        self.mp: MotionPlanner = env.mp
        self.mlam = env.mlam
        self.config = config

        # ---------------- Goal spaces & KB ----------------
        self.goal_spaces: List[GoalSpace] = create_goal_spaces()
        self.kb = KnowledgeBase(len(self.goal_spaces), config=config)
        self.goal_space_swms: List[GoalSpaceSWM] = []
        self.update_goal_space_swms()

        # ---------------- Q-learning policy ----------------
        self.q_policy = QLearningPolicy(
            obs_dim=None,
            num_actions=len(HighLevelActions) + len(self.goal_spaces),
            config=config
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

        # --- restored semantics ---
        self.leader: bool = False
        self.own_episode: bool = False

        self.total_rollouts = 0

    # =====================================================
    # Goal-space utilities
    # =====================================================
    def update_goal_space_swms(self):
        self.goal_space_swms = update_goal_space_swm(
            self.kb, self.goal_spaces, self.config
        )

    # =====================================================
    # Reset
    # =====================================================
    def reset(
        self,
        mdp: Optional[OvercookedGridworld] = None,
        exploit: bool = False,
        other_goal_space_swms: Optional[List[GoalSpaceSWM]] = None,
        other_kb: Optional[KnowledgeBase] = None,
        leader: bool = False
    ):
        super().reset()

        self.mdp = mdp
        self.leader = leader

        self.t = 0
        self.path = []
        self.goal_reach_time_step = None
        self.rollout_fitness = 0.0
        self.rollout_intrinsic_reward = 0.0

        self.previous_state = None
        self.previous_obs = None
        self.previous_action = None

        reset_goal_spaces(self.goal_spaces)

        # ---------------- LEADER-AWARE GOAL SELECTION ----------------
        chosen_goal, exploit, self.own_episode = select_goal(
            self.goal_space_swms,
            goal_spaces=self.goal_spaces,
            exploit=exploit,
            other_goal_space_swms=other_goal_space_swms,
            other_kb=other_kb,
            own_kb=self.kb,
            config=self.config,
            leader=self.leader
        )

        if chosen_goal is None:
            self.main_goal = random.choice(self.goal_spaces)
        else:
            self.main_goal = chosen_goal

    # =====================================================
    # Action
    # =====================================================
    def action(self, state: OvercookedState) -> Action:
        if self.goal_reach_time_step is not None:
            return Action.STAY

        self.t += 1
        goal = self.main_goal

        # -------- Goal success detection --------
        if (
            self.previous_state
            and goal
            and goal.success(self.agent_id, state, self.previous_state, self.mdp)
            and self.goal_reach_time_step is None
        ):
            self.goal_reach_time_step = self.t

        # -------- Fitness accumulation --------
        if self.previous_state and goal:
            self.rollout_fitness += goal.fitness(
                pick_step=self.goal_reach_time_step,
                state=state,
                previous_state=self.previous_state,
                agent_id=self.agent_id,
                mdp=self.mdp
            )

        # -------- Follow existing plan --------
        if self.path:
            return self._return_action(self.path.pop(0), state)

        obs_vec = get_goal_policy_input_vector(
            state, self.mdp, self.mp, self.agent_id
        )

        # -------- Leader vs follower action selection --------
        greedy = True if self.leader else GREEDY
        action_token = self.q_policy.select_action(obs_vec, greedy=greedy)

        # Leader explores goal tokens more decisively
        if self.leader and action_token < len(HighLevelActions):
            if random.random() < 0.3:
                action_token = random.randrange(
                    len(HighLevelActions),
                    len(HighLevelActions) + len(self.goal_spaces)
                )

        # -------- Store transition --------
        if self.previous_obs is not None:
            self.q_policy.store_transition(
                self.previous_obs,
                self.previous_action,
                -0.01,
                obs_vec,
                False
            )

            if self.t % 5 == 0 and len(self.q_policy.replay_buffer) >= 32:
                for _ in range(2):
                    self.q_policy.train_step()

        self.previous_obs = obs_vec
        self.previous_action = action_token

        # -------- Interpret action token --------
        if action_token < len(HighLevelActions):
            token = HighLevelActions(action_token)
            motion_goals = get_motion_goals(self.mlam, self.mdp, token, state)
            self.path = get_plan(
                state.players[self.agent_id].pos_and_or,
                motion_goals,
                self.mlam
            )

        if not self.path:
            return self._return_action(
                random.choice(list(Action.MOTION_ACTIONS)), state
            )

        return self._return_action(self.path.pop(0), state)

    def _return_action(self, action, state):
        self.previous_state = state
        return action

    # =====================================================
    # Finish rollout
    # =====================================================
    def finish_rollout(self, info, partner_goal_id: int = -1):
        goal = self.main_goal

        prev_record = self.kb.nearest(
            goal=goal,
            k=self.config.ir_avg_prev_records,
            exploit=self.own_episode
        )
        prev_f = (
            sum(r.fitness for r in prev_record) / len(prev_record)
            if prev_record else 0.0
        )

        rollout_fitness = max(0.0, self.rollout_fitness)
        fitness_difference = rollout_fitness - prev_f
        self.rollout_intrinsic_reward = max(0.0, fitness_difference)

        # -------- Final Q-learning reward --------
        if self.previous_obs is not None and self.previous_action is not None:
            ir_scale = 0.15 if self.own_episode else 0.05
            final_reward = -0.01 + ir_scale * self.rollout_intrinsic_reward

            self.q_policy.store_transition(
                self.previous_obs,
                self.previous_action,
                final_reward,
                None,
                True
            )

        prev_idx = self.kb.nearest(1)
        rollout_idx = prev_idx[0].rollout_idx + 1 if prev_idx else 0

        # -------- LEADER-AWARE KB RECORDING --------
        self.kb.add_record(
            RolloutRecord(
                goal_space_id=goal.goal_id.value if goal else -1,
                theta=np.zeros(1, dtype=np.float32),
                fitness=rollout_fitness,
                intrinsic_reward=self.rollout_intrinsic_reward,
                exploit=True if self.own_episode else False,
                rollout_idx=rollout_idx,
                partner_goal_id=partner_goal_id
            )
        )

        # -------- Training intensity --------
        self.total_rollouts += 1
        if self.total_rollouts < 100:
            train_steps = 30
        elif self.total_rollouts < 500:
            train_steps = 20
        else:
            train_steps = 15

        if self.own_episode:
            train_steps += 5

        for _ in range(train_steps):
            self.q_policy.train_step()

        self.update_goal_space_swms()
