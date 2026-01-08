import math
import random
from collections import defaultdict
from typing import Dict, List, Tuple

from base_dir.hyper_parameters import AgentConfig, IR_BONUS_CAP, IR_BONUS_SLOPE
from base_dir.proto4.knowledge_base import KnowledgeBase, RolloutRecord
from base_dir.shared_files.goal_spaces import GoalSpace, get_goal_by_goal_id
from base_dir.shared_files.helpers.softmax_sampler import GoalSpaceSWM, _softmax_sample

EPSILON = 1e-6  # numeric tolerance


def _top_swm_candidates(gs_swms: List[GoalSpaceSWM], eps: float = EPSILON) -> List[GoalSpaceSWM]:
    """Return all GS-EMA objects whose EMA is within eps of the maximum."""
    if not gs_swms:
        return []
    m = max(gs.swm for gs in gs_swms)
    return [gs for gs in gs_swms if abs(gs.swm - m) < eps]


def select_goal(
        own_goal_space_swms: List[GoalSpaceSWM],
        other_goal_space_swms: List[GoalSpaceSWM],
        other_kb: KnowledgeBase,
        goal_spaces: List[GoalSpace],
        own_kb: KnowledgeBase,
        config: AgentConfig,
        exploit: bool = False,
        leader: bool = False,  # <- added for compatibility with proto3-style callers
) -> Tuple[GoalSpace, bool, bool]:
    """
    Select a goal space to pursue.

    Returns:
      (chosen_goal, exploit_flag, own_episode_flag)

    Notes:
      - `leader` is accepted for compatibility; the current implementation does not use it,
        but callers (e.g. agent.reset) may pass it as a keyword.
    """
    # ---------- 0) trivial guards ----------
    if not own_goal_space_swms:
        raise ValueError("select_goal() called with empty own_goal_space_emas")

    # ---------- 1) how strong is everyone’s motivation? ----------
    own_max = max(gs.swm for gs in own_goal_space_swms)
    other_max = max(gs.swm for gs in other_goal_space_swms) if other_goal_space_swms else -math.inf
    our_turn = (own_max + EPSILON) >= other_max

    # ---------- 2) OUR TURN --------------------------------------------------
    if our_turn or not other_goal_space_swms:
        if exploit:  # *greedy* path
            # filter near-best goals and pick one uniformly
            best = [gs for gs in own_goal_space_swms
                    if abs(gs.swm - own_max) < EPSILON]
            chosen_gs = random.choice(best)
        else:  # *exploration* path
            chosen_gs = _softmax_sample(
                own_goal_space_swms,
                tau=getattr(config, "softmax_temperature", 0.3)
            )
        return get_goal_by_goal_id(chosen_gs.goal.goal_id, goal_spaces), exploit, True

    # ---------- 3) PARTNER ACCOMMODATION ------------------------------------
    partner_gs = random.choice(other_goal_space_swms)  # they are already EMA-sorted
    partner_goal = partner_gs.goal

    # ask KB whether we have a mapping that helps them
    lookups = [
        other_kb.nearest(1, goal=partner_goal, exploit=True),
        other_kb.nearest(1, goal=partner_goal, exploit=False),
    ]
    other_recent = next((res for res in lookups if res), [])
    if other_recent:
        mapped_id = other_recent[0].partner_goal_id
        mapped_goal = get_goal_by_goal_id(mapped_id, goal_spaces)

        ours = own_kb.nearest(10, goal=mapped_goal, exploit=True)
        if ours and max(r.fitness for r in ours) > config.minimum_goal_fitness:
            # we help partner → always exploit
            return mapped_goal, True, False

    # ---------- 4) fallback to our side -------------------------------------
    # exploit? greedy; explore? soft-max.
    if exploit:
        fallback = max(own_goal_space_swms, key=lambda g: g.swm)
    else:
        fallback = _softmax_sample(
            own_goal_space_swms,
            tau=getattr(config, "softmax_temperature", 0.3)
        )
    return get_goal_by_goal_id(fallback.goal.goal_id, goal_spaces), exploit, True


def update_goal_space_swm(
        kb: KnowledgeBase,
        goal_spaces: List[GoalSpace],
        config: AgentConfig,
) -> List[GoalSpaceSWM]:
    """
    Compute an (approximate) EMA of intrinsic reward per goal space,
    adding a small bonus for sparsely tried goals so they remain eligible.
    """
    # 1) get the N most recent exploit rollouts
    recent: List[RolloutRecord] = kb.nearest(config.n_recent, exploit=True)

    # 2) bucket by goal_space_id
    by_goal: Dict[int, List[RolloutRecord]] = defaultdict(list)
    for r in recent:
        by_goal[r.goal_space_id].append(r)

    # 3) compute a per-goal EMA + bonus
    bonus_cutoff = int(IR_BONUS_CAP / IR_BONUS_SLOPE)
    out: List[GoalSpaceSWM] = []
    for g in goal_spaces:
        recs = by_goal.get(g.goal_id, [])
        n = len(recs)
        avg_ir = (sum(r.intrinsic_reward for r in recs) / n) if n > 0 else 0.0

        # if very sparse, give a decreasing bonus
        if n < bonus_cutoff:
            bonus = IR_BONUS_CAP - IR_BONUS_SLOPE * n
            avg_ir += bonus

        out.append(GoalSpaceSWM(g, avg_ir))

    return out
