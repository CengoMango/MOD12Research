import copy
import os
import random

from base_dir.hyper_parameters import AgentConfig, HORIZON, LAYOUT_ID, LOAD_KB, ROLLOUTS, USE_COUNTERS
from base_dir.visualise.create_gif import create_gif
from overcooked_ai_py.data.layouts.layouts import layouts
from overcooked_ai_py.mdp.overcooked_env import OvercookedEnv
from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld

layout_name = layouts[LAYOUT_ID]
config = AgentConfig()
# ---------------------------------------------------------------- env + agents
from base_dir.proto1.agent import IMGEPAgent

mdp: OvercookedGridworld = OvercookedGridworld.from_layout_name(layout_name)
if USE_COUNTERS:
    base_params = {
        "start_orientations": False,
        "wait_allowed": False,
        "counter_goals": mdp.terrain_pos_dict["X"],
        "counter_drop": mdp.terrain_pos_dict["X"],
        "counter_pickup": mdp.terrain_pos_dict["X"],
        "same_motion_goals": True,
    }
else:
    base_params = {
        "start_orientations": False,
        "wait_allowed": False,
        "counter_goals": [],
        "counter_drop": [],
        "counter_pickup": [],
        "same_motion_goals": True,
    }
env: OvercookedEnv = OvercookedEnv.from_mdp(mdp, horizon=HORIZON, info_level=0, mlam_params=base_params)

mp = env.mp
mlam = env.mlam

agents = [IMGEPAgent(env, mdp, agent_id, config=config) for agent_id in range(2)]
if LOAD_KB:
    for ag in agents:
        kb_path = f"../kb/buffer_rollouts{ag.agent_id}.npz"
        ag.kb = ag.kb.load_buffer(kb_path, config=config)
        ag.update_goal_space_swms()
# ---------------------------------------------------------------- run one roll-out
for roll in range(ROLLOUTS):
    print(roll)
    env.reset(regen_mdp=True)
    exploit = random.random() < config.exploit_prob
    for ag in agents: ag.reset(mdp=mdp, exploit=exploit)
    done = False
    state = env.state
    # -------- record trajectory -----------------------------------------
    ep_states = [copy.deepcopy(state)]  # include start state
    while not done:
        joint = [ag.action(state) for ag in agents]
        state, _, done, info = env.step(joint)
        ep_states.append(copy.deepcopy(state))  # save each next state

    # -------- finish roll-out bookkeeping -------------------------------
    for idx, ag in enumerate(agents):
        ag.finish_rollout(info)

    if roll == ROLLOUTS - 1:
        create_gif(ep_states, mdp, roll, False, 'save_dir')


def get_next_run_dir(base_dir):
    os.makedirs(base_dir, exist_ok=True)
    existing_runs = [
        int(d) for d in os.listdir(base_dir)
        if d.isdigit() and os.path.isdir(os.path.join(base_dir, d))
    ]
    next_index = max(existing_runs, default=0) + 1
    new_dir = os.path.join(base_dir, str(next_index))
    os.makedirs(new_dir)
    return new_dir


# === Usage ===

base_dir = f"base_dir/proto1/layout{LAYOUT_ID}/buffer_rollouts"
save_dir = get_next_run_dir(base_dir)

for ag in agents:
    ag.kb.save_buffer(os.path.join(save_dir, f"buffer_agent{ag.agent_id}"))

# make_graphs()
