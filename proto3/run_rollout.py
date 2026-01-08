import copy
import os
import random
import tempfile
from pathlib import Path

from base_dir.hyper_parameters import AgentConfig, HORIZON, LAYOUT_ID, LOAD_KB, ROLLOUTS, USE_COUNTERS
from overcooked_ai_py.data.layouts.layouts import layouts
from base_dir.visualise.create_gif import create_gif
from overcooked_ai_py.mdp.overcooked_env import OvercookedEnv
from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld

layout_name = layouts[LAYOUT_ID]
config = AgentConfig()
config.use_q_learning = True  # Set True or False

# ----------------------------------------------------------------
# Import agent
from base_dir.proto3.agent import IMGEPAgent

# ----------------------------------------------------------------
# Create MDP and environment
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

# ----------------------------------------------------------------
# Initialize agents
agents = [IMGEPAgent(env, mdp, agent_id, config=config) for agent_id in range(2)]

# Load KB if required
if LOAD_KB:
    for ag in agents:
        kb_path = f"../kb/buffer_rollouts{ag.agent_id}.npz"
        ag.kb = ag.kb.load_buffer(kb_path, config=config)
        ag.update_goal_space_swms()

# ----------------------------------------------------------------
# Run rollouts
leader = 0
for roll in range(ROLLOUTS):
    print(f"{roll}")
    env.reset(regen_mdp=True)
    exploit = random.random() < config.exploit_prob
    leader = 1 - leader  # alternate leader

    # Reset agents
    for ag in agents:
        ag.reset(
            mdp=mdp,
            exploit=exploit,
            other_goal_space_swms=agents[1 - ag.agent_id].goal_space_swms,
            other_kb=agents[1 - ag.agent_id].kb,
            leader=ag.agent_id == leader
        )

    done = False
    state = env.state
    ep_states = [copy.deepcopy(state)]

    while not done:
        joint = [ag.action(state) for ag in agents]
        state, _, done, info = env.step(joint)
        ep_states.append(copy.deepcopy(state))

    # Finish roll-out bookkeeping
    for idx, ag in enumerate(agents):
        # Use main_goal.goal_id instead of goal_policy_stack
        partner_goal_id = agents[1 - idx].main_goal.goal_id.value if agents[1 - idx].main_goal else -1
        ag.finish_rollout(info, partner_goal_id=partner_goal_id)

    # Save GIF of last rollout
    if roll == ROLLOUTS - 1:
        print("Last roll-out, saving GIF...")
        create_gif(ep_states, mdp, roll, False, f"layout{LAYOUT_ID}/buffer_rollouts/rollout_{roll}")

# ----------------------------------------------------------------
# Create unique directory for buffers
def get_next_run_dir(base_dir):
    os.makedirs(base_dir, exist_ok=True)
    existing_runs = [
        int(d) for d in os.listdir(base_dir)
        if d.isdigit() and os.path.isdir(os.path.join(base_dir, d))
    ]
    next_index = max(existing_runs, default=0) + 1
    new_dir = os.path.join(base_dir, str(next_index))
    os.makedirs(new_dir, exist_ok=True)
    return new_dir

# ----------------------------------------------------------------
# Save buffers
BASE_DIR = Path(__file__).resolve().parents[1]
proto_name = Path(__file__).parent.name
base_dir = os.path.join(str(BASE_DIR), proto_name, f"layout{LAYOUT_ID}", "buffer_rollouts")
save_dir = get_next_run_dir(base_dir)

for ag in agents:
    ag.kb.save_buffer(os.path.join(save_dir, f"buffer_agent{ag.agent_id}"))

print(f"Buffers saved to {save_dir}")
