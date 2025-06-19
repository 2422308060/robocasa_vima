# To run in (robocasa) environment: conda activate robocasa; python server_robocasa.py

import numpy as np
import os
import time
from robocasa.utils.env_utils import create_env
from robocasa.environments.kitchen.single_stage.kitchen_pnp import PnPCounterToCab


print("--- Robocasa Server ---")
print("Creating environment and waiting for 'action.npy' file to appear...")

# Use a temporary directory in the user's home for communication
COMM_DIR = os.path.expanduser("~/comm_dir") 
os.makedirs(COMM_DIR, exist_ok=True) # Ensure the directory exists

# Define communication file paths
ACTION_FILE = os.path.join(COMM_DIR, "action.npy")
ACTION_FLAG = os.path.join(COMM_DIR, "action.flag")
OBS_FILE = os.path.join(COMM_DIR, "obs.npz")
OBS_FLAG = os.path.join(COMM_DIR, "obs.flag")
LANG_FILE = os.path.join(COMM_DIR, "lang.txt")

# Clean up old communication files from previous runs
if os.path.exists(ACTION_FILE): os.remove(ACTION_FILE)
if os.path.exists(ACTION_FLAG): os.remove(ACTION_FLAG)
if os.path.exists(OBS_FILE): os.remove(OBS_FILE)
if os.path.exists(OBS_FLAG): os.remove(OBS_FLAG)
if os.path.exists(LANG_FILE): os.remove(LANG_FILE)

controller_config = {
    "type": "OSC_POSE",
    "input_max": 1,
    "input_min": -1,
    "output_max": [0.05, 0.05, 0.05, 0.5, 0.5, 0.5],
    "output_min": [-0.05, -0.05, -0.05, -0.5, -0.5, -0.5],
    "kp": 150,
    "damping": 1,
    "impedance_mode": "fixed",
    "kp_limits": [0, 300],
    "damping_limits": [0, 10],
    "position_limits": None,
    "orientation_limits": None,
    "uncouple_pos_ori": True,
    "control_delta": True,
    "interpolation": None,
    "ramping_up": False,
}
CAMERA_NAME = 'robot0_robotview'
# 1. Create the environment
print("Creating the environment...")
env = PnPCounterToCab(

    robots=['IIWA'],
    # Standard rendering and camera parameters for Robosuite/Robocasa
    has_renderer=True,
    has_offscreen_renderer=True,
    use_camera_obs=True,
    
    # Specify camera names, resolution, and other settings
    camera_names='robot0_robotview',
    render_camera='robot0_eye_in_hand',
    camera_heights=256,
    camera_widths=256,
    camera_depths=True,
    controller_configs=controller_config,
    
    reward_shaping=True,
    control_freq=20,
    ignore_done=False,
    hard_reset=False,
)
layout_id = 1
style_id = 0
env.layout_and_style_ids = [[layout_id, style_id]]
# 2. Reset the environment and send the first observation
obs = env.reset()

print(f"[Server Debug] Observation dictionary keys: {obs.keys()}")
lang_prompt = env.get_ep_meta()["lang"]
print(f"Environment reset. Task: {lang_prompt}")

# Save the first observation and language instruction
np.savez(OBS_FILE, **obs)
with open(LANG_FILE, 'w') as f:
    f.write(lang_prompt)
open(OBS_FLAG, 'w').close() # Create a flag file to signal that the observation is ready
print("First observation sent. Waiting for client response...")

done = False
max_steps = 500
i = 0
while not done and i < max_steps:
    # 3. Wait for the client's action
    while not os.path.exists(ACTION_FLAG):
        time.sleep(0.01) # Sleep briefly to avoid busy-waiting

    # 4. Read the action and execute it
    action = np.load(ACTION_FILE)
    os.remove(ACTION_FILE) # Delete file after reading
    os.remove(ACTION_FLAG) # Delete flag file
    print(f"[Server] Received action, executing step {i+1}...")
    
    obs, reward, done, info = env.step(action)
    env.render()

    # 5. Send the new observation
    np.savez(OBS_FILE, **obs)
    open(OBS_FLAG, 'w').close() # Create a flag file to signal that the new observation is ready
    print(f"[Server] Step {i+1} complete, new observation sent.")
    i += 1

print("Evaluation finished. Server is shutting down.")
env.close()

# Clean up the last communication files
if os.path.exists(OBS_FILE): os.remove(OBS_FILE)
if os.path.exists(OBS_FLAG): os.remove(OBS_FLAG)
