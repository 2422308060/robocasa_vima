# Filename: client_brs.py
# To run in (brs) environment: conda activate brs; python client_brs.py

import torch
import numpy as np
from omegaconf import OmegaConf
from collections import deque
import tree 
import os
import time
import hydra

from brs_algo.learning.module import DiffusionModule
from brs_algo.utils.tree_utils import stack_sequence_fields

print("--- BRS-Algo Client ---")
print("Loading model and waiting for 'obs.npz' file to appear...")

# Configuration
CKPT_PATH = "/home/salt_fish/brs-algo/my_brs_experiments/wbvima_put_items_onto_shelves_To2_Ta8_b4_20250611-154435/ckpt/last-v1.pth"
CONFIG_PATH = "/home/salt_fish/brs-algo/my_brs_experiments/wbvima_put_items_onto_shelves_To2_Ta8_b4_20250611-154435/conf.yaml"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

COMM_DIR = os.path.expanduser("~/comm_dir")
# Communication file paths
ACTION_FILE = os.path.join(COMM_DIR, "action.npy")
ACTION_FLAG = os.path.join(COMM_DIR, "action.flag")
OBS_FILE = os.path.join(COMM_DIR, "obs.npz")
OBS_FLAG = os.path.join(COMM_DIR, "obs.flag")
LANG_FILE = os.path.join(COMM_DIR, "lang.txt")

# Load the model
print(">>> Loading model...")
cfg = OmegaConf.load(CONFIG_PATH)

# Instantiate the entire module using Hydra, which recursively creates all objects with a _target_ key.
module = hydra.utils.instantiate(cfg.module)

# Load only the weights (state dictionary) from the checkpoint.
checkpoint = torch.load(CKPT_PATH, map_location=DEVICE, weights_only=False) 
module.load_state_dict(checkpoint['state_dict']) 

module.to(DEVICE)
module.eval()
policy = module.policy
print("Model loaded successfully.")


# Read action definitions from the configuration.
ACTION_KEYS = cfg.action_keys
ACTION_KEY_DIMS = cfg.action_key_dims

# Calculate slice indices for each action part in the 21-dimensional vector.
slice_indices = {}
start_idx = 0
for key in ACTION_KEYS:
    end_idx = start_idx + ACTION_KEY_DIMS[key]
    slice_indices[key] = slice(start_idx, end_idx)
    start_idx = end_idx


OBS_WINDOW_SIZE = cfg.module.policy.num_latest_obs 
obs_history = deque(maxlen=OBS_WINDOW_SIZE)

# Main loop
max_steps = 500
for i in range(max_steps):
    # Wait for the server's observation
    while not os.path.exists(OBS_FLAG):
        time.sleep(0.01)

    # Read observation and language instruction
    obs_data = np.load(OBS_FILE)
    obs = {key: obs_data[key] for key in obs_data.keys()}
    with open(LANG_FILE, 'r') as f:
        lang_prompt = f.read()
    
    os.remove(OBS_FILE)
    os.remove(OBS_FLAG)
    print(f"[Client] Received observation for step {i+1}, making a decision...")
    
    # Format the observation data from robocasa to the model's expected format

    # Parse the flattened proprioception data
    flat_proprio = obs["robot0_proprio-state"]
    prop_keys_config = cfg.module.policy.prop_keys
    prop_dims = {
        "odom/base_velocity": 3, "qpos/torso": 4, "qpos/left_arm": 6, 
        "qpos/left_gripper": 1, "qpos/right_arm": 6, "qpos/right_gripper": 1
    }
    parsed_proprio = {"odom": {}, "qpos": {}}
    start_idx = 0
    for key_path in cfg.module.policy.prop_keys:
        group, key = key_path.split('/')
        dim = prop_dims[key_path]
        parsed_proprio[group][key] = flat_proprio[start_idx : start_idx + dim]
        start_idx += dim

    # Construct point cloud data
    # Assume robocasa returns an RGB image of shape (H, W, 3)
    rgb_image = obs["robot0_robotview_image"]
    H, W, _ = rgb_image.shape
    
    # Create a pixel coordinate grid (x, y)
    x, y = np.meshgrid(np.arange(W), np.arange(H))
    
    # Stack pixel coordinates and color information.
    # PointNet typically expects a shape of (N, D), where N is the number of points and D is the dimension of each point.
    # We treat each pixel as a point.
    # For xyz coordinates, we temporarily use pixel (x, y) coordinates with z=0.
    
    # Create simplified, pseudo-"xyz" coordinates by normalizing pixel coordinates to the [-1, 1] range.
    xyz_coords = np.zeros((H, W, 3), dtype=np.float32)
    xyz_coords[..., 0] = (x / (W - 1)) * 2 - 1
    xyz_coords[..., 1] = (y / (H - 1)) * 2 - 1
    
    # Normalize RGB colors to the [0, 1] range.
    rgb_colors = rgb_image.astype(np.float32) / 255.0
    
    # Reshape them to (N, D) where N = H * W.
    num_points = H * W
    xyz_flat = xyz_coords.reshape(num_points, 3)
    rgb_flat = rgb_colors.reshape(num_points, 3)
    
    # The PointNet might expect a dictionary containing both xyz and rgb.
    pointcloud_dict = {
        'xyz': xyz_flat,
        'rgb': rgb_flat
    }
    
    # Construct the final observation dictionary for the model.
    current_obs_for_model = {
        "pointcloud": pointcloud_dict,
        "odom": parsed_proprio["odom"],
        "qpos": parsed_proprio["qpos"]
    }
    obs_history.append(current_obs_for_model)
    
    
    if len(obs_history) < OBS_WINDOW_SIZE:
        print(f"  ...Collecting observation history... ({len(obs_history)}/{OBS_WINDOW_SIZE})")
        action_to_send = np.zeros(7) # Send a 7-dimensional zero action while collecting history.
    else:
        # Prepare the batch data for model input
        # Stack the history of observations from a list of dicts into a dict of stacked arrays.
        # In the output stacked_obs, the first dimension of the tensors is time T (e.g., 2).
        stacked_obs = stack_sequence_fields(list(obs_history))
        
        # Convert numpy arrays to tensors and adjust dimensions.
        # We need to convert from (T, ...) to (B, T, ...), where B=1 for batch size.
        def process_leaf(x):
            # Convert to tensor
            t = torch.from_numpy(x).to(device=DEVICE).float()
            # Add a batch dimension (B=1) at the beginning
            t = t.unsqueeze(0)
            return t

        input_tensors = tree.map_structure(process_leaf, stacked_obs)
        
        # Execute the inference process
        with torch.no_grad():
            transformer_output = policy(input_tensors)
            pred_actions_dict = policy.inference(
                transformer_output=transformer_output,
                return_last_timestep_only=True
            )

            # Debugging Code
            print("\n--- DEBUG: Action Dictionary Contents ---")
            for key, value in pred_actions_dict.items():
                print(f"  Key: {key}, Shape: {value.shape}")
            print("----------------------------------------\n")

            # Concatenate the predicted action dictionary into a single vector.
            action_parts = [pred_actions_dict[key] for key in ACTION_KEYS]
            
            # Assuming the inference output has shape (B, A) without a horizon dimension.
            action_sequence = torch.cat(action_parts, dim=-1)

        # Post-process and send the action
        
        # The shape of action_sequence is (B, T, D_action), e.g., (1, 8, 21)
        # Squeeze the batch dimension to get the action sequence, shape (T, D_action), e.g., (8, 21)
        full_action_21d_sequence = action_sequence[0].cpu().numpy()
        
        # Take only the first step from the predicted action sequence.
        first_step_action_21d = full_action_21d_sequence[0] # Shape: (21,)
        print(f"  ...Model predicted an action sequence, taking the first step action with dimension: {first_step_action_21d.shape}")
        
        # Extract the required 7 dimensions for the robot arm from the 21-dimensional vector.
        start_index = 7
        end_index = start_index + 7
        action_to_send = first_step_action_21d[start_index:end_index] # Shape: (7,)
        
        print(f"  ...Extracted 7-dimensional action (dimension: {action_to_send.shape}) and preparing to send.")
    # Send the action to the server
    np.save(ACTION_FILE, action_to_send)
    open(ACTION_FLAG, 'w').close()
    print(f"[Client] Action for step {i+1} sent.")

print("Client reached max steps, shutting down.")
