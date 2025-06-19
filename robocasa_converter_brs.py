import os
import h5py
import numpy as np
import json
from tqdm import tqdm
import cv2 
import datetime

# Define BRS target image size
BRS_IMG_HEIGHT = 94
BRS_IMG_WIDTH = 168
BRS_TARGET_FREQ = 10.0 # Unified frequency for BRS compatibility

def ensure_string(value):
    if isinstance(value, bytes):
        return value.decode('utf-8')
    return str(value)

def create_zeros_dataset(group, name, shape, dtype, compression='gzip'):
    """Helper function to create a dataset filled with zeros if it doesn't exist."""
    if name not in group:
        group.create_dataset(name, data=np.zeros(shape, dtype=dtype), compression=compression)

def convert_robocasa_to_brs_hdf5(input_dir, output_hdf5_filepath, source_dataset_name_for_task_attr):
    """
    Converts RoboCasa dataset (HDF5 format) into a single HDF5 file that matches the format expected by the BRS algorithm.
    Episodes from 'data/demo_X' within each original HDF5 file will be merged into the /demo_Y structure in the output HDF5 file.
    """
    output_hdf5_dir = os.path.dirname(output_hdf5_filepath)
    os.makedirs(output_hdf5_dir, exist_ok=True)

    print(f"Starting dataset conversion to a single HDF5 file (BRS format)")
    print(f"Source path: {input_dir}")
    print(f"Target HDF5 file: {output_hdf5_filepath}")

    episode_file_parent_dirs = [os.path.join(input_dir, d) for d in os.listdir(input_dir) if os.path.isdir(os.path.join(input_dir, d))]
    episode_file_parent_dirs.sort()

    global_demo_counter = 0
    source_hdf5_files_processed = []

    with h5py.File(output_hdf5_filepath, 'w') as out_f:
        # Top-level attributes like 'merged_data_files', 'merging_time', 'num_demos' will be added at the end.

        for ep_file_parent_dir in tqdm(episode_file_parent_dirs, desc="Processing parent directories of HDF5 files"):
            target_filename = "demo_gentex_im128_randcams_train.hdf5"
            found_hdf5_files = []
            if os.path.exists(os.path.join(ep_file_parent_dir, target_filename)):
                found_hdf5_files.append(target_filename)

            if not found_hdf5_files:
                print(f"Warning: Target file '{target_filename}' not found in directory {ep_file_parent_dir}. Skipping this directory.")
                continue

            for hdf5_filename_in_parent in found_hdf5_files:
                source_hdf5_path = os.path.join(ep_file_parent_dir, hdf5_filename_in_parent)
                print(f"--- Preparing to process source file: {source_hdf5_path} ---")
                source_hdf5_files_processed.append(source_hdf5_path)

                try:
                    with h5py.File(source_hdf5_path, 'r') as f_in:
                        if 'data' not in f_in:
                            print(f"Warning: 'data' group missing in HDF5 file {source_hdf5_path}. Skipping.")
                            continue

                        demo_keys_in_source = sorted([key for key in f_in['data'].keys() if key.startswith('demo_')])
                        if not demo_keys_in_source:
                            print(f"Warning: No 'demo_X' subgroups found in 'data' group for file {source_hdf5_path}. Skipping.")
                            continue
                        
                        print(f"  Found {len(demo_keys_in_source)} demo episodes in {source_hdf5_path}.")

                        for demo_key_in_source in tqdm(demo_keys_in_source, desc=f"  Converting Episodes in {os.path.basename(source_hdf5_path)}", leave=False):
                            source_demo_group = f_in['data'][demo_key_in_source]

                            # --- Extract data from RoboCasa source demo ---
                            lang_instruction = ""
                            if 'language_instruction' in source_demo_group.attrs:
                                lang_instruction = ensure_string(source_demo_group.attrs['language_instruction'])
                            elif 'ep_meta' in source_demo_group.attrs:
                                ep_meta_str = ensure_string(source_demo_group.attrs['ep_meta'])
                                if ep_meta_str:
                                    try:
                                        ep_meta_json = json.loads(ep_meta_str)
                                        lang_instruction = ensure_string(ep_meta_json.get('lang', ''))
                                    except json.JSONDecodeError:
                                        print(f"Warning: Could not decode ep_meta JSON for data/{demo_key_in_source}: '{ep_meta_str}'")
                            if not lang_instruction:
                                print(f"Warning: Language instruction not found in data/{demo_key_in_source}.")

                            # Check for essential keys
                            image_key = 'robot0_eye_in_hand_image'
                            depth_key = 'robot0_eye_in_hand_depth' # Check if depth data exists
                            joint_pos_key = 'robot0_joint_pos'
                            gripper_qpos_key = 'robot0_gripper_qpos'
                            eef_pos_key = 'robot0_eef_pos'
                            eef_quat_key = 'robot0_eef_quat'
                            actions_key = 'actions'

                            if 'obs' not in source_demo_group or \
                               image_key not in source_demo_group['obs'] or \
                               joint_pos_key not in source_demo_group['obs'] or \
                               gripper_qpos_key not in source_demo_group['obs'] or \
                               actions_key not in source_demo_group:
                                print(f"Warning: data/{demo_key_in_source} is missing essential observation or action keys. Skipping this demo.")
                                continue

                            num_frames = len(source_demo_group['obs'][image_key])
                            if num_frames == 0:
                                print(f"Warning: No frame data in data/{demo_key_in_source}. Skipping.")
                                continue

                            # Prepare lists for extracted data
                            rgb_images_list = []
                            depth_images_list = []
                            joint_pos_list = []
                            gripper_qpos_list = []
                            eef_pos_list = []
                            eef_quat_list = []
                            actions_list = []
                            
                            has_depth = depth_key in source_demo_group['obs']
                            has_eef_pose = eef_pos_key in source_demo_group['obs'] and eef_quat_key in source_demo_group['obs']


                            for t in range(num_frames):
                                # RGB Image
                                img_orig = source_demo_group['obs'][image_key][t]
                                img_resized = cv2.resize(img_orig, (BRS_IMG_WIDTH, BRS_IMG_HEIGHT), interpolation=cv2.INTER_AREA)
                                rgb_images_list.append(img_resized)

                                # Depth Image (if available)
                                if has_depth:
                                    depth_orig = source_demo_group['obs'][depth_key][t]
                                    depth_resized = cv2.resize(depth_orig, (BRS_IMG_WIDTH, BRS_IMG_HEIGHT), interpolation=cv2.INTER_NEAREST)
                                    depth_images_list.append(depth_resized)
                                
                                joint_pos_list.append(source_demo_group['obs'][joint_pos_key][t])
                                gripper_qpos_list.append(source_demo_group['obs'][gripper_qpos_key][t])
                                
                                if has_eef_pose:
                                    eef_pos_list.append(source_demo_group['obs'][eef_pos_key][t])
                                    eef_quat_list.append(source_demo_group['obs'][eef_quat_key][t])

                                actions_list.append(source_demo_group[actions_key][t])

                            # Convert lists to numpy arrays
                            rgb_images_np = np.array(rgb_images_list, dtype=np.uint8)
                            depth_images_np = np.array(depth_images_list, dtype=np.float32) if has_depth else np.zeros((num_frames, BRS_IMG_HEIGHT, BRS_IMG_WIDTH), dtype=np.float32)
                            joint_pos_np = np.array(joint_pos_list, dtype=np.float64)
                            gripper_qpos_np = np.array(gripper_qpos_list, dtype=np.float64) # Shape: (N, 2)
                            actions_np = np.array(actions_list, dtype=np.float64) # Shape: (N, D_action)
                            
                            eef_pos_np = np.array(eef_pos_list, dtype=np.float64) if has_eef_pose else np.zeros((num_frames, 3), dtype=np.float64)
                            eef_quat_np = np.array(eef_quat_list, dtype=np.float64) if has_eef_pose else np.zeros((num_frames, 4), dtype=np.float64)
                            if has_eef_pose: # Ensure w is last for (pos, quat_xyzw)
                                # Assuming RoboCasa eef_quat is (w,x,y,z), convert to (x,y,z,w)
                                eef_quat_np = eef_quat_np[:, [1, 2, 3, 0]] # Convert (w,x,y,z) to (x,y,z,w)
                            eef_pose_np = np.concatenate((eef_pos_np, eef_quat_np), axis=1) if has_eef_pose else np.zeros((num_frames, 7), dtype=np.float64)


                            # --- Create demo_X group in output HDF5 ---
                            demo_out_group = out_f.create_group(f"demo_{global_demo_counter}")

                            # Attributes for demo_X
                            demo_out_group.attrs['collected_time'] = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
                            demo_out_group.attrs['horizon'] = num_frames
                            demo_out_group.attrs['recording_freq'] = BRS_TARGET_FREQ # Set to target frequency
                           
                            # Obs frequencies (aligned with BRS examples using BRS_TARGET_FREQ)
                            demo_out_group.attrs[f'obs_freq/depth_head'] = BRS_TARGET_FREQ
                            demo_out_group.attrs[f'obs_freq/depth_left_wrist'] = BRS_TARGET_FREQ
                            demo_out_group.attrs[f'obs_freq/depth_right_wrist'] = BRS_TARGET_FREQ
                            demo_out_group.attrs[f'obs_freq/gripper_state_left_gripper'] = BRS_TARGET_FREQ
                            demo_out_group.attrs[f'obs_freq/gripper_state_right_gripper'] = BRS_TARGET_FREQ
                            demo_out_group.attrs[f'obs_freq/joint_state_left_arm'] = BRS_TARGET_FREQ
                            demo_out_group.attrs[f'obs_freq/joint_state_right_arm'] = BRS_TARGET_FREQ
                            demo_out_group.attrs[f'obs_freq/joint_state_torso'] = BRS_TARGET_FREQ
                            demo_out_group.attrs[f'obs_freq/odom'] = BRS_TARGET_FREQ
                            demo_out_group.attrs[f'obs_freq/pcd_fused'] = BRS_TARGET_FREQ
                            demo_out_group.attrs[f'obs_freq/rgb_head'] = BRS_TARGET_FREQ
                            demo_out_group.attrs[f'obs_freq/rgb_left_wrist'] = BRS_TARGET_FREQ
                            demo_out_group.attrs[f'obs_freq/rgb_right_wrist'] = BRS_TARGET_FREQ
                            

                            # --- Populate /demo_X/action ---
                            action_out_group = demo_out_group.create_group('action')
                            robocasa_action_dim = actions_np.shape[1]

                            # Assuming RoboCasa action: [arm_6dof, gripper_1dof]
                            action_out_group.create_dataset('left_arm', data=actions_np[:, :6] if robocasa_action_dim >=6 else np.zeros((num_frames,6), dtype=np.float64), dtype=np.float64, compression='gzip')
                            action_out_group.create_dataset('left_gripper', data=actions_np[:, 6] if robocasa_action_dim >=7 else np.zeros((num_frames,), dtype=np.float64), dtype=np.float64, compression='gzip')
                            create_zeros_dataset(action_out_group, 'mobile_base', (num_frames, 3), np.float64)
                            create_zeros_dataset(action_out_group, 'right_arm', (num_frames, 6), np.float64)
                            create_zeros_dataset(action_out_group, 'right_gripper', (num_frames,), np.float64)
                            create_zeros_dataset(action_out_group, 'torso', (num_frames, 4), np.float64)

                            # --- Populate /demo_X/obs ---
                            obs_out_group = demo_out_group.create_group('obs')

                            # RGB
                            obs_rgb_group = obs_out_group.create_group('rgb')
                            obs_rgb_head_group = obs_rgb_group.create_group('head')
                            obs_rgb_head_group.create_dataset('img', data=rgb_images_np, dtype=np.uint8, compression='gzip')
                            obs_rgb_lw_group = obs_rgb_group.create_group('left_wrist')
                            create_zeros_dataset(obs_rgb_lw_group, 'img', (num_frames, BRS_IMG_HEIGHT, BRS_IMG_WIDTH, 3), np.uint8)
                            obs_rgb_rw_group = obs_rgb_group.create_group('right_wrist')
                            create_zeros_dataset(obs_rgb_rw_group, 'img', (num_frames, BRS_IMG_HEIGHT, BRS_IMG_WIDTH, 3), np.uint8)

                            # Depth
                            obs_depth_group = obs_out_group.create_group('depth')
                            obs_depth_head_group = obs_depth_group.create_group('head')
                            obs_depth_head_group.create_dataset('depth', data=depth_images_np, dtype=np.float32, compression='gzip')
                            obs_depth_lw_group = obs_depth_group.create_group('left_wrist')
                            create_zeros_dataset(obs_depth_lw_group, 'depth', (num_frames, BRS_IMG_HEIGHT, BRS_IMG_WIDTH), np.float32)
                            obs_depth_rw_group = obs_depth_group.create_group('right_wrist')
                            create_zeros_dataset(obs_depth_rw_group, 'depth', (num_frames, BRS_IMG_HEIGHT, BRS_IMG_WIDTH), np.float32)
                            
                            # Gripper State
                            obs_gs_group = obs_out_group.create_group('gripper_state')
                            obs_gs_lg_group = obs_gs_group.create_group('left_gripper')
                            # RoboCasa gripper_qpos is often (N,2) for two fingers. BRS expects (N,). Taking the first finger's value.
                            lg_gripper_pos_flat = gripper_qpos_np[:, 0] if gripper_qpos_np.ndim == 2 and gripper_qpos_np.shape[1] > 0 else np.zeros((num_frames,), dtype=np.float64)
                            obs_gs_lg_group.create_dataset('gripper_position', data=lg_gripper_pos_flat, dtype=np.float64, compression='gzip')
                            create_zeros_dataset(obs_gs_lg_group, 'gripper_effort', (num_frames,), np.float64) # Shape: (N,)
                            create_zeros_dataset(obs_gs_lg_group, 'gripper_velocity', (num_frames,), np.float64) # Shape: (N,)
                            create_zeros_dataset(obs_gs_lg_group, 'seq', (num_frames,), np.int64)
                            create_zeros_dataset(obs_gs_lg_group, 'stamp', (num_frames,), np.float64)
                            
                            obs_gs_rg_group = obs_gs_group.create_group('right_gripper')
                            create_zeros_dataset(obs_gs_rg_group, 'gripper_position', (num_frames,), np.float64) # Shape: (N,)
                            create_zeros_dataset(obs_gs_rg_group, 'gripper_effort', (num_frames,), np.float64) # Shape: (N,)
                            create_zeros_dataset(obs_gs_rg_group, 'gripper_velocity', (num_frames,), np.float64) # Shape: (N,)
                            create_zeros_dataset(obs_gs_rg_group, 'seq', (num_frames,), np.int64)
                            create_zeros_dataset(obs_gs_rg_group, 'stamp', (num_frames,), np.float64)

                            # Joint State
                            obs_js_group = obs_out_group.create_group('joint_state')
                            # Left Arm (assuming joint_pos_np is (N,7) for 7DoF arm)
                            obs_js_la_group = obs_js_group.create_group('left_arm')
                            la_joint_pos = joint_pos_np # Should be (N,7)
                            if la_joint_pos.shape[1] != 7: # Pad or truncate if necessary
                                print(f"  Warning: Left arm joint position dimension is not 7 (is {la_joint_pos.shape[1]}). Padding/truncating.")
                                temp_la_joint_pos = np.zeros((num_frames, 7), dtype=np.float64)
                                copy_dim = min(la_joint_pos.shape[1], 7)
                                temp_la_joint_pos[:, :copy_dim] = la_joint_pos[:, :copy_dim]
                                la_joint_pos = temp_la_joint_pos
                            obs_js_la_group.create_dataset('joint_position', data=la_joint_pos, dtype=np.float64, compression='gzip')
                            create_zeros_dataset(obs_js_la_group, 'joint_effort', (num_frames,7), np.float64)
                            create_zeros_dataset(obs_js_la_group, 'joint_velocity', (num_frames,7), np.float64)
                            create_zeros_dataset(obs_js_la_group, 'seq', (num_frames,), np.int64)
                            create_zeros_dataset(obs_js_la_group, 'stamp', (num_frames,), np.float64)
                            
                            obs_js_ra_group = obs_js_group.create_group('right_arm')
                            create_zeros_dataset(obs_js_ra_group, 'joint_position', (num_frames,7), np.float64)
                            create_zeros_dataset(obs_js_ra_group, 'joint_effort', (num_frames,7), np.float64)
                            create_zeros_dataset(obs_js_ra_group, 'joint_velocity', (num_frames,7), np.float64)
                            create_zeros_dataset(obs_js_ra_group, 'seq', (num_frames,), np.int64)
                            create_zeros_dataset(obs_js_ra_group, 'stamp', (num_frames,), np.float64)

                            obs_js_torso_group = obs_js_group.create_group('torso')
                            create_zeros_dataset(obs_js_torso_group, 'joint_position', (num_frames,4), np.float64)
                            create_zeros_dataset(obs_js_torso_group, 'joint_effort', (num_frames,4), np.float64)
                            create_zeros_dataset(obs_js_torso_group, 'joint_velocity', (num_frames,4), np.float64)
                            create_zeros_dataset(obs_js_torso_group, 'seq', (num_frames,), np.int64)
                            create_zeros_dataset(obs_js_torso_group, 'stamp', (num_frames,), np.float64)

                            # Link Poses
                            obs_lp_group = obs_out_group.create_group('link_poses')
                            obs_lp_group.create_dataset('left_eef', data=eef_pose_np, dtype=np.float64, compression='gzip')
                            create_zeros_dataset(obs_lp_group, 'head', (num_frames,7), np.float64)
                            create_zeros_dataset(obs_lp_group, 'right_eef', (num_frames,7), np.float64)
                            
                            # Odom
                            obs_odom_group = obs_out_group.create_group('odom')
                            create_zeros_dataset(obs_odom_group, 'base_velocity', (num_frames,3), np.float32)

                            # Point Cloud
                            obs_pc_group = obs_out_group.create_group('point_cloud')
                            obs_pc_fused_group = obs_pc_group.create_group('fused')
                            create_zeros_dataset(obs_pc_fused_group, 'padding_mask', (num_frames, 4096), bool)
                            create_zeros_dataset(obs_pc_fused_group, 'rgb', (num_frames, 4096, 3), np.uint8)
                            create_zeros_dataset(obs_pc_fused_group, 'xyz', (num_frames, 4096, 3), np.float32)
                            
                            global_demo_counter += 1
                
                except Exception as e:
                    print(f"Critical error processing demo {demo_key_in_source if 'demo_key_in_source' in locals() else ''} in HDF5 file {source_hdf5_path}: {e}")
                    import traceback
                    traceback.print_exc()
                    continue # Continue with the next demo or HDF5 file
        
        # Update top-level attributes after all processing
        out_f.attrs['merged_data_files'] = [ensure_string(s) for s in source_hdf5_files_processed]
        out_f.attrs['merging_time'] = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        out_f.attrs['num_demos'] = global_demo_counter

    print(f"Dataset conversion complete! Converted a total of {global_demo_counter} demos to {output_hdf5_filepath}.")

if __name__ == '__main__':
    robocasa_base_dir = os.path.expanduser('~/robocasa/datasets/v0.1/single_stage/kitchen_pnp')
    robocasa_dataset_name = 'PnPCounterToCab' # Example: 'PnPCounterToCab'
    input_data_path = os.path.join(robocasa_base_dir, robocasa_dataset_name)
    
    output_hdf5_dir = os.path.expanduser('~/brs-algo/brs_data_repo/converted_robocasa_brs_format') # Define and create the output directory
    os.makedirs(output_hdf5_dir, exist_ok=True)
    
    output_hdf5_filename = f"robocasa_{robocasa_dataset_name.lower().replace('-', '_')}_brs_target_format.hdf5" # Define the output filename
    output_hdf5_full_path = os.path.join(output_hdf5_dir, output_hdf5_filename)

    source_dataset_name_for_task_attr = f"robocasa_{robocasa_dataset_name.lower().replace('-', '_')}" # Used for metadata attributes

    if not os.path.isdir(input_data_path):
        print(f"Error: Input directory {input_data_path} does not exist or is not a directory. Please check your path configuration.")
    else:
        convert_robocasa_to_brs_hdf5(input_data_path, output_hdf5_full_path, source_dataset_name_for_task_attr)