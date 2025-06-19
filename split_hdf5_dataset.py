import h5py
import numpy as np
import os
import random
from tqdm import tqdm

def copy_demos(source_file, dest_file, demo_keys):
    """
    A helper function to copy specified demos from a source HDF5 file to a destination HDF5 file.
    """
    if 'data' not in dest_file:
        dest_file.create_group('data')
    
    source_data_group = source_file['data']
    dest_data_group = dest_file['data']
    
    for key in tqdm(demo_keys, desc=f"Copying to {os.path.basename(dest_file.filename)}"):
        # The copy method of h5py can recursively copy the entire group
        source_data_group.copy(key, dest_data_group)

def split_hdf5_file(source_hdf5_path, train_hdf5_path, val_hdf5_path, val_split_ratio=0.1):
    """
    Splits an HDF5 file containing multiple demos into two HDF5 files for training and validation sets.

    Args:
        source_hdf5_path (str): Path to the original HDF5 file.
        train_hdf5_path (str): Path to the output training set HDF5 file.
        val_hdf5_path (str): Path to the output validation set HDF5 file.
        val_split_ratio (float): The proportion of the validation set, e.g., 0.1 for 10%.
    """
    print(f"Opening source file: {source_hdf5_path}")
    
    with h5py.File(source_hdf5_path, 'r') as f_source:
        if 'data' not in f_source:
            raise ValueError("Group 'data' not found in the source HDF5 file!")
            
        # Get all demo keys (e.g., 'demo_0', 'demo_1', ...)
        demo_keys = list(f_source['data'].keys())
        num_demos = len(demo_keys)
        print(f"Found {num_demos} demos.")

        # Shuffle the keys randomly to ensure the split is random
        random.shuffle(demo_keys)

        # Calculate the split point
        split_index = int(num_demos * (1 - val_split_ratio))
        
        train_keys = demo_keys[:split_index]
        val_keys = demo_keys[split_index:]

        print(f"Splitting into {len(train_keys)} training samples and {len(val_keys)} validation samples.")

        # Ensure the output directories exist
        os.makedirs(os.path.dirname(train_hdf5_path), exist_ok=True)
        os.makedirs(os.path.dirname(val_hdf5_path), exist_ok=True)

        # Create and write the training set file
        print("\n--- Creating training set file ---")
        with h5py.File(train_hdf5_path, 'w') as f_train:
            copy_demos(f_source, f_train, train_keys)
        print(f"Training set file saved to: {train_hdf5_path}")

        # Create and write the validation set file
        print("\n--- Creating validation set file ---")
        with h5py.File(val_hdf5_path, 'w') as f_val:
            copy_demos(f_source, f_val, val_keys)
        print(f"Validation set file saved to: {val_hdf5_path}")

    print("\nFile splitting complete!")


if __name__ == '__main__':
    # --- Please configure your paths ---

    # 1. Input file: Your existing, merged HDF5 file.
    #    Please replace this with the [full path] to your file.
    source_file = os.path.expanduser("~/robocasa/datasets/v0.1/single_stage/kitchen_pnp/PnPCounterToCab/2024-04-24/demo_gentex_im128_randcams.hdf5")

    # 2. Output files: Define the save paths and names for the split training and validation files.
    #    It's recommended to add 'train' and 'val' identifiers to the original filename.
    output_dir = os.path.dirname(source_file) # Get the directory of the source file
    base_filename = os.path.basename(source_file).replace('.hdf5', '') # Get the filename without extension

    train_output_file = os.path.join(output_dir, f"{base_filename}_train.hdf5")
    val_output_file = os.path.join(output_dir, f"{base_filename}_val.hdf5")
    
    # 3. Validation split ratio.
    validation_ratio = 0.1

    # Run the split
    split_hdf5_file(source_file, train_output_file, val_output_file, validation_ratio)
