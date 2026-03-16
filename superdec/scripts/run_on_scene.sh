#!/bin/bash
set -euo pipefail  # stop on error, undefined variable, or pipeline failure

# trap to print the failing command
trap 'echo "Error occurred at command: $BASH_COMMAND"' ERR

OBJECTS_SCENE_DIR=data/scenes/scene_example # path to the folder containing the .ply files of all the segmented objects in the scene
OUTPUT_NPZ_DIR=data/output_npz # path to the folder where to save the output .npz files
SCENE_NAME=scene_example # name of the scene (used to name the output .npz file)
Z_UP=true

python superdec/utils/ply_to_npz.py --input_path="$OBJECTS_SCENE_DIR" --scene_name="$SCENE_NAME"

python superdec/evaluate/to_npz.py checkpoints_folder="checkpoints/normalized" output_dir="$OUTPUT_NPZ_DIR" dataset=scene scene.name="$SCENE_NAME" scene.z_up="$Z_UP"

python superdec/visualization/object_visualizer.py dataset=scene split="$SCENE_NAME" npz_folder="$OUTPUT_NPZ_DIR"