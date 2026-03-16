
OUTPUT_NPZ_DIR=data/output_npz
# Convert results to NPZ format
python superdec/evaluate/to_npz.py output_dir="$OUTPUT_NPZ_DIR" 

# Visualize results using viser
python superdec/visualization/object_visualizer.py npz_folder="$OUTPUT_NPZ_DIR"