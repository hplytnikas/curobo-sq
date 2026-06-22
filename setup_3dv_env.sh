#!/usr/bin/env bash
# Creates the '3dv' conda environment for running motion_planning_sq.py
# Uses CUDA 12.8 toolkit with PyTorch cu128 wheels.
set -euo pipefail

ENV_NAME="3dv"
CUROBO_DIR="/home/haroldas/3DV/curobov2/curobo"
SUPERDEC_DIR="/home/haroldas/3DV/superdec"

echo "=== Creating conda env: ${ENV_NAME} (Python 3.11) ==="
conda create -n "${ENV_NAME}" python=3.11 -y

CONDA_BASE=$(conda info --base)
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate "${ENV_NAME}"

echo "=== Installing PyTorch 2.7 (cu128) ==="
pip install torch torchvision \
    --index-url https://download.pytorch.org/whl/cu128

echo "=== Installing CUDA runtime / cuda-core (cu12) ==="
pip install "cuda-core[cu12]>=0.7" "nvidia-cuda-runtime-cu12>=12"

echo "=== Installing curobo v2 (editable, cu12-torch extras) ==="
# cu12-torch extra pulls cuda-core[cu12] + nvidia-cuda-runtime-cu12 + torch>=2.5
# We install without those extras since torch is already present; just install base deps.
pip install -e "${CUROBO_DIR}" \
    --no-build-isolation

echo "=== Installing SuperDec dependencies ==="
pip install \
    h5py \
    omegaconf \
    hydra-core \
    plyfile \
    wandb \
    open3d \
    gdown \
    Cython \
    ninja \
    scipy \
    scikit-learn

echo "=== Installing SuperDec (editable) ==="
pip install -e "${SUPERDEC_DIR}" --no-build-isolation

# torch downgrades cuda-pathfinder to 1.2.2 which lacks the public
# find_nvidia_header_directory API needed by curobo's kernel_cache.py.
echo "=== Re-pinning cuda-pathfinder>=1.5.0 ==="
pip install "cuda-pathfinder>=1.5.0"

echo ""
echo "Done. Activate with:  conda activate ${ENV_NAME}"
echo "Run the example with:"
echo "  conda run -n ${ENV_NAME} python ${CUROBO_DIR}/curobo/examples/getting_started/motion_planning_sq.py \\"
echo "    --ply_path <your_cloud.ply> --world_representation superquadrics"
