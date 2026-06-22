#!/bin/bash
#SBATCH --job-name=sf_exp_tt
#SBATCH --account=3dv
#SBATCH --time=48:00:00
#SBATCH --gpus=5060ti:1
#SBATCH --output=/work/courses/3dv/team15/logs/sf_exp_tt_%j.out
#SBATCH --error=/work/courses/3dv/team15/logs/sf_exp_tt_%j.err

# Phase B: Teacher (rigid SuperFlex) training on tabletop+GSO with IoU+SDF loss.
# Resumes from checkpoints/shapenet_foundation/ckpt.pt.
# Full run is ~1000 epochs; if you hit the 24h wall, re-submit and the
# trainer can pick up via checkpoints.resume_from + checkpoints.keep_epoch=true
# (set those overrides when re-submitting).

set -euo pipefail
. /etc/profile.d/modules.sh
module add cuda/12.9

TEAM_DIR="/work/courses/3dv/team15"
REPO_DIR="$TEAM_DIR/superdec_concave"
VENV_DIR="$TEAM_DIR/superdec/.venv"   # reusing the working venv from the v3 repo

echo "Node: $(hostname)  GPU: ${CUDA_VISIBLE_DEVICES:-?}  $(date)"
nvcc --version | tail -1

cd "$REPO_DIR"
source "$VENV_DIR/bin/activate"
export PYTHONPATH="$REPO_DIR:${PYTHONPATH:-}"
export TORCH_EXTENSIONS_DIR="$TEAM_DIR/.torch_extensions"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p "$TORCH_EXTENSIONS_DIR" "$TEAM_DIR/logs"

# Build the C++ FPS sampler used by the geometric loss / sampler. Idempotent.
python setup_sampler.py build_ext --inplace --quiet

# Sanity: data symlink + foundation ckpt must exist before training starts.
test -d "$REPO_DIR/data/ShapeNet/gso" || { echo "ERROR: data/ShapeNet/gso missing"; exit 2; }
test -f "$REPO_DIR/checkpoints/shapenet_foundation/ckpt.pt" \
    || { echo "ERROR: foundation checkpoint missing"; exit 2; }

# Optional overrides:
#   BATCH_SIZE=16 sbatch ...                     # smaller GPU
#   RESUME=checkpoints/exp_tt/epoch_75.pt sbatch ...   # resume after walltime
#   WANDB_API_KEY=xxxxx sbatch --export=ALL ...  # if not in ~/.netrc
BATCH_SIZE="${BATCH_SIZE:-32}"
RESUME="${RESUME:-}"

# wandb auth: prefer env var, else expect ~/.netrc to be set up.
if [ -n "${WANDB_API_KEY:-}" ]; then
    export WANDB_API_KEY
fi

EXTRA_OVERRIDES=()
if [ -n "$RESUME" ]; then
    test -f "$REPO_DIR/$RESUME" || { echo "ERROR: RESUME path $RESUME not found"; exit 2; }
    EXTRA_OVERRIDES+=("checkpoints.resume_from=$RESUME")
    EXTRA_OVERRIDES+=("checkpoints.keep_epoch=true")
fi

python train/train.py \
    --config-name exp_tt \
    trainer.batch_size="$BATCH_SIZE" \
    "${EXTRA_OVERRIDES[@]}"

echo "Done. $(date)"
