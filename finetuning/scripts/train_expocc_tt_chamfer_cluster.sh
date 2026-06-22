#!/bin/bash
#SBATCH --job-name=sf_expocc_tt_chamfer
#SBATCH --account=3dv
#SBATCH --time=24:00:00
#SBATCH --gpus=5060ti:1
#SBATCH --output=/work/courses/3dv/team15/logs/sf_expocc_tt_chamfer_%j.out
#SBATCH --error=/work/courses/3dv/team15/logs/sf_expocc_tt_chamfer_%j.err

# Phase D (Chamfer variant): Student training. Hungarian + Chamfer
# (loss.type=geom) supervised against pseudo-GT from the Chamfer teacher,
# with full SO(3) augmentation and occlusions.
# Resumes from checkpoints/exp_tt_chamfer/epoch_500.pt.

set -euo pipefail
. /etc/profile.d/modules.sh
module add cuda/12.9

TEAM_DIR="/work/courses/3dv/team15"
REPO_DIR="$TEAM_DIR/superdec_concave"
VENV_DIR="$TEAM_DIR/superdec/.venv"

echo "Node: $(hostname)  GPU: ${CUDA_VISIBLE_DEVICES:-?}  $(date)"
nvcc --version | tail -1

cd "$REPO_DIR"
source "$VENV_DIR/bin/activate"
export PYTHONPATH="$REPO_DIR:${PYTHONPATH:-}"
export TORCH_EXTENSIONS_DIR="$TEAM_DIR/.torch_extensions"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export WANDB_DIR=/work/scratch/eylsen/wandb_data
export WANDB_CACHE_DIR=/work/scratch/eylsen/wandb_cache
export WANDB_ARTIFACTS_DIR=/work/scratch/eylsen/wandb_artifacts
mkdir -p "$TORCH_EXTENSIONS_DIR" "$TEAM_DIR/logs" \
    "$WANDB_DIR" "$WANDB_CACHE_DIR" "$WANDB_ARTIFACTS_DIR"

python setup_sampler.py build_ext --inplace --quiet

# Pseudo-GT preconditions. Default expects iou_balanced/; flip to iou/ if you
# decided to use the paper's global threshold without rebalancing.
GT_DIR="${GT_DIR:-data/output_npz/exp_tt_chamfer/iou_balanced}"
test -f "$REPO_DIR/$GT_DIR/train.npz" \
    || { echo "ERROR: pseudo-GT $GT_DIR/train.npz missing. Run Phase C first."; exit 2; }
test -f "$REPO_DIR/$GT_DIR/val.npz" \
    || { echo "ERROR: pseudo-GT $GT_DIR/val.npz missing."; exit 2; }
test -f "$REPO_DIR/checkpoints/exp_tt_chamfer/epoch_500.pt" \
    || { echo "ERROR: teacher checkpoint missing"; exit 2; }

BATCH_SIZE="${BATCH_SIZE:-32}"
RESUME="${RESUME:-}"

EXTRA_OVERRIDES=("shapenet.gt_train_path=$GT_DIR/train.npz" "shapenet.gt_val_path=$GT_DIR/val.npz")
if [ -n "$RESUME" ]; then
    test -f "$REPO_DIR/$RESUME" || { echo "ERROR: RESUME path $RESUME not found"; exit 2; }
    EXTRA_OVERRIDES+=("checkpoints.resume_from=$RESUME")
    EXTRA_OVERRIDES+=("checkpoints.keep_epoch=true")
fi

python train/train.py \
    --config-name expocc_tt_chamfer \
    trainer.batch_size="$BATCH_SIZE" \
    "${EXTRA_OVERRIDES[@]}"

echo "Done. $(date)"
