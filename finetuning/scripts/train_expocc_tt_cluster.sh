#!/bin/bash
#SBATCH --job-name=sf_expocc_tt
#SBATCH --account=3dv
#SBATCH --time=24:00:00
#SBATCH --gpus=5060ti:1
#SBATCH --output=/work/courses/3dv/team15/logs/sf_expocc_tt_%j.out
#SBATCH --error=/work/courses/3dv/team15/logs/sf_expocc_tt_%j.err

# Phase D: Student training. Supervised Hungarian + Chamfer (loss.type=geom)
# against the rigid pseudo-GT generated in Phase C, with full SO(3)
# augmentation and HRP/random-spherical occlusions.
#
# Resumes from checkpoints/exp_tt/epoch_1000.pt. The 4-D quat rot_head from
# the teacher is auto-deleted and re-initialized to 6-D by the SuperDec
# load_state_dict path because expocc_tt.yaml sets rotation6d=true.

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
# Keep wandb cache off /home (20GB quota) -> /work/scratch (100GB).
export WANDB_DIR=/work/scratch/eylsen/wandb_data
export WANDB_CACHE_DIR=/work/scratch/eylsen/wandb_cache
export WANDB_ARTIFACTS_DIR=/work/scratch/eylsen/wandb_artifacts
mkdir -p "$TORCH_EXTENSIONS_DIR" "$TEAM_DIR/logs" \
    "$WANDB_DIR" "$WANDB_CACHE_DIR" "$WANDB_ARTIFACTS_DIR"

python setup_sampler.py build_ext --inplace --quiet

# Hard preconditions: pseudo-GT must exist before this job starts.
test -f "$REPO_DIR/data/output_npz/exp_tt/iou/train.npz" \
    || { echo "ERROR: pseudo-GT train.npz missing. Run generate_gt_tt_cluster.sh first."; exit 2; }
test -f "$REPO_DIR/data/output_npz/exp_tt/iou/val.npz" \
    || { echo "ERROR: pseudo-GT val.npz missing. Run generate_gt_tt_cluster.sh first."; exit 2; }
test -f "$REPO_DIR/checkpoints/exp_tt/epoch_1000.pt" \
    || { echo "ERROR: teacher checkpoint missing"; exit 2; }

BATCH_SIZE="${BATCH_SIZE:-32}"
RESUME="${RESUME:-}"

EXTRA_OVERRIDES=()
if [ -n "$RESUME" ]; then
    test -f "$REPO_DIR/$RESUME" || { echo "ERROR: RESUME path $RESUME not found"; exit 2; }
    EXTRA_OVERRIDES+=("checkpoints.resume_from=$RESUME")
    EXTRA_OVERRIDES+=("checkpoints.keep_epoch=true")
    echo "Resuming from $RESUME (keep_epoch=true)"
fi

python train/train.py \
    --config-name expocc_tt \
    trainer.batch_size="$BATCH_SIZE" \
    "${EXTRA_OVERRIDES[@]}"

echo "Done. $(date)"
