#!/bin/bash
#SBATCH --job-name=sf_expocc_tt_bent
#SBATCH --account=3dv
#SBATCH --time=24:00:00
#SBATCH --gpus=5060ti:1
#SBATCH --output=/work/courses/3dv/team15/logs/sf_expocc_tt_bent_%j.out
#SBATCH --error=/work/courses/3dv/team15/logs/sf_expocc_tt_bent_%j.err

# Bent fine-tune: extended (tapering + bending) superquadric model on
# tabletop+GSO. Resumes from the supervisor's expocc2.1 checkpoint
# (checkpoints/expocc2.1/epoch_100.pt) and fine-tunes with loss.type=iou
# (bending-aware SDF + volumetric IoU against occupancy GT).
#
# rotation6d=false in expocc_tt_bent.yaml -> matches the checkpoint's 4-D
# quat rot_head, so no head deletion/re-init at load time.

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

# Hard precondition: the bent checkpoint must exist before this job starts.
# No pseudo-GT needed -- IoULoss supervises purely from occupancy.
test -f "$REPO_DIR/checkpoints/expocc2.1/epoch_100.pt" \
    || { echo "ERROR: bent checkpoint checkpoints/expocc2.1/epoch_100.pt missing"; exit 2; }

BATCH_SIZE="${BATCH_SIZE:-16}"   # 16GB GPU: 32 OOMs with the bent IoU loss
RESUME="${RESUME:-}"

EXTRA_OVERRIDES=()
if [ -n "$RESUME" ]; then
    test -f "$REPO_DIR/$RESUME" || { echo "ERROR: RESUME path $RESUME not found"; exit 2; }
    EXTRA_OVERRIDES+=("checkpoints.resume_from=$RESUME")
    EXTRA_OVERRIDES+=("checkpoints.keep_epoch=true")
    echo "Resuming from $RESUME (keep_epoch=true)"
fi

python train/train.py \
    --config-name expocc_tt_bent \
    trainer.batch_size="$BATCH_SIZE" \
    "${EXTRA_OVERRIDES[@]}"

echo "Done. $(date)"
