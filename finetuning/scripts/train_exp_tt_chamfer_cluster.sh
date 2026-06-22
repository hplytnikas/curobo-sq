#!/bin/bash
#SBATCH --job-name=sf_exp_tt_chamfer
#SBATCH --account=3dv
#SBATCH --time=24:00:00
#SBATCH --gpus=5060ti:1
#SBATCH --output=/work/courses/3dv/team15/logs/sf_exp_tt_chamfer_%j.out
#SBATCH --error=/work/courses/3dv/team15/logs/sf_exp_tt_chamfer_%j.err

# Phase B (Chamfer variant): Teacher training using v3 recipe
# (loss.type=original, w_cd=1.0, w_cub=0.1, w_sps=0.1, w_ext=0.01).
# Resumes from checkpoints/shapenet_foundation/ckpt.pt (Chamfer-trained).
# 500 epochs; if walltime hit, re-submit with RESUME=checkpoints/exp_tt_chamfer/epoch_NNN.pt

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
mkdir -p "$TORCH_EXTENSIONS_DIR" "$TEAM_DIR/logs"

python setup_sampler.py build_ext --inplace --quiet

test -d "$REPO_DIR/data/ShapeNet/gso" || { echo "ERROR: data/ShapeNet/gso missing"; exit 2; }
test -f "$REPO_DIR/checkpoints/shapenet_foundation/ckpt.pt" \
    || { echo "ERROR: foundation checkpoint missing"; exit 2; }

BATCH_SIZE="${BATCH_SIZE:-32}"
RESUME="${RESUME:-}"

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
    --config-name exp_tt_chamfer \
    trainer.batch_size="$BATCH_SIZE" \
    "${EXTRA_OVERRIDES[@]}"

echo "Done. $(date)"
