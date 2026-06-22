#!/bin/bash
#SBATCH --job-name=sf_gt_tt
#SBATCH --account=3dv
#SBATCH --time=12:00:00
#SBATCH --gpus=5060ti:1
#SBATCH --output=/work/courses/3dv/team15/logs/sf_gt_tt_%j.out
#SBATCH --error=/work/courses/3dv/team15/logs/sf_gt_tt_%j.err

# Phase C: Generate pseudo-GT for the student.
#   1) Forward the trained teacher over train + val splits, dump raw
#      predictions to data/output_npz/exp_tt/shapenet_<split>.npz
#   2) Run per-shape test-time optimization (rigid: tapering/bending OFF)
#      to tighten the predictions, write data/output_npz/exp_tt/iou/<split>.npz
#      and per-shape <split>_metrics.csv.
#
# After this completes, configs/expocc_tt.yaml's gt_train_path/gt_val_path will
# resolve correctly and Phase D can launch.

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

# Allow callers to override teacher checkpoint via env, e.g. CKPT_FILE=epoch_500.pt
CKPT_FOLDER="${CKPT_FOLDER:-exp_tt}"
CKPT_FILE="${CKPT_FILE:-epoch_1000.pt}"
SOURCE_FOLDER="${SOURCE_FOLDER:-exp_tt}"

test -f "$REPO_DIR/checkpoints/$CKPT_FOLDER/$CKPT_FILE" \
    || { echo "ERROR: teacher checkpoint checkpoints/$CKPT_FOLDER/$CKPT_FILE missing"; exit 2; }

# --- Step 1: dump teacher raw predictions ---
for SPLIT in train val; do
    echo "[$(date)] Step 1: to_npz split=$SPLIT"
    # to_npz lives at superdec/evaluate/to_npz.py -> need ../../configs/...
    python -m superdec.evaluate.to_npz \
        --config-path="../../configs/optim_tt" --config-name=save_npz \
        "checkpoints_folder=checkpoints/$CKPT_FOLDER" \
        "checkpoint_file=$CKPT_FILE" \
        "output_dir=data/output_npz/$SOURCE_FOLDER" \
        "dataloader.split=$SPLIT"
done

# --- Step 2: per-shape test-time optimization (rigid) ---
for SPLIT in train val; do
    echo "[$(date)] Step 2: batch_evaluate split=$SPLIT (rigid optim)"
    python -m superoptim.batch_evaluate \
        --config-path="../configs/optim_tt" --config-name=batch_optim \
        "+source_folder=$SOURCE_FOLDER" \
        "shapenet.split=$SPLIT"
done

# --- Sanity: peek at filter-pass rates so we can gate Phase D ---
python - <<'PY'
import csv, glob, os
for f in sorted(glob.glob("data/output_npz/exp_tt/iou/*_metrics.csv")):
    rows = list(csv.DictReader(open(f)))
    n = len(rows)
    iou_pass = sum(1 for r in rows if float(r['iou']) >= 0.8) if n else 0
    iou_avg  = sum(float(r['iou']) for r in rows) / max(n,1)
    print(f"{f}: n={n}  mean_iou={iou_avg:.3f}  pass(iou>=0.8)={iou_pass} ({iou_pass/max(n,1):.1%})")
PY

echo "Done. $(date)"
