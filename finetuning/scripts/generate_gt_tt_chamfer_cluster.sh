#!/bin/bash
#SBATCH --job-name=sf_gt_tt_chamfer
#SBATCH --account=3dv
#SBATCH --time=12:00:00
#SBATCH --gpus=5060ti:1
#SBATCH --output=/work/courses/3dv/team15/logs/sf_gt_tt_chamfer_%j.out
#SBATCH --error=/work/courses/3dv/team15/logs/sf_gt_tt_chamfer_%j.err

# Phase C (Chamfer variant): Generate pseudo-GT for the student from the
# Chamfer teacher checkpoint. Same pipeline as generate_gt_tt_cluster.sh
# but reads checkpoints/exp_tt_chamfer/ and writes data/output_npz/exp_tt_chamfer/.
#
# Step 1: dump teacher predictions (data/output_npz/exp_tt_chamfer/shapenet_<split>.npz)
# Step 2: per-shape rigid IoU optimization (data/output_npz/exp_tt_chamfer/iou/<split>.npz)
# Step 3: print per-category metrics for filter decision.

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

CKPT_FOLDER="${CKPT_FOLDER:-exp_tt_chamfer}"
CKPT_FILE="${CKPT_FILE:-epoch_500.pt}"
SOURCE_FOLDER="${SOURCE_FOLDER:-exp_tt_chamfer}"

test -f "$REPO_DIR/checkpoints/$CKPT_FOLDER/$CKPT_FILE" \
    || { echo "ERROR: teacher checkpoint checkpoints/$CKPT_FOLDER/$CKPT_FILE missing"; exit 2; }

# --- Step 1: dump teacher raw predictions ---
for SPLIT in train val; do
    echo "[$(date)] Step 1: to_npz split=$SPLIT"
    python -m superdec.evaluate.to_npz \
        --config-path="../../configs/optim_tt" --config-name=save_npz \
        "checkpoints_folder=checkpoints/$CKPT_FOLDER" \
        "checkpoint_file=$CKPT_FILE" \
        "output_dir=data/output_npz/$SOURCE_FOLDER" \
        "dataloader.split=$SPLIT"
done

# --- Step 2: per-shape rigid test-time optimization (IoU-based) ---
for SPLIT in train val; do
    echo "[$(date)] Step 2: batch_evaluate split=$SPLIT (rigid IoU optim)"
    python -m superoptim.batch_evaluate \
        --config-path="../configs/optim_tt" --config-name=batch_optim \
        "+source_folder=$SOURCE_FOLDER" \
        "shapenet.split=$SPLIT"
done

# --- Step 3: per-category IoU stats so we can pick the filter ---
python - <<PY
import csv, glob, os
from collections import defaultdict
CATS = {"02876657":"bottle","02880940":"bowl","03624134":"knife",
        "03642806":"laptop","03797390":"mug","gso":"gso"}
def cat_of(name, ids_by_cat):
    for c,ids in ids_by_cat.items():
        if name in ids: return c
    return "?"
def load_ids(split):
    out={}
    for c in CATS:
        p=f"data/ShapeNet/{c}/{split}.lst"
        if os.path.exists(p):
            out[c]=set(l.strip() for l in open(p) if l.strip())
    return out
for split in ("train","val"):
    f=f"data/output_npz/$SOURCE_FOLDER/iou/{split}_metrics.csv"
    if not os.path.exists(f):
        print(f"missing {f}"); continue
    rows=list(csv.DictReader(open(f)))
    ids=load_ids(split)
    by_cat=defaultdict(list)
    for r in rows: by_cat[cat_of(r["name"],ids)].append(float(r["iou"]))
    print(f"\\n=== {split} ===  n={len(rows)}  mean_iou={sum(float(r['iou']) for r in rows)/max(len(rows),1):.3f}")
    print(f"{'cat':<8}{'name':<8}{'n':>6}{'mean':>8}{'p>=0.8':>9}{'p>=0.7':>9}{'p>=0.6':>9}")
    for c in sorted(by_cat):
        vs=by_cat[c]; n=len(vs)
        mean=sum(vs)/max(n,1)
        p8=sum(1 for v in vs if v>=0.8)/max(n,1)
        p7=sum(1 for v in vs if v>=0.7)/max(n,1)
        p6=sum(1 for v in vs if v>=0.6)/max(n,1)
        print(f"{c:<8}{CATS.get(c,'?'):<8}{n:>6}{mean:>8.3f}{p8:>9.1%}{p7:>9.1%}{p6:>9.1%}")
PY

echo "Done. $(date)"
echo
echo "Next: choose filter strategy based on per-cat stats above, then either"
echo "  (A) point student config at data/output_npz/exp_tt_chamfer/iou/{train,val}.npz with threshold 0.8, OR"
echo "  (B) python scripts/rebalance_pseudo_gt.py --top_frac 0.7 --min_iou 0.6 (and use iou_balanced/)"
