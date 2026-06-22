#!/bin/bash
#SBATCH --job-name=sf_eval_partial
#SBATCH --account=3dv
#SBATCH --time=03:00:00
#SBATCH --gpus=5060ti:1
#SBATCH --output=/work/courses/3dv/team15/logs/sf_eval_partial_%j.out
#SBATCH --error=/work/courses/3dv/team15/logs/sf_eval_partial_%j.err

# Partial-view fit quality: model is fed OCCLUDED point clouds
# (trainer.force_occlusions=true) and scored (Chamfer-L1/L2, F-score@1%) against
# the FULL unoccluded cloud -- the regime the fine-tunes were built for.
# Seeded so all three models see the same occlusions.

set -euo pipefail
. /etc/profile.d/modules.sh
module add cuda/12.9
TEAM_DIR="/work/courses/3dv/team15"
REPO_DIR="$TEAM_DIR/superdec_concave"
cd "$REPO_DIR"
source "$TEAM_DIR/superdec/.venv/bin/activate"
export PYTHONPATH="$REPO_DIR:${PYTHONPATH:-}"
export TORCH_EXTENSIONS_DIR="$TEAM_DIR/.torch_extensions"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python setup_sampler.py build_ext --inplace --quiet

run() {  # name folder file normalize
  echo; echo "########### $1 ###########"
  python -m superdec.evaluate.evaluate_partial \
    "checkpoints_folder=$2" "checkpoint_file=$3" \
    "dataloader.dataset=shapenet" "shapenet.split=test" "shapenet.normalize=$4" \
    "+trainer.force_occlusions=true"
}

run "BEFORE: normalized baseline"          "$TEAM_DIR/superdec/checkpoints/normalized" ckpt.pt       true
run "AFTER (chamfer): expocc_tt_chamfer/500" checkpoints/expocc_tt_chamfer              epoch_500.pt false
run "AFTER (bent): expocc_tt_bent/100"       checkpoints/expocc_tt_bent                 epoch_100.pt true

echo; echo "Done. $(date)"
