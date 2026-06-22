#!/bin/bash
#SBATCH --job-name=sf_eval_fit
#SBATCH --account=3dv
#SBATCH --time=02:00:00
#SBATCH --gpus=5060ti:1
#SBATCH --output=/work/courses/3dv/team15/logs/sf_eval_fit_%j.out
#SBATCH --error=/work/courses/3dv/team15/logs/sf_eval_fit_%j.err

# Reconstruction fit quality (Chamfer-L1/L2 + avg #primitives) BEFORE vs AFTER
# the tabletop fine-tune, on the same ShapeNet test split.
#
#   BEFORE = original SuperDec "normalized" pretrained checkpoint
#            (../superdec/checkpoints/normalized/ckpt.pt, 4D-quat rot head,
#             trained on normalize=true inputs)
#   AFTER  = our fine-tune
#            (checkpoints/expocc_tt_chamfer/epoch_500.pt, 6D rot head,
#             trained on normalize=false inputs)
#
# Each run loads the model architecture from the checkpoint folder's own
# config.yaml, so the rotation head (quat vs 6D) matches the weights and nothing
# gets reinitialized. We feed each model the normalization it was trained with;
# evaluate.py denormalizes predictions back to the original object frame, so the
# Chamfer numbers are measured in the same frame and are directly comparable.

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

echo
echo "###########################################################"
echo "# BEFORE fine-tune: original SuperDec 'normalized' baseline"
echo "###########################################################"
python -m superdec.evaluate.evaluate \
  "checkpoints_folder=$TEAM_DIR/superdec/checkpoints/normalized" \
  "checkpoint_file=ckpt.pt" \
  "dataloader.dataset=shapenet" \
  "shapenet.split=test" \
  "shapenet.normalize=true"

echo
echo "###########################################################"
echo "# AFTER fine-tune: expocc_tt_chamfer / epoch_500"
echo "###########################################################"
python -m superdec.evaluate.evaluate \
  "checkpoints_folder=checkpoints/expocc_tt_chamfer" \
  "checkpoint_file=epoch_500.pt" \
  "dataloader.dataset=shapenet" \
  "shapenet.split=test" \
  "shapenet.normalize=false"

echo
echo "###########################################################"
echo "# AFTER fine-tune (bent/extended SQs): expocc_tt_bent / epoch_100"
echo "###########################################################"
# extended=true (tapering+bending), rotation6d=false (quat); trained normalize=true.
python -m superdec.evaluate.evaluate \
  "checkpoints_folder=checkpoints/expocc_tt_bent" \
  "checkpoint_file=epoch_100.pt" \
  "dataloader.dataset=shapenet" \
  "shapenet.split=test" \
  "shapenet.normalize=true"

echo
echo "Done. $(date)"
