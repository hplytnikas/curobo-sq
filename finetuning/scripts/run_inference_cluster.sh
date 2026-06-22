#!/bin/bash
#SBATCH --job-name=sf_infer
#SBATCH --account=3dv
#SBATCH --time=02:00:00
#SBATCH --gpus=5060ti:1
#SBATCH --output=/work/courses/3dv/team15/logs/sf_infer_%j.out
#SBATCH --error=/work/courses/3dv/team15/logs/sf_infer_%j.err

# Phase E inference + visualization driver.
# Runs the trained student over the 6-category test set in the four
# eval modes: clean / partial / rotated / partial_rotated. For each mode
# it writes per-object .glb files (input cloud, predicted SQs, combined)
# plus an aggregate predictions.npz. Drag the .glb's into MeshLab / Blender
# / a glTF viewer for visual checks.
#
# Optional env overrides:
#   CKPT=checkpoints/expocc_tt   (default: checkpoints/expocc_tt)
#   EPOCH=epoch_500.pt           (default: latest)
#   OUT_ROOT=predictions         (default: predictions)
#   MAX=120                       (cap shapes per mode for speed; default: 120)
#   MODES="clean partial"        (subset of modes; default: all 4)
#   MESH_RES=80

set -euo pipefail
. /etc/profile.d/modules.sh
module add cuda/12.9

TEAM_DIR="/work/courses/3dv/team15"
REPO_DIR="$TEAM_DIR/superdec_concave"
VENV_DIR="$TEAM_DIR/superdec/.venv"

cd "$REPO_DIR"
source "$VENV_DIR/bin/activate"
export PYTHONPATH="$REPO_DIR:${PYTHONPATH:-}"
export TORCH_EXTENSIONS_DIR="$TEAM_DIR/.torch_extensions"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p "$TORCH_EXTENSIONS_DIR" "$TEAM_DIR/logs"

python setup_sampler.py build_ext --inplace --quiet

CKPT="${CKPT:-checkpoints/expocc_tt}"
EPOCH="${EPOCH:-}"
OUT_ROOT="${OUT_ROOT:-predictions}"
MAX="${MAX:-120}"
MESH_RES="${MESH_RES:-80}"
NO_MESHES="${NO_MESHES:-0}"   # set to 1 to skip per-object .glb dumps (npz only)
MODES_DEFAULT="clean partial rotated partial_rotated"
MODES="${MODES:-$MODES_DEFAULT}"

# Sanity
test -d "$REPO_DIR/$CKPT" || { echo "ERROR: checkpoint dir $CKPT not found"; exit 2; }
test -f "$REPO_DIR/$CKPT/config.yaml" || { echo "ERROR: $CKPT/config.yaml missing"; exit 2; }

EPOCH_FLAGS=()
if [ -n "$EPOCH" ]; then
    EPOCH_FLAGS+=(--epoch "$EPOCH")
fi

echo "Node: $(hostname)  GPU: ${CUDA_VISIBLE_DEVICES:-?}  $(date)"
echo "ckpt=$CKPT  epoch=${EPOCH:-latest}  max=$MAX  modes=[$MODES]"

for MODE in $MODES; do
    case "$MODE" in
        clean)            FLAGS=() ;;
        partial)          FLAGS=(--partial) ;;
        rotated)          FLAGS=(--rotate) ;;
        partial_rotated)  FLAGS=(--partial --rotate) ;;
        *) echo "Unknown MODE=$MODE"; exit 2 ;;
    esac
    OUT_DIR="$OUT_ROOT/$(basename "$CKPT")_${MODE}"
    echo "=== mode=$MODE -> $OUT_DIR ==="
    EXTRA_FLAGS=()
    if [ "$NO_MESHES" = "1" ]; then
        EXTRA_FLAGS+=(--no_meshes)
    fi
    if [ -n "$MAX" ] && [ "$MAX" -gt 0 ] 2>/dev/null; then
        EXTRA_FLAGS+=(--max_objects "$MAX")
    fi
    python scripts/run_inference.py \
        --checkpoint "$CKPT" \
        "${EPOCH_FLAGS[@]}" \
        --output "$OUT_DIR" \
        --mesh_resolution "$MESH_RES" \
        "${FLAGS[@]}" \
        "${EXTRA_FLAGS[@]}"
done

echo "Done. $(date)"
echo "Outputs under $OUT_ROOT/. scp the dirs to your laptop and view the .glb files."
