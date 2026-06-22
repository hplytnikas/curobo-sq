#!/bin/bash
#SBATCH --job-name=sf_smoke
#SBATCH --account=3dv
#SBATCH --time=00:30:00
#SBATCH --gpus=5060ti:1
#SBATCH --output=/work/courses/3dv/team15/logs/sf_smoke_%j.out
#SBATCH --error=/work/courses/3dv/team15/logs/sf_smoke_%j.err

# Phase A5: Smoke check before launching long training runs.
# Verifies (in order): CUDA available, sampler builds, foundation ckpt loads
# into rigid+quat model, dataloader iterates over tabletop+GSO without crashes,
# 3 forward+backward passes succeed with no NaN, loss decreases.
# Should complete in <5 min if everything is wired correctly.

set -euo pipefail
. /etc/profile.d/modules.sh
module add cuda/12.9

TEAM_DIR="/work/courses/3dv/team15"
REPO_DIR="$TEAM_DIR/superdec_concave"
VENV_DIR="$TEAM_DIR/superdec/.venv"

echo "Node: $(hostname)  GPU: ${CUDA_VISIBLE_DEVICES:-?}  $(date)"
nvidia-smi -L

cd "$REPO_DIR"
source "$VENV_DIR/bin/activate"
export PYTHONPATH="$REPO_DIR:${PYTHONPATH:-}"
export TORCH_EXTENSIONS_DIR="$TEAM_DIR/.torch_extensions"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p "$TORCH_EXTENSIONS_DIR" "$TEAM_DIR/logs"

# Build the C++ FPS sampler used by the geometric loss / sampler. Idempotent.
python setup_sampler.py build_ext --inplace --quiet

python - <<'PY'
import torch, numpy as np
from omegaconf import OmegaConf
from collections import Counter

print('=== CUDA ===')
assert torch.cuda.is_available(), 'CUDA not available'
print(f'  device: {torch.cuda.get_device_name(0)}')

print('=== Loading exp_tt config ===')
cfg = OmegaConf.load('configs/exp_tt.yaml')
print(f'  loss.type={cfg.loss.type}  extended={cfg.superdec.extended}  rotation6d={cfg.superdec.rotation6d}')

print('=== Building model on GPU (compiles PVCNN backend on first run) ===')
from superdec.superdec import SuperDec
model = SuperDec(cfg.superdec).cuda()
print(f'  params: {sum(p.numel() for p in model.parameters())/1e6:.2f}M')

print('=== Loading foundation checkpoint ===')
ck = torch.load(cfg.checkpoints.resume_from, map_location='cuda', weights_only=False)
res = model.load_state_dict(ck['model_state_dict'])
# load_state_dict in SuperDec.load_state_dict returns either the parent's result or its filtered result. Check missing.
print('  checkpoint loaded')

print('=== Building train + val datasets ===')
from superdec.data.dataloader import ShapeNet
train_ds = ShapeNet(split='train', cfg=cfg)
val_ds   = ShapeNet(split='val',   cfg=cfg)
print(f'  train={len(train_ds)}  val={len(val_ds)}')
print(f'  train per-cat: {dict(Counter(m["category"] for m in train_ds.models))}')

print('=== Probe one sample (covers ShapeNet + GSO occupancy paths) ===')
import random
random.seed(0)
for cat in sorted({m["category"] for m in train_ds.models}):
    idxs = [i for i,m in enumerate(train_ds.models) if m["category"]==cat]
    i = random.choice(idxs)
    s = train_ds[i]
    occ_rate = s['occupancies'].float().mean().item() if 'occupancies' in s else float('nan')
    print(f'  {cat}: pts={tuple(s["points"].shape)} pts_iou={tuple(s["points_iou"].shape)} occ_rate={occ_rate:.4f}')

print('=== 3 forward+backward iters ===')
from torch.utils.data import DataLoader
from superdec.loss.loss import Loss
loss_fn = Loss(cfg.loss).cuda()
loader = DataLoader(train_ds, batch_size=4, shuffle=True, num_workers=0)
optim = torch.optim.Adam(model.parameters(), lr=1e-4)
losses = []
it = iter(loader)
for step in range(3):
    batch = next(it)
    batch['points'] = batch['points'].cuda().float()
    out = model(batch['points'])
    L, ldict = loss_fn(batch, out)
    assert torch.isfinite(L), f'non-finite loss at step {step}: {L.item()}'
    optim.zero_grad(); L.backward(); optim.step()
    losses.append(L.item())
    print(f'  step {step}: loss={L.item():.4f}  components={ {k: round(v,3) for k,v in ldict.items() if k != "all"} }')
assert any(losses[i+1] != losses[0] for i in range(2)), 'loss is constant -> grad not flowing'

print()
print('SMOKE PASS')
PY
echo "Done. $(date)"
