# Tabletop fine-tuning of SuperDec/SuperFlex

Fine-tunes the SuperFlex superquadric decomposition model on a tabletop object
set (5 ShapeNet categories + GSO) to produce cuRobo-compatible rigid superquadric
fits.

This directory is an **overlay on the upstream SuperFlex training repo** (a
SuperDec fork, internally `superdec_concave`) — it does not run standalone. Clone
that repo, apply our patch, and drop these files in (see Setup).

## Setup

1. **Clone SuperFlex** (code link on the project page, https://superflex3d.github.io/):
   ```bash
   git clone <superflex-repo-url> superflex && cd superflex
   ```
2. **Apply our patch** — dataloader lazy-FPS + GSO occupancy unpack, rigid-safe
   geometric loss, wandb-artifact opt-out (touches `superdec/data/dataloader.py`,
   `superdec/loss/loss.py`, `train/trainer.py`):
   ```bash
   git apply /path/to/finetuning/patches/upstream_edits.patch
   ```
3. **Overlay this directory** so Hydra and the scripts resolve:
   ```bash
   cp -r /path/to/finetuning/configs/* configs/
   cp -r /path/to/finetuning/scripts/  scripts/
   cp /path/to/finetuning/superdec/evaluate/evaluate_partial.py superdec/evaluate/
   ```
4. **Build the FPS sampler:**
   ```bash
   python setup_sampler.py build_ext --inplace
   ```

## Dependencies

Tested with Python 3.9. External libraries used by the code in this directory:

| library | version | link |
|---------|---------|------|
| PyTorch | 2.8.0 | https://pytorch.org/ |
| NumPy | 2.0.2 | https://numpy.org/ |
| SciPy | 1.13.1 | https://scipy.org/ |
| trimesh | 4.11.5 | https://trimesh.org/ |
| OmegaConf | 2.3.0 | https://github.com/omry/omegaconf |
| Hydra | 1.3.2 | https://hydra.cc/ |
| viser | 1.0.26 | https://github.com/nerfstudio-project/viser |
| tqdm | 4.67.3 | https://github.com/tqdm/tqdm |
| PyTorch Geometric | 2.6.1 | https://github.com/pyg-team/pytorch_geometric |

The training/eval entry points (`train/train.py`, the `superdec` package,
`superoptim`, `setup_sampler.py`) come from the upstream **SuperFlex** repo, which
itself builds on **SuperDec** — these are external code, not ours (see Setup).
Our own contribution is this overlay: the configs, scripts, the partial-view
evaluator, and the edits in `patches/upstream_edits.patch`.

## Data

Expected layout: `data/ShapeNet/<category>/<instance>/`, with categories
bottle `02876657`, bowl `02880940`, knife `03624134`, laptop `03642806`,
mug `03797390`, and `gso`.
- ShapeNet objects need `models/model_normalized.solid.binvox`.
- GSO objects need `pointcloud.npz` (surface points + normals).

Copy in the committed splits, then generate occupancy:
```bash
# 1) splits — the exact train/val/test ids used in the report
for c in 02876657 02880940 03624134 03642806 03797390 gso; do
  cp /path/to/finetuning/splits/$c/*.lst data/ShapeNet/$c/
done
# 2) occupancy -> points.npz per object
python scripts/preprocess_shapenet_occupancy.py   # from binvox
python scripts/preprocess_gso_occupancy.py        # from normals
```

Two ready-to-use example objects (one ShapeNet, one GSO) are committed under
`data_samples/ShapeNet/` in this exact layout — see `data_samples/README.md`.

## Checkpoints

Our two trained checkpoints are committed under `finetuning/checkpoints/`
(each ~23 MB, with `config.yaml`). Copy them into the SuperFlex repo:
```bash
cp -r /path/to/finetuning/checkpoints/* checkpoints/
```
| name  | path                                         | role |
|-------|----------------------------------------------|------|
| rigid | `checkpoints/expocc_tt_chamfer/epoch_500.pt` | deployed rigid student |
| bent  | `checkpoints/expocc_tt_bent/epoch_100.pt`    | experimental deformable variant |

The base weights (`normalized/ckpt.pt` eval baseline) are **not** included — download them from
the upstream SuperDec repo.

## Pipeline (reproduce the rigid checkpoint)

Training/eval run through SLURM drivers in `scripts/*_cluster.sh`. **Edit the
`#SBATCH` header and the `TEAM_DIR`/`REPO_DIR`/`VENV_DIR` block at the top of each
for your environment** (ours targets `/work/courses/3dv/team15`, `cuda/12.9`, a
5060ti GPU).

1. `train_exp_tt_chamfer_cluster.sh` — Stage 1: rigid teacher.
2. `generate_gt_tt_chamfer_cluster.sh` + `rebalance_pseudo_gt.py` — emit and
   per-category-IoU-rebalance the pseudo-GT.
3. `train_expocc_tt_chamfer_cluster.sh` — Stage 2: supervised student.

Deformable variant: `train_expocc_tt_bent_cluster.sh` (single stage from the
SuperFlex deformable checkpoint).

## Evaluation & visualization

- `eval_fit_quality_cluster.sh` / `eval_fit_quality_partial_cluster.sh` +
  `superdec/evaluate/evaluate_partial.py` → `result/fit_quality_table.md`.
- `demo_viser_tabletop.py` — interactive viser viewer over the test set
  (rotation/occlusion controls; auto-detects rigid vs deformable). Flags:
  `--shapenet_root`, `--extra_ply`.
- `make_synth_scene.py` — synthesize TO-Scene-style tabletop scenes from
  ShapeNet + GSO point clouds (`--shapenet_root` defaults to `data/ShapeNet`).

## What we changed vs stock SuperFlex

Recipe inherited verbatim (supervised geom loss + weights, pseudo-GT
distillation, occlusion augmentation). Our changes: deformation heads disabled
(rigid fits), full-SO(3) rotation augmentation with 6D parametrization,
unnormalized-frame training, per-category IoU rebalancing (vs a global IoU≥0.8
filter), and a 5× longer schedule.

## Directory layout

| dir | contents |
|-----|----------|
| `configs/`     | fine-tuning configs (teacher, student, bent) |
| `scripts/`     | preprocessing, training, pseudo-GT, eval drivers |
| `superdec/`    | partial-view evaluation code (overlay) |
| `splits/`      | committed train/val/test id lists |
| `data_samples/`| two example objects (ShapeNet + GSO) in dataloader layout |
| `checkpoints/` | committed rigid + bent weights (+ `config.yaml`) |
| `patches/`     | `upstream_edits.patch` for the SuperFlex repo |
| `result/`      | reconstruction-quality table |
</content>
