# Tabletop fine-tuning of SuperDec/SuperFlex

## Layout
- `configs/`     — fine-tuning configs (rigid teacher → student, bent variant)
- `scripts/`     — pseudo-GT generation, training, inference, evaluation
- `superdec/`    — partial-view evaluation code
- `result/`      — reconstruction-quality table (paper Table 1)
- `demo_viser.py`— qualitative superquadric-fit visualizations
- `patches/`     — my edits to upstream files (dataloader, loss, trainer)

## Pipeline (reproduce the deployed rigid checkpoint)
0. `scripts/preprocess_shapenet_occupancy.py`   — occupancy for ShapeNet tabletop objects (from binvox)
1. `scripts/preprocess_gso_occupancy.py`        — occupancy for GSO objects
2. `scripts/train_exp_tt_chamfer_cluster.sh`    — Stage 1: rigid teacher (configs/exp_tt_chamfer.yaml)
3. `scripts/generate_gt_tt_chamfer_cluster.sh` + `scripts/rebalance_pseudo_gt.py`
                                                — emit + per-category-IoU-rebalance pseudo-GT
4. `scripts/train_expocc_tt_chamfer_cluster.sh` — Stage 2: supervised student (configs/expocc_tt_chamfer.yaml)

Deformable (experimental) variant: `scripts/train_expocc_tt_bent_cluster.sh`
(single-stage from the SuperFlex deformable checkpoint, configs/expocc_tt_bent.yaml).

## Evaluation & visualization
- `scripts/eval_fit_quality_cluster.sh` / `eval_fit_quality_partial_cluster.sh`
  + `superdec/evaluate/evaluate_partial.py` → `result/fit_quality_table.md`
- `scripts/make_synth_scene.py` — standalone scene builder: synthesizes
  TO-Scene-style tabletop scenes from ShapeNet + GSO point clouds
- `demo_viser.py` — qualitative superquadric-fit visualizations
- `demo_viser_tabletop.py` — single-checkpoint viser viewer across all tabletop
  test objects, with interactive rotation/occlusion controls (auto-detects
  rigid vs deformable from the checkpoint config)

## What is changed vs stock SuperFlex
Recipe inherited verbatim (supervised geom loss + weights, pseudo-GT
distillation, occlusion augmentation). Our changes: deformation heads disabled
(rigid fits), full-SO(3) rotation augmentation + 6D parametrization,
unnormalized-frame training, per-category IoU rebalancing (vs IoU≥0.8 filter),
5× longer schedule.

## Note on checkpoints
Trained weights (`expocc_tt_chamfer/epoch_500.pt`,
`expocc_tt_bent/epoch_100.pt`, ~24 MB each) are not included here.
