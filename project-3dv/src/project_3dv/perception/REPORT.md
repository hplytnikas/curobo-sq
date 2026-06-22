# Perception Pipeline: Implementation Report
## Project: Robotic Path Planning with Superquadric Primitives
## Team: Isabelle Cretton, Haroldas Plytnikas, Eylül Öykü Şen, Valentin Veluppillai
## Institution: ETH Zürich, 3DV Course 2025
## Last Updated: 2026-03-17

---

## 1  System Overview

The tabletop perception pipeline converts raw sensor input into compact superquadric collision geometry for cuRobo motion planning on a Franka Panda. Execution proceeds through six stages: `pointcloud_from_depth()` unprojection, `remove_table()` RANSAC plane removal, `segment_instances_dual()` PointGroup clustering, `classify_shape_hint()` geometry labelling, `fit_superquadrics()` dispatching to either `SuperdecFitter` (primary, PVCNN neural encoder, Fedele et al. ICCV 2025) or `SuperquadricFitter` (LM baseline), and `postprocess_fits()` inverting all preprocessing transforms. The convenience entry point `single_frame_pipeline()` orchestrates all six stages. The full test suite reports **110 passed, 5 skipped** (115 total, runtime ~11 s). The 5 skipped tests require a CUDA GPU plus the fine-tuned tabletop checkpoint and are in `TestSuperdecFitterSmoke`, gated by combined `@_SKIP_SUPERDEC` and `@_SKIP_NO_CUDA` marks.

---

## 2  Input and Point Cloud Formation

### 2.1  Pre-registered Point Cloud Loading

`RGBDScenesV2` (`datasets/rgbd_scenes.py`) loads binary big-endian PLY files via `plyfile.PlyData.read()`, extracting fields `x y z` (float32) and `diffuse_red diffuse_green diffuse_blue` (uint8) from the `"vertex"` element. The `.label` format has one integer point count on line 1 followed by N per-point instance labels (0 = background). `load()` caches the result; `get_object_clouds()` returns `{label_id: (M,3) float32}` excluding label 0. The dataset has 14 scenes; scene 1 contains at least two distinct non-zero labels (verified by test).

### 2.2  Depth Image Unprojection

`depth_to_pointcloud()` (line 541) accepts uint16 or float depth, a `depth_scale` divisor (default 1000 for mm PNGs), and returns (N, 3) float64 in camera space, clipping beyond `max_depth=4.0` m. `pointcloud_from_depth()` (line 574) accepts float32 metres directly, returns float32, and accepts an optional (4×4) `extrinsic` camera-to-world transform applied via homogeneous multiplication. The two functions serve complementary cases: the former for raw Kinect streams, the latter as the building block for `single_frame_pipeline`.

### 2.3  Single-Frame Pipeline Entry Point

`single_frame_pipeline()` (line 1587) exists. Its stage sequence: `pointcloud_from_depth()` → `remove_table()` (graceful fallback to full cloud on failure) → `segment_instances_dual(adaptive_eps=True)` → per-segment `preprocess_pointcloud()` + `fitter.fit_adaptive()` + `postprocess_fits()`. The `fitter` parameter accepts `'superdec'`, `'lm'`, or a pre-built instance; `'superdec'` falls back transparently to LM when no GPU or checkpoint is available. A `PerceptionTimer` is created automatically if none is passed. Returns a three-tuple of `List[MultiSQFit]`, `SQWorldModel`, and timing dict.

---

## 3  Table Removal

`remove_table()` (line 629) uses Open3D `segment_plane()` with `plane_dist_threshold=0.012` m, `ransac_n=3`, `num_iterations=1000`. Three spatial filters follow. The **depth gate** rejects background points whose camera-Z exceeds `table_z_max + depth_margin` (default 0.25 m), handling walls in angled views. The **height filter** retains points in `(min_height_above_table=0.005, max_height_above_table=0.25)` m above the table, computed via `pts @ table_normal − table_height`. The **XY radius crop** discards points beyond `xy_radius=0.80` m from the camera origin. For RGB-D Scenes v2 evaluation `xy_radius` is widened to 3.0 m because the Kinect field of view spans several metres. The `plane_hint` parameter accepts a `(normal, height)` pair from a previous frame, bypassing RANSAC (~200 ms saving per frame). Five values are returned: `obj_pts`, `table_normal`, `table_height`, `table_pts`, and `n_table_pts`.

---

## 4  Point Cloud Preprocessing

### 4.1  Statistical Outlier Removal

`preprocess_pointcloud()` applies an axis-aligned box test: points outside `centroid ± outlier_std × std` on any axis are dropped (default `outlier_std=2.5`). Per-axis rather than Euclidean removal is used to avoid discarding valid extremal points on elongated objects (e.g., bottle tips lie far from the centroid but pass the per-axis test).

### 4.2  Scale Normalisation and the for_superdec Parameter

`for_superdec=True` (line 293) disables scale normalisation: `meta['scale']` is always 1.0 and `meta['centroid']` is always zero. SuperdecFitter normalises to unit sphere internally via `normalize_points()` in `fit_adaptive()`; applying a second normalisation shifts the input away from the ShapeNet training distribution (Liu et al. CVPR 2022 Sec. 3.4 and Supplementary Sec. 2). Outlier removal, FPS resampling, and table-frame rotation still run when `for_superdec=True`. The parameter appears at lines 293, 340–349, and 1713 of `pipeline.py`.

### 4.3  Table-Frame Canonical Coordinates

When `table_normal` is passed to `preprocess_pointcloud()` (line 374), Step 0A translates the cloud by the segment's height component along the normal (stored in `meta['table_centroid']`), and Step 0B applies the Rodrigues rotation mapping `table_normal → [0,0,1]` computed from `np.cross(n, z)` and `np.eye(3)` without scipy (stored in `meta['table_rotation']`). PCA canonicalisation is skipped. `postprocess_fits()` inverts the table frame as the outermost operation (lines 525–528): `R_table.T @ t_orig + tbl_centroid`. Tests confirm the round-trip recovers the original centroid within 1e-4 m. A `--no-table-frame` ablation flag exists in `test_on_superdec_split.py`. The RGB-D runs show `table_normal` Z-component ≈ −0.35 to −0.39, indicating ~20° camera tilt; raw-Z height filtering would be incorrect in these scenes. Cf. Kim et al. CoRL 2022 (SQPD-Net, Sec. 2).

### 4.4  Per-Category Canonical Rotation

`SHAPENET_CATEGORY_ROTATIONS` (line 254) maps ShapeNet category IDs to rotation matrices applied after preprocessing. Bottle (02876657) and mug (03797390) use identity: the ablation showed R_x(90°) raises their orig-L2 from ~0.074 to ~0.15 and from ~0.088 to ~0.18 respectively, because the fine-tuned model was trained on ShapeNet's native Y-up orientation. Bowl (02880940) keeps R_x(90°): ablation showed 0.105 vs 0.115 orig-L2. Knife (03624134) provisionally retains R_x(90°) (not ablated). Laptop (03642806) and gso have identity rotations.

### 4.5  Uniform Resampling

After any rotation, the cloud is resampled to exactly `target_n=4096` points. Downsampling uses `_fps_numpy()` (Farthest Point Sampling, line 264), which iteratively selects the point furthest from all already-selected points (Qi et al. NeurIPS 2017). Upsampling uses random duplication plus Gaussian jitter σ=0.001 m. The 4096 figure matches the SuperDec training resolution in `superdec/data/dataloader.py`.

---

## 5  Instance Segmentation

### 5.1  Standard DBSCAN Segmentation

`segment_instances()` (line 1009) applies Open3D DBSCAN followed by four splitting passes on clusters exceeding `split_min_points=600`: horizontal density saddle (`_split_cluster()`), vertical density saddle (`_split_cluster_vertical()`), concavity/neck (`_split_by_concavity()`), and height-layer gaps (`_split_by_height_layers()`). `adaptive_cluster_eps()` (line 962) estimates eps as `multiplier × median(nn_distances)` with an optional `eps_max` cap. The height filter uses `pts @ table_normal − table_height` rather than raw camera Z; in tilted-camera scenes (table_normal Z ≈ −0.35) raw-Z filtering would incorrectly reject valid foreground points.

### 5.2  Dual-Set Segmentation

`segment_instances_dual()` (line 1116) implements the PointGroup dual-set algorithm (Jiang et al. CVPR 2020). Set C_p is standard DBSCAN on raw XYZ. The centroid-offset shift replaces each point with `p_i + (mean(kNN(p_i)) − p_i)` using k=20 nearest neighbours, pulling scattered points toward object centres. Set C_q is DBSCAN on the shifted coordinates (original coordinates recovered for output). The union of C_p and C_q is deduplicated: any C_q cluster with point-set IoU > 0.5 against an already-kept cluster is dropped. Unlike the original PointGroup (which uses a learned offset network), this implementation uses geometric k-NN centroids, is CPU-only, and requires no normals. The motivation follows proposal Sec. 2.1: touching same-category objects are the central failure mode of single-set DBSCAN. The `--dual-segment` flag in the eval script controls dispatch, defaulting to standard DBSCAN.

---

## 6  Superquadric Fitting

### 6.1  LM Baseline Fitter

`SuperquadricFitter` (line 608 of `superquadric.py`) optimises via the custom `lm_optimize()` (line 399), not `scipy`. The cost function is the radial distance error: `_residuals()` transforms points to canonical frame and calls `sq_radial_distance()`. The Jacobian is computed by finite differences (step `eps=1e-5`). Defaults: `n_restarts=3`, `n_lm_rounds=15`, `subsample=512`. Each restart uses `init_from_bbox()` (PCA + aspect-ratio hypotheses) with Gaussian noise `±0.05` on subsequent restarts. `fit_adaptive()` recursively splits primitives, accepting the split only when mean L2 improves by more than 20%. `BOUNDS_LO` (line 83) sets e1/e2 lower bound to 0.1; `_clamp_exponents()` (in `superdec_fitter.py`) clips all exponents to [0.1, 1.9] after fitting, following Paschalidou et al. CVPR 2019 Sec. 3.3.

### 6.2  EMS Robust Fitter (Liu et al. CVPR 2022)

NOT YET IMPLEMENTED — see TODO 10. No TRF optimizer, `_compute_inlier_weights` E-step, `_axis_mismatch_candidates` S-step, or `w_o` outlier weight exist in either `superquadric.py` or `superdec_fitter.py`. Partial-view robustness relies entirely on LM restarts, which Liu et al. show are insufficient at partial ratio 0.4.

### 6.3  SuperDec Neural Fitter

`SuperdecFitter` (line 506 of `superdec_fitter.py`) loads `config.yaml` and a `.pt` checkpoint at construction. Default checkpoint: `../checkpoints/superdec_tabletop/superdec_tabletop_finetune_v2/` relative to `superdec_dir`. Network configuration: `n_queries=16`, `exist_threshold=0.3`. Input is XYZ-only at 4096 points; no normals or colour. `_check_input_contract()` (line 578) asserts three preconditions: N ≥ 100 points, all coordinates satisfy `|x| ≤ 2.0` (catching world-scale metres input), and global std > 0.01 (catching degenerate clouds). The coordinate check error message names double normalisation as the likely cause and refers to `for_superdec=True` as the fix (Liu et al. CVPR 2022 Supplementary Sec. 2).

### 6.4  Degenerate Primitive Filtering

`_filter_degenerate_primitives()` (line 217) removes primitives with `min(sx,sy,sz) < min_scale=0.002` m. The threshold was lowered from 0.005 m after ablation found the stricter value removed 63 primitives across the test split. The safety fallback returns the largest primitive by volume when all are degenerate, preventing empty-result outputs.

### 6.5  Primitive Merging

`merge_overlapping_primitives()` (line 289) greedily discards smaller primitives whose AABB IoU with a kept primitive exceeds `iou_threshold` (0.3 for ShapeNet, 0.1 for RGB-D). The `distance_weights` parameter (implemented, line 292) enables global cross-fit merging with effective threshold `iou_threshold / (w_i × w_j)` per primitive pair: low-weight (far) segments resist merging; high-weight (near) segments merge readily. Adapted from SAM2Object (Zhao et al. CVPR 2025, Sec. 3.3.2). The `--no-distance-merge` flag disables this mode.

### 6.6  Exponent Clamping

`_clamp_exponents()` (line 404) clips e1 and e2 in-place to [SQ_EXPONENT_MIN, SQ_EXPONENT_CONVEX_MAX] = [0.1, 1.9], called after inference (lines 700, 786) before degenerate filtering. Range from Paschalidou et al. CVPR 2019 Sec. 3.3: values outside (0, 2] produce non-convex surfaces and numerical instability near e=0.

---

## 7  Surface Sampling and Chamfer L2

### 7.1  Equal-Distance Surface Sampling

`sample_surface_equal_distance()` (line 479 of `superquadric.py`) implements Liu et al. CVPR 2022 Supplementary Sec. 4. It resamples the principal superellipse `{e1, sz}` at equal physical arc-lengths using `_arclength_resample()` (line 450), determines `n_lat ≈ sqrt(n_points)` latitude samples, samples each latitude ring at equal arc-length, then sub/upsamples to exactly `n_points` before applying the SQ pose. Physical coordinates `(sx × cos^e1(η), z)` are used for arc-length so that barrel and polar contributions are weighted by actual extent. The uniformity test confirms max/min latitude band ratio ≤ 3.0, versus ~5–8× for uniform (η, ω) sampling. `_chamfer_l2_from_surface()` uses this with `n_points=500` by default; `--surface-samples` controls the budget.

### 7.2  Bi-directional Chamfer L2

`_chamfer_l2_from_surface()` computes `d_ps + d_sp` with equal weights 1.0 + 1.0 (Paschalidou et al. CVPR 2019, eq. 3). Paschalidou uses asymmetric weights 1.2/0.8 during training; equal weights are standard for evaluation. The test `test_bidirectional_chamfer_larger_than_unidirectional_for_partial_cloud` confirms `d_sp > 0` and contributes meaningfully for partial-view clouds.

### 7.3  Parsimony Score

The parsimony column exists in `test_on_superdec_split.py` (lines 562, 779, 949, 1049). Formula: `n_prims / _N_QUERIES` where `_N_QUERIES = 16` (the SuperDec max primitives from `train_tabletop.yaml`). For LM single-primitive fits: 1/16 = 0.0625. Motivated by Paschalidou et al. CVPR 2019 Sec. 3.2 eq. 12. Actual values from the full 286-sample split are pending GPU evaluation.

### 7.4  Confidence-Weighted Chamfer L2

`confidence_weighted_chamfer()` (line 170 of `superdec_fitter.py`) implements Paschalidou et al. CVPR 2019 eq. 6: `Σ_k(γ_k × L_D) / Σγ_k`. When `total_conf < 1e-12` (LM path, where `fit_superquadrics()` overwrites `prim.shape_conf` from `ObjectSegment.shape_conf=0.0`), the function falls back to the unweighted mean, ensuring the conf-L2 column is always populated in the summary table.

---

## 8  Postprocessing and Coordinate Recovery

`postprocess_fits()` (line 467) inverts preprocessing transforms in order: scale semi-axes by `meta['scale']` (currently 1.0), apply `R_pca.T` to translation and rotation, then — as the outermost step — invert the table frame via `R_table.T @ t + tbl_centroid` when `'table_rotation'` is present in `meta`. Missing keys are graceful no-ops. Tests verify round-trip accuracy to 1e-4 m. The inversion order (table frame outermost) matches the forward order (table frame applied first in preprocessing).

---

## 9  Signed Distance Function and Collision Geometry

### 9.1  SQ Signed Distance Function

`sq_signed_distance_batch(points, params)` (line 246 of `superquadric.py`) is the standalone SQ SDF: it transforms points to canonical frame, evaluates `sq_implicit()`, and returns `sign(f−1) × dr`. It is exposed as `SuperquadricFit.signed_distance(points)` (line 142), which accepts world-frame points. A separate named function `sq_signed_distance(pts, fit)` taking a `SuperquadricFit` object does not exist; `fit.signed_distance(pts)` provides the equivalent interface.

### 9.2  cuRobo Export

`fits_to_curobo_obstacles()` (line 778 of `superquadric.py`) exports each primitive as a dict containing `'type'`, `'id'`, `'shape_type'`, `'shape_conf'`, `'quality_ok'`, `'params'` (scale, shape, translation, rotation, margin), and **`'sdf_fn': fit.signed_distance`** (line 800) — a callable accepting world-frame point arrays. A `'state'` field for dynamic world model transitions is absent. `SQWorldModel.to_curobo_obstacles()` (line 192 of `pipeline.py`) delegates to this function with `margin=0.005` m.

---

## 10  Dynamic World Model

`SQWorldModel` (line 160 of `pipeline.py`) is a dataclass wrapping `List[MultiSQFit]` with properties `n_objects`, `n_primitives`, `all_primitives`, and method `to_curobo_obstacles()`. State-transition methods (`on_grasp`, `on_release`, `on_remove`, `get_active_obstacles`) are **NOT YET IMPLEMENTED** — see TODO 6. Proposal Goal 2 (Sec. 2.2) requires transitioning the target object from a scene obstacle to an attached-payload representation during transport; this lifecycle management is absent.

---

## 11  Latency Instrumentation

`PerceptionTimer` (line 115 of `pipeline.py`) provides `start(stage)`, `stop(stage)` → elapsed seconds, `to_dict()` → `{stage: seconds}`, and `total` property. Stage durations accumulate across repeated calls (safe in per-segment loops). `single_frame_pipeline()` times four stages: `unproject`, `table`, `segment`, and `fit`. The proposal's 200 ms perception budget (Sec. 2.4) is not enforced with a warning; the timing dict is returned for the caller to inspect. On a shared CPU-only cluster node, a fast LM configuration (1 restart, 5 rounds) completes a 120×160 synthetic scene in ~850–920 ms, dominated by fitting.

---

## 12  Datasets and Evaluation

### 12.1  ShapeNet Test Split

The split covers 286 samples across six categories (bottle 50, bowl 18, knife 42, laptop 45, mug 22, gso 103). The eval script generates 4-panel visualisations (raw cloud, segmented cloud, SQ wireframes, overlay) with Y-up → Z-up transform. Two Chamfer L2 metrics are reported: pre-L2 (preprocessed frame) and orig-L2 (after `postprocess_fits()`). EXPERIMENTS.md does not exist in the repository; quantitative results are pending GPU evaluation.

### 12.2  RGB-D Scenes v2

The 5-scene LM evaluation (xy_radius=3.0 m) produced 23–36 detected segments against 7 ground-truth objects per scene. Root cause: `adaptive_cluster_eps()` clamped at `eps_max=0.05` m is too small at 3 m radius, over-segmenting individual objects. `table_normal` Z-component ≈ −0.35 to −0.39 across scenes confirms ~20° camera downward tilt. Dual-set clustering is the proposed fix.

### 12.3  Collision Coverage Metric

`collision_coverage()` is **NOT YET IMPLEMENTED** — see TODO 8. Without it, no quantitative collision-safety claim is possible. Chamfer L2 does not capture the case where a slightly undersized SQ allows the robot arm to pass through the real object geometry (proposal Sec. 2.4: collision rate assessment).

---

## 13  Test Suite

110 passed, 5 skipped (115 total). File breakdown: `test_table_removal.py` — 13 tests; `test_instance_segmentation.py` — 27 tests; `test_preprocessing.py` — 8 tests; `test_sq_fitting.py` — 34 tests (2 GPU-gated in `TestSuperdecFitterSmoke`, 1 in `TestSingleFramePipeline`); `test_sq_improvements.py` — 13 tests; `test_rgbd_scenes.py` — 3 tests (all skipped); `test_pipeline_integration.py` — 12 tests. Registered markers in `pyproject.toml`: `requires_data` (OCID dataset), `requires_rgbd_data` (RGB-D Scenes v2). The 5 skipped tests: 2 GPU/checkpoint in `TestSuperdecFitterSmoke`, 3 data-absent in `test_rgbd_scenes.py`. Fast subset: `pytest tests/ -m "not requires_data and not requires_rgbd_data"`.

---

## 14  Known Limitations

**L1 — Single-view entry point.** `pointcloud_from_depth()` and `single_frame_pipeline()` are implemented. Missing for Franka deployment: calibrated K from Azure Kinect factory calibration, correct depth scale, and a ROS subscriber feeding live depth frames.

**L2 — SuperDec GPU evaluation.** All evaluation uses the LM baseline; the `for_superdec=True` path has never been exercised end-to-end. The GPU-gated smoke tests are skipped on the login node, so `_check_input_contract()` has not been tested with a live model.

**L3 — Segment count gap.** 23–36 segments vs 7 GT objects on RGB-D Scenes v2. `adaptive_cluster_eps()` clamped at 0.05 m over-segments at 3 m xy_radius. `segment_instances_dual()` reduces duplicate proposals but does not fix the over-segmentation from too-small eps.

**L4 — Stacked object detection.** Vertically touching objects form a contiguous DBSCAN cluster. `detect_support_planes()` is not implemented. The splitting heuristics address simple cases but not the general stacked-object scenario from proposal Sec. 2.1.

**L5 — Collision coverage.** `collision_coverage()` is not implemented. Chamfer L2 does not measure whether the SQ representation safely contains the real object geometry.

**L6 — cuRobo true SQ integration.** `fits_to_curobo_obstacles()` exports `'sdf_fn'` as the correct callable. However, native cuRobo support for SQ implicit functions (proposal Goal 3, Sec. 2.3) requires a custom obstacle backend kernel that has not been written.

**L7 — Dynamic world model.** `SQWorldModel` is a read-only dataclass. The state-transition interface for proposal Goal 2 (Sec. 2.2: carried-object representation) is absent.

**L8 — EMS robust fitter.** Liu et al. CVPR 2022 EMS (TRF, E-step, S-step) is not implemented. Partial-view robustness relies on LM restarts, which are insufficient at partial ratio 0.4.

**L9 — No simulation validation.** No Isaac Sim or MuJoCo harness. Proposal Sec. 2.4 treats simulation as the intermediate step before hardware deployment.

**L10 — No ROS interface.** Without a ROS subscriber node, the pipeline cannot run on the Franka Panda in real time.

---

## TODO Status Table

| ID | Task | Status | Notes |
|---|---|---|---|
| **1** | **Single-frame depth entry point** | **DONE** | Resolves L1, Goal 1. `pointcloud_from_depth` (line 574) and `single_frame_pipeline` (line 1587) both exist. |
| **2** | **SuperDec double-normalisation fix** | **DONE** | Resolves L2, Goal 1. `for_superdec` at line 293 of `pipeline.py`; `_check_input_contract` at line 578 of `superdec_fitter.py`. |
| **3** | **Dual-set segmentation** | **DONE** | Resolves L3 partially, Goal 1. `segment_instances_dual` at line 1116 of `pipeline.py`. |
| **4** | **Confidence-weighted Chamfer L2** | **DONE** | Goal 1. `confidence_weighted_chamfer` at line 170 of `superdec_fitter.py` with unweighted-mean fallback. |
| **5** | **True SQ SDF and cuRobo export** | **PARTIAL** | Resolves L5, L6, Goal 3. `fits_to_curobo_obstacles` exports `'sdf_fn': fit.signed_distance` (line 800). No `'state'` field and no native cuRobo SQ kernel yet. |
| **6** | **Dynamic world model** | **PARTIAL** | Resolves L7, Goal 2. `SQWorldModel` dataclass exists (line 160) with properties and `to_curobo_obstacles()`; state-transition methods are still missing. |
| **7** | **Latency instrumentation** | **DONE** | Resolves L8 (measurement), Goal 4. `PerceptionTimer` at line 115; four stages timed in `single_frame_pipeline`. |
| **8** | **Collision coverage metric** | **TODO** | Resolves L5, Goal 4. Not started. |
| **9** | **Stacked object detection** | **TODO** | Resolves L4, Goal 1. Not started. |
| **10** | **EMS robust fitter (Liu et al. CVPR 2022)** | **TODO** | Resolves L8, Goal 1. Not started — no TRF, no outlier weighting, no S-step in either `superquadric.py` or `superdec_fitter.py`. |
| **11** | **Equal-distance surface sampling and Chamfer** | **DONE** | Goal 1. `sample_surface_equal_distance` at line 479 of `superquadric.py`; parsimony and conf-L2 columns in eval script. |
| **12** | **GPU end-to-end SuperDec evaluation** | **TODO** | Depends on TODO 2. Not started — no GPU on login node. |
| **13** | **Isaac Sim / MuJoCo simulation** | **TODO** | Resolves L9, Goal 4. Not started — depends on TODOs 5 and 6. |
| **14** | **ROS interface** | **TODO** | Resolves L10, Goal 4.  Not started — depends on TODOs 1, 5, 6, 7. |

## Summary by Status

| Status | Items |
|---|---|
| **DONE** | 1, 2, 3, 4, 7, 11 |
| **PARTIAL** | 5, 6, 15 |
| **TODO** | 8, 9, 10, 12, 13, 14 |