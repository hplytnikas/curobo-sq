# project_3dv.perception

Tabletop RGB-D perception for robot manipulation.
Converts a depth image (or pre-built point cloud) into a list of fitted
superquadric primitives ready for cuRobo motion planning and Franka Panda
grasp planning.

---

## Pipeline overview

```
depth image + K
      │
      ▼ Stage 1
depth_to_pointcloud()          (N,3) float64 cloud in camera space
      │
      ▼ Stage 2
remove_table()                 (M,3) foreground cloud  +  table geometry
      │
      ▼ Stage 3
segment_instances()            List[(N_i,3)]  one array per object
      │
      ▼ Stage 4a
classify_shape_hint()          ("Ellipsoid"|"Cylinder"|"Cuboid"|"Other", conf)
      │                        called per cluster by TabletopPerception.run()
      ▼ Stage 4b
fit_superquadrics()            List[MultiSQFit]  one set of primitives per object
```

`TabletopPerception.run()` orchestrates Stages 2–4a and returns a
`PerceptionResult`.  Call `fit_superquadrics(result.objects)` for Stage 4b.

---

## Stage details

### Stage 1 — `depth_to_pointcloud(depth, K, depth_scale=1000.0, max_depth=4.0)`

| | |
|---|---|
| **Input** | `depth`: `(H, W)` uint16 or float depth image; `K`: `(3,3)` camera intrinsic matrix |
| **Output** | `(N, 3)` float64 XYZ in camera space (Z forward, Y down, OpenCV convention) |
| **Failure modes** | Returns empty array if all pixels are zero or beyond `max_depth` |

Pixels with depth ≤ 0 or > `max_depth` are discarded.  `depth_scale` defaults
to 1000 (PNG millimetre encoding → metres).  Skip this stage if you already
have a point cloud.

---

### Stage 2 — `remove_table(pts, ..., plane_hint=None)`

| | |
|---|---|
| **Input** | `(N, 3)` cloud (already denoised); optional `plane_hint=(normal, height)` from a prior frame |
| **Output** | `obj_pts (M,3)`, `table_normal (3,)`, `table_height float`, `table_pts (K,3)`, `n_table_pts int` |
| **Failure modes** | Raises `ValueError` if RANSAC returns a degenerate plane (very sparse input) |

Three sequential filters after RANSAC:

1. **Depth gate** — discards points at camera-Z > `table_z_max + depth_margin`.
   Removes background walls visible in tilted camera setups (e.g. ARID20
   "bottom" orientation).
2. **Height filter** — keeps only points `min_height_above_table … max_height_above_table`
   above the fitted plane.
3. **XY radius crop** — discards points beyond `xy_radius` metres from the
   camera origin in the XY plane.

**Coordinate convention** — camera at origin, Z forward.  After RANSAC the
normal is flipped so `table_height ≤ 0`.  Objects *above* the table are
*closer* to the camera (smaller Z), so `height = pts @ table_normal − table_height > 0`.

**`plane_hint`** — pass `(table_normal, table_height)` from the previous
frame to skip RANSAC and save ~200 ms/frame in video sequences.

---

### Stage 3 — `segment_instances(obj_pts, obj_col=None, ...)`

| | |
|---|---|
| **Input** | `(M, 3)` foreground cloud; optional `(M, 3)` float32 RGB colours |
| **Output** | `List[np.ndarray]` — one `(N_i, 3)` array per detected instance |
| **Failure modes** | Returns `[]` on empty input; tiny/noisy clouds may merge into one cluster |

DBSCAN clustering (`eps=0.018 m`, `min_points=20`) followed by four splitting
passes to separate touching or stacked objects:

| Pass | Function | Trigger |
|---|---|---|
| 1 | Horizontal saddle (`_split_cluster`) | density valley in dominant XY-PCA axis |
| 2 | Vertical saddle (`_split_cluster_vertical`) | density valley in Y, only if Y-extent ≥ 40 % of XZ-extent |
| 3A | Normal discontinuity | **disabled** — `open3d` `estimate_normals` segfaults on macOS after DBSCAN |
| 3B | Concavity / neck (`_split_by_concavity`) | narrowest XZ column < 30 % of overall width |
| 3C | Height-layer gaps (`_split_by_height_layers`) | empty Y band ≥ `height_gap_threshold` metres |

Passes are only applied to clusters with ≥ `split_min_points` points (default 600).

**DBSCAN density requirement** — `open3d 0.19` DBSCAN needs dense clouds.
Use volume-uniform sphere sampling (`r = R × U(0,1)^(1/3)`, **not**
`U(0,R)^(1/3)`) in tests.  With n=2000 pts in a sphere of radius 0.04 m,
eps=0.010 m is reliable.

---

### Stage 4a — `classify_shape_hint(cluster)`

| | |
|---|---|
| **Input** | `(N, 3)` cluster array |
| **Output** | `(shape_type: str, confidence: float)` |

Pure geometry heuristic — no trained model required.  Uses bounding-box
extent ratios to return one of `"Ellipsoid"`, `"Cylinder"`, `"Cuboid"`,
`"Other"`.  Used only as a warm-start hint for the SQ fitter; the
authoritative shape type comes from the fitted exponents via
`sq_type_from_exponents(e1, e2)`.

---

### Stage 4b — `fit_superquadrics(segments, fitter=None, l2_threshold=0.008, max_primitives=3)`

| | |
|---|---|
| **Input** | `List[ObjectSegment]` from `PerceptionResult.objects` |
| **Output** | `List[MultiSQFit]` — one per segment, same order |
| **Failure modes** | LM fitter may return high-error fits on very partial views |

`fitter=None` creates a default `SuperquadricFitter` (Levenberg-Marquardt).
Pass a `SuperdecFitter` instance to use the fine-tuned neural fitter:

```python
from project_3dv.perception import fit_superquadrics, SuperdecFitter
fitter  = SuperdecFitter()              # loads fine-tuned tabletop v2 by default
sq_fits = fit_superquadrics(result.objects, fitter=fitter)
```

---

## Public API

```python
from project_3dv.perception import (
    # Pipeline stages
    depth_to_pointcloud,      # Stage 1
    remove_table,             # Stage 2
    segment_instances,        # Stage 3
    classify_shape_hint,      # Stage 4a
    fit_superquadrics,        # Stage 4b

    # Orchestrator + data structures
    TabletopPerception,       # runs Stages 2–4a, returns PerceptionResult
    ObjectSegment,            # one segmented object (points, bbox, shape_type)
    PerceptionResult,         # full pipeline output

    # Superquadric fitting
    SuperquadricFitter,       # classical Levenberg-Marquardt fitter
    SuperdecFitter,           # neural fitter (requires torch + superdec package)
    SuperquadricFit,          # single fitted primitive — 11 params + quality
    MultiSQFit,               # set of primitives for one object
    fits_to_curobo_obstacles, # export to cuRobo collision-checking format

    # Scene / planning interface
    Scene,                    # wraps Superquadrics; .get_signed_distance() for cuRobo
    sq_fits_to_npz,           # export SuperquadricFit list to SuperDec .npz format

    # Grasp planning
    GraspSelector,            # per-primitive grasp candidates for Franka Panda
    GraspCandidate,           # single grasp (pose, gripper_width, score)
)
```

`SuperdecFitter` is imported lazily and set to `None` if `torch` or the
`superdec` package is not installed, so the rest of the module works in
CPU-only CI environments.

---

## Superquadric fit — data structures

### `SuperquadricFit`
```
scales      (3,)    semi-axes a1, a2, a3 in metres
exponents   (2,)    shape exponents e1, e2 ∈ (0, 2]
translation (3,)    centre in world coordinates
rotation    (3, 3)  SO(3) orientation matrix
shape_type  str     "Ellipsoid" | "Cylinder" | "Cuboid" | "Other"
chamfer_l2  float   symmetric Chamfer L2 (quality metric, lower = better)
```

### `MultiSQFit`
```
primitives  List[SuperquadricFit]
n_points    int    number of object points this fit covers
```

---

## Downstream consumers

| Consumer | Interface |
|---|---|
| cuRobo collision avoidance | `fits_to_curobo_obstacles(flat_fits)` |
| cuRobo SDF queries | `Scene.from_fits(flat_fits).get_signed_distance(query_pts)` |
| Grasp planning | `GraspSelector(sq_fit).grasp_candidates()` |
| SuperDec fine-tuning export | `sq_fits_to_npz(fits, path)` |

---

## Checkpoint configuration (`SuperdecFitter`)

`SuperdecFitter` resolves the checkpoint at construction time:

```
superdec_dir/
└── ../checkpoints/
    └── superdec_tabletop/
        └── superdec_tabletop_finetune_v2/   ← default
            └── epoch_300.pt                 ← picked automatically
```

The default `checkpoint_dir` is
`<superdec_dir>/../checkpoints/superdec_tabletop/superdec_tabletop_finetune_v2`.

To override:
```python
fitter = SuperdecFitter(checkpoint_dir="/path/to/my/checkpoint/")
```

The loader handles both naming conventions:
- `ckpt.pt` (base pre-trained checkpoint)
- `epoch_NNN.pt` (fine-tuned; highest epoch is selected automatically)

Fallback: if the tabletop checkpoint directory does not exist, the loader
falls back to the `normalized` base checkpoint inside `superdec_dir`.

---

## Input / output contract with neighbouring modules

### Inputs (from upstream)
| Source | Format |
|---|---|
| Depth camera (ROS `sensor_msgs/Image`) | `(H, W)` uint16 mm, `(3,3)` intrinsic matrix |
| Pre-built point cloud | `(N, 3)` float64 numpy array, camera frame, metres |

### Outputs (to downstream)
| Consumer | What is passed |
|---|---|
| `motion_planning/` (cuRobo) | `fits_to_curobo_obstacles(flat_fits)` → `List[dict]` with `sdf_fn` |
| `grasp_planning/` | `List[ObjectSegment]` (raw) + `List[MultiSQFit]` |
| SuperDec fine-tuning | `sq_fits_to_npz(fits, path)` → `.npz` file |

---

## Running the tests

```bash
source /work/courses/3dv/team15/superdec/.venv/bin/activate
python3 -m pytest tests/perception/ -v
```

Expected output: **75 passed**.  No GPU required — `SuperdecFitter` is not
exercised by the unit tests (requires torch + checkpoint).

Individual test files:

| File | Stage | Tests |
|---|---|---|
| `tests/perception/test_table_removal.py` | Stages 1–2 | 13 |
| `tests/perception/test_instance_segmentation.py` | Stage 3 | 24 |
| `tests/perception/test_sq_fitting.py` | Stage 4b + data structures + exponent clamping | 30 |
| `tests/perception/test_pipeline_integration.py` | End-to-end | 13 |

---

## Testing

### Synthetic tests (no dataset required)

All 75 tests use synthetic point clouds and can be run immediately without any
external data:

```bash
pytest tests/ -m "not requires_data"
```

### Dataset tests (OCID required)

Three additional integration tests run the full pipeline on a real OCID
RGB-D frame and compare against ground-truth instance masks.  They are
marked `@pytest.mark.requires_data` and are **automatically skipped** with a
clear message when the dataset is absent.

**Step 1 — download the dataset** (~15 GB):

```bash
bash scripts/download_ocid.sh          # → data/ocid/ (default)
# or to a custom path:
bash scripts/download_ocid.sh /my/data/ocid
```

The script downloads from the
[OCID dataset page](https://www.acin.tuwien.ac.at/vision-for-robotics/software-tools/object-cluttered-indoor-dataset/)
(TU Wien), verifies the SHA-256 checksum, and extracts the three subsets
(`ARID20/`, `ARID10/`, `YCB10/`) into the destination directory.

**Step 2 — run the dataset tests**:

```bash
pytest tests/ -m requires_data -v
```

| Test | What it checks |
|---|---|
| `test_table_removed` | `remove_table()` reduces the point count on a real OCID frame |
| `test_instances_match_gt` | `segment_instances()` count is within ±2 of the GT label count |
| `test_sq_fits_are_valid` | Every `MultiSQFit` has finite pose, scale > 0, exponents in (0, 2] |

**Run all tests (synthetic + dataset)**:

```bash
pytest tests/ -v
```

---

## Directory structure

```
src/project_3dv/perception/
├── __init__.py              # public API exports
├── pipeline.py              # Stages 1–4a + TabletopPerception orchestrator
├── superquadric.py          # SuperquadricFitter (LM), SuperquadricFit, MultiSQFit
├── superdec_fitter.py       # SuperdecFitter — neural SQ fitting via SuperDec
├── superdec_utils.py        # Scene, Superquadrics, sq_fits_to_npz
├── grasp_from_sq.py         # GraspSelector, GraspCandidate
└── README.md                # this file

tests/perception/
├── test_table_removal.py
├── test_instance_segmentation.py
├── test_sq_fitting.py
└── test_pipeline_integration.py
```

---

## Design decisions and downstream compatibility

### Convex SQ regime

All fitted superquadric primitives are clamped so that both shape exponents
satisfy `SQ_EXPONENT_MIN ≤ e₁, e₂ ≤ SQ_EXPONENT_CONVEX_MAX` (constants
defined at the top of `superdec_fitter.py`; currently 1e-3 and 2.0).

**Why this matters** — the implicit superquadric function

```
F(x,y,z) = ((|x/sx|^(2/e2) + |y/sy|^(2/e2))^(e2/e1) + |z/sz|^(2/e1))
```

has ∂F/∂x ∝ |x|^(2/e2 − 1).  For e2 > 2 the exponent `2/e2 − 1 < 0`,
so the gradient diverges as x → 0.  Gradient-based collision-avoidance
planners (cuRobo CBF-QP) and safety filters that query `∂SDF/∂x` would
receive NaN or ±∞ values, causing the optimiser to fail silently or diverge.
Clamping to the convex regime (e₁, e₂ ≤ 2) guarantees Lipschitz-bounded
gradients throughout the workspace.

The lower bound `SQ_EXPONENT_MIN = 1e-3` prevents division-by-zero in
`sq_radial_distance()`, which computes `norms * |1 − F^(−e1/2)|`; with
e1 → 0 the exponent `−e1/2 → 0` but intermediate powers can still underflow
to zero creating 0/0.  The numerical floor avoids this degenerate case.

If `SuperdecFitter` returns a primitive whose raw exponents fall outside this
range (which can happen for uncommon object shapes not well represented in the
tabletop fine-tuning set), the clamping is applied silently and a
`logging.WARNING` is emitted so the caller can inspect the affected frames.

---

### Surface sampling resolution (`n_u`, `n_v`)

`SuperquadricFit.surface_points(n_u, n_v)` and the underlying
`sq_sample_surface(params, n_u, n_v)` use a deterministic `n_u × n_v`
angular grid (linspace in u ∈ [−π/2, π/2] and v ∈ [−π, π]) rather than
random sampling.  This gives uniform angular coverage and reproducible
Chamfer L2 values across calls.

| Use case | Recommended values | Total surface points |
|---|---|---|
| Real-time perception (default) | `n_u=50, n_v=50` | 2 500 |
| Offline evaluation / fine-tuning | `n_u=100, n_v=100` | 10 000 |
| Fast CI / unit tests | `n_u=16, n_v=16` | 256 |

The Chamfer L2 reported in `SuperquadricFit.chamfer_l2` is computed with the
default 50×50 grid.  For publishable benchmarks, recompute with 100×100 to
reduce discretisation error.

---

### Comparison to registration-based pipelines

This pipeline (depth image → RANSAC plane → DBSCAN → SQ fitting) does **not**
require known object models, CAD meshes, or ICP registration.  It is suitable
for novel tabletop objects seen for the first time, which is a deliberate
design choice for robot manipulation in unstructured environments.

The trade-off relative to model-based pipelines:
- **Lower per-object pose accuracy** — SQ fitting gives an approximate convex
  hull, not a precise 6-DOF pose.  For tasks that require sub-centimetre
  placement accuracy, object-specific templates or dense pose estimators
  (FoundPose, GigaPose) would be more appropriate.
- **Broader generality** — any rigid tabletop object can be handled without
  retraining or adding it to a model database.

SuperDec fine-tuning on tabletop objects (handled by a separate collaborator)
improves SQ decomposition quality for objects in the tabletop distribution
while preserving this model-free property.

---

### What this module does NOT do

The perception module's contract ends at a `List[MultiSQFit]` containing
convex superquadric primitives with valid pose, scale, and shape parameters.
The following are explicitly **downstream responsibilities** (planning module):

| Responsibility | Module |
|---|---|
| GJK distance queries between robot geometry and obstacles | `motion_planning/` |
| CBF-QP safety filter formulation | `motion_planning/` |
| SDF gradient computation for cuRobo trajectory optimisation | `motion_planning/` (via `Scene.get_signed_distance`) |
| Arm-specific collision geometry (Franka Panda) | `motion_planning/` |
| Grasp pose selection and execution | `grasp_planning/` |

The `Scene` and `fits_to_curobo_obstacles` helpers in `superdec_utils.py`
provide the interface bridge, but the planning logic itself is out of scope
for this module.
