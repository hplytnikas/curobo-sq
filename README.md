# Superquadric Obstacle Integration for CuRobo

This repository integrates **superquadric obstacle representations** into
[CuRobo](https://curobo.org/) - NVIDIA's CUDA-accelerated robot motion planning
library. Superquadrics enable more compact and differentiable scene
representations than triangle meshes, and avoid the scalability issues of
voxel grids.

The pipeline has two components:

| Component | Directory | Role |
|-----------|-----------|------|
| **SuperDec** | `superdec/` | Neural network that decomposes a point cloud into superquadric primitives |
| **CuRobo (fork)** | `curobov2/` | CuRobo v2 fork with superquadric support implemented in NVIDIA Warp (no C++ compilation) |

---

## Table of Contents

1. [Install](#install)
2. [Reproducing the Paper Results](#reproducing-the-paper-results)
3. [Superquadric SDF Math](#superquadric-sdf-math)
4. [CuRobo - Changed Files](#curobo--changed-files)
5. [CuRobo - Python API](#curobo--python-api)

---

## Install

```bash
# Run all install commands from the repo root
cd /path/to/3DV   # adjust to wherever you cloned the repo

# (Ubuntu 25.04+) GCC 15 is incompatible with CUDA 12.8 headers.
# Install GCC 14 — SuperDec's JIT backend selects it automatically.
sudo apt install gcc-14 g++-14

# Environment
conda create -n 3dv python=3.11 -y
conda activate 3dv

# PyTorch (CUDA 12.8 wheels)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128

# CuRobo v2 kernel backend (no compilation required)
pip install 'cuda-core[cu12]'

# CuRobo v2 (Warp backend - no CUDA C++ build), then SuperDec, both editable
pip install -e curobov2/curobo --no-build-isolation
pip install -r superdec/requirements.txt && pip install -e superdec --no-build-isolation

# Benchmark / plotting extras
pip install pandas scipy scikit-learn

# Sanity check
python -c "import curobo, superdec, warp, torch; print('ok', torch.__version__)"
```

GPU: an Ada-class card (RTX 4070 Ti Super / arch 8.9) with CUDA 12.8 is the reference setup.

---

## Reproducing the Paper Results

The paper experiments live in `curobov2/curobo/curobo/examples/paper/` and run
inside the **`3dv` conda environment** on the **CuRobo v2** (Warp) backend. All
downloaded assets go into a **single folder at the repo root, `data/paper/`**,
and every script looks there by default - so there is nothing to configure once
the files are in place. Run the commands from the paper folder:

```bash
cd curobov2/curobo/curobo/examples/paper
```

Finetuning of SuperDec/SuperFlex is described seperately - please refer to the
`finetuning` directory.

### 1. Download the assets

Download the four files below and extract them into `data/paper/` (paths are
relative to the repository root):

| Asset | Download | Extract to | Needed for |
|-------|----------|------------|------------|
| **scenes_cache.pkl** - prebuilt benchmark scenes (SuperDec already run) | [link](https://polybox.ethz.ch/index.php/s/CHTEFx52QoBfdyG) | `data/paper/scenes_cache.pkl` | fast benchmark |
| **tabletop_finetuned** - SuperDec checkpoint (`ckpt.pt` + `config.yaml`) | [link](https://polybox.ethz.ch/index.php/s/s4rdAKmTLjgsfsP) | `data/paper/tabletop_finetuned/` | demo, building scenes from scratch |
| **ShapeNet_test** - test-split object point clouds | [link](https://polybox.ethz.ch/index.php/s/tttTR5ikcbeiHAS) | `data/paper/ShapeNet_test/` | demo, building scenes from scratch |
| **chair.ply** - single-object cloud | [link](https://polybox.ethz.ch/index.php/s/doY3dfcXH6GYsJ6) | `data/paper/chair.ply` | sofa-scaling benchmark |

Final layout:

```
data/paper/
├── scenes_cache.pkl      # prebuilt benchmark scenes (fast path)
├── tabletop_finetuned/   # SuperDec checkpoint: ckpt.pt + config.yaml
├── ShapeNet_test/        # {synset}/test.lst + {synset}/{model}/pointcloud.npz
└── chair.ply             # single object for the sofa-scaling benchmark
```

For just the **fast path** (reproduce the benchmark numbers and figures) you only
need `scenes_cache.pkl` - the cache already contains the SuperDec predictions, so
no checkpoint and no ShapeNet are required.

### 2. Run - fast path (only `scenes_cache.pkl`)

Reproduces the benchmark numbers and figures with no checkpoint or ShapeNet (run inside the 3dv conda environment):

```bash
python benchmark_sq_vs_mesh.py benchmark   # → eval_out/results.csv
python plot_benchmark.py                   # → eval_out/objects_vs_*.png
python plot_benchmark_paper.py             # → eval_out/paper_*.png
```

### 3. Run - full pipeline (needs `ShapeNet_test` + `tabletop_finetuned`)

Regenerates every scene from the dataset and re-runs SuperDec inference:

```bash
# Interactive Viser demo on http://localhost:8080 (optional sanity check)
conda run -n 3dv python motion_planning_sq_demo.py

# Benchmark, in order: build scenes → set 4 targets/scene → plan SQ vs mesh
python benchmark_sq_vs_mesh.py build          # Builds random scenes → data/paper/scenes_cache.pkl
python benchmark_sq_vs_mesh.py set-targets --port 8080   # GUI for setting targets → eval_out/targets.json
python benchmark_sq_vs_mesh.py benchmark      # Runs the benchmark for planning time → eval_out/results.csv

# Sofa-scaling benchmark (needs data/paper/chair.ply)
python benchmark_sofa_scaling.py --counts 1,3,6,21,51
```

Restrict object counts with `--counts 1,5,10`; the mesh-fidelity sweep is
`benchmark_sq_vs_mesh.py benchmark --fidelity`. `targets.json` and `results.csv`
are in the repository, so the figures can be regenerated without re-running anything.

### 4. What each file in `paper/` does

| File | Role |
|------|------|
| `motion_planning_sq_demo.py` | Interactive Viser demo - tabletop scenes from ShapeNet/GSO, decomposed by SuperDec into superquadrics, with live SQ↔mesh switching and motion planning. Also the shared library (scene generation, planner construction) imported by the benchmarks. |
| `benchmark_sq_vs_mesh.py` | Main benchmark harness (`build` / `set-targets` / `benchmark` subcommands). Plans a sequential tour over scenes of increasing object count for the SQ, mesh and point-cloud representations; `--fidelity` runs the mesh tessellation sweep. |
| `benchmark_sofa_scaling.py` | Object-size scaling ("sofa") benchmark - replicates one large object N times and measures planning time vs. number of primitives. |
| `plot_benchmark.py` | Figures from `results.csv`: planning time, motion time and collision rate vs. object count. |
| `plot_benchmark_paper.py` | Paper-styled figures from `results.csv`: planning-time and collision bar charts plus a combined figure. |
| `plot_fidelity.py` | Figures from `results_fidelity.csv`: planning time and SQ-vs-mesh speedup vs. mesh fidelity. |
| `plot_sofa_scaling.py` | Figure from `results_sofa.csv`: planning time vs. number of primitives. |
| `eval_out/` | Generated outputs - `results*.csv`, the committed `targets.json`, and the PNG figures. |

---

## Superquadric SDF Math

This section documents the exact formulas evaluated by the superquadric
collision kernel.

The implementation is the Warp kernel in:
`curobov2/curobo/curobo/_src/geom/data/data_superquadric.py`.

For a query sphere center `p_world = (px, py, pz)` and radius `r`:

1. Transform to SQ local frame using inverse quaternion rotation:

    `p_local = (x, y, z) = R_world_to_local * (p_world - c)`

2. Define normalized magnitudes:

    `ax = |x|/sx, ay = |y|/sy, az = |z|/sz`

3. Superquadric implicit function:

    `p1 = 2/eps1, p2 = 2/eps2, er = eps2/eps1`

    `F(x,y,z) = (ax^p2 + ay^p2)^er + az^p1`

    Surface is `F = 1`, outside is `F > 1`, inside is `F < 1`.

#### Outside distance (Taubin first-order approximation)

The raw sphere-vs-SQ signed distance in kernel sign convention
(positive outside, negative inside) is:

`d_raw_out ~= (F - 1)/||gradF|| - r`

with local gradient:

`gradF = (dF/dx, dF/dy, dF/dz)`

`dF/dx = (2/eps1) * (1/sx) * sign(x) * (ax^p2 + ay^p2)^(er - 1) * ax^(p2 - 1)`

`dF/dy = (2/eps1) * (1/sy) * sign(y) * (ax^p2 + ay^p2)^(er - 1) * ay^(p2 - 1)`

`dF/dz = (2/eps1) * (1/sz) * sign(z) * az^(p1 - 1)`

and `||gradF|| = sqrt((dF/dx)^2 + (dF/dy)^2 + (dF/dz)^2)`.

The implementation then applies conservative lower bounds and returns:

`d_raw = max(d_raw_out, lb_sphere, lb_aabb)`

where:

- `lb_sphere = ||p_world - c|| - max(sx,sy,sz) - r`
- `lb_aabb = ||max(|p_local| - (sx,sy,sz), 0)||_2 - r`

#### Inside distance (Newton radial projection)

For `F < 1`, the kernel avoids flat-cost behavior by solving for `lambda` such
that:

`F(lambda * p_local) = 1`

Newton iteration:

`lambda_{k+1} = lambda_k - (F(lambda_k p_local) - 1) / (gradF(lambda_k p_local) dot p_local)`

with initialization:

`lambda_0 = max(1, F_init^(-eps1/2))`

and the inside signed distance:

`d_raw_in = (1 - lambda) * ||p_local|| - r`

Special case near the SQ center (`||p_local|| < 1e-6`) returns
`-min(sx,sy,sz) - r`.

#### Analytical normal and world-frame gradient direction

The unit outward normal is computed in local frame and rotated to world frame:

`n_local = gradF / ||gradF||`

`n_world = R_local_to_world * n_local`

- Outside path: `gradF` is evaluated at the query point.
- Inside path: `gradF` is evaluated at the converged surface point
  `lambda * p_local`.

#### CuRobo sign convention and collision-cost gradient

The kernel distance uses `d_raw > 0` outside. CuRobo uses the opposite sign:

`sdf_curobo = -d_raw`

so `sdf_curobo > 0` means penetration.

Collision cost per obstacle uses a smooth quadratic-linear activation with
activation distance `a`:

- `cost = 0`, if `sdf <= 0`
- `cost = 0.5/a * sdf^2`, if `0 < sdf <= a`
- `cost = sdf - 0.5*a`, if `sdf > a`

Its derivative w.r.t. `sdf` is:

- `cost' = 0`, if `sdf <= 0`
- `cost' = sdf/a`, if `0 < sdf <= a`
- `cost' = 1`, if `sdf > a`

Given outward normal `n_world`, the position gradient used by autograd is:

`d(cost)/d(p_world) = -weight * cost'(sdf_curobo) * n_world`

For sum-collisions mode, this is summed over all obstacles; for min-distance
mode, it uses the closest obstacle normal.

---

## CuRobo - Changed Files

The superquadric integration (`curobov2/`) is implemented in
[NVIDIA Warp](https://github.com/NVIDIA/warp) - GPU kernels written in Python and
compiled at runtime, with no separate build step. The SDF algorithm is the
Newton radial projection + Taubin approximation described in
[Superquadric SDF Math](#superquadric-sdf-math).

No compilation step is required. Set `PYTHONPATH` to the source tree (run from
the repository root):

```bash
export PYTHONPATH="$PWD/curobov2/curobo"
export PATH=/usr/local/cuda-12.8/bin:/usr/bin:$PATH
```

### New files

#### `curobov2/curobo/curobo/_src/geom/data/data_superquadric.py` *(new)*

All Warp GPU kernels and Python tensor management for superquadric obstacles.

**Warp struct exposed to kernels:**

```python
@wp.struct
class SuperquadricDataWarp:
    params:    wp.array2d(dtype=wp.float32)   # (num_envs*max_n, 8) → [sx,sy,sz, ε1,ε2, pad,pad,pad]
    inv_pose:  wp.array2d(dtype=wp.float32)   # (num_envs*max_n, 8) → [x,y,z, qw,qx,qy,qz, pad]
    enable:    wp.array(dtype=wp.uint8)        # (num_envs*max_n,) enable mask
    n_per_env: wp.array(dtype=wp.int32)        # (num_envs,) active count per env
    max_n:     wp.int32
    num_envs:  wp.int32
```

**Warp SDF functions (called by the generic collision kernel via the plugin registry):**

| Function | Signature | Description |
|----------|-----------|-------------|
| `is_obs_enabled` | `(obs_set, env_idx, local_idx) → wp.bool` | Enable mask lookup |
| `load_obstacle_transform` | `(obs_set, env_idx, local_idx) → wp.transform` | Inverse pose for frame transform |
| `compute_local_sdf` | `(obs_set, env_idx, local_idx, pt) → float32` | SDF value only |
| `compute_local_sdf_with_grad` | `(obs_set, env_idx, local_idx, pt) → wp.vec4` | `(sdf, ∂x, ∂y, ∂z)` |

**Python dataclass `SuperquadricData`:**

| Method | Description |
|--------|-------------|
| `create_cache(max_n, num_envs, device_cfg)` | Allocate empty GPU tensors |
| `from_scene_cfg(scene_cfg, device_cfg, ...)` | Load from a `SceneCfg` |
| `from_batch_scene_cfg(scene_cfg_list, ...)` | Multi-environment load |
| `add(sq, env_idx)` | Add one obstacle; returns its index |
| `load_batch(sqs, env_idx)` | Replace all obstacles in an environment |
| `update_pose(name, w_obj_pose, obj_w_pose, env_idx)` | In-place pose update |
| `set_enabled(name, enabled, env_idx)` | Toggle obstacle visibility |
| `get_idx(name, env_idx)` | Obstacle index lookup |
| `get_active_count(env_idx)` | Number of enabled obstacles |
| `to_warp()` | Convert to `SuperquadricDataWarp` for kernel calls |

#### `curobov2/curobo/docs/guides/superquadric_obstacles.md` *(new)*

Full user guide: environment setup, scene definition, collision query API,
`SceneCollision` integration, multi-environment use, test instructions, and
implementation notes. Canonical reference for the v2 API.

---

### Modified files

#### `curobov2/curobo/curobo/_src/geom/types.py`

**New `Superquadric` dataclass** (alongside existing `Cuboid`, `Mesh`, `Blox`):

```python
@dataclass
class Superquadric(Obstacle):
    radii: List[float] = field(default_factory=lambda: [0.1, 0.1, 0.1])  # [a₁, a₂, a₃]
    shape: List[float] = field(default_factory=lambda: [1.0, 1.0])       # [ε₁, ε₂]

    def get_trimesh_mesh(self) -> trimesh.Trimesh: ...  # 32×32 parametric surface
```

The shape exponents `[ε₁, ε₂]` are stored in the `shape` field; `radii` are the
semi-axes `[sx, sy, sz]`.

**`SceneCfg` additions:**

- New field: `superquadric: Optional[List[Superquadric]] = None`
- `__post_init__`: appends SQs to the unified `objects` list
- `create()`: parses `data_dict["superquadric"]` from dict/YAML
- `add_obstacle()`: routes `Superquadric` instances correctly
- `create_mesh_scene()`: tessellates SQs for mesh-based comparison
- `clone()`: deep-copies the superquadric list

#### `curobov2/curobo/curobo/_src/geom/data/registry.py`

Added `data_superquadric` to the obstacle plugin list:

```python
OBSTACLE_SDF_MODULES = [
    "curobo._src.geom.data.data_cuboid",
    "curobo._src.geom.data.data_mesh",
    "curobo._src.geom.data.data_voxel",
    "curobo._src.geom.data.data_superquadric",   # ← added
]
```

The generic Warp collision kernel iterates this list and dispatches to each
module's `compute_local_sdf_with_grad`. New obstacle types can be added by
implementing that interface and appending a module here.

#### `curobov2/curobo/curobo/_src/geom/data/data_scene.py`

- New field: `superquadrics: Optional[SuperquadricData] = None`
- `create_cache(..., superquadric_cache: Optional[int] = None)` - allocates GPU buffer
- `from_scene_cfg(...)` / `from_batch_scene_cfg(...)` - pass-through for SQ cache
- `add_obstacle(sq)` - routes `Superquadric` to `self.superquadrics.add()`
- `update_obstacle_pose(name, pose, env_idx)` - checks SQs if not found elsewhere
- `enable_obstacle(name, enabled, env_idx)` - toggles SQ visibility
- `get_obstacle_names()` - includes SQ names
- `has_superquadrics() → bool`
- `get_active_types() → dict` - returns `{"superquadric": bool, ...}`

#### `curobov2/curobo/curobo/_src/geom/collision/collision_scene.py`

- `SceneCollisionCfg.cache` accepts `"superquadric": int` key
- `SceneCollision.from_config()` extracts and passes `superquadric_cache`
- `collision_types` property now includes `"superquadric": bool`

---

### Examples & tests

| File | Description |
|------|-------------|
| `curobov2/curobo/curobo/examples/getting_started/superquadric_motion_planning.py` | Full Isaac Sim demo: 3 SQ obstacles, live collision query, pose update, optional GUI |
| `curobov2/curobo/curobo/examples/getting_started/motion_planning_sq.py` | Production example: SuperDec inference → SQ/mesh scene → motion planning with timing logs |
| `curobov2/curobo/curobo/examples/getting_started/motion_gen_sq_simple.py` | Minimal benchmark: single SQ obstacle, SQ vs mesh timing, CSV output |
| `curobov2/curobo/curobo/tests/_src/geom/test_superquadric_sdf.py` | 11 unit/integration tests covering tensor creation, SDF sign/value, gradients |

Run tests:

```bash
PYTHONPATH="$PWD/curobov2/curobo" \
PATH=/usr/local/cuda-12.8/bin:/usr/bin:$PATH \
~/isaacsim/python.sh \
  curobov2/curobo/curobo/tests/_src/geom/test_superquadric_sdf.py
```

Run the demo:

```bash
PYTHONPATH="$PWD/curobov2/curobo" \
PATH=/usr/local/cuda-12.8/bin:/usr/bin:$PATH \
~/isaacsim/python.sh \
  curobov2/curobo/curobo/examples/getting_started/superquadric_motion_planning.py
```

---

## CuRobo - Python API

### `curobo._src.geom.types.Superquadric`

```python
from curobo._src.geom.types import Superquadric

sq = Superquadric(
    name="obstacle",
    pose=[x, y, z, qw, qx, qy, qz],   # world-frame pose
    radii=[a1, a2, a3],                 # semi-axes in metres
    shape=[e1, e2],                     # shape exponents (0, 2]
)
```

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `name` | `str` | - | Unique obstacle identifier |
| `pose` | `List[float]` (7) | - | `[x, y, z, qw, qx, qy, qz]` |
| `radii` | `List[float]` (3) | `[0.1, 0.1, 0.1]` | Semi-axes `[a₁, a₂, a₃]` |
| `shape` | `List[float]` (2) | `[1.0, 1.0]` | Exponents `[ε₁, ε₂]` |
| `color` | `Optional[List[float]]` | `None` | RGBA for visualisation |

### Scene setup

```python
import warp as wp
wp.init()

from curobo._src.geom.types import SceneCfg, Superquadric
from curobo._src.geom.collision.collision_scene import SceneCollision, SceneCollisionCfg
from curobo._src.types.device_cfg import DeviceCfg

device_cfg = DeviceCfg(device="cuda")

scene_cfg = SceneCfg(
    superquadric=[
        Superquadric(name="box",  pose=[0.5,0,0.4, 1,0,0,0], radii=[0.1,0.1,0.1], shape=[0.1,0.1]),
        Superquadric(name="ball", pose=[0.3,0,0.5, 1,0,0,0], radii=[0.08,0.08,0.08], shape=[1,1]),
    ]
)

cfg = SceneCollisionCfg(
    device_cfg=device_cfg,
    scene_model=scene_cfg,
    cache={"cuboid": 0, "superquadric": 8},   # pre-allocate for up to 8 SQs
)
scene = SceneCollision.from_config(cfg)

print(scene.collision_types)
# → {'cuboid': False, 'mesh': False, 'voxel': False, 'superquadric': True}
```

### Collision query

```python
import torch
from curobo._src.geom.collision.buffer_collision import CollisionBuffer
from curobo._src.geom.collision.checker_collision import CollisionChecker

# Shape: [batch, horizon, n_spheres, 4] - (x, y, z, radius)
query = torch.tensor([[[[0.5, 0.0, 0.4, 0.02]]]], dtype=torch.float32, device="cuda")

buf      = CollisionBuffer.from_shape(query.shape[:3], device_cfg)
weight   = device_cfg.to_device([1.0])
act_dist = device_cfg.to_device([0.0])
env_idx  = torch.zeros(1, dtype=torch.int32, device="cuda")

checker = CollisionChecker(device_cfg=device_cfg)
cost = checker.get_sphere_distance(
    scene.data, query, buf, weight, act_dist, env_query_idx=env_idx
)
# cost > 0 → collision; cost = 0 → free
```

### Dynamic obstacle management

```python
from curobo._src.types.pose import Pose

new_pose = Pose.from_list([0.6, 0.0, 0.4, 1.0, 0.0, 0.0, 0.0], device_cfg=device_cfg)
scene.data.update_obstacle_pose("box", new_pose, env_idx=0)

scene.data.enable_obstacle("box", enable=False, env_idx=0)   # hide
scene.data.enable_obstacle("box", enable=True,  env_idx=0)   # show
```

### Multi-environment

```python
cfg = SceneCollisionCfg(
    device_cfg=device_cfg,
    scene_model=[scene_cfg_0, scene_cfg_1, scene_cfg_2],   # num_envs inferred
    cache={"superquadric": 8},
)
```
