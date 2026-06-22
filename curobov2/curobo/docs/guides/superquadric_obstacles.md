# Superquadric Obstacles in cuRobo v2

This guide explains how to define superquadric (SQ) obstacles and use them for
collision checking and motion planning in cuRobo v2 with Isaac Sim Python.

---

## What are superquadrics?

A superquadric is a family of shapes parameterized by three semi-axes
(a₁, a₂, a₃) and two exponents (ε₁, ε₂):

```
F(x,y,z) = ( (|x/a₁|^{2/ε₂} + |y/a₂|^{2/ε₂})^{ε₂/ε₁} + |z/a₃|^{2/ε₁} )
```

The surface is `F = 1`. Special cases:

| ε₁  | ε₂  | Shape          |
|-----|-----|----------------|
| 1.0 | 1.0 | Ellipsoid / sphere (when a₁=a₂=a₃) |
| 0.1 | 0.1 | Box-like        |
| 1.0 | 0.1 | Cylinder-like   |

Compared to mesh-based obstacles, superquadrics provide **exact, differentiable
signed distances** without approximation, making them ideal for gradient-based
trajectory optimisation (MPPI, iCEM, etc.).

---

## Environment setup

cuRobo v2 uses [NVIDIA Warp](https://github.com/NVIDIA/warp) for GPU kernels.
Isaac Sim ships with Warp 1.12.0.  No separate installation is needed — just
point `PYTHONPATH` at the curobov2 source tree:

```bash
export PYTHONPATH=/home/haroldas/3DV/curobov2/curobo
export PATH=/usr/local/cuda-12.8/bin:/usr/bin:$PATH

# Verify Warp is visible
~/isaacsim/python.sh -c "import warp; print(warp.__version__)"
# → 1.12.0
```

> **Alias**: the `omni_python` shell alias wraps `PATH=... ~/isaacsim/python.sh`.
> You can use it instead of the full command below.

---

## Defining superquadric obstacles

Import and instantiate `Superquadric` objects, then include them in a `SceneCfg`:

```python
import warp as wp
wp.init()

from curobo._src.geom.types import SceneCfg, Superquadric, Cuboid

scene_cfg = SceneCfg(
    superquadric=[
        # Sphere-like (ε₁=ε₂=1)
        Superquadric(
            name="sq_ellipsoid",
            pose=[0.5, 0.0, 0.4,  1.0, 0.0, 0.0, 0.0],  # [x,y,z, qw,qx,qy,qz]
            radii=[0.08, 0.08, 0.08],                      # semi-axes [a₁, a₂, a₃]
            shape=[1.0, 1.0],                              # exponents [ε₁, ε₂]
        ),
        # Cylinder-like (ε₁=1, ε₂→0)
        Superquadric(
            name="sq_cylinder",
            pose=[0.4, 0.2, 0.5,  1.0, 0.0, 0.0, 0.0],
            radii=[0.05, 0.05, 0.15],
            shape=[1.0, 0.1],
        ),
        # Box-like (ε₁≈ε₂≈0)
        Superquadric(
            name="sq_box",
            pose=[0.35, -0.2, 0.35,  1.0, 0.0, 0.0, 0.0],
            radii=[0.06, 0.10, 0.08],
            shape=[0.1, 0.1],
        ),
    ],
    cuboid=[
        Cuboid(
            name="floor",
            pose=[0.0, 0.0, -0.025,  1.0, 0.0, 0.0, 0.0],
            dims=[2.0, 2.0, 0.05],
        ),
    ],
)
```

### `Superquadric` field reference

| Field  | Type        | Description |
|--------|-------------|-------------|
| `name` | `str`       | Unique obstacle identifier |
| `pose` | `List[float]` (7) | `[x, y, z, qw, qx, qy, qz]` in world frame |
| `radii`| `List[float]` (3) | Semi-axes `[a₁, a₂, a₃]` in metres |
| `shape`| `List[float]` (2) | Exponents `[ε₁, ε₂]`; both in `(0, 2]`; avoid exactly 0 |

---

## Running the collision-query example

The ready-made example at
`curobov2/curobo/curobo/examples/getting_started/superquadric_motion_planning.py`
queries three probe spheres against the scene above and reports their collision
status:

```bash
PYTHONPATH=/home/haroldas/3DV/curobov2/curobo \
PATH=/usr/local/cuda-12.8/bin:/usr/bin:$PATH \
~/isaacsim/python.sh \
  curobov2/curobo/curobo/examples/getting_started/superquadric_motion_planning.py
```

Expected output:

```
Scene: 3 superquadrics, 1 cuboids
  sq_ellipsoid: radii=[0.08, 0.08, 0.08], shape=[1.0, 1.0]
  sq_cylinder:  radii=[0.05, 0.05, 0.15], shape=[1.0, 0.1]
  sq_box:       radii=[0.06, 0.1, 0.08],  shape=[0.1, 0.1]

--- Collision Query Demo ---
  Probe 0 (inside sq_ellipsoid): cost=0.0900  [COLLISION]
  Probe 1 (free space):          cost=0.0000  [FREE]
  Probe 2 (near sq_cylinder):   cost=0.0700  [COLLISION]

--- Motion Planning Scene Setup ---
  Active collision types: {'cuboid': True, 'mesh': False, 'voxel': False, 'superquadric': True}
  Superquadric count (env 0): 3
  Updated sq_ellipsoid pose to [0.6, 0, 0.4, identity]
  SceneCollision with superquadrics is ready for trajectory optimisation.

Done.
```

---

## Writing your own collision query

```python
import warp as wp
wp.init()

import torch
from curobo._src.geom.types import SceneCfg, Superquadric
from curobo._src.geom.data.data_scene import SceneData
from curobo._src.geom.collision.buffer_collision import CollisionBuffer
from curobo._src.geom.collision.checker_collision import CollisionChecker
from curobo._src.types.device_cfg import DeviceCfg

device_cfg = DeviceCfg(device="cuda")

scene_cfg = SceneCfg(
    superquadric=[
        Superquadric(
            name="obstacle",
            pose=[0.5, 0.0, 0.4, 1.0, 0.0, 0.0, 0.0],
            radii=[0.1, 0.1, 0.1],
            shape=[1.0, 1.0],
        )
    ]
)

# Load scene into GPU tensors
scene_data = SceneData.from_scene_cfg(scene_cfg, device_cfg)

# Query spheres: shape [batch, horizon, n_spheres, 4] → [x, y, z, radius]
query_spheres = torch.tensor(
    [[[[0.5, 0.0, 0.4, 0.02]]]],     # sphere at obstacle centre
    dtype=device_cfg.dtype,
    device=device_cfg.device,
)

buf     = CollisionBuffer.from_shape(query_spheres.shape[:3], device_cfg)
weight  = device_cfg.to_device([1.0])
act_dist = device_cfg.to_device([0.0])
env_idx = torch.zeros(1, dtype=torch.int32, device=device_cfg.device)

checker = CollisionChecker(device_cfg=device_cfg)
cost = checker.get_sphere_distance(
    scene_data, query_spheres, buf, weight, act_dist,
    env_query_idx=env_idx,
)

print(f"cost = {cost[0, 0, 0].item():.4f}")  # > 0 → collision
```

---

## Using SceneCollision (motion planning integration)

`SceneCollision` bundles scene data with the collision checker for use inside
the cuRobo trajectory optimiser:

```python
from curobo._src.geom.collision.collision_scene import SceneCollision, SceneCollisionCfg

cfg = SceneCollisionCfg(
    device_cfg=device_cfg,
    scene_model=scene_cfg,      # your SceneCfg
    num_envs=1,
    # Pre-allocate GPU buffers; set larger than you need
    cache={"cuboid": 10, "superquadric": 16},
)
scene = SceneCollision.from_config(cfg)

print(scene.collision_types)
# {'cuboid': True, 'mesh': False, 'voxel': False, 'superquadric': True}
```

### Dynamic obstacle updates

```python
from curobo._src.types.pose import Pose

new_pose = Pose.from_list(
    [0.6, 0.0, 0.4, 1.0, 0.0, 0.0, 0.0], device_cfg=device_cfg
)
scene.data.update_obstacle_pose("obstacle", new_pose, env_idx=0)
scene.data.enable_obstacle("obstacle", enable=False, env_idx=0)  # hide
scene.data.enable_obstacle("obstacle", enable=True,  env_idx=0)  # show
```

### Multi-environment setup

Pass a list of `SceneCfg` objects — one per environment — and each environment
gets its own independent set of obstacles on the GPU:

```python
cfg = SceneCollisionCfg(
    device_cfg=device_cfg,
    scene_model=[scene_cfg_0, scene_cfg_1, scene_cfg_2],  # num_envs inferred
    cache={"superquadric": 8},
)
```

---

## Running the tests

```bash
PYTHONPATH=/home/haroldas/3DV/curobov2/curobo \
PATH=/usr/local/cuda-12.8/bin:/usr/bin:$PATH \
~/isaacsim/python.sh \
  curobov2/curobo/curobo/tests/_src/geom/test_superquadric_sdf.py
```

Expected output (all 11 tests):

```
=== Superquadric SDF Integration Tests ===

PASS: test_superquadric_data_create
PASS: test_superquadric_scene_cfg
PASS: test_superquadric_data_from_scene_cfg
PASS: test_to_warp_conversion
PASS: test_scene_data_with_superquadrics
PASS: test_sphere_outside_ellipsoid
PASS: test_sphere_inside_ellipsoid
PASS: test_sphere_surface_ellipsoid
PASS: test_boxy_superquadric
PASS: test_translated_superquadric
PASS: test_gradient_flows  (|grad|=...)

=== All tests passed! ===
```

---

## Implementation notes

### SDF algorithm

The kernel uses two branches:

- **Outside** (`F ≥ 1`): Taubin first-order approximation,
  `d ≈ (F−1) / ‖∇F‖`, clamped below by the bounding-sphere and AABB lower
  bounds so gradients are always conservative.
- **Inside** (`F < 1`): Newton radial projection (4 iterations) starting from
  `λ = max(1, F^{−ε₁/2})` to avoid overshoot for boxy shapes.

Gradients are computed analytically and passed back through
`torch.autograd` via the Warp autograd bridge in
`collision/wp_autograd.py`.

### Quaternion convention

Poses are stored as `[x, y, z, qw, qx, qy, qz]` — **`qw` first** in the
rotation part.  This matches the rest of cuRobo v2.

### Adding more obstacle types

The collision system uses a plugin registry. To add a new shape:

1. Create `curobo/_src/geom/data/data_<shape>.py` implementing
   `is_obs_enabled`, `load_obstacle_transform`, `compute_local_sdf_with_grad`.
2. Add the module path to `OBSTACLE_SDF_MODULES` in
   `curobo/_src/geom/data/registry.py`.

No changes to the generic collision kernel are required.

---

## File map

| Path | Purpose |
|------|---------|
| `curobo/_src/geom/types.py` | `Superquadric` dataclass, `SceneCfg` |
| `curobo/_src/geom/data/data_superquadric.py` | Warp struct + SDF functions |
| `curobo/_src/geom/data/data_scene.py` | `SceneData` — loads SQs to GPU |
| `curobo/_src/geom/collision/collision_scene.py` | `SceneCollision` facade |
| `curobo/_src/geom/data/registry.py` | Plugin module list |
| `curobo/tests/_src/geom/test_superquadric_sdf.py` | Integration tests |
| `curobo/examples/getting_started/superquadric_motion_planning.py` | Runnable example |
