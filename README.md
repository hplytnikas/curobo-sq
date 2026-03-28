# Superquadric Obstacle Integration for CuRobo

This repository integrates **superquadric obstacle representations** into
[CuRobo](https://curobo.org/) — NVIDIA's CUDA-accelerated robot motion planning
library. Superquadrics enable more compact and differentiable scene
representations than triangle meshes, and avoid the scalability issues of
voxel grids.

The pipeline has three components:

| Component | Directory | Role |
|-----------|-----------|------|
| **SuperDec** | `superdec/` | Neural network that decomposes a point cloud into superquadric primitives |
| **CuRobo (fork)** | `curobo/` | Motion planner extended with native SQ SDF/collision kernels |
| **OpenGJK** | `openGJK/` | Standalone GJK library — **not used in the planning pipeline**, kept for independent geometry validation |

---

## Table of Contents

1. [Environment & Build](#environment--build)
2. [Quick Start](#quick-start)
3. [New Files Added](#new-files-added)
4. [Python API — Superquadric Types](#python-api--superquadric-types)
5. [Python API — Collision World](#python-api--collision-world)
6. [CUDA Kernel API](#cuda-kernel-api)
7. [Integration Demo CLI Reference](#integration-demo-cli-reference)
8. [Architecture](#architecture)

---

## Environment & Build

### Python interpreter

All Python commands use the Isaac Sim interpreter with CUDA 12.8 in `PATH`:

```bash
PATH=/usr/local/cuda-12.8/bin:/usr/bin:$PATH ~/isaacsim/python.sh <script.py>
```

There is an `omni_python` alias that wraps this (aliase ~/isaacsim/python.sh to omni_python if you haven't done so). The virtual environment at
`.venv/` is separate and used only for SuperDec training/inference.

### Build CuRobo CUDA extensions

After modifying any `.cu` or `.cpp` file under `curobo/src/curobolib/cpp/`:

```bash
PATH=/usr/local/cuda-12.8/bin:/usr/bin:$PATH ~/isaacsim/python.sh \
    -m pip install -e curobo/ --no-build-isolation
```

GPU architecture target: **8.9** (RTX 4090 / Ada Lovelace).
NVCC flags: `-O3 --ftz=true --fmad=true`.

---

## Quick Start

```bash
# Run the Isaac Sim integration demo (native SQ collision)
omni_python curobo/examples/isaac_sim/motion_gen_reacher_superquadrics.py \
    --world_representation superquadrics \
    --ply_path superdec/examples/chair.ply

# Compare against the mesh-based collision representation
omni_python curobo/examples/isaac_sim/motion_gen_reacher_superquadrics.py \
    --world_representation mesh \
    --ply_path superdec/examples/chair.ply


# Export SuperDec meshes for Isaac Sim visualisation
cd superdec && python export_for_isaacsim.py

# Superquadric SDF regression tests
omni_python curobo/tests/test_sq_rotation.py
omni_python curobo/tests/test_sq_motion_gen_headless.py
omni_python curobo/tests/test_sq_clpt.py
```

---

## New Files Added

The files below were added on top of the upstream CuRobo and SuperDec
repositories to implement the superquadric integration.

### SuperDec

#### `superdec/export_for_isaacsim.py`

Standalone script that runs SuperDec inference on a `.ply` point cloud and
exports the inferred superquadrics as triangle meshes for use in Isaac Sim.

```python
# Hardcoded config at the top of the script:
checkpoints_folder = "checkpoints/normalized"
path_to_point_cloud = "examples/chair.ply"
output_dir          = "/tmp/superdec_meshes"
resolution          = 30   # mesh tessellation resolution
```

Outputs written to `output_dir`:
- `chair_superquadrics.obj` — all primitives merged into a single mesh
- `sq_0.obj`, `sq_1.obj`, … — one `.obj` per active primitive

#### `superdec/superdec_to_curobo_world.py`

Utility that runs inference and returns a CuRobo `WorldConfig` with `Mesh`
obstacles (the mesh-based predecessor to the native SQ kernel):

```python
from superdec_to_curobo_world import superdec_to_curobo_world

world: WorldConfig = superdec_to_curobo_world(
    ply_path="examples/chair.ply",
    checkpoint_folder="checkpoints/normalized",
    output_dir="/tmp/superdec_meshes",
    resolution=30,
)
```

---

### CuRobo — CUDA Kernels

#### `curobo/src/curobolib/cpp/superquadric_radial_distance_kernel.cu` *(new)*

The **active superquadric collision kernel**. Implements batched
sphere-vs-superquadric SDF evaluation and analytical gradient computation
entirely in CUDA, with no external library dependencies.

Key components:

- **`SQData` struct** — 48-byte aligned GPU layout per superquadric:
  `cx, cy, cz` (centre), `sx, sy, sz` (semi-axes), `eps1, eps2` (shape
  exponents), `qw, qx, qy, qz` (orientation quaternion).

- **`pack_env_sq`** — reorders the Python-side `[nenv, maxobs, 12]` parameter
  tensor into packed `SQData` arrays, clamps exponents to [0.05, 4.0], and
  normalises quaternions. Returns only enabled obstacles.

- **`sphere_superquadric_clpt`** — main entry point for static collision
  queries. Evaluates all spheres against all SQs in their assigned environment
  and writes per-sphere cost/ESDF and analytical gradients.

- **`swept_sphere_superquadric_clpt`** — swept-sphere variant for trajectory
  optimisation. Integrates velocity via a `speed_dt` tensor; gradients via
  3-point finite differences.

The SDF uses two regimes:
- **Outside** (`F ≥ 1`): Taubin first-order approximation for fast evaluation.
- **Inside** (`F < 1`): Newton radial projection — finds λ such that
  `F(λ·p_local) = 1`, giving smooth gradients near and inside obstacles.

#### `curobo/src/curobolib/cpp/superquadric_distance_kernel.cu` *(legacy, not compiled)*

Original GJK/OpenGJK-based kernel from an earlier prototype. **Not listed in
`setup.py` and not compiled.** Retained for reference; a header comment makes
this explicit.

---

### CuRobo — Python Integration

#### `curobo/src/curobo/geom/types.py` — `Superquadric` class *(new)*

New obstacle dataclass added to CuRobo's geometry system. See
[Python API — Superquadric Types](#python-api--superquadric-types).

#### `curobo/src/curobo/geom/sdf/world.py` — superquadric loading *(extended)*

Extended `WorldPrimitiveCollision` to load `Superquadric` objects from
`WorldConfig`, pack them into GPU tensors, and route collision queries to the
radial-distance kernel.

#### `curobo/src/curobo/curobolib/geom.py` — autograd wrappers *(new)*

`SdfSphereSuperquadric` and `SdfSweptSphereSuperquadric` — PyTorch
`autograd.Function` subclasses that call the CUDA kernel and expose
gradients to the trajectory optimiser.

#### `curobo/src/curobolib/cpp/geom_cuda.cpp` — pybind11 bindings *(extended)*

Exposes the new CUDA functions to Python via the `geom_cu` module:
- `geom_cu.closest_point_superquadric`
- `geom_cu.swept_closest_point_superquadric`

---

### CuRobo — Examples & Tests

| File | Description |
|------|-------------|
| `curobo/examples/isaac_sim/motion_gen_reacher_superquadrics.py` | Full Isaac Sim demo: SuperDec inference → SQ collision world → motion generation. Supports `--world_representation superquadrics\|mesh` for comparison. |
| `curobo/examples/isaac_sim/motion_gen_reacher_superquadrics_simple.py` | Lightweight headless version; omits Isaac Sim scene setup. Useful for debugging the collision pipeline without a full simulator. |
| `curobo/tests/test_sq_rotation.py` | Regression tests for quaternion/rotation conventions in `pack_env_sq`. 5 scenarios covering axis-aligned and rotated SQs. |
| `curobo/tests/test_sq_motion_gen_headless.py` | Integration test: gradient-descent loop drives spheres out of SQ obstacles using the analytical kernel gradient. Verifies no NaN, monotone loss decrease, and convergence to zero cost. |
| `curobo/tests/test_sq_clpt.py` | Checks the radial-distance kernel against known Franka retract-pose spheres and scene SQs captured from a real planning session. |

---

## Python API — Superquadric Types

### `curobo.geom.types.Superquadric`

```python
from curobo.geom.types import Superquadric

sq = Superquadric(
    name="chair_back",                        # unique obstacle identifier
    pose=[x, y, z, qw, qx, qy, qz],          # world-frame pose (metres + unit quaternion)
    radii=[a1, a2, a3],                       # semi-axes along local x, y, z (metres)
    eps=[e1, e2],                             # shape exponents
    color=[r, g, b, a],                       # optional, for visualisation (0–1 floats)
)
```

**Fields:**

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `name` | `str` | — | Unique obstacle name (required) |
| `pose` | `List[float]` | — | `[x, y, z, qw, qx, qy, qz]` world-frame pose |
| `radii` | `List[float]` | `[0.05, 0.05, 0.05]` | Semi-axes [a₁, a₂, a₃] in metres |
| `eps` | `List[float]` | `[1.0, 1.0]` | Shape exponents [ε₁, ε₂] |
| `color` | `Optional[List[float]]` | `None` | RGBA for visualisation |
| `scale` | `Optional[List[float]]` | `None` | Optional uniform/non-uniform scale |

**Shape exponent guide:**

| ε₁, ε₂ | Shape |
|---------|-------|
| 0.1–0.4 | Super-ellipsoid (boxy, sharp edges) |
| 1.0 | Ellipsoid |
| 2.0 | Biconcave / pillow-like |

**Method:**

```python
sq.get_trimesh_mesh(process=True) -> trimesh.Trimesh
```

Returns a triangulated surface for visualisation. Collision checks always use
the native SDF representation, not this mesh.

---

### `curobo.geom.types.WorldConfig`

Pass superquadrics to the planner via `WorldConfig`:

```python
from curobo.geom.types import WorldConfig, Superquadric

world = WorldConfig(superquadric=[sq1, sq2, sq3])
```

`WorldConfig.superquadric` is a `List[Superquadric]`. All other obstacle types
(`cuboid`, `mesh`, `blox`) can be combined in the same `WorldConfig`.

---

## Python API — Collision World

### Setup

```python
from curobo.geom.sdf.world import (
    WorldPrimitiveCollision,
    WorldCollisionConfig,
    CollisionQueryBuffer,
)
from curobo.types.base import TensorDeviceType

tensor_args = TensorDeviceType(device="cuda:0", dtype=torch.float32)

cfg = WorldCollisionConfig(
    tensor_args=tensor_args,
    world_model=WorldConfig(superquadric=[sq1, sq2]),
    cache={"obb": 0, "superquadric": 4},  # pre-allocate space for up to 4 SQs
)
world = WorldPrimitiveCollision(cfg)
```

### Querying the SDF

```python
# Sphere tensor: [batch, horizon, n_spheres, 4] = (x, y, z, radius)
spheres = torch.tensor([[x, y, z, r]], device="cuda:0").view(1, 1, 1, 4)

query_buf = CollisionQueryBuffer.initialize_from_shape(
    spheres.shape, tensor_args, world.collision_types
)
weight   = tensor_args.to_device([1.0])
act_dist = tensor_args.to_device([0.0])
env_idx  = tensor_args.to_device([0]).to(torch.int32)

# --- Cost mode: 0 = outside, positive = penetrating ---
dist = world.get_sphere_distance(
    spheres, query_buf, weight, act_dist,
    env_query_idx=env_idx,
    compute_esdf=False,
    sum_collisions=True,
)

# --- ESDF mode: negative = outside (|val| = gap), positive = penetration depth ---
dist = world.get_sphere_distance(
    spheres, query_buf, weight, act_dist,
    env_query_idx=env_idx,
    compute_esdf=True,
    sum_collisions=False,
)
```

**`compute_esdf` return value conventions:**

| Mode | Sign | Meaning |
|------|------|---------|
| cost (`False`) | `= 0` | Sphere is outside all obstacles |
| cost (`False`) | `> 0` | Sphere penetrates; value = weighted cost |
| ESDF (`True`) | `< 0` | Sphere is outside; `|value|` = distance to nearest SQ surface |
| ESDF (`True`) | `> 0` | Sphere penetrates; value = penetration depth |

### Reading analytical gradients

After a forward pass, the analytical gradient (direction to move the sphere to
exit collision) is in:

```python
grad = query_buf.superquadric_collision_buffer.grad_distance_buffer
# shape: [batch, horizon, n_spheres, 4] — first 3 components are ∇xyz
```

---

## CUDA Kernel API

These functions are accessible as `geom_cu.closest_point_superquadric` and
`geom_cu.swept_closest_point_superquadric` after building the extension. They
are normally called through the Python autograd wrappers in `geom.py`, not
directly.

### `closest_point_superquadric` (static spheres)

```c
std::vector<torch::Tensor> sphere_superquadric_clpt(
    torch::Tensor sphere_position,     // [B, H, N, 4]   (x,y,z,radius)
    torch::Tensor distance,            // [B, H, N]       output: cost or ESDF
    torch::Tensor closest_point,       // [B, H, N, 4]   output: analytical gradient direction
    torch::Tensor sparsity_idx,        // [B, H, N]       output: active-obstacle mask
    torch::Tensor weight,              // [1]             collision weight scalar
    torch::Tensor activation_distance, // [1]             activation threshold (metres)
    torch::Tensor sq_params,           // [nenv, maxobs, 12]  SQ parameters (see layout below)
    torch::Tensor sq_enable,           // [nenv, maxobs]  uint8 per-obstacle enable mask
    torch::Tensor n_env_sq,            // [nenv]          int32 active obstacle count per env
    torch::Tensor env_query_idx,       // [B]             int32 environment index per batch entry
    int  max_nobs,                     // maximum obstacles per environment
    int  batch_size,                   // B
    int  horizon,                      // H
    int  n_spheres,                    // N
    bool compute_distance,             // reserved, set true
    bool use_batch_env,                // true when each batch entry references a different env
    bool sum_collisions,               // true: sum costs across obstacles; false: max (ESDF)
    bool compute_esdf                  // true: return signed distance instead of cost
);
// Returns: {distance, closest_point, sparsity_idx}
```

**`sq_params` tensor layout** — each row of the `[nenv, maxobs, 12]` tensor:

```
index:  0    1    2     3     4    5   6   7    8    9   10   11
field: sx   sy   sz  eps1  eps2   cx  cy  cz   qx   qy   qz   qw
```

This is the Python-side storage order. `pack_env_sq` inside the kernel
reorders to the internal `SQData` struct layout and applies clamping before
evaluation.

---

### `swept_closest_point_superquadric` (trajectory)

```c
std::vector<torch::Tensor> swept_sphere_superquadric_clpt(
    torch::Tensor sphere_position,     // [B, H, N, 4]
    torch::Tensor distance,            // [B, H, N]
    torch::Tensor closest_point,       // [B, H, N, 4]
    torch::Tensor sparsity_idx,        // [B, H, N]
    torch::Tensor weight,              // [1]
    torch::Tensor activation_distance, // [1]
    torch::Tensor speed_dt,            // [B, H]   velocity-weighted timestep per waypoint
    torch::Tensor sq_params,           // [nenv, maxobs, 12]
    torch::Tensor sq_enable,           // [nenv, maxobs]
    torch::Tensor n_env_sq,            // [nenv]
    torch::Tensor env_query_idx,       // [B]
    int  max_nobs,
    int  batch_size,
    int  horizon,
    int  n_spheres,
    int  sweep_steps,                  // interpolation steps between consecutive waypoints
    bool enable_speed_metric,          // weight cost by instantaneous speed
    bool compute_distance,
    bool use_batch_env,
    bool sum_collisions
);
// Returns: {distance, closest_point, sparsity_idx}
// Note: gradients use 3-point finite differences with ε = 1e-3
```

---

## Integration Demo CLI Reference

### `motion_gen_reacher_superquadrics.py`

Full Isaac Sim motion generation demo with live SuperDec inference.

**Collision world:**

| Argument | Default | Description |
|----------|---------|-------------|
| `--world_representation` | `superquadrics` | `superquadrics` (native SDF kernel) or `mesh` (tessellated comparison) |
| `--ply_path` | `superdec/examples/chair.ply` | Input point cloud for SuperDec inference |
| `--checkpoint_folder` | `superdec/checkpoints/normalized` | SuperDec model checkpoint |
| `--superquadric_collision_tolerance` | `0.01` | Shrink each SQ radius by this margin (m) to add clearance |
| `--superquadric_min_eps` | `0.1` | Lower clamp for inferred shape exponents |
| `--superquadric_max_eps` | `2.0` | Upper clamp for inferred shape exponents |
| `--superquadric_max_radius` | `1.5` | Upper clamp per principal radius (m) |
| `--max_superquadrics` | `48` | Maximum number of SQs used for collision |
| `--print_superquadric_stats` | `False` | Print parameter ranges and clamp counts after inference |

**Scene placement:**

| Argument | Default | Description |
|----------|---------|-------------|
| `--superquadric_translation` | `[-0.300, -0.684, 0.136]` | World-frame translation of the SQ scene (m) |
| `--superquadric_orientation` | `[0.707, 0.707, 0.0, 0.0]` | World-frame orientation `[qw, qx, qy, qz]` |
| `--superquadric_scale` | `1.0` | Uniform scale applied to all inferred SQs |
| `--seed` | `0` | Random seed for point cloud subsampling |
| `--surface_resolution` | `40` | Tessellation resolution for visualisation meshes |

**Planner:**

| Argument | Default | Description |
|----------|---------|-------------|
| `--superquadric_graph_seeds` | `8` | Graph planner seeds in SQ mode |
| `--disable_superquadric_graph` | `False` | Disable graph planner, use trajopt only |
| `--auto_fallback_to_mesh` | `False` | Rebuild planner with mesh collision on persistent SQ failures |
| `--superquadric_auto_relax_max_tolerance` | `0.05` | Max radius shrink during start-state auto-relax (m) |
| `--superquadric_auto_relax_max_eps_blend` | `0.35` | Max ε blend toward ellipsoid during auto-relax |

---

## Architecture

```
point_cloud.ply
     │
     ▼
 SuperDec (neural network, superdec/)
     │  per-primitive: translation, rotation, scale (radii), exponents, exist score
     ▼
 Superquadric objects  ──  pose=[x,y,z,qw,qx,qy,qz], radii=[a1,a2,a3], eps=[ε1,ε2]
     │
     ▼
 WorldConfig(superquadric=[...])
     │
     ▼
 WorldPrimitiveCollision  (curobo/src/curobo/geom/sdf/world.py)
     │  _superquadric_to_tensor → [sx,sy,sz, ε1,ε2, cx,cy,cz, qx,qy,qz,qw]
     │  stored as  [nenv, maxobs, 12]  GPU float32 tensor
     ▼
 SdfSphereSuperquadric  (curobo/src/curobo/curobolib/geom.py)
     │  PyTorch autograd.Function — connects loss to ∇trajectory
     ▼
 geom_cu.closest_point_superquadric  (pybind11, geom_cuda.cpp)
     │
     ▼
 superquadric_radial_distance_kernel.cu
     ├─ pack_env_sq()          reorder params, clamp, normalise quaternion
     ├─ evaluate_all_sq()      batched SDF per sphere:
     │                           F ≥ 1  →  Taubin approximation (fast, outside)
     │                           F < 1  →  Newton radial solve  (accurate, inside)
     └─ evaluate_all_sq_grad() analytical gradient: ∇F/‖∇F‖ rotated to world frame
     │
     ▼
 distance [B,H,N]  +  gradient [B,H,N,4]
     │
     ▼
 CuRobo trajopt / graph planner
```

### Tensor conventions

| Context | Layout |
|---------|--------|
| Pose (Python) | `[x, y, z, qw, qx, qy, qz]` |
| `sq_params` Python storage | `[sx, sy, sz, eps1, eps2, cx, cy, cz, qx, qy, qz, qw]` |
| `SQData` inside kernel | `[cx, cy, cz, sx, sy, sz, eps1, eps2, qw, qx, qy, qz]` |

**CUDA graphs** are disabled when using superquadrics (`use_cuda_graph=False`)
because `pack_env_sq` calls `mask.nonzero()`, which produces dynamically-shaped
tensors incompatible with CUDA graph stream capture.
