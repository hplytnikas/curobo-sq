# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This project integrates **superquadric obstacle representations** into [CuRobo](https://curobo.org/) (NVIDIA's CUDA-accelerated robot motion planning library). Superquadrics enable more efficient collision checking than mesh-based approaches. The pipeline is:

1. **SuperDec** (`superdec/`): Neural network (transformer encoder-decoder) that decomposes 3D point clouds into superquadric primitives
2. **CuRobo** (`curobo/`): Fork with native superquadric SDF/collision kernels added

## Python Environment

All Python commands use the `3dv` conda environment:
```bash
conda run -n 3dv python <script.py>
```

The virtual env at `.venv/` is separate and used for SuperDec training/inference.

## Building CuRobo CUDA Extensions

After modifying any `.cu` or `.cpp` file in `curobo/src/curobolib/cpp/`:
```bash
conda run -n 3dv python -m pip install -e curobo/ --no-build-isolation
```
GPU architecture target: **8.9** (RTX 4090 / Ada). NVCC flags include `-O3 --ftz=true --fmad=true`.

## Running Tests

```bash
# Superquadric rotation/quaternion regression test (most relevant)
conda run -n 3dv python tests/test_sq_rotation.py

# Main integration demo
conda run -n 3dv python \
  curobo/examples/isaac_sim/motion_gen_reacher_superquadrics.py \
  --world_representation superquadrics
```

## Architecture: Superquadric Integration in CuRobo

### Data flow
1. `Superquadric` geometry type is defined in `curobo/src/curobo/geom/types.py`
2. `pack_env_sq()` in `curobo/src/curobolib/cpp/superquadric_radial_distance_kernel.cu` packs SQ parameters (radii a/b/c, shape eps1/eps2, pose) into a flat GPU buffer — **quaternion order is `[qw, qx, qy, qz]`** (was a bug when it was `[qx,qy,qz,qw]`)
3. `superquadric_radial_distance_kernel.cu` implements batched sphere-vs-SQ queries using an analytical Newton radial projection (no external library required)
4. `geom_cuda.cpp` exposes the CUDA functions via pybind11 as the `geom` extension module

### Key files for SQ collision
| File | Role |
|------|------|
| `curobo/src/curobolib/cpp/superquadric_radial_distance_kernel.cu` | **Active kernel**: analytical SDF, gradient, `pack_env_sq`, swept sphere |
| `curobo/src/curobolib/cpp/superquadric_distance_kernel.cu` | Legacy file — **not compiled**, kept for reference only |
| `curobo/src/curobolib/cpp/geom_cuda.cpp` | pybind11 C++ wrapper |
| `curobo/src/curobo/geom/types.py` | `Superquadric` class definition |
| `tests/test_sq_rotation.py` | Regression tests (5 scenarios) |

### SuperDec model architecture
- **Encoder**: `StackedPVConv` (voxel-based point cloud encoding, resolution 32³)
- **Decoder**: `TransformerDecoder` (n_queries=16 superquadrics, n_layers=3, n_heads=1)
- **Head**: outputs per-SQ: radii (a,b,c), shape (eps1,eps2), position (x,y,z), quaternion
- Training uses Hydra config at `superdec/configs/train.yaml`
