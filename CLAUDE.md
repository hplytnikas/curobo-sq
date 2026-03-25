# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This project integrates **superquadric obstacle representations** into [CuRobo](https://curobo.org/) (NVIDIA's CUDA-accelerated robot motion planning library). Superquadrics enable more efficient collision checking than mesh-based approaches. The pipeline is:

1. **SuperDec** (`superdec/`): Neural network (transformer encoder-decoder) that decomposes 3D point clouds into superquadric primitives
2. **CuRobo** (`curobo/`): Fork with native superquadric SDF/collision kernels added
3. **OpenGJK** (`openGJK/`): GJK collision detection with GPU support, used for validation

## Python Environment

All Python commands must use the Isaac Sim interpreter with CUDA 12.8 in `PATH`:
```bash
PATH=/usr/local/cuda-12.8/bin:/usr/bin:$PATH ~/isaacsim/python.sh <script.py>
```

There is also an `omni_python` alias that wraps this. The virtual env at `.venv/` is separate and used for SuperDec training/inference.

## Building CuRobo CUDA Extensions

After modifying any `.cu` or `.cpp` file in `curobo/src/curobolib/cpp/`:
```bash
PATH=/usr/local/cuda-12.8/bin:/usr/bin:$PATH ~/isaacsim/python.sh \
  -m pip install -e curobo/ --no-build-isolation
```
GPU architecture target: **8.9** (RTX 4090 / Ada). NVCC flags include `-O3 --ftz=true --fmad=true`.

## Running Tests

```bash
# Superquadric rotation/quaternion regression test (most relevant)
PATH=/usr/local/cuda-12.8/bin:/usr/bin:$PATH ~/isaacsim/python.sh tests/test_sq_rotation.py

# Main integration demo
PATH=/usr/local/cuda-12.8/bin:/usr/bin:$PATH ~/isaacsim/python.sh \
  curobo/examples/isaac_sim/motion_gen_reacher_superquadrics.py \
  --world_representation superquadrics
```

## Architecture: Superquadric Integration in CuRobo

### Data flow
1. `Superquadric` geometry type is defined in `curobo/src/curobo/geom/types.py`
2. `pack_env_sq()` in `curobo/src/curobolib/cpp/superquadric_radial_distance_kernel.cu` packs SQ parameters (radii a/b/c, shape eps1/eps2, pose) into a flat GPU buffer — **quaternion order is `[qw, qx, qy, qz]`** (was a bug when it was `[qx,qy,qz,qw]`)
3. `superquadric_distance_kernel.cu` calls the radial distance kernel for batched sphere-vs-SQ queries
4. `geom_cuda.cpp` exposes the CUDA functions via pybind11 as the `geom` extension module

### Key files for SQ collision
| File | Role |
|------|------|
| `curobo/src/curobolib/cpp/superquadric_radial_distance_kernel.cu` | Core SDF math, `pack_env_sq` |
| `curobo/src/curobolib/cpp/superquadric_distance_kernel.cu` | Batched wrapper, swept sphere |
| `curobo/src/curobolib/cpp/geom_cuda.cpp` | pybind11 C++ wrapper |
| `curobo/src/curobo/geom/types.py` | `Superquadric` class definition |
| `tests/test_sq_rotation.py` | Regression tests (5 scenarios) |

### SuperDec model architecture
- **Encoder**: `StackedPVConv` (voxel-based point cloud encoding, resolution 32³)
- **Decoder**: `TransformerDecoder` (n_queries=16 superquadrics, n_layers=3, n_heads=1)
- **Head**: outputs per-SQ: radii (a,b,c), shape (eps1,eps2), position (x,y,z), quaternion
- Training uses Hydra config at `superdec/configs/train.yaml`

## Building OpenGJK (for validation/standalone use)
```bash
cmake -B openGJK/build -DCMAKE_BUILD_TYPE=Release -DBUILD_GPU=ON \
  -S openGJK -DCMAKE_CUDA_ARCHITECTURES=89
cmake --build openGJK/build
ctest --test-dir openGJK/build
```
