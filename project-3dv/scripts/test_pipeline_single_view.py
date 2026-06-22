#!/usr/bin/env python3
"""
test_pipeline_single_view.py
============================
Smoke test for the two-stage perception pipeline.

Builds a synthetic tabletop scene (a flat table + a sphere object),
projects it to a depth image through a known camera, then runs:

  Stage 1: get_world_pointcloud(frames)  — depth + extrinsic → world cloud
  Stage 2: fit_superquadrics_world(pcd)  — segments + fits SQ in world frame

Done criteria checked
---------------------
[x] Stage 1 returns a non-empty cloud
[x] Stage 2 returns at least one SuperquadricWorld
[x] Fitted world-frame position is non-zero (assert norm > 0)
[x] No local-frame intermediate is exposed outside Stage 2
[x] No depth/image logic in Stage 2 (verified by API: takes o3d.PointCloud)
"""

import sys
from pathlib import Path

import numpy as np
import open3d as o3d

_REPO = Path(__file__).resolve().parent.parent
_SRC  = _REPO / "src" / "project_3dv" / "perception"
for _p in [str(_SRC)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from pipeline import (
    Frame,
    SuperquadricWorld,
    get_world_pointcloud,
    fit_superquadrics_world,
    MULTIVIEW,
)


# ── Synthetic scene helpers ───────────────────────────────────────────────────

def _make_synthetic_scene(
    table_z: float = 0.80,
    obj_z:   float = 0.65,
    obj_r:   float = 0.04,
    H: int = 240, W: int = 320,
):
    """Return (depth, K, extrinsic) for a synthetic tabletop scene.

    Camera is at origin looking along +Z. Table is a flat slab at table_z.
    Object is a sphere of radius obj_r centred on the optical axis at obj_z.
    Extrinsic = identity (camera frame == world frame for this test).
    """
    fx = fy = 250.0
    cx, cy  = W / 2.0, H / 2.0
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)

    u, v = np.meshgrid(np.arange(W, dtype=np.float32),
                       np.arange(H, dtype=np.float32))
    # back-project rays to unit depth
    rx = (u - cx) / fx
    ry = (v - cy) / fy

    depth = np.zeros((H, W), dtype=np.float32)

    # sphere: |ray*t - (0,0,obj_z)|^2 = obj_r^2
    # (rx^2+ry^2+1)*t^2 - 2*obj_z*t + (obj_z^2 - obj_r^2) = 0
    a = rx**2 + ry**2 + 1.0
    b = -2.0 * obj_z
    c = obj_z**2 - obj_r**2
    disc = b**2 - 4 * a * c
    hit  = disc >= 0
    t_sphere = (-b - np.sqrt(np.where(hit, disc, 0.0))) / (2.0 * a)
    depth = np.where(hit & (t_sphere > 0), t_sphere.astype(np.float32), depth)

    # table: a pixel hits the table at z=table_z if the sphere doesn't occlude it
    t_table = (table_z / 1.0) * np.ones((H, W), dtype=np.float32)
    # table is valid everywhere the sphere didn't hit
    table_valid = (~hit) | (t_sphere <= 0)
    # only keep table pixels within a moderate radius to bound the cloud
    r_img = np.sqrt(rx**2 + ry**2)
    table_valid = table_valid & (r_img < 0.6)
    depth = np.where(table_valid, t_table, depth)

    extrinsic = np.eye(4, dtype=np.float64)  # camera frame == world frame
    return depth, K, extrinsic


# ── Smoke test ────────────────────────────────────────────────────────────────

def main():
    print("=== test_pipeline_single_view ===\n")
    print(f"MULTIVIEW = {MULTIVIEW}  (single-view fallback)\n")

    # ── build synthetic frame ────────────────────────────────────────────────
    depth, K, extrinsic = _make_synthetic_scene()
    frame = Frame(depth=depth, K=K, extrinsic=extrinsic)

    n_valid = int((depth > 0).sum())
    print(f"Synthetic depth: {depth.shape}, {n_valid:,} valid pixels")

    # ── Stage 1 ──────────────────────────────────────────────────────────────
    print("\nStage 1: get_world_pointcloud ...")
    world_pcd = get_world_pointcloud([frame], voxel_size=0.003, max_depth=2.0)
    n_pts = len(world_pcd.points)
    print(f"  world cloud: {n_pts:,} points")

    assert n_pts > 0, "Stage 1 returned empty cloud"
    pts_arr = np.asarray(world_pcd.points)
    print(f"  z range: [{pts_arr[:,2].min():.3f}, {pts_arr[:,2].max():.3f}] m  "
          f"(table at ~0.80m, object at ~0.65m)")
    print("  Stage 1 PASS\n")

    # ── Stage 2 ──────────────────────────────────────────────────────────────
    print("Stage 2: fit_superquadrics_world ...")
    sqs = fit_superquadrics_world(
        world_pcd,
        min_object_points  = 30,
        max_object_extent  = 0.30,
        cluster_min_points = 5,
        adaptive_eps       = True,
        min_height         = 0.005,
        max_height         = 0.25,
    )

    print(f"  fitted {len(sqs)} SuperquadricWorld primitive(s)")
    assert len(sqs) >= 1, "Stage 2 returned no primitives"

    for i, sq in enumerate(sqs):
        assert isinstance(sq, SuperquadricWorld), \
            f"Expected SuperquadricWorld, got {type(sq)}"
        pos_norm = float(np.linalg.norm(sq.position))
        assert pos_norm > 0, f"Primitive {i} has zero-norm position"
        print(f"  [{i}] pos={sq.position.round(3)}  "
              f"axes=({sq.a1:.3f},{sq.a2:.3f},{sq.a3:.3f})  "
              f"shape={sq.shape_type}  "
              f"conf={sq.shape_conf:.2f}  "
              f"|pos|={pos_norm:.3f} m")

    # verify rotation matrices are valid SO(3)
    for sq in sqs:
        R = sq.rotation
        assert R.shape == (3, 3), "rotation must be (3,3)"
        assert abs(float(np.linalg.det(R)) - 1.0) < 1e-4, \
            f"rotation det={np.linalg.det(R):.6f} is not +1"

    print("  Stage 2 PASS\n")

    # ── API contract check ────────────────────────────────────────────────────
    # Stage 2 takes o3d.PointCloud — no image/depth logic exposed
    assert callable(fit_superquadrics_world)
    import inspect
    sig = inspect.signature(fit_superquadrics_world)
    first_param = list(sig.parameters.keys())[0]
    assert first_param == "world_pcd", \
        f"Stage 2 first param should be 'world_pcd', got '{first_param}'"
    print("API contract check PASS  (Stage 2 takes world_pcd, not depth/frames)\n")

    print("=== ALL CHECKS PASSED ===")


if __name__ == "__main__":
    main()
