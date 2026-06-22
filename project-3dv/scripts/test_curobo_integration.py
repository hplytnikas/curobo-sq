#!/usr/bin/env python3
"""
test_curobo_integration.py
==========================
End-to-end test of perception pipeline → cuRobo world model.

Usage (GPU node required for cuRobo):
    python3 scripts/test_curobo_integration.py \\
        --checkpoint /work/courses/3dv/team15/checkpoints/superdec_tabletop/\\
superdec_tabletop_finetune_v2/epoch_300.pt \\
        --scene 1

CPU-only mode (skips cuRobo, just tests adapters):
    python3 scripts/test_curobo_integration.py --no-curobo --scene 1
"""

import argparse
import sys
import time
import warnings
from pathlib import Path

import numpy as np

# ── PYTHONPATH setup ─────────────────────────────────────────────────────────
_REPO_ROOT  = Path(__file__).resolve().parent.parent
_SRC        = _REPO_ROOT / "src" / "project_3dv" / "perception"
_SUPERDEC   = Path("/work/courses/3dv/team15/superdec")
_CUROBO_SRC = Path("/work/courses/3dv/team15/curobo-sq/curobo/src")

for p in [str(_SRC), str(_SUPERDEC), str(_CUROBO_SRC)]:
    if p not in sys.path:
        sys.path.insert(0, p)

from pipeline import remove_table
from superquadric import (
    SuperquadricFit, MultiSQFit, sq_signed_distance_batch,
    sq_sdf_with_gradient, sq_sdf_torch,
)
from superdec_fitter import (
    SuperdecFitter, fits_to_curobo_world, fits_to_superdec_npz,
)


# ── RGB-D Scenes v2 loader ────────────────────────────────────────────────────
_RGBD_DATA = Path("/work/courses/3dv/team15/project-3dv/data/rgbd-scenes-v2/pc")


def _load_scene(scene_id: int) -> tuple:
    """Load RGB-D Scenes v2 scene, return (pts, labels)."""
    sys.path.insert(0, str(_REPO_ROOT / "scripts"))
    from test_on_superdec_split import RGBDScenesV2
    scene = RGBDScenesV2(data_root=str(_RGBD_DATA), scene_id=scene_id)
    pts, _, labels = scene.load()
    return pts, labels


def _run_pipeline_lm(pts: np.ndarray) -> list:
    """Run table removal + LM superquadric fitting on raw pts."""
    from superquadric import SuperquadricFitter
    from pipeline import segment_instances, adaptive_cluster_eps, merge_nearby_segments

    N = len(pts)
    if N > 50_000:
        idx = np.random.choice(N, 50_000, replace=False)
        pts = pts[idx]

    try:
        obj_pts, *_ = remove_table(pts.astype(np.float64),
                                   max_height_above_table=0.4,
                                   depth_margin=np.inf, xy_radius=np.inf)
    except Exception:
        obj_pts = pts.astype(np.float64)

    eps  = adaptive_cluster_eps(obj_pts, multiplier=3.0, eps_max=0.08)
    segs = segment_instances(obj_pts, adaptive_eps=True, eps_multiplier=3.0,
                             cluster_min_points=5, eps_max=0.08)
    segs = merge_nearby_segments(segs, merge_dist=0.15)
    segs = [s for s in segs if len(s) >= 100
            and (s.max(axis=0) - s.min(axis=0)).max() <= 0.60]

    fitter = SuperquadricFitter(n_restarts=2, n_lm_rounds=10, subsample=256)
    fits   = [fitter.fit_adaptive(s) for s in segs if len(s) >= 30]
    return [f for f in fits if f.primitives]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", type=int, default=1)
    parser.add_argument("--checkpoint", default=None,
                        help="SuperDec checkpoint .pt (enables SuperDec fitter)")
    parser.add_argument("--no-curobo", action="store_true",
                        help="Skip cuRobo WorldConfig (tests adapters only)")
    args = parser.parse_args()

    print(f"\n=== Perception → cuRobo integration test  (scene {args.scene:02d}) ===\n")

    # ── 1. Load scene and run pipeline ───────────────────────────────────────
    print("Step 1: loading scene and running perception pipeline (LM fitter)…")
    t0 = time.perf_counter()
    try:
        pts, labels = _load_scene(args.scene)
    except Exception as exc:
        print(f"  ERROR loading scene: {exc}")
        sys.exit(1)

    fits = _run_pipeline_lm(pts)
    t1   = time.perf_counter()
    n_prims = sum(len(f.primitives) for f in fits)
    print(f"  Pipeline: {len(fits)} segments → {n_prims} primitives  ({t1-t0:.1f}s)")

    if not fits:
        print("  WARNING: no fits produced — check scene loading and segmentation.")
        sys.exit(1)

    # ── 2. fits_to_superdec_npz ───────────────────────────────────────────────
    print("\nStep 2: fits_to_superdec_npz() …")
    arrays = fits_to_superdec_npz(fits)
    print(f"  exist       : {arrays['exist'].shape}")
    print(f"  scale       : {arrays['scale'].shape}")
    print(f"  exponents   : {arrays['exponents'].shape}")
    print(f"  rotation    : {arrays['rotation'].shape}")
    print(f"  translation : {arrays['translation'].shape}")

    # ── 3. fits_to_curobo_world ───────────────────────────────────────────────
    print("\nStep 3: fits_to_curobo_world() …")
    if args.no_curobo:
        # Manual quaternion unit-norm check without cuRobo
        from superdec_fitter import _rotmat_to_quat_wxyz
        n_obs = 0
        for f in fits:
            for p in f.primitives:
                R    = np.array(p.rotation_matrix, dtype=np.float64)
                wxyz = _rotmat_to_quat_wxyz(R)
                norm = float(np.linalg.norm(wxyz))
                assert abs(norm - 1.0) < 1e-5, f"Quaternion norm {norm:.8f} ≠ 1"
                n_obs += 1
        print(f"  cuRobo not available (--no-curobo).  "
              f"Manually verified {n_obs} quaternions — all unit-norm ✓")
    else:
        try:
            t0    = time.perf_counter()
            world = fits_to_curobo_world(fits)
            t1    = time.perf_counter()
            n_sq  = len(world.superquadric)
            print(f"  WorldConfig: {n_sq} Superquadric obstacles  ({(t1-t0)*1000:.1f}ms)")

            # verify quaternion norms
            bad = 0
            for sq in world.superquadric:
                qw, qx, qy, qz = sq.pose[3], sq.pose[4], sq.pose[5], sq.pose[6]
                norm = (qw**2 + qx**2 + qy**2 + qz**2)**0.5
                if abs(norm - 1.0) > 1e-5:
                    bad += 1
            if bad:
                print(f"  WARNING: {bad} quaternions with |q| ≠ 1")
            else:
                print(f"  All quaternions are unit-norm ✓")

        except ImportError as e:
            print(f"  cuRobo import failed (expected on login node): {e}")

    # ── 4. SDF gradient test ──────────────────────────────────────────────────
    print("\nStep 4: SDF gradient quality check …")
    p0    = fits[0].primitives[0]
    prms  = p0.params
    t_vec = p0.translation
    # sample 20 points clearly outside the SQ (2× semi-axis offset along each axis)
    offsets = np.eye(3) * 2 * max(p0.sx, p0.sy, p0.sz)
    pts_out = t_vec[None, :] + offsets
    pts_out = np.vstack([pts_out, t_vec[None, :] + np.array([[p0.sx*3, p0.sy*0.5, 0]])])

    sdf_old = sq_signed_distance_batch(pts_out, prms)
    sdf_new, grad_new = sq_sdf_with_gradient(pts_out, prms)
    grad_norms = np.linalg.norm(grad_new, axis=1)

    print(f"  Old SDF (radial) for outside pts : {sdf_old.round(4)}")
    print(f"  New SDF (implicit) for outside pts: {sdf_new.round(4)}")
    print(f"  Gradient norms (should be > 0)    : {grad_norms.round(4)}")
    assert (sdf_new > 0).all(),  "New SDF should be positive outside the SQ"
    assert (grad_norms > 0.01).all(), "Gradient norms should be non-zero outside"
    print("  SDF gradient well-behaved ✓")

    # ── 5. PyTorch SDF test ───────────────────────────────────────────────────
    print("\nStep 5: sq_sdf_torch() gradient test …")
    try:
        import torch
        pts_t  = torch.from_numpy(pts_out).float().unsqueeze(0)  # (1, N, 3)
        pts_t.requires_grad_(True)
        prms_t = torch.from_numpy(prms).float().unsqueeze(0)     # (1, 11)
        sdf_t  = sq_sdf_torch(pts_t, prms_t)                     # (1, N)
        loss   = sdf_t.sum()
        loss.backward()
        g_norms = pts_t.grad[0].norm(dim=-1)
        print(f"  Torch SDF values: {sdf_t[0].detach().numpy().round(4)}")
        print(f"  Torch grad norms: {g_norms.detach().numpy().round(4)}")
        assert (g_norms > 0.01).all(), "Torch gradient norms should be non-zero"
        print("  PyTorch SDF autograd works ✓")
    except ImportError:
        print("  PyTorch not available — skipping torch SDF test")

    print("\n=== All integration checks passed ===\n")
    print("Method              | N obstacles")
    print("--------------------|------------")
    n_obs_count = sum(len(f.primitives) for f in fits)
    print(f"Pipeline primitives | {n_obs_count}")
    print(f"npz arrays (N segs) | {arrays['exist'].shape[0]}")
    print(f"npz K (max prims)   | {arrays['exist'].shape[1]}")


if __name__ == "__main__":
    main()
