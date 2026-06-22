#!/usr/bin/env python3
"""
smoke_test_pipeline.py
======================
Self-contained smoke test for the full perception pipeline.

Runs:
  load → table removal → segmentation → LM fitting → fits_to_curobo_world

Usage:
  python3 scripts/smoke_test_pipeline.py [--scene N]   (default: scene 5)

Expected environment:
  source /work/courses/3dv/team15/superdec/.venv/bin/activate
  export PYTHONPATH="src:${PYTHONPATH:-}"

cuRobo is not installed in the superdec venv — fits_to_curobo_world will
raise ImportError, which is expected and handled gracefully.
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np

# ── path setup ────────────────────────────────────────────────────────────────
_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO / "src"))

# ── imports (all exact signatures confirmed via inspect) ──────────────────────
from project_3dv.perception.datasets.rgbd_scenes import RGBDScenesV2
from project_3dv.perception.pipeline import (
    remove_table,           # -> (obj_pts, table_normal, table_height, table_pts, n_table_pts)
    segment_instances,      # -> List[np.ndarray]
    merge_nearby_segments,  # -> List[np.ndarray]
)
from project_3dv.perception.superquadric import SuperquadricFitter
from project_3dv.perception.superdec_fitter import fits_to_curobo_world
# fits_to_curobo_world(fits: List[MultiSQFit], active_only: bool = True) -> WorldConfig


def main():
    parser = argparse.ArgumentParser(
        description="Perception pipeline smoke test"
    )
    parser.add_argument("--scene", type=int, default=5,
                        help="RGB-D Scenes v2 scene ID (1–14, default: 5)")
    args = parser.parse_args()

    data_root = str(_REPO / "data" / "rgbd-scenes-v2" / "pc")
    print(f"\n=== smoke_test_pipeline.py  scene={args.scene:02d} ===\n")

    # ── 1. Load ───────────────────────────────────────────────────────────────
    print("Step 1: loading scene …")
    t0 = time.perf_counter()
    ds = RGBDScenesV2(data_root, scene_id=args.scene)
    pts, rgb, labels = ds.load()           # (N,3) float32, (N,3) uint8, (N,) int32
    print(f"  {len(pts):,} points  ({time.perf_counter()-t0:.2f}s)")
    assert pts.shape[1] == 3, f"Expected (N,3) pts, got {pts.shape}"

    # ── 2. Table removal ──────────────────────────────────────────────────────
    print("Step 2: table removal …")
    t0 = time.perf_counter()
    obj_pts, table_normal, table_height, table_pts, n_table_pts = remove_table(
        pts, depth_margin=np.inf, xy_radius=np.inf
    )
    print(f"  {len(obj_pts):,} foreground pts  "
          f"normal={table_normal.round(3)}  "
          f"height={table_height:.4f}m  "
          f"({time.perf_counter()-t0:.2f}s)")
    assert obj_pts.shape[1] == 3
    assert table_normal.shape == (3,)
    assert isinstance(table_height, float)

    # ── 3. Segmentation ───────────────────────────────────────────────────────
    print("Step 3: segmentation …")
    t0 = time.perf_counter()
    segs = segment_instances(
        obj_pts,
        adaptive_eps=True, eps_multiplier=3.0,
        cluster_min_points=5, eps_max=0.08,
    )
    segs = merge_nearby_segments(segs, merge_dist=0.15)
    segs = [s for s in segs
            if len(s) >= 100 and (s.max(0) - s.min(0)).max() <= 0.70]
    print(f"  {len(segs)} valid segments  ({time.perf_counter()-t0:.2f}s)")
    assert len(segs) > 0, "No segments found — check scene data or table removal"

    # ── 4. LM fitting ─────────────────────────────────────────────────────────
    print("Step 4: LM superquadric fitting …")
    t0 = time.perf_counter()
    lm = SuperquadricFitter(n_restarts=2, n_lm_rounds=10, subsample=256)
    fits = [lm.fit_adaptive(s) for s in segs if len(s) >= 30]
    fits = [f for f in fits if f and f.primitives]
    n_prims = sum(len(f.primitives) for f in fits)
    print(f"  {len(fits)} objects → {n_prims} primitives  ({time.perf_counter()-t0:.2f}s)")
    assert len(fits) > 0, "No fits produced"
    assert n_prims > 0, "No primitives produced"

    # Spot-check first primitive fields
    p0 = fits[0].primitives[0]
    for attr in ("sx", "sy", "sz", "e1", "e2", "tx", "ty", "tz"):
        assert hasattr(p0, attr) and np.isfinite(getattr(p0, attr)), \
            f"Primitive missing or non-finite field: {attr}"

    # ── 5. fits_to_curobo_world ───────────────────────────────────────────────
    # Signature: fits_to_curobo_world(fits: List[MultiSQFit], active_only: bool = True)
    # cuRobo is not installed in the superdec venv — ImportError is expected.
    print("Step 5: fits_to_curobo_world() …")
    try:
        world = fits_to_curobo_world(fits, active_only=True)
        n_sq = len(world.superquadric)
        print(f"  {n_sq} Superquadric obstacles in WorldConfig  ✓")
        assert n_sq > 0, "WorldConfig has no SQ obstacles"
    except ImportError as e:
        print(f"  cuRobo not in this venv (expected on cluster): {e}")
        print("  Step 5 skipped — OK")

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n  Scene        : {args.scene:02d}")
    print(f"  Raw points   : {len(pts):,}")
    print(f"  Foreground   : {len(obj_pts):,}")
    print(f"  Segments     : {len(segs)}")
    print(f"  Objects fit  : {len(fits)}")
    print(f"  Primitives   : {n_prims}")
    print("\nSMOKE TEST PASSED")


if __name__ == "__main__":
    main()
