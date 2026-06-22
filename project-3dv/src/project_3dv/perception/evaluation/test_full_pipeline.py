"""
test_full_pipeline.py
=====================
End-to-end test:
  RGB-D point cloud
    → TabletopPerception  (segments + shape_type)
    → SuperquadricFitter  (SQ params per segment, shape_hint from classifier)
    → Scene / superdec_utils  (signed distance for cuRobo, PLY visualisation)

Usage:
    python3 test_full_pipeline.py --ply /Volumes/T7/rgbd-scenes-v2/pc/01.ply
    python3 test_full_pipeline.py --ply /Volumes/T7/rgbd-scenes-v2/pc/01.ply --vis
"""

import argparse
import sys
import os
import time
import numpy as np
import open3d as o3d

from project_3dv.perception.pipeline    import TabletopPerception
from project_3dv.perception.superquadric import SuperquadricFitter, MultiSQFit
from project_3dv.perception.superdec_utils import Scene, sq_fits_to_npz

# ── paths ────────────────────────────────────────────────────────────────────
PERCEPTION_DIR   = os.path.dirname(os.path.abspath(__file__))
CLASSIFIER_PATH  = os.path.join(PERCEPTION_DIR, "sq_shape_classifier.pkl")
# fallback: look in project root
if not os.path.exists(CLASSIFIER_PATH):
    CLASSIFIER_PATH = os.path.join(os.getcwd(), "sq_shape_classifier.pkl")


def run_full_pipeline(ply_path: str,
                      save_vis:  bool = False,
                      save_npz:  bool = False):

    print(f"\n{'='*60}")
    print(f"  Full pipeline test")
    print(f"  Input : {ply_path}")
    print(f"  Classifier: {CLASSIFIER_PATH}")
    print(f"{'='*60}\n")

    # ── 1. load point cloud ──────────────────────────────────────────────────
    t0  = time.time()
    pcd = o3d.io.read_point_cloud(ply_path)
    pts = np.asarray(pcd.points, dtype=np.float32)
    col = np.asarray(pcd.colors, dtype=np.float32) if pcd.has_colors() else None
    print(f"[1] Loaded {len(pts):,} points  ({time.time()-t0:.2f}s)")

    # ── 2. perception ────────────────────────────────────────────────────────
    t1 = time.time()
    perception = TabletopPerception(
        classifier_path=CLASSIFIER_PATH,
        max_object_extent=0.30,    # tighter to reject furniture
        max_height_above_table=0.35,
    )
    result = perception.run(pts, rgb=col)
    t_perc = time.time() - t1

    print(f"\n[2] Perception  ({t_perc*1000:.0f} ms)")
    print(f"    {result.summary()}")

    if not result.objects:
        print("    No objects found — check scene or loosen pipeline params.")
        return

    # ── 3. SQ fitting ────────────────────────────────────────────────────────
    t2 = time.time()
    fitter    = SuperquadricFitter(n_restarts=3, n_lm_rounds=15, subsample=512)
    sq_multis = []

    print(f"\n[3] Superquadric fitting")
    print(f"    {'obj':>4}  {'shape_hint':>12}  {'n_prim':>6}  "
          f"{'chamfer_L2*1e3':>14}  {'converged':>9}  {'margin_cm':>9}")
    print(f"    {'-'*4}  {'-'*12}  {'-'*6}  {'-'*14}  {'-'*9}  {'-'*9}")

    for seg in result.objects:
        multi = fitter.fit_adaptive(
            seg.points,
            shape_hint=seg.shape_type,
            l2_threshold=0.008,
            max_primitives=3,
        )
        # carry shape_type and shape_conf onto each primitive
        for prim in multi.primitives:
            prim.shape_type = seg.shape_type
            prim.shape_conf = seg.shape_conf

        sq_multis.append(multi)

        mean_l2 = np.mean([p.chamfer_l2 for p in multi.primitives])
        conv    = all(p.converged for p in multi.primitives)
        margin  = np.mean([p.collision_margin for p in multi.primitives]) * 100

        print(f"    {seg.id:>4}  {seg.shape_type:>12}  {len(multi):>6}  "
              f"{mean_l2*1e3:>14.3f}  {str(conv):>9}  {margin:>8.1f}cm")

    t_sq = time.time() - t2
    flat_fits = [sq for m in sq_multis for sq in m.primitives]
    print(f"\n    Total primitives: {len(flat_fits)}  ({t_sq*1000:.0f} ms)")

    # ── 4. Scene / signed distance ───────────────────────────────────────────
    t3    = time.time()
    scene = Scene.from_fits(flat_fits)
    t_scene = time.time() - t3

    # test SDF on a grid of query points above the table
    query_pts = np.random.uniform(-0.3, 0.3, (1000, 3)).astype(np.float32)
    query_pts[:, 1] += float(result.table_height) + 0.1  # above table

    sd      = scene.get_signed_distance(query_pts)
    n_in    = (sd < 0).sum()
    n_out   = (sd >= 0).sum()

    print(f"\n[4] Scene SDF  ({t_scene*1000:.1f} ms to build)")
    print(f"    Query points : 1000")
    print(f"    Inside SQs   : {n_in}")
    print(f"    Outside SQs  : {n_out}")
    print(f"    SD range     : [{sd.min():.4f}, {sd.max():.4f}] m")

    # ── 5. timing summary ────────────────────────────────────────────────────
    print(f"\n[5] Timing summary")
    print(f"    Perception   : {t_perc*1000:>7.0f} ms")
    print(f"    SQ fitting   : {t_sq*1000:>7.0f} ms  "
          f"({t_sq/max(len(result.objects),1)*1000:.0f} ms/object)")
    print(f"    Scene build  : {t_scene*1000:>7.1f} ms")
    print(f"    TOTAL        : {(t_perc+t_sq+t_scene)*1000:>7.0f} ms")

    # ── 6. optional outputs ──────────────────────────────────────────────────
    stem = os.path.splitext(os.path.basename(ply_path))[0]

    if save_npz:
        npz_path = f"{stem}_sq.npz"
        sq_fits_to_npz(flat_fits, npz_path)
        print(f"\n[6] Saved NPZ → {npz_path}")

    if save_vis:
        ply_out = f"{stem}_sq_vis.ply"
        scene.save_superquadrics_vis(ply_out, resolution=20)
        print(f"\n[6] Saved visualisation → {ply_out}")
        print(f"    Open with: open3d.visualization or MeshLab")
        print(f"    Colour key: green=Ellipsoid  orange=Cylinder  "
              f"blue=Cuboid  red=Other")

    print(f"\n{'='*60}")
    print(f"  Pipeline test complete.")
    print(f"{'='*60}\n")

    return result, sq_multis, scene


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ply", required=True,
                        help="Input point cloud PLY file")
    parser.add_argument("--vis", action="store_true",
                        help="Save SQ mesh as PLY for visualisation")
    parser.add_argument("--npz", action="store_true",
                        help="Save SQ params as .npz for cuRobo")
    args = parser.parse_args()

    run_full_pipeline(args.ply, save_vis=args.vis, save_npz=args.npz)
