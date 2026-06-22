#!/usr/bin/env python3
"""
test_pipeline_ocid.py
=====================
Smoke test: OCID single-view frame → two-stage pipeline → superquadrics.

If OCID data is not present, the script prints download instructions and exits 0.
Run from project root:
    python scripts/test_pipeline_ocid.py

To get the data manually:
    # from a machine with access to TU Wien servers:
    wget -O OCID-dataset.zip \
        "https://data.acin.tuwien.ac.at/index.php/s/uMPFOBFjUtBPLCX/download"
    unzip OCID-dataset.zip -d data/OCID/
"""

import sys
from pathlib import Path

import numpy as np
import open3d as o3d
from PIL import Image

_REPO = Path(__file__).resolve().parent.parent
_SRC  = _REPO / "src" / "project_3dv" / "perception"
for _p in [str(_SRC)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from pipeline import Frame, get_world_pointcloud, fit_superquadrics_world

# ── config ────────────────────────────────────────────────────────────────────

OCID_ROOT = _REPO / "data" / "OCID"

# ASUS Xtion — fixed for entire dataset, no per-frame calibration files
K = np.array([
    [570.3422241210938, 0.0,              319.5],
    [0.0,              570.3422241210938, 239.5],
    [0.0,              0.0,              1.0  ],
], dtype=np.float64)

EXTRINSIC   = np.eye(4, dtype=np.float64)  # world = camera (fixed cam, no poses)
DEPTH_SCALE = 1000.0                        # uint16 mm → metres

# ── data check ────────────────────────────────────────────────────────────────

if not (OCID_ROOT / "ARID20").exists():
    print("OCID data not found.")
    print(f"Expected: {OCID_ROOT / 'ARID20'}")
    print()
    print("The TU Wien download server may be unreachable from the cluster.")
    print("Download manually on a machine with external access:")
    print()
    print("  wget -O OCID-dataset.zip \\")
    print('    "https://data.acin.tuwien.ac.at/index.php/s/uMPFOBFjUtBPLCX/download"')
    print(f"  unzip OCID-dataset.zip -d {OCID_ROOT}/")
    print()
    print("Then re-run this script.")
    sys.exit(0)

# ── find first scene ─────────────────────────────────────────────────────────

seq_dirs = sorted((OCID_ROOT / "ARID20" / "table" / "bottom").glob("seq*"))
assert seq_dirs, f"no seq* dirs under {OCID_ROOT / 'ARID20/table/bottom'}"
seq_dir = seq_dirs[0]

rgb_files   = sorted((seq_dir / "rgb").glob("*.png"))
depth_files = sorted((seq_dir / "depth").glob("*.png"))
label_files = sorted((seq_dir / "label").glob("*.png"))

assert rgb_files,   f"no rgb PNGs in {seq_dir / 'rgb'}"
assert depth_files, f"no depth PNGs in {seq_dir / 'depth'}"

rgb_path   = rgb_files[0]
depth_path = depth_files[0]
label_path = label_files[0] if label_files else None

print(f"scene : {seq_dir.name}")
print(f"rgb   : {rgb_path.name}")
print(f"depth : {depth_path.name}")
if label_path:
    print(f"label : {label_path.name}")

# ── load frame ────────────────────────────────────────────────────────────────

rgb   = np.array(Image.open(rgb_path))
depth = np.array(Image.open(depth_path)).astype(np.float32) / DEPTH_SCALE

print(f"\nrgb   shape={rgb.shape}  dtype={rgb.dtype}")
print(f"depth shape={depth.shape}  dtype={depth.dtype}  "
      f"range=[{depth[depth>0].min():.3f}, {depth.max():.3f}] m")

if label_path:
    label    = np.array(Image.open(label_path))
    n_gt_obj = int((np.unique(label) > 0).sum())
    print(f"label shape={label.shape}  dtype={label.dtype}  GT objects={n_gt_obj}")

frame = Frame(rgb=rgb, depth=depth, K=K, extrinsic=EXTRINSIC)

# ── stage 1: image → world point cloud ───────────────────────────────────────

print("\nStage 1: get_world_pointcloud ...")
world_pcd = get_world_pointcloud([frame], voxel_size=0.005, max_depth=2.0)
n_pts = len(world_pcd.points)
print(f"  {n_pts:,} points")

assert n_pts > 100, f"expected >100 points, got {n_pts}"
print("  PASS")

# ── stage 2: world cloud → superquadrics ─────────────────────────────────────

print("\nStage 2: fit_superquadrics_world ...")
sqs = fit_superquadrics_world(
    world_pcd,
    min_object_points  = 50,
    max_object_extent  = 0.40,
    cluster_min_points = 20,
    adaptive_eps       = True,
    min_height         = 0.005,
    max_height         = 0.25,
)
print(f"  {len(sqs)} superquadric(s) fitted")

assert len(sqs) >= 1, "expected at least 1 superquadric"
assert np.linalg.norm(sqs[0].position) > 0, "first SQ position is zero"
print("  PASS")

# ── save + print ──────────────────────────────────────────────────────────────

out_ply = Path("/tmp/ocid_world.ply")
o3d.io.write_point_cloud(str(out_ply), world_pcd)
print(f"\nsaved → {out_ply}  ({n_pts:,} points)")
print("inspect with:")
print(f"  python -c \"import open3d as o3d; "
      f"o3d.visualization.draw_geometries([o3d.io.read_point_cloud('{out_ply}')])\"")

print("\nfitted superquadrics (world frame):")
for i, sq in enumerate(sqs):
    print(f"  SQ {i:2d}: pos={sq.position.round(3)}  "
          f"a=({sq.a1:.3f},{sq.a2:.3f},{sq.a3:.3f})  "
          f"e=({sq.e1:.2f},{sq.e2:.2f})  "
          f"type={sq.shape_type}  conf={sq.shape_conf:.2f}")

print("\n=== ALL ASSERTIONS PASSED ===")
