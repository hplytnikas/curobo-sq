#!/usr/bin/env python3
"""
process_own_scan.py
===================
Process a Record3D .r3d scan file through the full SuperDec tabletop pipeline.

Steps
-----
1. Unzip .r3d, read metadata (per-frame intrinsics, camera-to-world poses).
2. Subsample frames (--frame-stride) for efficiency.
3. Decompress depth (LZFSE via liblzfse) and confidence maps.
4. Back-project depth pixels → 3-D points in camera frame; apply conf filter.
5. Transform to world frame via pose quaternion + translation.
6. Optionally extract per-point RGB from the aligned JPEG.
7. Voxel-downsample fused cloud (open3d), statistical outlier removal.
8. table removal → segmentation → merge → bbox filter → SuperDec fits.
9. Save coloured point cloud and 4-panel figure.

Usage
-----
  python3 scripts/process_own_scan.py \\
      --r3d   /work/courses/3dv/team15/data/own_scan/2026-03-30--20-51-15.r3d \\
      --checkpoint /work/courses/3dv/team15/checkpoints/superdec_tabletop/\\
superdec_tabletop_finetune_v3/epoch_300.pt \\
      --output outputs/own_scan/ \\
      --verbose

Coordinate conventions
-----------------------
Record3D (ARKit) world frame: X right, Y up, Z toward viewer.
Depth resolution:  256 × 192  (LiDAR sensor).
RGB  resolution: 1920 × 1440 (camera — shares focal length, scaled cx/cy).
Pose format: [qx, qy, qz, qw, tx, ty, tz]  (camera-to-world).
"""

import matplotlib
matplotlib.use("Agg")

import argparse
import io
import json
import sys
import time
import warnings
import zipfile
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from scipy.spatial.transform import Rotation

# ── path setup ────────────────────────────────────────────────────────────────
_HERE         = Path(__file__).resolve().parent
_ROOT         = _HERE.parent
_SUPERDEC_DIR = Path("/work/courses/3dv/team15/superdec")

sys.path.insert(0, str(_ROOT / "src"))
sys.path.insert(0, str(_SUPERDEC_DIR))

from project_3dv.perception.superquadric import (
    MultiSQFit,
    sample_surface_equal_distance,
)
from project_3dv.perception.pipeline import (
    remove_table,
    segment_instances,
    merge_nearby_segments,
)

# ── constants ─────────────────────────────────────────────────────────────────
DEPTH_W, DEPTH_H = 256, 192
MIN_SEG_PTS = 400
BBOX_MAX_DIM = 0.70   # m — drop wall/shelf remnants

PALETTE = [
    "#e63946", "#457b9d", "#2a9d8f", "#e76f51",
    "#8338ec", "#f4a261", "#06d6a0", "#ffd166",
    "#118ab2", "#ef476f", "#06a77d", "#d62246",
]


def _colour(i: int) -> str:
    return PALETTE[i % len(PALETTE)]


# ── LZFSE decompression ───────────────────────────────────────────────────────

def _import_lzfse():
    """Return the lzfse decompress function (pyliblzfse installs as 'liblzfse')."""
    try:
        import liblzfse
        return liblzfse.decompress
    except ImportError:
        raise RuntimeError(
            "liblzfse not found. Install via: "
            "pip install pyliblzfse"
        )


# ── depth back-projection ─────────────────────────────────────────────────────

def _depth_intrinsics(rgb_K: List[float],
                      rgb_w: int, rgb_h: int) -> Tuple[float, float, float, float]:
    """Scale RGB intrinsics [fx, fy, cx, cy] from actual RGB resolution to depth 256×192."""
    fx, fy, cx, cy = rgb_K
    sx = DEPTH_W / rgb_w
    sy = DEPTH_H / rgb_h
    return fx * sx, fy * sy, cx * sx, cy * sy


def _backproject(depth: np.ndarray,
                 fx: float, fy: float, cx: float, cy: float
                 ) -> Tuple[np.ndarray, np.ndarray]:
    """Return (N,3) camera-frame points and (N,) flat pixel indices for valid pixels."""
    v_idx, u_idx = np.where(depth > 0)
    d = depth[v_idx, u_idx]
    x = (u_idx - cx) / fx * d
    y = (v_idx - cy) / fy * d
    z = d
    pts_cam = np.stack([x, y, z], axis=1).astype(np.float32)
    flat_idx = v_idx * DEPTH_W + u_idx
    return pts_cam, flat_idx


def _pose_to_Rt(pose: List[float]) -> Tuple[np.ndarray, np.ndarray]:
    """Decode [qx, qy, qz, qw, tx, ty, tz] → rotation matrix (3,3), translation (3,)."""
    qx, qy, qz, qw, tx, ty, tz = pose
    R = Rotation.from_quat([qx, qy, qz, qw]).as_matrix()
    t = np.array([tx, ty, tz], dtype=np.float64)
    return R, t


# ── outlier removal ───────────────────────────────────────────────────────────

def _remove_outliers(pts: np.ndarray,
                     nb_neighbors: int = 20,
                     std_ratio: float = 2.0,
                     verbose: bool = False) -> Tuple[np.ndarray, np.ndarray]:
    """Statistical outlier removal; returns filtered pts and boolean keep mask."""
    from sklearn.neighbors import NearestNeighbors
    if len(pts) < nb_neighbors + 1:
        return pts, np.ones(len(pts), dtype=bool)
    nbrs = NearestNeighbors(n_neighbors=nb_neighbors + 1).fit(pts)
    dists, _ = nbrs.kneighbors(pts)
    mean_dists = dists[:, 1:].mean(axis=1)
    threshold = mean_dists.mean() + std_ratio * mean_dists.std()
    mask = mean_dists < threshold
    if verbose:
        n_rm = int((~mask).sum())
        print(f"  outlier removal: {n_rm} pts removed "
              f"({100.0 * n_rm / len(pts):.1f}%)")
    return pts[mask], mask


# ── normalized Chamfer ────────────────────────────────────────────────────────

def _compute_chamfer_normalized(pts: np.ndarray, result: MultiSQFit) -> float:
    if not result or not result.primitives:
        return float("nan")
    center = pts.mean(0).astype(np.float64)
    scale  = float(np.linalg.norm(pts - center, axis=1).max()) + 1e-8
    pts_n  = ((pts - center) / scale).astype(np.float32)
    surf_parts = []
    for p in result.primitives:
        try:
            s = sample_surface_equal_distance(p, n_points=200)
            if s is not None and len(s) > 0:
                surf_parts.append(((s - center) / scale).astype(np.float32))
        except Exception:
            pass
    if not surf_parts:
        return float("nan")
    surf = np.concatenate(surf_parts, axis=0)
    d_ps = ((pts_n[:, None] - surf[None]) ** 2).sum(-1).min(-1)
    d_sp = ((surf[:, None] - pts_n[None]) ** 2).sum(-1).min(-1)
    return float(d_ps.mean() + d_sp.mean())


# ── wireframe helper ──────────────────────────────────────────────────────────

def _draw_sq_wireframe(ax, prim, color, alpha: float = 0.45, lw: float = 0.8):
    u = np.linspace(-np.pi / 2, np.pi / 2, 20)
    v = np.linspace(-np.pi, np.pi, 40)
    _se = lambda x, e: np.sign(x) * (np.abs(x) + 1e-8) ** e
    Xb = prim.sx * _se(np.cos(u[:, None]), prim.e1) * _se(np.cos(v[None, :]), prim.e2)
    Yb = prim.sy * _se(np.cos(u[:, None]), prim.e1) * _se(np.sin(v[None, :]), prim.e2)
    Zb = prim.sz * _se(np.sin(u[:, None]), prim.e1) * np.ones_like(v[None, :])
    if hasattr(prim, "rotation_matrix"):
        R = np.array(prim.rotation_matrix, dtype=np.float64)
    else:
        R = Rotation.from_euler("xyz", [prim.rx, prim.ry, prim.rz]).as_matrix()
    body  = np.stack([Xb.ravel(), Yb.ravel(), Zb.ravel()], axis=1)
    world = body @ R.T + np.array([prim.tx, prim.ty, prim.tz])
    Xw = world[:, 0].reshape(Xb.shape)
    Yw = world[:, 1].reshape(Yb.shape)
    Zw = world[:, 2].reshape(Zb.shape)
    ax.plot_wireframe(Xw, Yw, Zw, rstride=2, cstride=2,
                      color=color, alpha=alpha, linewidth=lw)


def _set_ax_limits(ax, pts: np.ndarray, margin: float = 0.1):
    if len(pts) == 0:
        return
    centre = pts.mean(axis=0)
    half_r = max((pts.max(axis=0) - pts.min(axis=0)).max() / 2 * (1.0 + margin), 0.01)
    ax.set_xlim(centre[0] - half_r, centre[0] + half_r)
    ax.set_ylim(centre[1] - half_r, centre[1] + half_r)
    ax.set_zlim(centre[2] - half_r, centre[2] + half_r)
    ax.set_box_aspect([1, 1, 1])
    ax.set_xlabel("X", fontsize=7)
    ax.set_ylabel("Y", fontsize=7)
    ax.set_zlabel("Z", fontsize=7)
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_zticklabels([])


# ── fuse frames ───────────────────────────────────────────────────────────────

def fuse_frames(r3d_path: str,
                frame_stride: int = 15,
                conf_threshold: int = 1,
                verbose: bool = False
                ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Fuse depth frames from a Record3D .r3d file into a single world-frame point cloud.

    Parameters
    ----------
    r3d_path      : path to the .r3d file
    frame_stride  : process every N-th frame (default 15 ≈ 4 fps from 60 fps)
    conf_threshold: minimum ARKit confidence level to keep (0=all, 1=medium+, 2=high only)
    verbose       : print per-frame progress

    Returns
    -------
    pts_world   : (N, 3) float32 world-frame points
    colors_world: (N, 3) uint8 RGB colors, or None if RGB extraction failed
    """
    decompress = _import_lzfse()
    from PIL import Image as _PIL_Image

    with zipfile.ZipFile(r3d_path, "r") as zf:
        meta    = json.loads(zf.read("metadata"))
        poses   = meta["poses"]                      # list of [qx,qy,qz,qw,tx,ty,tz]
        frame_K = meta["perFrameIntrinsicCoeffs"]    # list of [fx,fy,cx,cy] at RGB resolution

        # Fix 2: sort all frame files numerically; pose_idx = sorted position
        def _frame_num(path, ext):
            return int(path.split("/")[1].replace(ext, ""))

        all_depth = sorted(
            [f for f in zf.namelist() if f.startswith("rgbd/") and f.endswith(".depth")],
            key=lambda x: _frame_num(x, ".depth"),
        )
        all_conf = sorted(
            [f for f in zf.namelist() if f.startswith("rgbd/") and f.endswith(".conf")],
            key=lambda x: _frame_num(x, ".conf"),
        )
        all_jpg = sorted(
            [f for f in zf.namelist() if f.startswith("rgbd/") and f.endswith(".jpg")],
            key=lambda x: _frame_num(x, ".jpg"),
        )

        # Fix 3: detect actual RGB resolution from the first JPEG
        with zf.open(all_jpg[0]) as _f:
            _img0 = _PIL_Image.open(io.BytesIO(_f.read()))
            rgb_w, rgb_h = _img0.size
        if verbose:
            print(f"  RGB resolution: {rgb_w}×{rgb_h}")
            sx = DEPTH_W / rgb_w
            sy = DEPTH_H / rgb_h
            fx0, fy0, cx0, cy0 = _depth_intrinsics(frame_K[0], rgb_w, rgb_h)
            print(f"  depth intrinsics (frame 0): fx={fx0:.1f} fy={fy0:.1f} "
                  f"cx={cx0:.1f} cy={cy0:.1f}")

        n_frames = min(len(all_depth), len(poses))
        frame_indices = list(range(0, n_frames, frame_stride))
        if verbose:
            print(f"  total frames: {n_frames}, processing every {frame_stride}th "
                  f"= {len(frame_indices)} frames")

        all_pts    = []
        all_colors = []
        has_colors = True

        for fi, pose_idx in enumerate(frame_indices):
            df = all_depth[pose_idx]
            cf = all_conf[pose_idx]
            jf = all_jpg[pose_idx]
            fname_num = _frame_num(df, ".depth")

            # --- depth ---
            depth = np.frombuffer(decompress(zf.read(df)),
                                  dtype=np.float32).copy().reshape(DEPTH_H, DEPTH_W)

            # --- confidence ---
            conf = np.frombuffer(decompress(zf.read(cf)),
                                 dtype=np.uint8).reshape(DEPTH_H, DEPTH_W)
            depth[conf < conf_threshold] = 0.0

            # --- intrinsics — Fix 3: use actual rgb_w/rgb_h ---
            fx, fy, cx, cy = _depth_intrinsics(frame_K[pose_idx], rgb_w, rgb_h)

            # --- back-project ---
            pts_cam, flat_idx = _backproject(depth, fx, fy, cx, cy)
            if len(pts_cam) == 0:
                continue

            # --- transform to world frame — Fix 2: pose_idx not fname_num ---
            R, t = _pose_to_Rt(poses[pose_idx])
            pts_world_i = (pts_cam.astype(np.float64) @ R.T + t).astype(np.float32)
            all_pts.append(pts_world_i)

            # --- RGB color at depth pixels (optional) ---
            if has_colors:
                try:
                    img = _PIL_Image.open(io.BytesIO(zf.read(jf))).convert("RGB")
                    img_arr = np.array(img)
                    v_pix = flat_idx // DEPTH_W
                    u_pix = flat_idx %  DEPTH_W
                    u_rgb = np.clip(
                        np.round(u_pix * (img_arr.shape[1] / DEPTH_W)).astype(int),
                        0, img_arr.shape[1] - 1)
                    v_rgb = np.clip(
                        np.round(v_pix * (img_arr.shape[0] / DEPTH_H)).astype(int),
                        0, img_arr.shape[0] - 1)
                    all_colors.append(img_arr[v_rgb, u_rgb])
                except Exception:
                    has_colors = False
                    all_colors = []

            if verbose and (fi % 50 == 0 or fi == len(frame_indices) - 1):
                print(f"    frame {fname_num:5d} (pose {pose_idx:5d}/{n_frames-1})"
                      f"  pts: {len(pts_cam):6d}  total: {sum(len(p) for p in all_pts):8d}")

    if not all_pts:
        raise RuntimeError("No valid depth frames found in .r3d file.")

    pts = np.concatenate(all_pts, axis=0)
    colors = np.concatenate(all_colors, axis=0) if has_colors and all_colors else None

    # Fix 1: apply initPose global transform to align world frame
    init   = meta["initPose"]   # [qx, qy, qz, qw, tx, ty, tz]
    R_init = Rotation.from_quat(init[:4]).as_matrix()
    t_init = np.array(init[4:], dtype=np.float64)
    pts    = (pts.astype(np.float64) @ R_init.T + t_init).astype(np.float32)

    # fix Record3D coordinate frame → Z-up (table horizontal)
    # diagnostic confirmed: X was table normal, rotate -90° around Y to make Z vertical
    R_fix = Rotation.from_euler('y', -90, degrees=True).as_matrix()
    pts   = (pts @ R_fix.T).astype(np.float32)

    if verbose:
        print(f"  initPose + coord-fix applied  bbox after: "
              f"x=[{pts[:,0].min():.2f},{pts[:,0].max():.2f}]  "
              f"y=[{pts[:,1].min():.2f},{pts[:,1].max():.2f}]  "
              f"z=[{pts[:,2].min():.2f},{pts[:,2].max():.2f}]")

    return pts, colors


# ── voxel downsample ──────────────────────────────────────────────────────────

def voxel_downsample(pts: np.ndarray,
                     colors: Optional[np.ndarray],
                     voxel_size: float = 0.005,
                     verbose: bool = False
                     ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Voxel-grid downsample using open3d."""
    import open3d as o3d
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts.astype(np.float64))
    if colors is not None:
        pcd.colors = o3d.utility.Vector3dVector(colors.astype(np.float64) / 255.0)

    pcd_ds = pcd.voxel_down_sample(voxel_size=voxel_size)
    pts_ds = np.asarray(pcd_ds.points, dtype=np.float32)
    colors_ds = None
    if colors is not None and pcd_ds.has_colors():
        colors_ds = (np.asarray(pcd_ds.colors) * 255).astype(np.uint8)

    if verbose:
        print(f"  voxel downsample ({voxel_size*100:.1f} cm): "
              f"{len(pts):,} → {len(pts_ds):,} pts")
    return pts_ds, colors_ds


# ── figure ────────────────────────────────────────────────────────────────────

def _make_figure(pts_fused: np.ndarray,
                 colors_fused: Optional[np.ndarray],
                 obj_pts: np.ndarray,
                 table_normal: np.ndarray,
                 table_height: float,
                 segs: List[np.ndarray],
                 seg_fits: List[MultiSQFit],
                 out_path: Path):
    """4-panel figure: fused cloud | table removed | segments | SQ fits."""
    fig = plt.figure(figsize=(22, 5.5))
    axes = [fig.add_subplot(1, 4, k + 1, projection="3d") for k in range(4)]
    titles = ["Fused cloud", "Table removed", "Segments", "SuperDec fits"]

    # ── panel 1: fused cloud ──────────────────────────────────────────────────
    ax = axes[0]
    if colors_fused is not None:
        c = colors_fused.astype(np.float32) / 255.0
        ax.scatter(pts_fused[:, 0], pts_fused[:, 1], pts_fused[:, 2],
                   c=c, s=0.3, linewidths=0, rasterized=True)
    else:
        ax.scatter(pts_fused[:, 0], pts_fused[:, 1], pts_fused[:, 2],
                   c="steelblue", s=0.3, linewidths=0, rasterized=True)
    _set_ax_limits(ax, pts_fused)

    # ── panels 2-4: zoom to foreground ────────────────────────────────────────
    if len(segs) > 0:
        fg_pts = np.concatenate(segs, axis=0)
        fg_min = fg_pts.min(axis=0)
        fg_max = fg_pts.max(axis=0)
        fg_center = (fg_min + fg_max) / 2
        fg_range  = max((fg_max - fg_min).max() * 0.6, 0.05)

        def _lim_fg(ax):
            ax.set_xlim(fg_center[0] - fg_range, fg_center[0] + fg_range)
            ax.set_ylim(fg_center[1] - fg_range, fg_center[1] + fg_range)
            ax.set_zlim(fg_center[2] - fg_range, fg_center[2] + fg_range)
            ax.set_box_aspect([1, 1, 1])
            ax.set_xlabel("X", fontsize=7)
            ax.set_ylabel("Y", fontsize=7)
            ax.set_zlabel("Z", fontsize=7)
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.set_zticklabels([])
    else:
        def _lim_fg(ax):
            _set_ax_limits(ax, obj_pts if len(obj_pts) > 0 else pts_fused)

    # panel 2: table removed
    ax = axes[1]
    ax.scatter(obj_pts[:, 0], obj_pts[:, 1], obj_pts[:, 2],
               c="steelblue", s=0.8, linewidths=0, rasterized=True)
    # draw table plane at table_height
    if table_normal is not None:
        xs = np.array([obj_pts[:, 0].min(), obj_pts[:, 0].max()])
        zs = np.array([obj_pts[:, 2].min(), obj_pts[:, 2].max()])
        XX, ZZ = np.meshgrid(xs, zs)
        # table_normal is approximately [0, 1, 0] in ARKit world frame
        a, b, c_ = table_normal
        if abs(b) > 1e-6:
            YY = (table_height - a * XX - c_ * ZZ) / b
        else:
            YY = np.full_like(XX, table_height)
        ax.plot_surface(XX, YY, ZZ, alpha=0.2, color="tan")
    _lim_fg(ax)

    # panel 3: segments
    ax = axes[2]
    for si, seg in enumerate(segs):
        c = _colour(si)
        ax.scatter(seg[:, 0], seg[:, 1], seg[:, 2],
                   c=c, s=1.5, linewidths=0, rasterized=True)
    _lim_fg(ax)

    # panel 4: SQ fits
    ax = axes[3]
    for si, (seg, fit) in enumerate(zip(segs, seg_fits)):
        c = _colour(si)
        ax.scatter(seg[:, 0], seg[:, 1], seg[:, 2],
                   c=c, s=0.8, alpha=0.3, linewidths=0, rasterized=True)
        if fit and fit.primitives:
            for prim in fit.primitives:
                _draw_sq_wireframe(ax, prim, color=c)
    _lim_fg(ax)

    for ax, title in zip(axes, titles):
        ax.set_title(title, fontsize=9, pad=4)

    fig.suptitle("Own scan — SuperDec pipeline", fontsize=11, y=1.01)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  figure saved → {out_path}")


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="Process Record3D .r3d scan with SuperDec.")
    ap.add_argument("--r3d", default=None,
                    help="Path to the Record3D .r3d file")
    ap.add_argument("--input-cloud", default=None,
                    help="Skip .r3d extraction: load pre-fused .npz (keys: pts, colors)")
    ap.add_argument("--no-superdec", action="store_true",
                    help="Skip SuperDec fitting (table removal + segmentation only)")
    ap.add_argument("--checkpoint", default=None,
                    help="SuperDec checkpoint .pt (omit for LM-only mode)")
    ap.add_argument("--output", default="outputs/own_scan",
                    help="Output directory (default: outputs/own_scan)")
    ap.add_argument("--frame-stride", type=int, default=None,
                    help="Process every N-th frame (overrides --n-frames)")
    ap.add_argument("--n-frames", type=int, default=None,
                    help="Target number of frames to process (computes stride automatically)")
    ap.add_argument("--exist-threshold", type=float, default=0.3,
                    help="(unused, accepted for compatibility)")
    ap.add_argument("--conf-threshold", type=int, default=1, choices=[0, 1, 2],
                    help="Min ARKit LiDAR confidence: 0=all, 1=medium+, 2=high only")
    ap.add_argument("--voxel-size", type=float, default=0.005,
                    help="Voxel grid size in metres for downsampling (default 0.005)")
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--min-seg-pts", type=int, default=80,
                    help="Minimum points per segment to keep (default 80 for single frames)")
    ap.add_argument("--single-frame", type=int, default=None, metavar="N",
                    help="Skip fusion: extract only frame N (sorted index) from .r3d")
    ap.add_argument("--xy-radius", type=float, default=3.0,
                    help="remove_table xy_radius in metres (default 3.0 for own scans)")
    ap.add_argument("--min-height", type=float, default=0.01,
                    help="remove_table min_height_above_table in metres (default 0.01)")
    ap.add_argument("--max-height", type=float, default=0.50,
                    help="remove_table max_height_above_table in metres (default 0.50)")
    args = ap.parse_args()

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(42)

    if args.r3d is None and args.input_cloud is None:
        ap.error("one of --r3d or --input-cloud is required")

    # ── 1. obtain fused point cloud ───────────────────────────────────────────
    if args.single_frame is not None:
        if args.r3d is None:
            ap.error("--single-frame requires --r3d")
        pose_idx = args.single_frame
        print(f"[1/6] Extracting single frame {pose_idx} from {args.r3d} …")
        decompress = _import_lzfse()
        from PIL import Image as _PIL_Image
        with zipfile.ZipFile(args.r3d, "r") as zf:
            meta    = json.loads(zf.read("metadata"))
            poses   = meta["poses"]
            frame_K = meta["perFrameIntrinsicCoeffs"]
            def _frame_num(path, ext):
                return int(path.split("/")[1].replace(ext, ""))
            all_depth = sorted(
                [f for f in zf.namelist() if f.startswith("rgbd/") and f.endswith(".depth")],
                key=lambda x: _frame_num(x, ".depth"),
            )
            all_conf = sorted(
                [f for f in zf.namelist() if f.startswith("rgbd/") and f.endswith(".conf")],
                key=lambda x: _frame_num(x, ".conf"),
            )
            all_jpg = sorted(
                [f for f in zf.namelist() if f.startswith("rgbd/") and f.endswith(".jpg")],
                key=lambda x: _frame_num(x, ".jpg"),
            )
            with zf.open(all_jpg[0]) as _f:
                _img0 = _PIL_Image.open(io.BytesIO(_f.read()))
                rgb_w, rgb_h = _img0.size
            depth = np.frombuffer(decompress(zf.read(all_depth[pose_idx])),
                                  dtype=np.float32).copy().reshape(DEPTH_H, DEPTH_W)
            conf  = np.frombuffer(decompress(zf.read(all_conf[pose_idx])),
                                  dtype=np.uint8).reshape(DEPTH_H, DEPTH_W)
            depth[conf < args.conf_threshold] = 0.0
            fx, fy, cx, cy = _depth_intrinsics(frame_K[pose_idx], rgb_w, rgb_h)
            pts_cam, flat_idx = _backproject(depth, fx, fy, cx, cy)
            R, t = _pose_to_Rt(poses[pose_idx])
            pts_raw = (pts_cam.astype(np.float64) @ R.T + t).astype(np.float32)
            try:
                img = _PIL_Image.open(io.BytesIO(zf.read(all_jpg[pose_idx]))).convert("RGB")
                img_arr = np.array(img)
                v_pix = flat_idx // DEPTH_W
                u_pix = flat_idx %  DEPTH_W
                u_rgb = np.clip(np.round(u_pix * (img_arr.shape[1] / DEPTH_W)).astype(int), 0, img_arr.shape[1]-1)
                v_rgb = np.clip(np.round(v_pix * (img_arr.shape[0] / DEPTH_H)).astype(int), 0, img_arr.shape[0]-1)
                colors_raw = img_arr[v_rgb, u_rgb]
            except Exception:
                colors_raw = None
            # apply initPose + coord fix
            init   = meta["initPose"]
            R_init = Rotation.from_quat(init[:4]).as_matrix()
            t_init = np.array(init[4:], dtype=np.float64)
            pts_raw = (pts_raw.astype(np.float64) @ R_init.T + t_init).astype(np.float32)
            R_fix   = Rotation.from_euler('y', -90, degrees=True).as_matrix()
            pts_raw = (pts_raw @ R_fix.T).astype(np.float32)
        print(f"      single frame {pose_idx}: {len(pts_raw)} pts")
    elif args.input_cloud:
        print(f"[1/6] Loading pre-fused cloud from {args.input_cloud} …")
        _data = np.load(args.input_cloud)
        pts_raw    = _data["pts"].astype(np.float32)
        colors_raw = _data["colors"] if "colors" in _data else None
        print(f"      loaded {len(pts_raw):,} pts")
        if args.verbose:
            print(f"      cloud bbox: "
                  f"x=[{pts_raw[:,0].min():.2f},{pts_raw[:,0].max():.2f}]  "
                  f"y=[{pts_raw[:,1].min():.2f},{pts_raw[:,1].max():.2f}]  "
                  f"z=[{pts_raw[:,2].min():.2f},{pts_raw[:,2].max():.2f}]")
    else:
        # ── compute frame stride ──────────────────────────────────────────────
        if args.frame_stride is not None:
            stride = args.frame_stride
        elif args.n_frames is not None:
            import json as _json, zipfile as _zf
            with _zf.ZipFile(args.r3d) as _z:
                _meta = _json.loads(_z.read("metadata"))
                _total = len(_meta["poses"])
            stride = max(1, _total // args.n_frames)
            if args.verbose:
                print(f"  --n-frames {args.n_frames}: stride={stride} "
                      f"(total={_total} frames)")
        else:
            stride = 15   # default ~4 fps from 60 fps

        print("[1/6] Fusing depth frames …")
        t0 = time.perf_counter()
        pts_raw, colors_raw = fuse_frames(
            args.r3d,
            frame_stride=stride,
            conf_threshold=args.conf_threshold,
            verbose=args.verbose,
        )
        print(f"      fused {len(pts_raw):,} pts in {time.perf_counter()-t0:.1f}s")
        if args.verbose:
            print(f"      cloud bbox: "
                  f"x=[{pts_raw[:,0].min():.2f},{pts_raw[:,0].max():.2f}]  "
                  f"y=[{pts_raw[:,1].min():.2f},{pts_raw[:,1].max():.2f}]  "
                  f"z=[{pts_raw[:,2].min():.2f},{pts_raw[:,2].max():.2f}]")

    # ── 2. voxel downsample ───────────────────────────────────────────────────
    print("[2/6] Voxel downsampling …")
    pts_ds, colors_ds = voxel_downsample(
        pts_raw, colors_raw, voxel_size=args.voxel_size, verbose=args.verbose
    )

    # ── 3. outlier removal ────────────────────────────────────────────────────
    print("[3/6] Statistical outlier removal …")
    pts_clean, keep_mask = _remove_outliers(pts_ds, verbose=args.verbose)
    colors_clean = colors_ds[keep_mask] if colors_ds is not None else None

    # ── 4. table removal + segmentation ──────────────────────────────────────
    print("[4/6] Table removal + segmentation …")
    try:
        obj_pts, table_normal, table_height, _table_pts, _n_tbl = remove_table(
            pts_clean,
            min_height_above_table=args.min_height,
            max_height_above_table=args.max_height,
            depth_margin=np.inf, xy_radius=np.inf,
        )
    except Exception as e:
        warnings.warn(f"Table removal failed ({e}); using full cloud.")
        obj_pts      = pts_clean
        table_normal = np.array([0.0, 1.0, 0.0])
        table_height = pts_clean[:, 1].min()

    if args.verbose:
        print(f"      table removed: {len(pts_clean)-len(obj_pts):,} table pts, "
              f"{len(obj_pts):,} foreground pts remain")

    # outlier removal on foreground
    obj_pts, _ = _remove_outliers(obj_pts, verbose=args.verbose)

    segs_raw = segment_instances(obj_pts)
    segs_raw = merge_nearby_segments(segs_raw)

    if args.verbose:
        print(f"      segments before filter: {len(segs_raw)}")

    # bbox filter + min-pts filter
    segs = []
    for si, seg in enumerate(segs_raw):
        if len(seg) < args.min_seg_pts:
            if args.verbose:
                print(f"      seg {si:02d}: {len(seg)} pts < {args.min_seg_pts} → dropped (too few pts)")
            continue
        dims = seg.max(axis=0) - seg.min(axis=0)
        if dims.max() > BBOX_MAX_DIM:
            if args.verbose:
                print(f"      seg {si:02d}: max dim {dims.max():.3f}m > {BBOX_MAX_DIM}m → "
                      f"dropped (wall/shelf)")
            continue
        segs.append(seg)

    # flatness filter — drop chair backs and planar wall remnants
    def _is_flat(pts_s, threshold=0.015):
        centered = pts_s - pts_s.mean(0)
        _, sv, _ = np.linalg.svd(centered, full_matrices=False)
        return (sv[2] / (sv[0] + 1e-8)) < threshold

    segs_before_flat = segs
    segs = []
    for si, seg in enumerate(segs_before_flat):
        if _is_flat(seg):
            if args.verbose:
                centered = seg - seg.mean(0)
                _, sv, _ = np.linalg.svd(centered, full_matrices=False)
                ratio = sv[2] / (sv[0] + 1e-8)
                print(f"      seg {si:02d}: flatness ratio={ratio:.4f} < 0.015 → dropped (flat plane)")
            continue
        segs.append(seg)

    print(f"      kept {len(segs)} segments after filters")

    if len(segs) == 0:
        print("WARNING: no segments survived filters — check scan or loosen parameters.")
        return

    if args.verbose:
        print(f"      table normal: [{table_normal[0]:.3f}, {table_normal[1]:.3f}, {table_normal[2]:.3f}]  "
              f"height: {table_height:.3f}m")

    # ── 5. SuperDec fitting ───────────────────────────────────────────────────
    print("[5/6] SuperDec fitting …")
    use_superdec = (args.checkpoint is not None) and not args.no_superdec

    if args.no_superdec:
        print("      --no-superdec: skipping fitting, producing empty fits")
        seg_fits = [MultiSQFit(primitives=[], n_points=len(s)) for s in segs]
        print(f"\n[6/6] Saving figure …")
        _make_figure(
            pts_fused   = pts_ds,
            colors_fused= colors_ds,
            obj_pts     = obj_pts,
            table_normal= table_normal,
            table_height= table_height,
            segs        = segs,
            seg_fits    = seg_fits,
            out_path    = out_dir / "result.png",
        )
        npz_path = out_dir / "fused_cloud.npz"
        save_dict = {"pts": pts_ds}
        if colors_ds is not None:
            save_dict["colors"] = colors_ds
        np.savez_compressed(npz_path, **save_dict)
        print(f"  fused cloud saved → {npz_path}")
        return

    if use_superdec:
        from project_3dv.perception.superdec_fitter import SuperdecFitter
        fitter = SuperdecFitter(checkpoint_path=args.checkpoint)
    else:
        from project_3dv.perception.superquadric import SuperquadricFitter
        fitter_sq = SuperquadricFitter()
        class _LMWrapper:
            def fit_batch(self, segs_):
                return [fitter_sq.fit(s) for s in segs_]
        fitter = _LMWrapper()

    # cap pts per segment before fitting
    capped_segs = []
    for seg in segs:
        if len(seg) > 8192:
            seg = seg[rng.choice(len(seg), 8192, replace=False)]
        capped_segs.append(seg)

    import torch
    BATCH_CHUNK = 8   # max segments per GPU forward+LM pass
    t_fit = time.perf_counter()
    all_results = []
    for i in range(0, len(capped_segs), BATCH_CHUNK):
        chunk = capped_segs[i:i + BATCH_CHUNK]
        chunk_results = fitter.fit_batch(chunk)
        all_results.extend(chunk_results)
        torch.cuda.empty_cache()
        if args.verbose:
            print(f"      chunk {i//BATCH_CHUNK + 1}/{(len(capped_segs)-1)//BATCH_CHUNK + 1}"
                  f"  segs {i}–{i+len(chunk)-1}  done")
    seg_fits = all_results
    print(f"      fit_batch: {time.perf_counter()-t_fit:.2f}s for {len(segs)} segments")

    # ── metrics ───────────────────────────────────────────────────────────────
    print("\n  Per-segment results:")
    print(f"  {'seg':>4}  {'pts':>6}  {'prims':>5}  {'norm-L2':>8}")
    for si, (seg, fit) in enumerate(zip(segs, seg_fits)):
        norm_l2 = _compute_chamfer_normalized(seg, fit)
        n_prims = len(fit.primitives) if fit and fit.primitives else 0
        print(f"  {si:4d}  {len(seg):6d}  {n_prims:5d}  {norm_l2:8.4f}")

    # ── 6. figure ─────────────────────────────────────────────────────────────
    print("\n[6/6] Saving figure …")
    _make_figure(
        pts_fused   = pts_ds,
        colors_fused= colors_ds,
        obj_pts     = obj_pts,
        table_normal= table_normal,
        table_height= table_height,
        segs        = segs,
        seg_fits    = seg_fits,
        out_path    = out_dir / "result.png",
    )

    # save fused cloud as .npz for inspection
    npz_path = out_dir / "fused_cloud.npz"
    save_dict = {"pts": pts_ds}
    if colors_ds is not None:
        save_dict["colors"] = colors_ds
    np.savez_compressed(npz_path, **save_dict)
    print(f"  fused cloud saved → {npz_path}")

    # save results.npz for visualise_own_scan.py
    # ── scale diagnostic: normalize_points uses scale = 2*max(|pts|) (diameter).
    # denormalize_outdict multiplies sx/sy/sz by this scale → world-frame semi-axes.
    # Expected: seg extent ~0.15m → norm_scale=0.15 → sx_world ~ 0.05–0.10m.
    # If translations are scene-scale (e.g. tx~1.0) but sx < 0.01 on all axes,
    # denorm failed. Flag and recover.
    import logging as _logging
    prim_list = []
    for seg_idx, (seg, fit) in enumerate(zip(segs, seg_fits)):
        if not (fit and fit.primitives):
            continue
        # normalize_points scale = 2 * max(|pts - center|)
        _center = seg.mean(0)
        _norm_scale = 2.0 * float(np.abs(seg - _center).max()) + 1e-8
        if args.verbose and seg_idx == 0:
            print(f"\n  seg 0: norm_scale={_norm_scale:.4f}  "
                  f"(primitives sx should be ~0.1–0.5 × {_norm_scale:.4f} "
                  f"= {0.1*_norm_scale:.4f}–{0.5*_norm_scale:.4f} m in world frame)")
        for p in fit.primitives:
            sx, sy, sz = float(p.sx), float(p.sy), float(p.sz)
            if sx < 0.01 and sy < 0.01 and sz < 0.01:
                _logging.warning(
                    f"seg {seg_idx} prim: all axes tiny after fit_batch "
                    f"(sx={sx:.4f} sy={sy:.4f} sz={sz:.4f}, "
                    f"norm_scale={_norm_scale:.4f}) — applying manual scale recovery"
                )
                p.sx = sx * _norm_scale
                p.sy = sy * _norm_scale
                p.sz = sz * _norm_scale
                p.tx = float(p.tx) * _norm_scale + _center[0]
                p.ty = float(p.ty) * _norm_scale + _center[1]
                p.tz = float(p.tz) * _norm_scale + _center[2]
            if args.verbose:
                print(f"    seg {seg_idx} prim: t=[{p.tx:.3f},{p.ty:.3f},{p.tz:.3f}] "
                      f"s=[{p.sx:.3f},{p.sy:.3f},{p.sz:.3f}]")
            prim_list.append({
                    "seg_idx": seg_idx,
                    "sx": float(p.sx), "sy": float(p.sy), "sz": float(p.sz),
                    "e1": float(p.e1), "e2": float(p.e2),
                    "tx": float(p.tx), "ty": float(p.ty), "tz": float(p.tz),
                    "rx": float(p.rx), "ry": float(p.ry), "rz": float(p.rz),
                    "rotation_matrix": np.array(p.rotation_matrix, dtype=np.float32),
                })
    results_path = out_dir / "results.npz"
    np.savez_compressed(
        results_path,
        seg_pts=np.array(segs, dtype=object),
        primitives=np.array(prim_list, dtype=object),
    )
    print(f"  results saved → {results_path}")


if __name__ == "__main__":
    main()
