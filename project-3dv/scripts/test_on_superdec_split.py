#!/usr/bin/env python3
"""
test_on_superdec_split.py
=========================

Loads samples from the SuperDec test split, runs them through the full
tabletop perception pipeline (remove_table → segment_instances →
fit_superquadrics), produces a 3-panel visualisation per sample, and
prints a summary table.

Test-split facts
----------------
  Category  | Model type     | Samples
  ----------|----------------|--------
  02876657  | bottle         |  50
  02880940  | bowl           |  18
  03624134  | knife          |  42
  03642806  | laptop         |  45
  03797390  | mug            |  22
  gso       | Google Scanned | 103
  ----------|----------------|--------
  TOTAL                      | 280  (286 incl. trailing-newline entries)

  All 286 paths are accessible on the cluster at:
      /work/courses/3dv/team15/superdec/data/ShapeNet/{category}/{model_id}/pointcloud.npz

  The .lst files contain bare model IDs (hex hashes for ShapeNet categories,
  product-name strings for gso).  The ShapeNet dataloader resolves the full
  path as:
      {shapenet.path}/{category}/{model_id}/pointcloud_4096.npz  (test, w/ fallback)
      {shapenet.path}/{category}/{model_id}/pointcloud.npz       (train/val)

Visualisation panels
--------------------
  Panel 1 – Raw input cloud, coloured by depth (Z value, plasma colormap).
  Panel 2 – Full cloud coloured by segment label (no table removed).
  Panel 3 – SQ wireframes only (plot_wireframe, 10×10 grid per primitive).
  Panel 4 – Overlay: grey cloud (α=0.3) + coloured SQ wireframes.

  All panels apply a Y-up → Z-up coordinate transform so ShapeNet objects
  appear upright in matplotlib (new_Y = −old_Z, new_Z = old_Y).

Note on table removal with ShapeNet data
-----------------------------------------
  ShapeNet / GSO point clouds are single-object scans stored with Y-up.
  There is no real table surface, so remove_table() is skipped by default
  (--no-table-removal is the default).  Pass --table-removal to enable it.

Usage
-----
    # default: 5 samples → outputs/test_split/
    python scripts/test_on_superdec_split.py

    # custom:
    python scripts/test_on_superdec_split.py --n-samples 10 \\
        --output-dir /tmp/sq_eval \\
        --data-root /path/to/ShapeNet

    # CPU-only login node (skips CUDA extension compilation entirely):
    python scripts/test_on_superdec_split.py --no-superdec

    # GPU node (via run_gpu.sh which sets CUDA_HOME + TORCH_CUDA_ARCH_LIST):
    bash scripts/run_gpu.sh --n-samples 20
"""

import argparse
import os
import sys
import warnings
from pathlib import Path

# ── Path setup ────────────────────────────────────────────────────────────────
_HERE        = Path(__file__).resolve().parent          # project-3dv/scripts/
_PROJ_ROOT   = _HERE.parent                             # project-3dv/
_PERCEPTION  = _PROJ_ROOT / "src" / "project_3dv" / "perception"
_SUPERDEC    = Path("/work/courses/3dv/team15/superdec")
_DEFAULT_DATA      = str(_SUPERDEC / "data" / "ShapeNet")
_DEFAULT_RGBD_DATA = str(_PROJ_ROOT / "data" / "rgbd-scenes-v2" / "pc")
_DEFAULT_CKPT = str(
    Path("/work/courses/3dv/team15/checkpoints")
    / "superdec_tabletop"
    / "superdec_tabletop_finetune_v2"
)

# Add perception/ to sys.path so bare `from pipeline import ...` works
# (same convention as the existing test files).
sys.path.insert(0, str(_PERCEPTION))
# Add superdec repo root so `superdec.*` imports resolve.
sys.path.insert(0, str(_SUPERDEC))

# ── Standard imports ──────────────────────────────────────────────────────────
import numpy as np
import matplotlib
matplotlib.use("Agg")               # headless – no display required
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  registers the 3D projection

# ── Project imports ───────────────────────────────────────────────────────────
from omegaconf import OmegaConf
from pipeline import (preprocess_pointcloud, postprocess_fits,
                      SHAPENET_CATEGORY_ROTATIONS,
                      adaptive_cluster_eps, _fps_numpy)
from superdec_fitter import merge_overlapping_primitives, confidence_weighted_chamfer
from superquadric import sample_surface_equal_distance

# RGB-D Scenes v2 loader
sys.path.insert(0, str(_PERCEPTION / "datasets"))
from rgbd_scenes import RGBDScenesV2

# SuperDec dataloader (unmodified – loaded from superdec repo)
from superdec.data.dataloader import ShapeNet

# Perception pipeline stages
from pipeline    import (remove_table, segment_instances, segment_instances_dual,
                         merge_nearby_segments, fit_superquadrics, ObjectSegment)
from superquadric import MultiSQFit, SuperquadricFitter
# SuperdecFitter import is safe at module level (CUDA backend is only loaded
# inside __init__); kept here so --help works without a GPU.
from superdec_fitter import SuperdecFitter

# ── Dataset metadata ─────────────────────────────────────────────────────────
CATEGORIES = ["02876657", "02880940", "03624134", "03642806", "03797390", "gso"]
CATEGORY_NAMES = {
    "02876657": "bottle",
    "02880940": "bowl",
    "03624134": "knife",
    "03642806": "laptop",
    "03797390": "mug",
    "gso":      "gso",
}

# Minimum foreground points for remove_table fallback.
# ShapeNet single-object clouds have no real table surface, so RANSAC often
# returns ≪ 200 points above the "detected plane".  Any result below this
# threshold is treated as a failure and the raw cloud is used instead.
MIN_FOREGROUND_PTS = 200

# ── Colour palette (same as visualize_sq.py) ─────────────────────────────────
PALETTE = [
    "#e63946", "#457b9d", "#2a9d8f", "#e76f51",
    "#8338ec", "#f4a261", "#06d6a0", "#ffd166",
    "#118ab2", "#ef476f", "#06a77d", "#d62246",
]
BG = "#1a1a2e"


def _colour(i: int) -> str:
    return PALETTE[i % len(PALETTE)]


def _hex_to_rgb(h: str):
    h = h.lstrip("#")
    return tuple(int(h[k:k+2], 16) / 255.0 for k in (0, 2, 4))


def _style_3d(ax, title: str = ""):
    for attr in ("xaxis", "yaxis", "zaxis"):
        pane = getattr(ax, attr).pane
        pane.fill = False
        pane.set_edgecolor("#333355")
    ax.set_xlabel("X", color="#888888", fontsize=7)
    ax.set_ylabel("Y", color="#888888", fontsize=7)
    ax.set_zlabel("Z", color="#888888", fontsize=7)
    ax.tick_params(colors="#888888", labelsize=5)
    ax.grid(True, alpha=0.08)
    if title:
        ax.set_title(title, color="white", fontsize=9, pad=3)


def _y_up_to_z_up(pts: np.ndarray) -> np.ndarray:
    """Convert Y-up (ShapeNet convention) to Z-up (matplotlib vertical)."""
    out = np.empty_like(pts)
    out[:, 0] = pts[:, 0]
    out[:, 1] = -pts[:, 2]
    out[:, 2] = pts[:, 1]
    return out


def _sq_wireframe_grid(prim, n_u: int = 10, n_v: int = 10, coord_transform=None):
    """
    Sample the SQ surface parametrically and return (X, Y, Z) 2-D grids
    for use with ax.plot_wireframe().

    The parametrization follows:
      Xs = sx * sp(cos U, e1) * sp(cos V, e2)
      Ys = sy * sp(cos U, e1) * sp(sin V, e2)
      Zs = sz * sp(sin U, e1)
    with u ∈ [−π/2, π/2] and v ∈ [−π, π].
    The local SQ coords are rotated/translated into world frame, then
    transformed via coord_transform (default: _y_up_to_z_up for ShapeNet).
    """
    def _sp(x, e):
        return np.sign(x) * (np.abs(x) + 1e-9) ** e

    u = np.linspace(-np.pi / 2, np.pi / 2, n_u)
    v = np.linspace(-np.pi,      np.pi,     n_v)
    U, V = np.meshgrid(u, v, indexing="ij")  # (n_u, n_v)

    Xs = prim.sx * _sp(np.cos(U), prim.e1) * _sp(np.cos(V), prim.e2)
    Ys = prim.sy * _sp(np.cos(U), prim.e1) * _sp(np.sin(V), prim.e2)
    Zs = prim.sz * _sp(np.sin(U), prim.e1)

    pts = np.stack([Xs.ravel(), Ys.ravel(), Zs.ravel()], axis=1)  # (n_u*n_v, 3)
    R = prim.rotation_matrix  # (3, 3)
    t = prim.translation       # (3,)
    pts = (R @ pts.T).T + t
    _tf = coord_transform if coord_transform is not None else _y_up_to_z_up
    pts = _tf(pts)

    shape = (n_u, n_v)
    return pts[:, 0].reshape(shape), pts[:, 1].reshape(shape), pts[:, 2].reshape(shape)


# ── Split inspection ──────────────────────────────────────────────────────────

def print_split_summary(data_root: str) -> int:
    """Print per-category test-split counts and path examples; return total."""
    print("\n" + "=" * 70)
    print("SuperDec test split summary")
    print("=" * 70)
    total = 0
    for cat in CATEGORIES:
        lst_path = os.path.join(data_root, cat, "test.lst")
        if not os.path.exists(lst_path):
            print(f"  {cat} ({CATEGORY_NAMES[cat]:<8}): [test.lst not found at {lst_path}]")
            continue
        with open(lst_path) as fh:
            ids = [ln.strip() for ln in fh if ln.strip()]
        example_id   = ids[0] if ids else "(empty)"
        example_path = os.path.join(data_root, cat, example_id, "pointcloud.npz")
        accessible   = "✓" if os.path.exists(example_path) else "✗"
        print(f"  {cat} ({CATEGORY_NAMES[cat]:<8}): {len(ids):4d} samples  "
              f"e.g. …/{cat}/{example_id[:20]}…  [{accessible} on disk]")
        total += len(ids)
    print(f"  {'TOTAL':<22}  {total:4d} samples")
    print(f"\n  Paths: {data_root}/{{category}}/{{model_id}}/pointcloud.npz")
    print("=" * 70 + "\n")
    return total


# ── Data loading ──────────────────────────────────────────────────────────────

def build_dataloader(data_root: str) -> ShapeNet:
    """Construct a ShapeNet test-split dataset using the existing dataloader."""
    cfg = OmegaConf.create({
        "shapenet": {
            "path":       data_root,
            "categories": CATEGORIES,
            "normalize":  False,   # keep raw world-scale coordinates
        },
        # get_transforms() checks cfg.trainer – provide a stub so it returns None
        "trainer": {"augmentations": False},
    })
    return ShapeNet(split="test", cfg=cfg)


# ── Pipeline ──────────────────────────────────────────────────────────────────

def run_pipeline(pts_raw: np.ndarray, fitter: SuperdecFitter,
                 skip_table_removal: bool = True,
                 dual_segment: bool = False):
    """
    Stage 2 – remove_table()  (optional, skipped by default for ShapeNet)
    Stage 3 – segment_instances()
    Stage 4 – fit_superquadrics() with the fine-tuned SuperdecFitter

    Parameters
    ----------
    pts_raw            : (N, 3) float64  raw point cloud from the dataloader
    fitter             : SuperdecFitter  pre-loaded fine-tuned model
    skip_table_removal : bool  if True, skip remove_table() entirely

    Returns
    -------
    obj_pts   : (M, 3) foreground cloud after remove_table (or raw)
    segments  : list of (N_i, 3) arrays, one per detected instance
    sq_fits   : list of MultiSQFit, same length as segments
    used_raw  : bool – True when table removal was skipped or fell back
    """
    # ── Stage 2: remove_table (optional) ────────────────────────────────────
    used_raw = False
    if skip_table_removal:
        obj_pts  = pts_raw
        used_raw = True
    else:
        try:
            obj_pts, *_ = remove_table(pts_raw, depth_margin=np.inf, xy_radius=np.inf)
            if len(obj_pts) < MIN_FOREGROUND_PTS:
                warnings.warn(
                    f"remove_table returned only {len(obj_pts)} pts "
                    f"(< {MIN_FOREGROUND_PTS}); using raw cloud",
                    stacklevel=2,
                )
                obj_pts  = pts_raw
                used_raw = True
        except Exception as exc:
            warnings.warn(
                f"remove_table raised {exc!r}; using raw cloud",
                stacklevel=2,
            )
            obj_pts  = pts_raw
            used_raw = True

    # ── Stage 3: segment_instances ────────────────────────────────────────────
    if dual_segment:
        segments = segment_instances_dual(obj_pts)
    else:
        segments = segment_instances(obj_pts)
    if not segments:
        # No DBSCAN clusters found – treat the entire foreground as one segment
        segments = [obj_pts]

    # Wrap arrays in ObjectSegment (required by fit_superquadrics)
    obj_segs = [
        ObjectSegment(
            id=i,
            points=pts.astype(np.float32),
            centroid=pts.mean(axis=0).astype(np.float32),
            bbox_min=pts.min(axis=0).astype(np.float32),
            bbox_max=pts.max(axis=0).astype(np.float32),
        )
        for i, pts in enumerate(segments)
    ]

    # ── Stage 4: fit_superquadrics ────────────────────────────────────────────
    sq_fits = fit_superquadrics(obj_segs, fitter=fitter)

    return obj_pts, segments, sq_fits, used_raw


# ── Visualisation ─────────────────────────────────────────────────────────────

def save_4panel(
    pts_raw:          np.ndarray,
    pts_pre:          np.ndarray,
    segments:         list,
    sq_fits:          list,
    sample_name:      str,
    save_path:        Path,
    gt_labels:        np.ndarray = None,
    coord_transform = None,
):
    """
    Save a 4-panel figure (24×6 inches, dark background):
      Panel 1 – raw input cloud, depth-coloured (Z, plasma)
      Panel 2 – if gt_labels provided: cloud coloured by GT instance label;
                otherwise: full preprocessed cloud coloured by nearest-segment
                centroid assignment
      Panel 3 – SQ wireframes only (plot_wireframe, 10×10)
      Panel 4 – overlay: grey cloud (α=0.3) + coloured SQ wireframes

    Parameters
    ----------
    gt_labels       : (N,) int array aligned with pts_raw.  When provided,
                      Panel 2 colours each point by its ground-truth label
                      rather than by the predicted segment centroid.
    coord_transform : callable (pts_array → pts_array), applied to all
                      display coordinates.  Default (None) uses
                      _y_up_to_z_up for ShapeNet; pass `lambda x: x` for
                      real-world (RGB-D) data that is already Z-up.
    """
    _tf = coord_transform if coord_transform is not None else _y_up_to_z_up

    fig = plt.figure(figsize=(24, 6), facecolor=BG)

    # ── helper: scatter a random subsample ───────────────────────────────────
    def _scatter(ax, pts, c, n_max=600, s=2, alpha=0.6):
        if len(pts) == 0:
            return
        idx = np.random.choice(len(pts), min(n_max, len(pts)), replace=False)
        p   = pts[idx]
        ax.scatter(p[:, 0], p[:, 1], p[:, 2],
                   c=c if isinstance(c, str) else [c] * len(idx),
                   s=s, alpha=alpha, depthshade=False)

    # ── Panel 1: depth-coloured raw cloud ────────────────────────────────────
    ax1 = fig.add_subplot(141, projection="3d", facecolor=BG)
    if len(pts_raw) > 0:
        p_tf = _tf(pts_raw)
        idx  = np.random.choice(len(p_tf), min(1500, len(p_tf)), replace=False)
        p    = p_tf[idx]
        z    = p[:, 2]
        norm_z = (z - z.min()) / (z.max() - z.min() + 1e-9)
        ax1.scatter(p[:, 0], p[:, 1], p[:, 2],
                    c=norm_z, cmap="plasma", s=2, alpha=0.6, depthshade=False)
    _style_3d(ax1, f"Raw input  ({len(pts_raw)} pts)")

    # ── Panel 2: GT labels (if provided) or nearest-segment coloring ─────────
    ax2 = fig.add_subplot(142, projection="3d", facecolor=BG)
    if gt_labels is not None and len(pts_raw) > 0:
        # Colour each point by its ground-truth instance label
        n_show   = min(2000, len(pts_raw))
        show_idx = np.random.choice(len(pts_raw), n_show, replace=False)
        p_tf     = _tf(pts_raw[show_idx])
        # Map each label to a palette colour; label 0 gets grey
        cols = np.array([
            _hex_to_rgb(_colour(int(gt_labels[j]))) if gt_labels[j] > 0
            else (0.35, 0.35, 0.35)
            for j in show_idx
        ])
        ax2.scatter(p_tf[:, 0], p_tf[:, 1], p_tf[:, 2],
                    c=cols, s=2, alpha=0.75, depthshade=False)
        n_gt_obj = len(np.unique(gt_labels[gt_labels > 0]))
        _style_3d(ax2, f"GT labels  ({n_gt_obj} object(s))")
    elif len(pts_pre) > 0 and segments:
        # Assign each point in pts_pre to the nearest segment centroid
        centroids = np.array([s.mean(axis=0) for s in segments])   # (K, 3)
        diff  = pts_pre[:, None, :] - centroids[None, :, :]        # (N, K, 3)
        dists = (diff ** 2).sum(axis=-1)                            # (N, K)
        nn    = dists.argmin(axis=1)                                # (N,)
        n_show   = min(1200, len(pts_pre))
        show_idx = np.random.choice(len(pts_pre), n_show, replace=False)
        p_tf  = _tf(pts_pre[show_idx])
        cols  = np.array([_hex_to_rgb(_colour(int(nn[j])))
                          for j in show_idx])
        ax2.scatter(p_tf[:, 0], p_tf[:, 1], p_tf[:, 2],
                    c=cols, s=2, alpha=0.75, depthshade=False)
        _style_3d(ax2, f"Segments  ({len(segments)} instance(s))")
    elif len(pts_pre) > 0:
        _scatter(ax2, _tf(pts_pre), c="#888888", n_max=1200)
        _style_3d(ax2, f"Segments  ({len(segments)} instance(s))")
    else:
        _style_3d(ax2, "Segments")

    # ── Panel 3: SQ wireframes only ───────────────────────────────────────────
    ax3 = fig.add_subplot(143, projection="3d", facecolor=BG)
    n_valid = 0
    prim_idx = 0
    for multi in sq_fits:
        for prim in multi.primitives:
            col = _colour(prim_idx)
            try:
                gX, gY, gZ = _sq_wireframe_grid(prim, n_u=10, n_v=10,
                                                 coord_transform=coord_transform)
                ax3.plot_wireframe(gX, gY, gZ, color=col,
                                   linewidth=0.6, alpha=0.85,
                                   rstride=1, cstride=1)
            except Exception:
                pass
            n_valid   += 1
            prim_idx  += 1
    _style_3d(ax3, f"SQ wireframes  ({n_valid} prim(s))")

    # ── Panel 4: overlay – grey cloud + SQ wireframes ─────────────────────────
    ax4 = fig.add_subplot(144, projection="3d", facecolor=BG)
    if len(pts_raw) > 0:
        _scatter(ax4, _tf(pts_raw), c="#888888", n_max=1000, s=1, alpha=0.3)
    prim_idx = 0
    for multi in sq_fits:
        for prim in multi.primitives:
            col = _colour(prim_idx)
            try:
                gX, gY, gZ = _sq_wireframe_grid(prim, n_u=10, n_v=10,
                                                 coord_transform=coord_transform)
                ax4.plot_wireframe(gX, gY, gZ, color=col,
                                   linewidth=0.8, alpha=0.9,
                                   rstride=1, cstride=1)
            except Exception:
                pass
            prim_idx += 1
    _style_3d(ax4, "Overlay  (cloud + wireframes)")

    # ── Super-title ───────────────────────────────────────────────────────────
    all_l2 = [p.chamfer_l2 for m in sq_fits for p in m.primitives]
    mean_l2_str = f"{np.mean(all_l2):.4f}" if all_l2 else "n/a"
    plt.suptitle(
        f"{sample_name}  ·  segs={len(segments)}  "
        f"prims={n_valid}  mean_chamfer_L2={mean_l2_str}",
        color="white", fontsize=10, y=1.01,
    )
    plt.tight_layout()
    fig.savefig(str(save_path), dpi=130, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    return n_valid, np.mean(all_l2) if all_l2 else float("nan")


# ── Chamfer L2 helpers ────────────────────────────────────────────────────────

def _chamfer_l2_cloud_vs_fits(pts: np.ndarray, sq_fits: list,
                               n_surface_pts: int = 500) -> float:
    """Symmetric Chamfer L2 between a point cloud and all SQ surface points.

    Uses equal arc-length surface sampling (Liu et al. CVPR 2022) and
    bi-directional Chamfer (Paschalidou et al. CVPR 2019, eq. 3, equal
    weights 1.0 + 1.0 for evaluation).

    Parameters
    ----------
    pts           : (N, 3) point cloud in the SAME coordinate frame as SQ poses.
    sq_fits       : List[MultiSQFit] — typically after postprocess_fits().
    n_surface_pts : surface points per primitive (--surface-samples flag).
    """
    all_l2 = []
    pts32 = pts.astype(np.float32)
    for multi in sq_fits:
        for prim in multi.primitives:
            try:
                surf = sample_surface_equal_distance(prim, n_points=n_surface_pts)
                if surf is None or len(surf) == 0:
                    continue
                surf32 = surf.astype(np.float32)
                diff_ps = pts32[:, None, :] - surf32[None, :, :]
                d_ps = (diff_ps ** 2).sum(-1).min(-1).mean()
                diff_sp = surf32[:, None, :] - pts32[None, :, :]
                d_sp = (diff_sp ** 2).sum(-1).min(-1).mean()
                all_l2.append(float(d_ps + d_sp))
            except Exception:
                pass
    return float(np.mean(all_l2)) if all_l2 else float("nan")


# ── RGB-D Scenes v2 pipeline ──────────────────────────────────────────────────

# Initial random subsample cap — keeps label-aligned pts_raw manageable.
_RGBD_MAX_PTS = 50_000
# Post-table-removal FPS target — dense enough for DBSCAN, cheap to cluster.
_RGBD_FPS_PTS = 8_192
# Minimum points for a segment to be kept (filters DBSCAN fragments).
_RGBD_MIN_SEG_PTS = 100
# DBSCAN adaptive-eps multiplier for RGB-D scenes (default, overridden by
# --rgbd-eps-mult CLI arg).
_RGBD_EPS_MULTIPLIER = 3.0


def _main_rgbd(args, out_dir: Path):
    """Process --n-samples RGB-D Scenes v2 scenes end-to-end.

    Pipeline per scene
    ------------------
    1. Load PLY + labels, random-subsample to _RGBD_MAX_PTS (keeps GT labels).
    2. remove_table()  — RANSAC plane removal; print pts-before / pts-after.
    3. FPS to _RGBD_FPS_PTS — uniform coverage for DBSCAN.
    4. adaptive_cluster_eps()  — median-NN based eps, printed per scene.
    5. segment_instances(..., adaptive_eps=True, eps_multiplier=2.0).
    6. Filter segments < _RGBD_MIN_SEG_PTS points.
    7. fit_superquadrics() per segment.
    """
    data_root = args.rgbd_data_root
    n_scenes  = min(args.n_samples, RGBDScenesV2.NUM_SCENES)

    _eps_mult = getattr(args, "rgbd_eps_mult", _RGBD_EPS_MULTIPLIER)
    _eps_max  = getattr(args, "rgbd_eps_max",  0.08)
    _merge_dist = getattr(args, "rgbd_merge_dist", 0.15)
    _no_merge   = getattr(args, "no_segment_merge", False)
    _no_fps     = getattr(args, "no_rgbd_fps", False)

    print(f"\nRGB-D Scenes v2 evaluation")
    print(f"  Data root    : {data_root}")
    print(f"  Scenes       : 1 – {n_scenes}  (max: {RGBDScenesV2.NUM_SCENES})")
    print(f"  DBSCAN cloud : {'full foreground (no FPS)' if not _no_fps else f'FPS to {_RGBD_FPS_PTS} pts'}")
    print(f"  eps_mult     : {_eps_mult}  eps_max: {_eps_max} m")
    print(f"  merge_dist   : {_merge_dist} m  no-merge: {_no_merge}\n")

    # Initialise fitter
    if args.no_superdec:
        fitter      = SuperquadricFitter(n_restarts=3, n_lm_rounds=15, subsample=512)
        fitter_name = "SuperquadricFitter (LM) — CPU-only mode (--no-superdec)"
        print(f"Fitter: {fitter_name}\n")
    else:
        fitter = SuperdecFitter(
            checkpoint_path=args.checkpoint,
            exist_threshold=0.3,
        )
        fitter_name = "SuperdecFitter (SuperDec) — GPU"
        print(f"Fitter: {fitter_name}\n")

    _N_QUERIES = 16   # SuperDec max primitives per object (train_tabletop.yaml)
    hdr = (f"{'#':<4}  {'scene':<12}  {'gt-obj':>6}  {'after-table':>11}  "
           f"{'eps':>8}  {'segs':>4}  {'prims':>5}  {'parsimony':>9}  "
           f"{'pre-L2':>8}  {'conf-L2':>8}  {'fitter':<9}")
    sep = "-" * len(hdr)
    print(hdr)
    print(sep)

    # Identity transform — RGB-D data is already in camera/world frame (Z-forward)
    _identity = lambda x: x  # noqa: E731

    for scene_id in range(1, n_scenes + 1):
        scene = RGBDScenesV2(data_root=data_root, scene_id=scene_id)
        try:
            pts_full, _, labels_full = scene.load()
        except Exception as exc:
            warnings.warn(f"Scene {scene_id:02d}: load failed ({exc!r}); skipping")
            continue

        # ── Step 1: random subsample to _RGBD_MAX_PTS (keep GT labels aligned) ─
        N = len(pts_full)
        if N > _RGBD_MAX_PTS:
            idx        = np.random.choice(N, _RGBD_MAX_PTS, replace=False)
            pts_raw    = pts_full[idx].astype(np.float64)
            labels_sub = labels_full[idx]
        else:
            pts_raw    = pts_full.astype(np.float64)
            labels_sub = labels_full

        n_gt = len(np.unique(labels_sub[labels_sub > 0]))

        # ── Step 2: remove_table — print pts-before / pts-after ───────────────
        used_raw     = False
        table_normal = None
        table_height = None
        try:
            obj_pts, table_normal, table_height, *_ = remove_table(
                pts_raw,
                max_height_above_table=0.4,
                depth_margin=np.inf, xy_radius=np.inf,
            )
            n_after = len(obj_pts)
            if n_after < MIN_FOREGROUND_PTS:
                warnings.warn(
                    f"Scene {scene_id:02d}: remove_table left only {n_after} pts "
                    f"(< {MIN_FOREGROUND_PTS}); using full cloud",
                    stacklevel=2,
                )
                obj_pts  = pts_raw
                used_raw = True
        except Exception as exc:
            warnings.warn(
                f"Scene {scene_id:02d}: remove_table failed ({exc!r}); using full cloud",
                stacklevel=2,
            )
            obj_pts  = pts_raw
            used_raw = True
        n_after = len(obj_pts)
        print(f"  scene_{scene_id:02d}: pts {len(pts_raw):,} → after table removal {n_after:,} "
              f"({'full cloud' if used_raw else 'foreground only'})")

        # ── Step 3: FPS subsample foreground to _RGBD_FPS_PTS ─────────────────
        # Stage 3: skip FPS before DBSCAN when --no-rgbd-fps is NOT set
        # (default behaviour is now to use the full foreground cloud for DBSCAN
        # so that object surfaces are dense enough for eps to connect fragments).
        if _no_fps and len(obj_pts) > _RGBD_FPS_PTS:
            fps_idx  = _fps_numpy(obj_pts, _RGBD_FPS_PTS)
            dbscan_pts = obj_pts[fps_idx]
        else:
            dbscan_pts = obj_pts

        # ── Step 4: compute adaptive eps and print ────────────────────────────
        eps_val = adaptive_cluster_eps(
            dbscan_pts, multiplier=_eps_mult, eps_max=_eps_max,
        )
        print(f"  scene_{scene_id:02d}: adaptive eps = {eps_val:.4f} m  "
              f"({len(dbscan_pts)} pts fed to DBSCAN)")

        # ── Step 5: segment_instances with adaptive eps ───────────────────────
        _seg_fn = segment_instances_dual if args.dual_segment else segment_instances
        segments = _seg_fn(
            dbscan_pts,
            adaptive_eps=True,
            eps_multiplier=_eps_mult,
            cluster_min_points=5,
            eps_max=_eps_max,
        )
        if not segments:
            segments = [dbscan_pts]

        # ── Step 5b: merge nearby fragments (Stage 2) ─────────────────────────
        if not _no_merge:
            n_before_merge = len(segments)
            segments = merge_nearby_segments(segments, merge_dist=_merge_dist)
            n_after_merge  = len(segments)
            print(f"  scene_{scene_id:02d}: merged {n_before_merge} → {n_after_merge} segments "
                  f"(merge_dist={_merge_dist} m)")

        # ── Step 6: filter tiny fragments ─────────────────────────────────────
        segments = [s for s in segments if len(s) >= _RGBD_MIN_SEG_PTS]
        if not segments:
            segments = [dbscan_pts]

        # ── Step 6b: height filter — discard floor / wall fragments ───────────
        # Use table_normal to project centroids onto the table-perpendicular axis
        # (same convention as remove_table's height filter).
        if table_normal is not None and table_height is not None:
            _H_MIN, _H_MAX = 0.01, 0.45
            kept, discarded = [], 0
            for s in segments:
                centroid = s.mean(axis=0)
                h = float(centroid @ table_normal - table_height)
                if _H_MIN <= h <= _H_MAX:
                    kept.append(s)
                else:
                    discarded += 1
            print(f"  scene_{scene_id:02d}: height filter  kept {len(kept)}, "
                  f"discarded {discarded}  "
                  f"(h window [{_H_MIN:.2f}, {_H_MAX:.2f}] m above table)")
            segments = kept if kept else segments  # never discard everything

        # ── Step 6c: bounding-box size filter ────────────────────────────────
        # Discard segments whose largest axis extent exceeds _MAX_SEG_EXTENT.
        # Tabletop objects are typically 0.05–0.40 m; anything larger is a
        # background wall patch, shelf, or floor strip that survived table
        # removal and will produce garbage SuperDec fits (the log warnings
        # "Segment has range X.Xm" come from exactly these oversized segments).
        _max_extent = getattr(args, "rgbd_max_seg_extent", 0.60)
        kept_sz, discarded_sz = [], 0
        for s in segments:
            extent = float((s.max(axis=0) - s.min(axis=0)).max())
            if extent <= _max_extent:
                kept_sz.append(s)
            else:
                discarded_sz += 1
        print(f"  scene_{scene_id:02d}: size filter  kept {len(kept_sz)}, "
              f"discarded {discarded_sz}  (max_extent ≤ {_max_extent:.2f} m)")
        segments = kept_sz if kept_sz else segments  # never discard everything

        # ── Step 6d: compute distance weights (SAM2Object eq. 7) ─────────────
        # d_i = Euclidean distance of segment centroid from camera origin.
        # D_i = 0.9*(d_max - d_i)/(d_max - d_min + 1e-6) + 0.1
        # Close segments → D≈1.0 (aggressive merge); far → D≈0.1 (conservative).
        seg_depths = np.array([
            float(np.linalg.norm(s.mean(axis=0))) for s in segments
        ])
        _d_min, _d_max = seg_depths.min(), seg_depths.max()
        seg_distance_weights = (
            0.9 * (_d_max - seg_depths) / (_d_max - _d_min + 1e-6) + 0.1
        ).tolist()
        print(f"  scene_{scene_id:02d}: seg depths/weights  " +
              "  ".join(f"d={d:.2f}m D={w:.2f}"
                        for d, w in zip(seg_depths, seg_distance_weights)))

        # ── Step 7: fit superquadrics ─────────────────────────────────────────
        # When table_normal is available, transform each segment to the table
        # frame (table_normal → +Z) before fitting so the fitter sees a
        # consistent up-direction.  postprocess_fits() inverts per segment.
        _use_tbl = (not getattr(args, "no_table_frame", False)
                    and table_normal is not None and table_height is not None)
        if _use_tbl:
            tbl_n_cam = table_normal  # already captured from remove_table()
            print(f"  scene_{scene_id:02d}: table_normal = "
                  f"({tbl_n_cam[0]:.3f},{tbl_n_cam[1]:.3f},{tbl_n_cam[2]:.3f})  "
                  f"(Z-component {tbl_n_cam[2]:.3f}, expect > 0.90 for flat table)")

        for_sd    = not args.no_superdec
        obj_segs  = []
        seg_metas = []
        for i, s in enumerate(segments):
            if _use_tbl:
                pts_prep, _, meta_tbl = preprocess_pointcloud(
                    s.astype(np.float64),
                    target_n=len(s),      # keep all segment points; fitter resamples
                    outlier_std=1e9,      # no outlier removal — already filtered
                    table_normal=table_normal,
                    table_height=float(table_height),
                    for_superdec=for_sd,
                )
            else:
                pts_prep = s.astype(np.float64)
                meta_tbl = {
                    "scale": 1.0, "centroid": np.zeros(3),
                    "rotation": np.eye(3), "n_outliers_removed": 0,
                }
            if not args.no_superdec:
                pts_range = float(pts_prep.max() - pts_prep.min())
                if pts_range > 2.0:
                    import logging as _logging
                    _logging.getLogger(__name__).warning(
                        f"Segment has range {pts_range:.2f}m before SuperDec — "
                        "SuperDec normalises internally but this is larger than "
                        "ShapeNet training distribution (typical ~1m objects). "
                        "Consider reducing xy_radius or using per-segment centring."
                    )
            # DEBUG: dump coordinate values for first segment of first scene
            if scene_id == 1 and i == 0:
                print(f"DEBUG seg centroid (world):  {s.mean(axis=0)}")
                print(f"DEBUG pts_pre centroid:      {pts_prep.mean(axis=0)}")
                print(f"DEBUG meta table_centroid:   {meta_tbl.get('table_centroid')}")
                print(f"DEBUG meta centroid:         {meta_tbl.get('centroid')}")
            seg_metas.append(meta_tbl)
            obj_segs.append(ObjectSegment(
                id=i,
                points=pts_prep.astype(np.float32),
                centroid=pts_prep.mean(axis=0).astype(np.float32),
                bbox_min=pts_prep.min(axis=0).astype(np.float32),
                bbox_max=pts_prep.max(axis=0).astype(np.float32),
            ))

        sq_fits = fit_superquadrics(obj_segs, fitter=fitter)

        # Undo table frame per segment (no-op when _use_tbl is False)
        if _use_tbl:
            postprocessed = []
            for seg_idx, (fit, meta) in enumerate(zip(sq_fits, seg_metas)):
                pp = postprocess_fits([fit], meta)[0]
                if scene_id == 1 and seg_idx == 0 and fit.primitives:
                    print(f"DEBUG fit prim[0] t (pre):   {fit.primitives[0].translation}")
                    print(f"DEBUG fit prim[0] t (post):  {pp.primitives[0].translation if pp.primitives else 'no prims'}")
                    print(f"DEBUG expected ~= seg centroid (world): {obj_segs[0].points.mean(axis=0)}")
                postprocessed.append(pp)
            sq_fits = postprocessed

        # ── Fix 2: cap primitives per segment to 6 (keep largest by volume) ───
        _RGBD_MAX_PRIMS_PER_SEG = 6
        for m in sq_fits:
            if len(m.primitives) > _RGBD_MAX_PRIMS_PER_SEG:
                m.primitives = sorted(
                    m.primitives,
                    key=lambda p: p.sx * p.sy * p.sz,
                    reverse=True,
                )[:_RGBD_MAX_PRIMS_PER_SEG]

        if not args.no_merge:
            _dw = None if args.no_distance_merge else seg_distance_weights
            sq_fits = merge_overlapping_primitives(
                sq_fits, iou_threshold=0.1, distance_weights=_dw,
            )

        n_segs  = len(segments)
        n_prims = sum(len(m.primitives) for m in sq_fits)

        pre_l2_vals = [p.chamfer_l2 for m in sq_fits for p in m.primitives]
        mean_pre_l2 = float(np.mean(pre_l2_vals)) if pre_l2_vals else float("nan")

        # ── Fix 1: diagnostic — bounding boxes before plotting ────────────────
        cmin, cmax = dbscan_pts.min(axis=0), dbscan_pts.max(axis=0)
        print(f"  scene_{scene_id:02d}: cloud bbox  X[{cmin[0]:.3f},{cmax[0]:.3f}] "
              f"Y[{cmin[1]:.3f},{cmax[1]:.3f}] Z[{cmin[2]:.3f},{cmax[2]:.3f}]")
        if sq_fits and sq_fits[0].primitives:
            p0 = sq_fits[0].primitives[0]
            print(f"  scene_{scene_id:02d}: prim[0] t=({p0.tx:.3f},{p0.ty:.3f},{p0.tz:.3f}) "
                  f"s=({p0.sx:.3f},{p0.sy:.3f},{p0.sz:.3f})")

        # ── Visualisation ─────────────────────────────────────────────────────
        scene_name = f"scene_{scene_id:02d}"
        png_path   = out_dir / f"sample_{scene_id}.png"
        save_4panel(
            pts_raw, dbscan_pts, segments, sq_fits,
            sample_name=scene_name,
            save_path=png_path,
            gt_labels=labels_sub,
            coord_transform=_identity,
        )

        mean_conf_l2 = confidence_weighted_chamfer(
            sq_fits, dbscan_pts, n_points=args.surface_samples)

        parsimony = n_prims / _N_QUERIES
        pre_str  = f"{mean_pre_l2:.4f}" if np.isfinite(mean_pre_l2) else "   n/a"
        conf_str = f"{mean_conf_l2:.4f}" if np.isfinite(mean_conf_l2) else "   n/a"
        pars_str  = f"{parsimony:.4f}"
        row = (f"{scene_id:<4}  {scene_name:<12}  {n_gt:>6}  {n_after:>11,}  "
               f"{eps_val:>8.4f}  {n_segs:>4}  {n_prims:>5}  {pars_str:>9}  "
               f"{pre_str:>8}  {conf_str:>8}  {fitter_name:<9}")
        print(row)

    print(sep)
    print(f"\nOutputs saved to: {out_dir}/")
    print(f"  sample_1.png … sample_{n_scenes}.png")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    # Reverse lookup: human name → category ID (e.g. "bottle" → "02876657")
    _name_to_id = {v: k for k, v in CATEGORY_NAMES.items()}

    parser = argparse.ArgumentParser(
        description="Evaluate SuperDec test split through the perception pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--n-samples", type=int, default=5,
        help="Samples to process.  With --categories, this many are taken from "
             "EACH specified category.",
    )
    parser.add_argument(
        "--categories", default="",
        help="Comma-separated category names to sample from, e.g. "
             "'bottle,mug,bowl'.  Default: all categories, first --n-samples overall.",
    )
    parser.add_argument(
        "--output-dir", default="outputs/test_split",
        help="Directory for 4-panel PNG outputs",
    )
    parser.add_argument(
        "--data-root", default=_DEFAULT_DATA,
        help="Path to ShapeNet data root (must contain {category}/{model_id}/pointcloud.npz)",
    )
    parser.add_argument(
        "--superdec-dir", default=str(_SUPERDEC),
        help="Path to the superdec repository root",
    )
    parser.add_argument(
        '--checkpoint',
        type=str,
        default=None,
        help=(
            'Path to SuperDec checkpoint .pth file. '
            'Required when --no-superdec is NOT set. '
            'Example: /work/courses/3dv/team15/superdec/checkpoints/'
            'tabletop_best.pth'
        ),
    )
    parser.add_argument(
        "--checkpoint-dir", default=_DEFAULT_CKPT,
        help="Path to the SuperdecFitter checkpoint directory "
             "(must contain config.yaml + epoch_*.pt or ckpt.pt). "
             "Used by the ShapeNet path and as fallback when --checkpoint is not set.",
    )
    parser.add_argument(
        "--no-superdec", action="store_true",
        help="Skip SuperdecFitter (avoids CUDA extension compilation) and use "
             "the classical Levenberg-Marquardt fitter instead.  Use this on "
             "CPU-only login nodes.  For the neural fitter run via run_gpu.sh.",
    )
    parser.add_argument(
        "--table-removal", action="store_true", default=False,
        help="Enable remove_table() (RANSAC plane removal).  Disabled by default "
             "because ShapeNet / GSO clouds are single-object scans with no real "
             "table surface.",
    )
    parser.add_argument(
        "--no-preprocess", action="store_true", default=False,
        help="Skip preprocess_pointcloud() (outlier removal, FPS resampling). "
             "Use this to compare raw vs. preprocessed pipeline quality.",
    )
    parser.add_argument(
        "--no-category-rotation", action="store_true", default=False,
        help="Skip per-category canonical rotation (SHAPENET_CATEGORY_ROTATIONS). "
             "Use this to ablate the orientation correction.",
    )
    parser.add_argument(
        "--no-merge", action="store_true", default=False,
        help="Skip merge_overlapping_primitives() post-processing. "
             "Use this to ablate primitive merging.",
    )
    parser.add_argument(
        "--no-distance-merge", action="store_true", default=False,
        help="Disable distance-weighted merging in merge_overlapping_primitives(). "
             "When set, distance_weights=None is passed, restoring flat IoU "
             "threshold behaviour.  Default: distance weighting is on.",
    )
    parser.add_argument(
        "--no-table-frame", action="store_true", default=False,
        help="Disable table-frame canonicalisation in the RGB-D path. "
             "When set, segment points are passed to the fitter in raw "
             "camera frame without aligning the table to +Z.  "
             "Default: table-frame alignment is on (requires table_normal "
             "from remove_table()).",
    )
    parser.add_argument(
        "--dataset", default="shapenet", choices=["shapenet", "rgbd_scenes"],
        help="Which dataset to evaluate: 'shapenet' (default, uses the SuperDec "
             "test split) or 'rgbd_scenes' (RGB-D Scenes v2 tabletop scenes).",
    )
    parser.add_argument(
        "--rgbd-data-root", default=_DEFAULT_RGBD_DATA,
        help="Path to the RGB-D Scenes v2 pc/ directory "
             "(used when --dataset rgbd_scenes).",
    )
    parser.add_argument(
        "--surface-samples", type=int, default=500,
        metavar="N",
        help="Number of surface sample points per primitive when computing "
             "Chamfer L2.  Uses equal arc-length sampling (Liu et al. CVPR "
             "2022).  Lower = faster; higher = more accurate.  Default: 500.",
    )
    parser.add_argument(
        "--dual-segment", action="store_true", default=False,
        help="Use segment_instances_dual() (PointGroup dual-set DBSCAN, "
             "Jiang et al. CVPR 2020) instead of the standard segment_instances(). "
             "Recommended for RGB-D scenes with non-uniform point density; "
             "disabled by default for ShapeNet (single-object clouds).",
    )
    parser.add_argument(
        "--rgbd-eps-mult", type=float, default=3.0,
        metavar="M",
        help="Adaptive-eps multiplier passed to adaptive_cluster_eps() for "
             "RGB-D scenes.  Default: 3.0.  Increase to merge more fragments "
             "(fewer, larger clusters); decrease to split tightly-spaced objects.",
    )
    parser.add_argument(
        "--rgbd-eps-max", type=float, default=0.08,
        metavar="M",
        help="Hard upper cap (metres) on the adaptive eps for RGB-D scenes.  "
             "Default: 0.08 m.  Prevents eps from growing so large that all "
             "objects merge into a single cluster.",
    )
    parser.add_argument(
        "--rgbd-merge-dist", type=float, default=0.15,
        metavar="M",
        help="Centroid distance threshold (metres) for post-DBSCAN segment "
             "merging.  Segments whose centroids are within this distance are "
             "merged into one.  Default: 0.15 m.",
    )
    parser.add_argument(
        "--no-segment-merge", action="store_true", default=False,
        help="Disable post-DBSCAN centroid-based segment merging "
             "(merge_nearby_segments).  Useful for ablation.",
    )
    parser.add_argument(
        "--rgbd-max-seg-extent", type=float, default=0.60,
        metavar="M",
        help="Maximum bounding-box extent (metres) for a segment to be kept "
             "before SuperDec fitting.  Segments larger than this are background "
             "walls/floor patches.  Default: 0.60 m.",
    )
    parser.add_argument(
        "--no-rgbd-fps", action="store_true", default=False,
        help="Re-enable FPS-before-DBSCAN (old behaviour): subsample foreground "
             "cloud to _RGBD_FPS_PTS before DBSCAN.  Default: disabled — the "
             "full foreground cloud is used for DBSCAN and FPS happens per-segment "
             "inside preprocess_pointcloud.",
    )
    args = parser.parse_args()

    if (args.dataset == "rgbd_scenes"
            and not args.no_superdec
            and args.checkpoint is None):
        parser.error(
            "--checkpoint is required when using SuperDec (--no-superdec not set). "
            "Pass --checkpoint /path/to/tabletop_best.pth or use --no-superdec "
            "for the LM baseline."
        )

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Dispatch to dataset-specific main loops ───────────────────────────────
    if args.dataset == "rgbd_scenes":
        _main_rgbd(args, out_dir)
        return

    # ── 1. Describe the test split ────────────────────────────────────────────
    print_split_summary(args.data_root)

    # ── 2. Load samples via ShapeNet dataloader ───────────────────────────────
    ds = build_dataloader(args.data_root)
    print(f"  Dataset has {len(ds)} test samples total.\n")

    # Build per-category index list from the flat ds.models list
    cat_indices: dict = {}
    for i, m in enumerate(ds.models):
        cat_indices.setdefault(m["category"], []).append(i)

    # Select which dataset indices to process
    if args.categories.strip():
        # Sample --n-samples from EACH specified category
        selected_indices = []
        for name in args.categories.split(","):
            name = name.strip()
            cat_id = _name_to_id.get(name, name)   # try name→ID; fall back to ID
            idxs   = cat_indices.get(cat_id, [])
            if not idxs:
                warnings.warn(f"Category '{name}' (id='{cat_id}') has no test samples; skipping")
                continue
            selected_indices.extend(idxs[:args.n_samples])
        print(f"Processing {len(selected_indices)} samples "
              f"({args.n_samples} each from: {args.categories})\n")
    else:
        selected_indices = list(range(min(args.n_samples, len(ds))))
        print(f"Processing first {len(selected_indices)} samples.\n")

    # ── 3. Initialise fitter ──────────────────────────────────────────────────
    if args.no_superdec:
        fitter = SuperquadricFitter(n_restarts=3, n_lm_rounds=15, subsample=512)
        fitter_name = "LM"
        print("Fitter: SuperquadricFitter (LM) — CPU-only mode (--no-superdec)\n")
    else:
        fitter = SuperdecFitter(
            superdec_dir=args.superdec_dir,
            checkpoint_dir=args.checkpoint_dir,
        )
        fitter_name = "SuperDec"
        print()

    # ── 4. Process each sample ────────────────────────────────────────────────
    _N_QUERIES = 16   # SuperDec max primitives per object (train_tabletop.yaml)
    hdr = (f"{'#':<4}  {'sample':<40}  {'category':<8}  "
           f"{'segs':>4}  {'prims':>5}  {'parsimony':>9}  "
           f"{'pre-L2':>8}  {'orig-L2':>8}  {'conf-L2':>8}  "
           f"{'fitter':<9}  {'note'}")
    sep = "-" * len(hdr)
    print(hdr)
    print(sep)

    for out_idx, ds_idx in enumerate(selected_indices):
        item       = ds[ds_idx]
        model_info = ds.models[ds_idx]
        cat        = model_info["category"]
        mid        = model_info["model_id"]
        cat_name   = CATEGORY_NAMES.get(cat, cat)
        sample_label = f"{cat_name}/{mid[:20]}"

        # Points (and optional normals) from dataloader: torch.Tensor → numpy
        pts_raw     = item["points"].numpy().astype(np.float64)
        normals_raw = item.get("normals")
        if normals_raw is not None:
            normals_raw = normals_raw.numpy().astype(np.float64)

        # Optional preprocessing stage
        if not args.no_preprocess:
            try:
                pts_pre, normals_pre, meta = preprocess_pointcloud(
                    pts_raw, normals=normals_raw,
                    # SuperDec normalises internally via normalize_points();
                    # skip external scale normalisation to avoid double scaling.
                    for_superdec=not args.no_superdec,
                )
            except Exception as exc:
                warnings.warn(f"preprocess_pointcloud failed ({exc!r}); using raw cloud")
                pts_pre, normals_pre, meta = pts_raw, normals_raw, None
        else:
            pts_pre, normals_pre, meta = pts_raw, normals_raw, None

        # Optional per-category canonical rotation (Improvement 1)
        if not args.no_category_rotation and meta is not None:
            R_cat = SHAPENET_CATEGORY_ROTATIONS.get(cat, None)
            if R_cat is not None and not np.allclose(R_cat, np.eye(3)):
                pts_pre = (R_cat @ pts_pre.T).T
                if normals_pre is not None:
                    normals_pre = (R_cat @ normals_pre.T).T
                # Prepend to meta rotation so postprocess_fits inverts it correctly
                meta = dict(meta)
                meta["rotation"] = R_cat @ meta["rotation"]

        skip_tr = not args.table_removal
        obj_pts, segments, sq_fits, used_raw = run_pipeline(
            pts_pre, fitter, skip_table_removal=skip_tr,
            dual_segment=args.dual_segment,
        )

        # Invert preprocessing transforms so SQ poses are in the original frame
        if meta is not None:
            sq_fits = postprocess_fits(sq_fits, meta)

        # Optional primitive merging (Improvement 3)
        if not args.no_merge:
            sq_fits = merge_overlapping_primitives(sq_fits)

        n_segs  = len(segments)
        n_prims = sum(len(m.primitives) for m in sq_fits)

        # ── Chamfer L2 in preprocessed frame (stored by fitter) ─────────────
        pre_l2_vals = [p.chamfer_l2 for m in sq_fits for p in m.primitives]
        mean_pre_l2 = float(np.mean(pre_l2_vals)) if pre_l2_vals else float("nan")

        # ── Chamfer L2 in original frame (recomputed against pts_raw) ────────
        mean_orig_l2 = _chamfer_l2_cloud_vs_fits(
            pts_raw, sq_fits, n_surface_pts=args.surface_samples)

        # ── Confidence-weighted Chamfer L2 (Paschalidou eq. 6) ──────────────
        mean_conf_l2 = confidence_weighted_chamfer(
            sq_fits, pts_raw, n_points=args.surface_samples)

        # Warn if the two L2 values diverge significantly (indicates a frame mismatch)
        if np.isfinite(mean_pre_l2) and np.isfinite(mean_orig_l2):
            ratio = abs(mean_orig_l2 - mean_pre_l2) / (mean_pre_l2 + 1e-9)
            if ratio > 0.5:
                warnings.warn(
                    f"Sample {out_idx+1}: orig-space L2 ({mean_orig_l2:.4f}) differs "
                    f"from pre-space L2 ({mean_pre_l2:.4f}) by {ratio*100:.0f}% — "
                    "check coordinate frame alignment.",
                    stacklevel=2,
                )

        png_path = out_dir / f"sample_{out_idx + 1}.png"
        save_4panel(pts_raw, pts_pre if pts_pre is not pts_raw else pts_raw,
                    segments, sq_fits, sample_label, png_path)

        notes = []
        if not used_raw:
            notes.append("table-removed")
        if args.no_preprocess:
            notes.append("no-preprocess")
        note = " ".join(notes)

        parsimony    = n_prims / _N_QUERIES
        pre_str  = f"{mean_pre_l2:.4f}"  if np.isfinite(mean_pre_l2)  else "   n/a"
        orig_str = f"{mean_orig_l2:.4f}" if np.isfinite(mean_orig_l2) else "   n/a"
        conf_str = f"{mean_conf_l2:.4f}" if np.isfinite(mean_conf_l2) else "   n/a"
        pars_str = f"{parsimony:.4f}"
        row = (f"{out_idx+1:<4}  {sample_label:<40}  {cat_name:<8}  "
               f"{n_segs:>4}  {n_prims:>5}  {pars_str:>9}  "
               f"{pre_str:>8}  {orig_str:>8}  {conf_str:>8}  "
               f"{fitter_name:<9}  {note}")
        print(row)

    print(sep)
    print(f"\nOutputs saved to: {out_dir}/")
    print(f"  sample_1.png … sample_{len(selected_indices)}.png")


if __name__ == "__main__":
    main()
