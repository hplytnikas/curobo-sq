#!/usr/bin/env python3
"""
evaluate_superdec.py
====================
Evaluate SuperDec v3 on RGB-D Scenes v2 and ShapeNet.

Full pipeline per sample:
  load → table removal → segmentation → preprocess_pointcloud
  → fit_adaptive → postprocess_fits → Chamfer L2 → figures

Usage
-----
  # GPU — SuperDec v3:
  omni_python scripts/evaluate_superdec.py \
      --checkpoint /work/courses/3dv/team15/checkpoints/superdec_tabletop/\
superdec_tabletop_finetune_v3/epoch_300.pt \
      --scenes 1,2,3,4,5 --categories bowl,bottle,mug --n-shapenet 3

  # CPU fallback (LM fitter, no checkpoint):
  python3 scripts/evaluate_superdec.py \
      --no-superdec --scenes 5,1 --categories bowl,bottle --n-shapenet 2

Synset map:
  bottle=02876657  bowl=02880940  mug=03797390
  knife=03624134   laptop=03642806
"""

# matplotlib backend MUST be set before any pyplot import
import matplotlib
matplotlib.use("Agg")

import argparse
import csv
import sys
import time
import warnings
from collections import defaultdict
from pathlib import Path
from typing import List, Optional

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

# ── path setup ────────────────────────────────────────────────────────────────
_HERE         = Path(__file__).resolve().parent
_ROOT         = _HERE.parent
_SUPERDEC_DIR = Path("/work/courses/3dv/team15/superdec")

sys.path.insert(0, str(_ROOT / "src"))
sys.path.insert(0, str(_SUPERDEC_DIR))

# ── perception imports (all signatures verified via inspect) ──────────────────
from project_3dv.perception.datasets.rgbd_scenes import RGBDScenesV2
from project_3dv.perception.superquadric import (
    SuperquadricFit,
    SuperquadricFitter,
    MultiSQFit,
    sample_surface_equal_distance,       # (sq, n_points=500, n_dense=1000) -> (N,3)
)
from project_3dv.perception.pipeline import (
    remove_table,           # -> (obj_pts, table_normal, table_height, table_pts, n_table_pts)
    segment_instances,      # -> List[np.ndarray]
    merge_nearby_segments,  # -> List[np.ndarray]
    preprocess_pointcloud,  # -> (pts_pre, normals_pre, meta)
    postprocess_fits,       # -> list[MultiSQFit]
)
from project_3dv.perception.superdec_fitter import _chamfer_l2_from_surface

# ── constants ─────────────────────────────────────────────────────────────────
SYNSET_MAP = {
    "bottle": "02876657",
    "bowl":   "02880940",
    "knife":  "03624134",
    "laptop": "03642806",
    "mug":    "03797390",
    "gso":    "gso",
}

PALETTE = [
    "#e63946", "#457b9d", "#2a9d8f", "#e76f51",
    "#8338ec", "#f4a261", "#06d6a0", "#ffd166",
    "#118ab2", "#ef476f", "#06a77d", "#d62246",
]

_TAB20 = cm.get_cmap("tab20")


def _colour(i: int) -> str:
    return PALETTE[i % len(PALETTE)]


# ── wireframe helper ──────────────────────────────────────────────────────────

def _draw_sq_wireframe(ax, prim, color, alpha: float = 0.45, lw: float = 0.8):
    u = np.linspace(-np.pi / 2, np.pi / 2, 20)
    v = np.linspace(-np.pi, np.pi, 40)
    _se = lambda x, e: np.sign(x) * (np.abs(x) + 1e-8) ** e  # noqa: E731
    Xb = prim.sx * _se(np.cos(u[:, None]), prim.e1) * _se(np.cos(v[None, :]), prim.e2)
    Yb = prim.sy * _se(np.cos(u[:, None]), prim.e1) * _se(np.sin(v[None, :]), prim.e2)
    Zb = prim.sz * _se(np.sin(u[:, None]), prim.e1) * np.ones_like(v[None, :])
    if hasattr(prim, "rotation_matrix"):
        R = np.array(prim.rotation_matrix, dtype=np.float64)
    else:
        from scipy.spatial.transform import Rotation
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


def _compute_chamfer(pts_pre: np.ndarray, result: MultiSQFit) -> float:
    """Bidirectional Chamfer L2 between pts_pre and union of all SQ surfaces."""
    if not result or not result.primitives:
        return float("nan")
    surf_parts = []
    for p in result.primitives:
        try:
            s = sample_surface_equal_distance(p, n_points=200)
            if s is not None and len(s) > 0:
                surf_parts.append(s.astype(np.float32))
        except Exception:
            pass
    if not surf_parts:
        return float("nan")
    surf  = np.concatenate(surf_parts, axis=0)
    pts32 = pts_pre.astype(np.float32)
    d_ps  = ((pts32[:, None] - surf[None]) ** 2).sum(-1).min(-1)
    d_sp  = ((surf[:, None] - pts32[None]) ** 2).sum(-1).min(-1)
    return float(d_ps.mean() + d_sp.mean())


def _compute_chamfer_normalized(pts_pre: np.ndarray, result: MultiSQFit) -> float:
    """Bidirectional Chamfer L2 in unit-sphere space — comparable across datasets.

    Both the input cloud and the SQ surface samples are normalized to the same
    unit sphere before computing distances.  This removes the world-scale factor
    that makes RGB-D L2 values (cm-scale objects) incomparable to ShapeNet L2
    values (unit-sphere objects).

    The normalisation is:
        pts_norm = (pts - pts.mean(0)) / max_norm,  max_norm = max(||pts - mean||)
    The same centre and scale are applied to SQ surface samples, so the relative
    fit quality is preserved.
    """
    if not result or not result.primitives:
        return float("nan")
    center = pts_pre.mean(0).astype(np.float64)
    scale  = float(np.linalg.norm(pts_pre - center, axis=1).max()) + 1e-8
    pts_n  = ((pts_pre - center) / scale).astype(np.float32)
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


# ── RGB-D scene evaluation ────────────────────────────────────────────────────

def _remove_outliers(pts: np.ndarray,
                     nb_neighbors: int = 20,
                     std_ratio: float = 2.0,
                     scene_id: int = 0) -> np.ndarray:
    """Remove points whose mean neighbor distance exceeds std_ratio * global std."""
    from sklearn.neighbors import NearestNeighbors
    if len(pts) < nb_neighbors + 1:
        return pts
    nbrs = NearestNeighbors(n_neighbors=nb_neighbors + 1).fit(pts)
    dists, _ = nbrs.kneighbors(pts)
    mean_dists = dists[:, 1:].mean(axis=1)   # exclude self
    threshold = mean_dists.mean() + std_ratio * mean_dists.std()
    mask = mean_dists < threshold
    n_removed = int((~mask).sum())
    if n_removed > 0:
        print(f"  [scene {scene_id:02d}] outlier removal: {n_removed} pts removed "
              f"({100.0 * n_removed / len(pts):.1f}% of foreground)")
    return pts[mask]


def evaluate_rgbd_scene(scene_id: int, data_rgbd: str, fitter,
                        use_superdec: bool, exist_thresh: float,
                        out_dir: Path,
                        single_object_mode: bool = False,
                        verbose: bool = False) -> dict:
    t0  = time.perf_counter()
    rng = np.random.default_rng(42)

    # 1. Load
    ds  = RGBDScenesV2(data_rgbd, scene_id=scene_id)
    pts, rgb, labels = ds.load()
    n_gt = int(np.unique(labels[labels > 0]).size)

    # 2. Table removal
    try:
        obj_pts, table_normal, table_height, _table_pts, _n_tbl = remove_table(
            pts, depth_margin=np.inf, xy_radius=np.inf
        )
        if len(obj_pts) < 200:
            warnings.warn(f"Scene {scene_id:02d}: only {len(obj_pts)} fg pts, using full cloud")
            obj_pts, table_normal, table_height = pts, np.array([0., 0., 1.]), 0.0
    except Exception as exc:
        warnings.warn(f"Scene {scene_id:02d}: remove_table failed ({exc}), using full cloud")
        obj_pts, table_normal, table_height = pts, np.array([0., 0., 1.]), 0.0

    if verbose:
        n_str = "[" + ", ".join(f"{v:.2f}" for v in table_normal) + "]"
        print(f"  [scene {scene_id:02d}] table normal={n_str} height={table_height:.3f}")

    # 2b. Statistical outlier removal (depth sensor noise that passed height filter)
    obj_pts = _remove_outliers(obj_pts, nb_neighbors=20, std_ratio=2.0,
                               scene_id=scene_id)

    # 3. Segment (or bypass for single-object mode)
    if single_object_mode:
        # Treat entire foreground cloud as one object — directly comparable to ShapeNet
        cap = obj_pts[rng.choice(len(obj_pts), 8192, replace=False)] \
              if len(obj_pts) > 8192 else obj_pts
        segs = [cap]
    else:
        segs_all = segment_instances(
            obj_pts, adaptive_eps=True, eps_multiplier=3.0, eps_max=0.08,
            cluster_min_points=100,
        )
        segs_all = merge_nearby_segments(segs_all, merge_dist=0.15)
        segs = [s for s in segs_all if len(s) >= 100 and (s.max(0) - s.min(0)).max() <= 0.70]
        if verbose:
            rejected = [s for s in segs_all
                        if not (len(s) >= 100 and (s.max(0) - s.min(0)).max() <= 0.70)]
            rej_dims = sorted([(s.max(0) - s.min(0)).max() for s in rejected], reverse=True)
            dim_str = (", max dims: " + " ".join(f"{d:.2f}m" for d in rej_dims[:7])
                       if rejected else "")
            print(f"  [scene {scene_id:02d}] bbox filter: {len(segs_all)} clusters "
                  f"→ {len(segs)} kept ({len(rejected)} rejected{dim_str})")
        # Drop tiny segments (noise clusters below MIN_SEG_PTS)
        _MIN_SEG_PTS = 400
        segs_tiny = [s for s in segs if len(s) < _MIN_SEG_PTS]
        segs = [s for s in segs if len(s) >= _MIN_SEG_PTS]
        if segs_tiny:
            print(f"  [scene {scene_id:02d}] dropped {len(segs_tiny)} segments "
                  f"with <{_MIN_SEG_PTS} pts (noise: sizes "
                  f"{sorted([len(s) for s in segs_tiny])})")
        if verbose:
            sizes_str = ", ".join(str(len(s)) for s in segs[:12])
            if len(segs) > 12:
                sizes_str += ", ..."
            print(f"  [scene {scene_id:02d}] seg sizes: [{sizes_str}]")
    n_segs = len(segs)

    # 4. Cap pts per segment — raw arrays, no preprocessing (fit_batch normalizes internally)
    orig_seg_sizes: List[int] = [len(s) for s in segs]
    capped_segs: List[np.ndarray] = []
    for seg in segs:
        if len(seg) > 8192:
            seg = seg[rng.choice(len(seg), 8192, replace=False)]
        capped_segs.append(seg)

    # 5. Batch inference — pass raw segments; fit_batch handles normalization internally
    t_fit0 = time.perf_counter()
    if use_superdec and capped_segs and hasattr(fitter, "fit_batch"):
        try:
            batch_results = fitter.fit_batch(capped_segs)
        except Exception as exc:
            warnings.warn(
                f"Scene {scene_id:02d}: fit_batch failed ({exc}), falling back to sequential"
            )
            batch_results = [fitter.fit_adaptive(seg) for seg in capped_segs]
    else:
        batch_results = []
        for seg in capped_segs:
            try:
                batch_results.append(fitter.fit_adaptive(seg))
            except Exception as exc:
                warnings.warn(f"Scene {scene_id:02d}: fit_adaptive failed ({exc})")
                batch_results.append(MultiSQFit(primitives=[], n_points=0))
    t_fit = time.perf_counter() - t_fit0

    # LM step count (fixed at optimizer.num_steps; None when LM is disabled)
    _lm_opt = getattr(fitter, '_lm_optimizer', None)
    lm_steps = _lm_opt.num_steps if _lm_opt is not None else None

    # 6. Per-segment metrics — fit_batch returns world-frame poses (denormalizes internally)
    # norm-L2 computed from raw segment (world frame), no postprocess_fits needed
    seg_fits_post: List[MultiSQFit] = batch_results
    seg_metrics   = []

    for i, (seg, result) in enumerate(zip(capped_segs, batch_results)):
        raw_l2   = _compute_chamfer(seg, result)
        norm_l2  = _compute_chamfer_normalized(seg, result)
        n_p      = len(result.primitives) if result else 0
        is_bad   = not np.isnan(norm_l2) and norm_l2 > 0.06
        seg_metrics.append({
            "raw_l2": raw_l2, "norm_l2": norm_l2, "n_prims": n_p,
            "n_pts": orig_seg_sizes[i],
        })
        if verbose:
            orig_n  = orig_seg_sizes[i]
            cap_n   = len(seg)
            cap_str = f" → {cap_n} sampled" if cap_n < orig_n else ""
            lm_str  = f" | LM iters={lm_steps}" if lm_steps is not None else ""
            bad_str = " | ⚠ BAD FIT" if is_bad else ""
            print(f"    seg {i:02d} [{orig_n} pts{cap_str}]: {n_p} prims "
                  f"| norm-L2={norm_l2:.3f}{bad_str} | raw-L2={raw_l2:.4f}{lm_str}")
        else:
            flag = "  ⚠ bad fit" if is_bad else ""
            print(f"    seg {i:02d}: {n_p} prims "
                  f"| raw-L2={raw_l2:.4f} | norm-L2={norm_l2:.4f}{flag}")

    # 7. Aggregate
    raw_l2s  = [m["raw_l2"]  for m in seg_metrics if not np.isnan(m.get("raw_l2", float("nan")))]
    norm_l2s = [m["norm_l2"] for m in seg_metrics if not np.isnan(m.get("norm_l2", float("nan")))]
    mean_raw  = float(np.nanmean(raw_l2s))  if raw_l2s  else float("nan")
    mean_norm = float(np.nanmean(norm_l2s)) if norm_l2s else float("nan")
    n_prims   = sum(m["n_prims"] for m in seg_metrics)
    n_bad     = sum(1 for m in seg_metrics
                    if not np.isnan(m["norm_l2"]) and m["norm_l2"] > 0.06)
    elapsed   = time.perf_counter() - t0

    if verbose:
        print(f"  [scene {scene_id:02d}] done: {n_segs} segs → {n_prims} prims "
              f"| mean norm-L2={mean_norm:.3f} | {n_bad} bad fits "
              f"| batch={t_fit:.1f}s | total={elapsed:.1f}s")
    else:
        print(f"  [scene {scene_id:02d}] {len(obj_pts):,} foreground pts "
              f"→ {n_segs} segs → {n_prims} prims "
              f"| raw-L2={mean_raw:.4f} | norm-L2={mean_norm:.4f} "
              f"| fit {t_fit:.1f}s | total {elapsed:.1f}s")

    # Collect bad fits for the end-of-run summary
    bad_fits = [
        {"scene": scene_id, "seg": i, "norm_l2": m["norm_l2"],
         "n_prims": m["n_prims"], "n_pts": m["n_pts"]}
        for i, m in enumerate(seg_metrics)
        if not np.isnan(m["norm_l2"]) and m["norm_l2"] > 0.06
    ]

    # 8. Figure
    _make_rgbd_figure(
        pts_raw=pts, obj_pts=obj_pts, segs=segs,
        seg_fits_post=seg_fits_post,
        scene_id=scene_id, n_prims=n_prims,
        mean_norm_l2=mean_norm, exist_thresh=exist_thresh,
        out_path=out_dir / f"rgbd_{scene_id:02d}.png",
    )

    return {
        "split": "rgbd", "name": f"scene_{scene_id:02d}",
        "gt_objs": n_gt, "n_segs": n_segs, "n_prims": n_prims,
        "mean_raw_l2": mean_raw, "mean_norm_l2": mean_norm, "time_s": elapsed,
        "bad_fits": bad_fits,
    }


def _make_rgbd_figure(pts_raw, obj_pts, segs, seg_fits_post,
                      scene_id, n_prims, mean_norm_l2, exist_thresh, out_path):
    n_segs  = len(segs)
    l2_str  = f"{mean_norm_l2:.4f}" if not np.isnan(mean_norm_l2) else "N/A"

    fig = plt.figure(figsize=(22, 5), dpi=130)
    fig.suptitle(
        f"Scene {scene_id:02d}  |  SuperDec v3  |  "
        f"{n_segs} segs  {n_prims} prims  |  norm-L2={l2_str}",
        fontsize=10,
    )

    # Panel 1 limits: full raw cloud (room-scale context)
    raw_ref = pts_raw[::max(1, len(pts_raw) // 20000)]
    raw_ctr = raw_ref.mean(0)
    raw_hr  = max((raw_ref.max(0) - raw_ref.min(0)).max() / 2 * 1.1, 0.05)

    def _lim_raw(ax):
        ax.set_xlim(raw_ctr[0] - raw_hr, raw_ctr[0] + raw_hr)
        ax.set_ylim(raw_ctr[1] - raw_hr, raw_ctr[1] + raw_hr)
        ax.set_zlim(raw_ctr[2] - raw_hr, raw_ctr[2] + raw_hr)
        ax.set_box_aspect([1, 1, 1])
        ax.set_xlabel("X", fontsize=7); ax.set_ylabel("Y", fontsize=7)
        ax.set_zlabel("Z", fontsize=7)
        ax.set_xticklabels([]); ax.set_yticklabels([]); ax.set_zticklabels([])

    # Panels 2/3/4 limits: foreground object bbox with padding
    if segs:
        fg_pts   = np.concatenate(segs, axis=0)
        fg_min, fg_max = fg_pts.min(0), fg_pts.max(0)
        fg_center = (fg_min + fg_max) / 2
        fg_range  = max((fg_max - fg_min).max() * 0.6, 0.05)  # 20% pad each side
    else:
        fg_center = raw_ctr
        fg_range  = raw_hr

    def _lim_fg(ax):
        ax.set_xlim(fg_center[0] - fg_range, fg_center[0] + fg_range)
        ax.set_ylim(fg_center[1] - fg_range, fg_center[1] + fg_range)
        ax.set_zlim(fg_center[2] - fg_range, fg_center[2] + fg_range)
        ax.set_box_aspect([1, 1, 1])
        ax.set_xlabel("X", fontsize=7); ax.set_ylabel("Y", fontsize=7)
        ax.set_zlabel("Z", fontsize=7)
        ax.set_xticklabels([]); ax.set_yticklabels([]); ax.set_zticklabels([])

    # Panel 1: raw cloud (room-scale)
    ax1 = fig.add_subplot(1, 4, 1, projection="3d")
    idx = np.random.choice(len(pts_raw), min(10000, len(pts_raw)), replace=False)
    p   = pts_raw[idx]
    ax1.scatter(p[:, 0], p[:, 1], p[:, 2],
                c=p[:, 2], cmap="viridis", s=0.4, alpha=0.7, rasterized=True)
    ax1.set_title(f"Scene {scene_id:02d} · Raw ({len(pts_raw):,} pts)", fontsize=9)
    _lim_raw(ax1)

    # Panel 2: segmentation (foreground zoom)
    ax2 = fig.add_subplot(1, 4, 2, projection="3d")
    bg  = obj_pts[np.random.choice(len(obj_pts), min(3000, len(obj_pts)), replace=False)]
    ax2.scatter(bg[:, 0], bg[:, 1], bg[:, 2],
                c="#888888", s=0.2, alpha=0.3, rasterized=True)
    for i, seg in enumerate(segs):
        idx = np.random.choice(len(seg), min(500, len(seg)), replace=False)
        q   = seg[idx]
        c   = _TAB20(i / max(n_segs, 1))
        ax2.scatter(q[:, 0], q[:, 1], q[:, 2],
                    c=[c] * len(idx), s=1.0, alpha=0.8, rasterized=True)
    ax2.set_title(f"Segmentation · {n_segs} segments", fontsize=9)
    _lim_fg(ax2)

    # Panel 3: SQ wireframes (foreground zoom)
    ax3 = fig.add_subplot(1, 4, 3, projection="3d")
    for i, fit in enumerate(seg_fits_post):
        if not fit or not fit.primitives:
            continue
        c = _TAB20(i / max(len(seg_fits_post), 1))
        for prim in fit.primitives:
            try:
                _draw_sq_wireframe(ax3, prim, color=c)
            except Exception:
                pass
    ax3.set_title(f"SuperDec v3 · {n_prims} prims · norm-L2={l2_str}", fontsize=9)
    _lim_fg(ax3)

    # Panel 4: overlay (foreground zoom)
    ax4 = fig.add_subplot(1, 4, 4, projection="3d")
    fg  = obj_pts[np.random.choice(len(obj_pts), min(5000, len(obj_pts)), replace=False)]
    ax4.scatter(fg[:, 0], fg[:, 1], fg[:, 2],
                c="#888888", s=0.3, alpha=0.15, rasterized=True)
    for i, fit in enumerate(seg_fits_post):
        if not fit or not fit.primitives:
            continue
        c = _TAB20(i / max(len(seg_fits_post), 1))
        for prim in fit.primitives:
            try:
                _draw_sq_wireframe(ax4, prim, color=c)
            except Exception:
                pass
    ax4.set_title(f"Overlay · exist_thresh={exist_thresh}", fontsize=9)
    _lim_fg(ax4)

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(out_path), dpi=130, bbox_inches="tight")
    plt.close("all")


# ── ShapeNet sample evaluation ────────────────────────────────────────────────

def evaluate_shapenet_sample(cat: str, idx: int, path: Path,
                              fitter, use_superdec: bool,
                              out_dir: Path,
                              verbose: bool = False) -> dict:
    t0 = time.perf_counter()

    # 1. Load + subsample
    data    = np.load(str(path))
    pts_raw = data["points"].astype(np.float32)          # key confirmed: 'points'
    if len(pts_raw) > 4096:
        rng = np.random.default_rng(42)
        pts_raw = pts_raw[rng.choice(len(pts_raw), 4096, replace=False)]

    # 2. Normalize to unit sphere
    pts = pts_raw - pts_raw.mean(axis=0)
    pts = pts / (pts.std() + 1e-8)

    # 3. preprocess_pointcloud
    try:
        pts_pre, _normals_pre, meta = preprocess_pointcloud(
            pts, target_n=4096, for_superdec=use_superdec,
        )
    except Exception as exc:
        warnings.warn(f"{cat}[{idx:04d}]: preprocess failed ({exc}), using pts directly")
        pts_pre = pts.astype(np.float64)
        meta    = {"rotation": np.eye(3), "scale": 1.0, "centroid": np.zeros(3)}

    # 4. Fit
    try:
        result = fitter.fit_adaptive(pts_pre)
    except Exception as exc:
        warnings.warn(f"{cat}[{idx:04d}]: fit failed ({exc})")
        result = MultiSQFit(primitives=[], n_points=0)

    # 5. Chamfer — ShapeNet input is already unit-sphere, so raw == norm here
    raw_l2  = _compute_chamfer(pts_pre, result)
    norm_l2 = _compute_chamfer_normalized(pts_pre, result)

    # 6. Postprocess to recover the original (normalized-sphere) frame
    try:
        post = postprocess_fits([result], meta)
    except Exception:
        post = [result]

    n_prims = sum(len(f.primitives) for f in post if f)
    elapsed = time.perf_counter() - t0

    l2_str = f"{norm_l2:.4f}" if not np.isnan(norm_l2) else "N/A"
    if verbose:
        pre_prims = getattr(result, 'primitives_before_lm', None)
        if pre_prims:
            # compute pre-LM in normalized space: same center/scale as norm_l2
            pre_lm_n = _compute_chamfer_normalized(
                pts_pre, MultiSQFit(primitives=pre_prims, n_points=0))
            lm_str = (f" | pre-LM={pre_lm_n:.4f} → post-LM={norm_l2:.4f}"
                      f" (Δ={pre_lm_n - norm_l2:+.4f})")
        else:
            lm_str = " | pre-LM=n/a"
        print(f"  [{cat} {idx:04d}] {len(pts_pre)} pts | {n_prims} prims "
              f"| norm-L2={l2_str}{lm_str} | {elapsed:.2f}s")
    else:
        print(f"  [{cat} {idx:04d}] {len(pts_pre)} pts → {n_prims} prims "
              f"| norm-L2={l2_str} | {elapsed:.2f}s")

    # 7. Figure — use pts_pre + result (preprocessed frame) for alignment
    _make_shapenet_figure(
        pts_pre=pts_pre.astype(np.float32), result=result,
        cat=cat, idx=idx, chamfer=norm_l2, n_prims=n_prims,
        out_path=out_dir / f"shapenet_{cat}_{idx:04d}.png",
    )

    return {
        "split": "shapenet", "name": f"{cat}_{idx:04d}", "cat": cat,
        "n_segs": 1, "n_prims": n_prims,
        "mean_raw_l2": raw_l2, "mean_norm_l2": norm_l2, "time_s": elapsed,
    }


def _make_shapenet_figure(pts_pre, result, cat, idx, chamfer, n_prims, out_path):
    l2_str = f"{chamfer:.4f}" if not np.isnan(chamfer) else "N/A"
    prims  = result.primitives if result else []

    fig = plt.figure(figsize=(20, 5), dpi=130)
    fig.suptitle(
        f"{cat} [{idx:04d}]  |  SuperDec v3  |  "
        f"{n_prims} prims  |  pre-L2={l2_str}",
        fontsize=10,
    )

    ref = pts_pre if len(pts_pre) > 0 else np.zeros((1, 3))
    ctr = ref.mean(0)
    hr  = max((ref.max(0) - ref.min(0)).max() / 2 * 1.1, 0.01)

    def _lim(ax):
        ax.set_xlim(ctr[0] - hr, ctr[0] + hr)
        ax.set_ylim(ctr[1] - hr, ctr[1] + hr)
        ax.set_zlim(ctr[2] - hr, ctr[2] + hr)
        ax.set_box_aspect([1, 1, 1])
        ax.set_xlabel("X", fontsize=7); ax.set_ylabel("Y", fontsize=7)
        ax.set_zlabel("Z", fontsize=7)
        ax.set_xticklabels([]); ax.set_yticklabels([]); ax.set_zticklabels([])

    colours = [PALETTE[i % len(PALETTE)] for i in range(16)]

    # Panel 1: input cloud
    ax1 = fig.add_subplot(1, 4, 1, projection="3d")
    if len(pts_pre) > 0:
        ax1.scatter(pts_pre[:, 0], pts_pre[:, 1], pts_pre[:, 2],
                    c=pts_pre[:, 2], cmap="viridis", s=1.0, alpha=0.6, rasterized=True)
    ax1.set_title(f"{cat} · {len(pts_pre)} pts", fontsize=9)
    _lim(ax1)

    # Panel 2: SQ wireframes only
    ax2 = fig.add_subplot(1, 4, 2, projection="3d")
    for pi, prim in enumerate(prims):
        try:
            _draw_sq_wireframe(ax2, prim, color=colours[pi % len(colours)])
        except Exception:
            pass
    ax2.set_title(f"SuperDec v3 · {n_prims} prims · pre-L2={l2_str}", fontsize=9)
    _lim(ax2)

    # Panel 3: SQ surface samples
    ax3      = fig.add_subplot(1, 4, 3, projection="3d")
    tot_surf = 0
    for pi, prim in enumerate(prims):
        try:
            s = sample_surface_equal_distance(prim, n_points=300)
            if s is not None and len(s) > 0:
                c = colours[pi % len(colours)]
                ax3.scatter(s[:, 0], s[:, 1], s[:, 2],
                            c=[c] * len(s), s=0.8, alpha=0.5, rasterized=True)
                tot_surf += len(s)
        except Exception:
            pass
    ax3.set_title(f"Surface samples · {tot_surf} pts", fontsize=9)
    _lim(ax3)

    # Panel 4: overlay
    ax4 = fig.add_subplot(1, 4, 4, projection="3d")
    if len(pts_pre) > 0:
        ax4.scatter(pts_pre[:, 0], pts_pre[:, 1], pts_pre[:, 2],
                    c="#888888", s=0.3, alpha=0.2, rasterized=True)
    for pi, prim in enumerate(prims):
        try:
            _draw_sq_wireframe(ax4, prim, color=colours[pi % len(colours)])
        except Exception:
            pass
    ax4.set_title("Overlay", fontsize=9)
    _lim(ax4)

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(out_path), dpi=130, bbox_inches="tight")
    plt.close("all")


# ── Summary tables ────────────────────────────────────────────────────────────

def _print_rgbd_table(rows: list, exist_thresh: float):
    if not rows:
        return
    print()
    print(f"┌──────────────────────────────────────────────────────────────────────────────────┐")
    print(f"│  RGB-D Scenes v2 — SuperDec v3 (exist_thresh={exist_thresh:.2f})                          │")
    print(f"├───────┬─────────┬───────┬───────┬────────────┬────────────┬─────────────────┤")
    print(f"│ scene │ gt_objs │  segs │ prims │  raw-L2    │  norm-L2   │   time (s)      │")
    print(f"├───────┼─────────┼───────┼───────┼────────────┼────────────┼─────────────────┤")
    for r in rows:
        sid   = r["name"].replace("scene_", "")
        raw   = f"{r['mean_raw_l2']:.4f}"  if not np.isnan(r["mean_raw_l2"])  else "  N/A "
        norm  = f"{r['mean_norm_l2']:.4f}" if not np.isnan(r["mean_norm_l2"]) else "  N/A "
        print(f"│  {sid}  │  {r['gt_objs']:>5}  │ {r['n_segs']:>5} │ {r['n_prims']:>5} "
              f"│  {raw:>8}  │  {norm:>8}  │  {r['time_s']:>12.2f}   │")
    raw_vals  = [r["mean_raw_l2"]  for r in rows if not np.isnan(r["mean_raw_l2"])]
    norm_vals = [r["mean_norm_l2"] for r in rows if not np.isnan(r["mean_norm_l2"])]
    mean_gt   = np.mean([r["gt_objs"] for r in rows])
    mean_segs = np.mean([r["n_segs"]  for r in rows])
    mean_prim = np.mean([r["n_prims"] for r in rows])
    mean_raw  = np.mean(raw_vals)  if raw_vals  else float("nan")
    mean_norm = np.mean(norm_vals) if norm_vals else float("nan")
    mean_t    = np.mean([r["time_s"] for r in rows])
    print(f"├───────┼─────────┼───────┼───────┼────────────┼────────────┼─────────────────┤")
    print(f"│  Mean │  {mean_gt:>5.1f}  │ {mean_segs:>5.1f} │ {mean_prim:>5.1f} "
          f"│  {mean_raw:>8.4f}  │  {mean_norm:>8.4f}  │  {mean_t:>12.2f}   │")
    print(f"└───────┴─────────┴───────┴───────┴────────────┴────────────┴─────────────────┘")


def _print_shapenet_table(rows: list):
    if not rows:
        return
    buckets: dict = defaultdict(list)
    for r in rows:
        buckets[r["cat"]].append(r)
    print()
    print(f"┌──────────────────────────────────────────────────────────────┐")
    print(f"│  ShapeNet — SuperDec v3                                      │")
    print(f"├──────────┬────┬────────────┬────────────┬───────────────────┤")
    print(f"│ category │ n  │ mean_prims │  mean_L2   │  mean_time (s)    │")
    print(f"├──────────┼────┼────────────┼────────────┼───────────────────┤")
    for cat in sorted(buckets):
        rlist = buckets[cat]
        n   = len(rlist)
        mp  = np.mean([r["n_prims"] for r in rlist])
        l2v = [r["mean_norm_l2"] for r in rlist if not np.isnan(r.get("mean_norm_l2", float("nan")))]
        ml  = np.mean(l2v) if l2v else float("nan")
        mt  = np.mean([r["time_s"] for r in rlist])
        print(f"│ {cat:8s} │ {n:>2d} │    {mp:>6.1f}  │   {ml:>8.4f} │     {mt:>12.2f}  │")
    print(f"└──────────┴────┴────────────┴────────────┴───────────────────┘")


def _save_csv(rows: list, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(str(path), "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["split", "name", "n_segs", "n_prims",
                        "mean_raw_l2", "mean_norm_l2", "time_s"],
        )
        writer.writeheader()
        for r in rows:
            def _fmt(v):
                return f"{v:.6f}" if not np.isnan(v) else "nan"
            writer.writerow({
                "split": r["split"], "name": r["name"],
                "n_segs": r["n_segs"], "n_prims": r["n_prims"],
                "mean_raw_l2":  _fmt(r.get("mean_raw_l2",  float("nan"))),
                "mean_norm_l2": _fmt(r.get("mean_norm_l2", float("nan"))),
                "time_s": f"{r['time_s']:.3f}",
            })
    print(f"\n  Results saved → {path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    _default_ckpt = (
        "/work/courses/3dv/team15/checkpoints/superdec_tabletop"
        "/superdec_tabletop_finetune_v3/epoch_300.pt"
    )
    p = argparse.ArgumentParser(
        description="Evaluate SuperDec v3 on RGB-D Scenes v2 and ShapeNet",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--checkpoint",      default=_default_ckpt)
    p.add_argument("--data-rgbd",       default="data/rgbd-scenes-v2/pc")
    p.add_argument("--data-shapenet",
                   default="/work/courses/3dv/team15/superdec/data/ShapeNet")
    p.add_argument("--scenes",          default="1,2,3,4,5")
    p.add_argument("--categories",      default="bowl,bottle,mug")
    p.add_argument("--n-shapenet",      type=int, default=3)
    p.add_argument("--output",          default="outputs/evaluate_superdec/")
    p.add_argument("--exist-threshold", type=float, default=0.3)
    p.add_argument("--no-rgbd",         action="store_true")
    p.add_argument("--no-shapenet",     action="store_true")
    p.add_argument("--no-superdec",          action="store_true",
                   help="Use LM SuperquadricFitter (CPU, no checkpoint needed)")
    p.add_argument("--single-object-mode",  action="store_true",
                   help="Skip segmentation; treat entire foreground as one object "
                        "(directly comparable to ShapeNet single-object evaluation)")
    p.add_argument("--verbose", "-v",       action="store_true",
                   help="Print table normal, bbox filter stats, seg sizes, LM iter counts, "
                        "and a bad-fit summary at the end")
    args = p.parse_args()

    out_dir      = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    use_superdec = not args.no_superdec
    scene_ids    = [int(s.strip()) for s in args.scenes.split(",") if s.strip()]
    categories   = [c.strip() for c in args.categories.split(",") if c.strip()]
    verbose      = args.verbose

    # ── Build fitter ──────────────────────────────────────────────────────────
    if use_superdec:
        from project_3dv.perception.superdec_fitter import SuperdecFitter
        print(f"\nLoading SuperDec fitter from {args.checkpoint} …")
        fitter = SuperdecFitter(
            superdec_dir=str(_SUPERDEC_DIR),
            checkpoint_path=args.checkpoint,
            exist_threshold=args.exist_threshold,
        )
    else:
        print("\nUsing LM SuperquadricFitter (--no-superdec mode) …")
        fitter = SuperquadricFitter(n_restarts=2, n_lm_rounds=10, subsample=256)

    all_rows: list = []

    all_bad_fits: list = []

    # ── RGB-D scenes ──────────────────────────────────────────────────────────
    if not args.no_rgbd:
        print(f"\n=== RGB-D Scenes v2 ({len(scene_ids)} scenes) ===")
        rgbd_rows: list = []
        for sid in scene_ids:
            try:
                row = evaluate_rgbd_scene(
                    sid, args.data_rgbd, fitter, use_superdec,
                    args.exist_threshold, out_dir,
                    single_object_mode=args.single_object_mode,
                    verbose=verbose,
                )
                rgbd_rows.append(row)
                all_rows.append(row)
                all_bad_fits.extend(row.get("bad_fits", []))
            except Exception as exc:
                print(f"  [scene {sid:02d}] ERROR: {exc}")
        _print_rgbd_table(rgbd_rows, args.exist_threshold)

    # ── ShapeNet ──────────────────────────────────────────────────────────────
    if not args.no_shapenet:
        print(f"\n=== ShapeNet ({len(categories)} categories × "
              f"{args.n_shapenet} samples) ===")
        sn_rows: list = []
        for cat in categories:
            synset = SYNSET_MAP.get(cat)
            if synset is None:
                print(f"  Unknown category '{cat}'. Known: {list(SYNSET_MAP)}")
                continue
            syn_dir = Path(args.data_shapenet) / synset
            if not syn_dir.exists():
                print(f"  Directory not found: {syn_dir}")
                continue
            paths = sorted(syn_dir.glob("*/pointcloud.npz"))
            if not paths:
                print(f"  No pointcloud.npz under {syn_dir}")
                continue
            for i in range(min(args.n_shapenet, len(paths))):
                try:
                    row = evaluate_shapenet_sample(
                        cat, i, paths[i], fitter, use_superdec, out_dir,
                        verbose=verbose,
                    )
                    sn_rows.append(row)
                    all_rows.append(row)
                except Exception as exc:
                    print(f"  [{cat} {i:04d}] ERROR: {exc}")
        _print_shapenet_table(sn_rows)

    # ── CSV ───────────────────────────────────────────────────────────────────
    if all_rows:
        _save_csv(all_rows, out_dir / "results.csv")

    # ── Bad-fit summary ───────────────────────────────────────────────────────
    if verbose and all_bad_fits:
        total_segs = sum(r.get("n_segs", 0) for r in all_rows if r.get("split") == "rgbd")
        print(f"\n=== Bad fits (norm-L2 > 0.06) ===")
        for bf in all_bad_fits:
            reason = "likely partial scan / irregular shape"
            print(f"  scene {bf['scene']:02d} seg {bf['seg']:02d}: "
                  f"norm-L2={bf['norm_l2']:.3f}, {bf['n_prims']} prims, "
                  f"{bf['n_pts']} pts — {reason}")
        pct = 100.0 * len(all_bad_fits) / total_segs if total_segs else float("nan")
        print(f"  total: {len(all_bad_fits)} / {total_segs} segments ({pct:.1f}%)")

    print("\nDone.")


if __name__ == "__main__":
    main()
