"""
superdec_fitter.py
==================
Drop-in replacement for SuperquadricFitter that uses the fine-tuned SuperDec
model for SQ decomposition on tabletop objects.

Default checkpoint
------------------
The default is the fine-tuned tabletop v2 model, which should be located at:
    ../checkpoints/superdec_tabletop/superdec_tabletop_finetune_v2/
relative to the superdec_dir.  You can override this via the `checkpoint_dir`
parameter (absolute path or relative to CWD).  The directory must contain:
    config.yaml   — Hydra training config (defines model architecture)
    ckpt.pt       — single checkpoint file, OR
    epoch_NNN.pt  — epoch-based saves (highest epoch is used automatically)

Base checkpoints (not fine-tuned) live at:
    {superdec_dir}/checkpoints/{normalized,shapenet}/ckpt.pt
and can still be used by passing checkpoint_dir explicitly.

Interface contract (compatible with SuperquadricFitter)
-------------------------------------------------------
Input:
    points: (N, 3) float32/float64 numpy array, world coordinates.
    SuperDec internally samples to 4096 points and normalises to unit sphere
    (per superdec/data/dataloader.py::normalize_points).

Output:
    MultiSQFit  — list of up to 16 SuperquadricFit primitives filtered by
                  existence score, with symmetric Chamfer L2 computed via
                  surface sampling.  Feeds directly into:
                    • fits_to_curobo_obstacles()  (collision avoidance)
                    • Scene.get_signed_distance()  (path planning SDF)
                    • GraspSelector.grasp_candidates()  (grasp planning)

Usage:
    from superdec_fitter import SuperdecFitter
    fitter = SuperdecFitter(superdec_dir='/path/to/superdec')
    result = fitter.fit_adaptive(points)   # (N,3) numpy array
"""

import logging
import os
import sys
import numpy as np
import torch
from dataclasses import dataclass, field
from typing import List, Optional
import scipy.spatial.transform as sst

# ---------------------------------------------------------------------------
# Superquadric exponent constraints
# ---------------------------------------------------------------------------
#: Maximum exponent for the convex SQ regime.
#: Paschalidou et al. CVPR 2019, Section 3.3: "we bound the values e1 and e2
#: to the range [0.1, 1.9] so as to prevent non-convex shapes which are less
#: likely to occur in practice."  Values outside (0, 2] produce non-convex
#: shapes and unbounded implicit-function gradients unsuitable for
#: gradient-based planners.  Note: the lower bound could be relaxed to 0.0
#: when using TRF optimisation (reflective bounds prevent the boundary
#: singularity that made LM unstable), but we keep 0.1 to match Paschalidou.
SQ_EXPONENT_CONVEX_MAX: float = 1.9

#: Minimum exponent — see Paschalidou et al. CVPR 2019, Section 3.3.
SQ_EXPONENT_MIN: float = 0.1

_log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# lazy import of superdec — path injected at construction time
# ---------------------------------------------------------------------------
_superdec_loaded = False

def _load_superdec(superdec_dir: str):
    global _superdec_loaded
    if _superdec_loaded:
        return
    if superdec_dir not in sys.path:
        sys.path.insert(0, superdec_dir)
    _superdec_loaded = True


# ---------------------------------------------------------------------------
# reuse SuperquadricFit / MultiSQFit from superquadric.py if available,
# otherwise define minimal stubs matching the canonical interface exactly
# ---------------------------------------------------------------------------
try:
    from superquadric import (SuperquadricFit, MultiSQFit, sq_type_from_exponents,
                               sample_surface_equal_distance)
except ImportError:
    def sample_surface_equal_distance(sq, n_points=500, n_dense=1000):
        """Fallback stub when superquadric.py is unavailable."""
        return sq.surface_points(n_u=50, n_v=50)

    @dataclass
    class SuperquadricFit:
        sx: float = 0.05
        sy: float = 0.05
        sz: float = 0.05
        e1: float = 1.0
        e2: float = 1.0
        tx: float = 0.0
        ty: float = 0.0
        tz: float = 0.0
        rx: float = 0.0
        ry: float = 0.0
        rz: float = 0.0
        shape_type: str = "Other"
        shape_conf: float = 1.0
        converged: bool = True
        chamfer_l2: float = 0.0

        def surface_points(self, n_u: int = 50, n_v: int = 50) -> np.ndarray:
            return np.zeros((0, 3), dtype=np.float32)

    @dataclass
    class MultiSQFit:
        primitives: List[SuperquadricFit] = field(default_factory=list)
        n_points: int = 0

    def sq_type_from_exponents(e1: float, e2: float) -> str:
        if e1 > 0.6 and e2 > 0.6:   return "Ellipsoid"
        if e1 < 0.45 and e2 > 0.6:  return "Cylinder"
        if e1 < 0.45 and e2 < 0.45: return "Cuboid"
        return "Other"


def _rotmat_to_euler_xyz(R: np.ndarray) -> tuple:
    """Convert 3×3 rotation matrix to Euler XYZ angles (radians)."""
    r = sst.Rotation.from_matrix(R)
    rx, ry, rz = r.as_euler('xyz')
    return float(rx), float(ry), float(rz)


def _chamfer_l2_from_surface(pts: np.ndarray, fit: 'SuperquadricFit',
                              n_points: int = 500) -> float:
    """Bi-directional Chamfer L2 between input points and sampled SQ surface.

    Surface points are sampled with equal arc-length spacing (Liu et al.
    CVPR 2022, Supplementary Sec. 4) to avoid the polar-crowding artifact
    of uniform (eta, omega) sampling.

    The loss is bi-directional (Paschalidou et al. CVPR 2019, eq. 3):
        L_D = 1.0 * d(cloud→surface) + 1.0 * d(surface→cloud)
    Paschalidou et al. use asymmetric weights (1.2 / 0.8) during neural
    network training; equal weight 1.0 is standard for evaluation.

    Parameters
    ----------
    n_points : int
        Number of surface sample points.  Default 500 (inference budget).
    """
    try:
        surf = sample_surface_equal_distance(fit, n_points=n_points)
        if surf is None or len(surf) == 0:
            return 0.0
        pts32  = pts.astype(np.float32)
        surf32 = surf.astype(np.float32)
        # cloud → surface
        diff_ps = pts32[:, None, :] - surf32[None, :, :]
        d_ps = (diff_ps ** 2).sum(-1).min(-1)
        # surface → cloud
        diff_sp = surf32[:, None, :] - pts32[None, :, :]
        d_sp = (diff_sp ** 2).sum(-1).min(-1)
        return float(d_ps.mean() + d_sp.mean())
    except Exception:
        return 0.0


def confidence_weighted_chamfer(
    fits: list,
    pts_original: np.ndarray,
    n_points: int = 500,
) -> float:
    """Confidence-weighted Chamfer L2 across all surviving primitives.

    Paschalidou et al. CVPR 2019, eq. 6:
        L_conf = sum_k(gamma_k · L_D(P, Q_k)) / sum_k(gamma_k)
    where gamma_k = shape_conf (existence probability) and L_D is the
    bi-directional Chamfer L2 for primitive k.

    For LM fits (shape_conf=1.0 for all primitives) this reduces to the
    plain unweighted mean, so the column is directly comparable between
    fitters.

    Parameters
    ----------
    fits         : List[MultiSQFit] — output of fit_adaptive() or postprocess.
    pts_original : (N, 3) point cloud in the SAME coordinate frame as SQ poses.
    n_points     : surface sample points per primitive (default 500).

    Returns
    -------
    float  — confidence-weighted mean Chamfer L2, or nan if no primitives.
    """
    all_prims = [p for multi in fits for p in multi.primitives]
    if not all_prims:
        return float("nan")

    confs  = np.array([p.shape_conf for p in all_prims], dtype=np.float64)
    l2s    = np.array(
        [_chamfer_l2_from_surface(pts_original, p, n_points=n_points)
         for p in all_prims],
        dtype=np.float64,
    )

    total_conf = confs.sum()
    if total_conf < 1e-12:
        # All shape_conf are zero (e.g. LM primitives whose shape_conf was
        # overwritten by ObjectSegment.shape_conf=0.0 default in
        # fit_superquadrics).  Fall back to the unweighted mean so the column
        # is always populated rather than showing 'n/a'.
        return float(l2s.mean()) if len(l2s) > 0 else float("nan")
    return float((confs * l2s).sum() / total_conf)


def _filter_degenerate_primitives(
    primitives: list, min_scale: float = 0.002
) -> list:
    """Remove SuperquadricFit objects whose smallest semi-axis < min_scale.

    Degenerate primitives (near-zero scale) cause numerical issues in the
    signed-distance function and pollute collision geometry.

    Parameters
    ----------
    primitives : list[SuperquadricFit]
        In-place filtering — the input list is NOT modified; a new list is
        returned.
    min_scale  : float
        Minimum allowed value for min(sx, sy, sz).  Default 0.002 (lowered
        from 0.005 after ablation showed 0.005 was too aggressive and caused
        zero-primitive outputs for some samples).

    Returns
    -------
    Filtered list.  If ALL primitives would be removed, the single largest
    primitive (by volume sx·sy·sz) is kept instead, and a warning is logged.
    """
    filtered = [p for p in primitives if min(p.sx, p.sy, p.sz) >= min_scale]
    n_removed = len(primitives) - len(filtered)
    if n_removed > 0:
        _log.warning(
            "SuperdecFitter: removed %d degenerate primitive(s) "
            "(min semi-axis < %.4f).",
            n_removed, min_scale,
        )
    if not filtered and primitives:
        # Safety fallback: return the largest primitive rather than an empty list.
        best = max(primitives, key=lambda p: p.sx * p.sy * p.sz)
        _log.warning(
            "SuperdecFitter: all %d primitive(s) were degenerate; "
            "keeping the largest one as a fallback.",
            len(primitives),
        )
        filtered = [best]
    return filtered


def _sq_aabb(prim) -> tuple:
    """Axis-aligned bounding box of a SuperquadricFit in world coordinates.

    Returns (min_corner, max_corner) each as (3,) float64 arrays.
    The tight AABB is computed via the rotation matrix applied to the
    axis-aligned semi-axis box in local frame.
    """
    R = np.array(prim.rotation_matrix, dtype=np.float64)   # (3, 3)
    t = np.array(prim.translation, dtype=np.float64)        # (3,)
    half_local = np.array([prim.sx, prim.sy, prim.sz], dtype=np.float64)
    # AABB of the rotated box: each axis half-extent is |R[:,i]| · half_local
    half_world = np.abs(R) @ half_local                     # (3,)
    return t - half_world, t + half_world


def _aabb_iou(min_a: np.ndarray, max_a: np.ndarray,
              min_b: np.ndarray, max_b: np.ndarray) -> float:
    """Intersection-over-union of two axis-aligned bounding boxes."""
    inter_lo = np.maximum(min_a, min_b)
    inter_hi = np.minimum(max_a, max_b)
    inter    = np.maximum(0.0, inter_hi - inter_lo)
    inter_vol = float(inter[0] * inter[1] * inter[2])
    if inter_vol <= 0.0:
        return 0.0
    vol_a = float(np.prod(max_a - min_a))
    vol_b = float(np.prod(max_b - min_b))
    return inter_vol / (vol_a + vol_b - inter_vol + 1e-12)


def merge_overlapping_primitives(
    fits: list,
    iou_threshold: float = 0.3,
    distance_weights: Optional[List[float]] = None,
) -> list:
    """Merge SQ primitives whose AABBs overlap more than *iou_threshold*.

    When *distance_weights* is None (default), primitives are merged within
    each MultiSQFit independently: sorted by volume (largest first), any
    smaller primitive that overlaps a kept primitive by IoU > iou_threshold
    is discarded.

    When *distance_weights* is provided (one float per MultiSQFit), merging
    is performed **globally** across all MultiSQFits.  For two primitives
    belonging to fits with weights w_i and w_j the effective threshold is::

        effective_threshold = iou_threshold / (w_i * w_j)

    This makes the merge conservative for segments far from the camera (low
    weight → high effective threshold → primitives resist being discarded)
    and aggressive for nearby segments (weight near 1 → threshold ≈
    iou_threshold → primitives merge readily).

    Parameters
    ----------
    fits             : List[MultiSQFit]
    iou_threshold    : base IoU threshold.
    distance_weights : parallel list of floats in (0, 1], one per MultiSQFit.
                       None restores the original per-MultiSQFit behaviour.

    Returns
    -------
    New list of MultiSQFit with merged primitives.  n_points fields are
    preserved; ordering of fits in the list is preserved.
    """
    from superquadric import MultiSQFit as _MultiSQFit

    if distance_weights is None:
        # ── original per-MultiSQFit behaviour ─────────────────────────────────
        result = []
        for multi in fits:
            prims = list(multi.primitives)
            if len(prims) <= 1:
                result.append(multi)
                continue

            prims.sort(key=lambda p: p.sx * p.sy * p.sz, reverse=True)
            aabbs = [_sq_aabb(p) for p in prims]

            kept = []
            for i, (p, (lo_i, hi_i)) in enumerate(zip(prims, aabbs)):
                dominated = False
                for j in range(len(kept)):
                    lo_j, hi_j = aabbs[kept[j][0]]
                    if _aabb_iou(lo_i, hi_i, lo_j, hi_j) > iou_threshold:
                        dominated = True
                        break
                if not dominated:
                    kept.append((i, p))

            new_prims = [p for _, p in kept]
            n_merged  = len(prims) - len(new_prims)
            if n_merged > 0:
                _log.debug(
                    "merge_overlapping_primitives: merged %d → %d primitives "
                    "(IoU threshold %.2f)",
                    len(prims), len(new_prims), iou_threshold,
                )
            result.append(_MultiSQFit(primitives=new_prims, n_points=multi.n_points))
        return result

    # ── distance-weighted global merge across all MultiSQFits ─────────────────
    # Flatten: (fit_index, primitive, volume, aabb)
    all_entries = []
    for fit_idx, multi in enumerate(fits):
        for prim in multi.primitives:
            vol  = prim.sx * prim.sy * prim.sz
            aabb = _sq_aabb(prim)
            all_entries.append((fit_idx, prim, vol, aabb))

    # Sort globally: largest volume first (keep larger on overlap)
    all_entries.sort(key=lambda e: e[2], reverse=True)

    kept_entries = []  # list of (fit_idx, prim, aabb)
    for fit_idx, prim, _, (lo_i, hi_i) in all_entries:
        w_i = float(distance_weights[fit_idx])
        dominated = False
        for k_fit_idx, k_prim, (lo_k, hi_k) in kept_entries:
            w_k = float(distance_weights[k_fit_idx])
            eff_thr = iou_threshold / (w_i * w_k)
            if _aabb_iou(lo_i, hi_i, lo_k, hi_k) > eff_thr:
                dominated = True
                break
        if not dominated:
            kept_entries.append((fit_idx, prim, (lo_i, hi_i)))

    # Rebuild per-MultiSQFit lists (preserving original fit order)
    prims_by_fit: List[List] = [[] for _ in fits]
    for fit_idx, prim, _ in kept_entries:
        prims_by_fit[fit_idx].append(prim)

    n_before = sum(len(m.primitives) for m in fits)
    n_after  = sum(len(pl) for pl in prims_by_fit)
    if n_before != n_after:
        _log.debug(
            "merge_overlapping_primitives (distance-weighted): %d → %d primitives",
            n_before, n_after,
        )

    return [
        _MultiSQFit(primitives=prims_by_fit[i], n_points=fits[i].n_points)
        for i in range(len(fits))
    ]


def _clamp_exponents(primitives: list) -> list:
    """Clamp e1 and e2 of every SuperquadricFit to the convex regime in-place.

    Ensures SQ_EXPONENT_MIN ≤ e1, e2 ≤ SQ_EXPONENT_CONVEX_MAX for all
    primitives.  Emits a single warning log if any clamping was needed so
    that callers know the raw network output was out of range.

    Parameters
    ----------
    primitives : list[SuperquadricFit]
        Modified in-place.

    Returns
    -------
    The same list (for call-chaining convenience).
    """
    clamped = False
    for fit in primitives:
        e1_orig, e2_orig = fit.e1, fit.e2
        fit.e1 = float(np.clip(fit.e1, SQ_EXPONENT_MIN, SQ_EXPONENT_CONVEX_MAX))
        fit.e2 = float(np.clip(fit.e2, SQ_EXPONENT_MIN, SQ_EXPONENT_CONVEX_MAX))
        if fit.e1 != e1_orig or fit.e2 != e2_orig:
            clamped = True
    if clamped:
        _log.warning(
            "SuperdecFitter: one or more primitives had exponents outside "
            "[%.4g, %.4g]; clamped to convex regime.",
            SQ_EXPONENT_MIN, SQ_EXPONENT_CONVEX_MAX,
        )
    return primitives


def _resolve_checkpoint_dir(superdec_dir: str, checkpoint_dir: Optional[str]) -> str:
    """Return the absolute path to a checkpoint directory.

    Resolution order:
      1. If checkpoint_dir is an absolute path, use it directly.
      2. If checkpoint_dir is a relative path, resolve from CWD.
      3. If checkpoint_dir is None, fall back to the default fine-tuned
         tabletop v2 checkpoint at
         ../checkpoints/superdec_tabletop/superdec_tabletop_finetune_v2
         (relative to superdec_dir).
    """
    if checkpoint_dir is not None:
        p = os.path.abspath(checkpoint_dir)
        if not os.path.isdir(p):
            raise FileNotFoundError(f"Checkpoint directory not found: {p}")
        return p

    # default: fine-tuned tabletop v2 checkpoint
    default = os.path.abspath(
        os.path.join(superdec_dir, '..', 'checkpoints',
                     'superdec_tabletop', 'superdec_tabletop_finetune_v2')
    )
    if os.path.isdir(default):
        return default

    # graceful fallback to base 'normalized' checkpoint bundled with superdec
    fallback = os.path.join(superdec_dir, 'checkpoints', 'normalized')
    if os.path.isdir(fallback):
        import warnings
        warnings.warn(
            f"Fine-tuned tabletop checkpoint not found at {default}. "
            f"Falling back to base 'normalized' checkpoint at {fallback}. "
            "For best tabletop performance use the fine-tuned model.",
            stacklevel=3,
        )
        return fallback

    raise FileNotFoundError(
        f"No checkpoint found at {default} or {fallback}. "
        "Pass checkpoint_dir= explicitly to specify the checkpoint path."
    )


def _find_checkpoint_file(checkpoint_dir: str) -> str:
    """Return path to the .pt checkpoint file inside checkpoint_dir.

    Supports both:
      • ckpt.pt          — single canonical checkpoint
      • epoch_NNN.pt     — epoch-based saves; highest epoch is used
    """
    ckpt_single = os.path.join(checkpoint_dir, 'ckpt.pt')
    if os.path.exists(ckpt_single):
        return ckpt_single

    import glob
    epoch_files = sorted(glob.glob(os.path.join(checkpoint_dir, 'epoch_*.pt')))
    if epoch_files:
        # sort by epoch number, return the latest
        def _epoch_num(p):
            try:
                return int(os.path.basename(p).split('_')[1].split('.')[0])
            except (IndexError, ValueError):
                return -1
        return max(epoch_files, key=_epoch_num)

    raise FileNotFoundError(
        f"No checkpoint file (ckpt.pt or epoch_*.pt) found in {checkpoint_dir}"
    )


class SuperdecFitter:
    """SuperDec-based superquadric fitter for tabletop objects.

    Defaults to the fine-tuned tabletop v2 model.  The base (non-fine-tuned)
    checkpoints can be used by passing checkpoint_dir explicitly.

    Parameters
    ----------
    superdec_dir : str
        Path to the cloned superdec repository root.
    checkpoint_dir : str, optional
        Path to the checkpoint directory (absolute, or relative to CWD).
        Must contain config.yaml and either ckpt.pt or epoch_*.pt files.
        Defaults to ../checkpoints/superdec_tabletop/superdec_tabletop_finetune_v2
        relative to superdec_dir.
    exist_threshold : float
        Primitives whose existence score is below this value are discarded.
        Lower → more primitives returned. Default 0.3.
    n_points : int
        Number of points sampled from each segment before inference.
        Must match the training resolution (4096). Do not change.
    device : str, optional
        'cuda' or 'cpu'. Defaults to CUDA if available.
    """

    def __init__(
        self,
        superdec_dir: str,
        checkpoint_dir: Optional[str] = None,
        exist_threshold: float = 0.3,
        n_points: int = 4096,
        device: Optional[str] = None,
    ):
        _load_superdec(superdec_dir)

        from omegaconf import OmegaConf
        from superdec.superdec import SuperDec
        self._normalize_points    = None   # initialised below
        self._denormalize_outdict = None

        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device          = device
        self.exist_threshold = exist_threshold
        self.n_points        = n_points
        self.superdec_dir    = superdec_dir

        ckpt_dir = _resolve_checkpoint_dir(superdec_dir, checkpoint_dir)
        ckp_path = _find_checkpoint_file(ckpt_dir)
        cfg_path = os.path.join(ckpt_dir, 'config.yaml')

        if not os.path.exists(ckp_path):
            raise FileNotFoundError(f"Checkpoint not found: {ckp_path}")

        ckp = torch.load(ckp_path, map_location=device, weights_only=False)
        cfg = OmegaConf.load(cfg_path)

        self.model = SuperDec(cfg.superdec).to(device)
        self.model.load_state_dict(ckp['model_state_dict'])
        self.model.eval()
        self.model.lm_optimization = False

        # import data utils
        from superdec.data.dataloader import normalize_points, denormalize_outdict
        self._normalize_points   = normalize_points
        self._denormalize_outdict = denormalize_outdict

        print(f"[SuperdecFitter] loaded checkpoint '{os.path.basename(ckp_path)}' "
              f"from {ckpt_dir} on {device}")

    # ------------------------------------------------------------------
    @staticmethod
    def _check_input_contract(pts: np.ndarray) -> None:
        """Assert that *pts* satisfies the SuperDec input contract.

        SuperDec normalises internally via normalize_points(), which maps the
        cloud to a unit sphere centred at the origin.  If the caller already
        normalised (e.g. preprocess_pointcloud with scale normalisation
        enabled), the input will be in [-1, 1] and the internal normalisation
        is a near-identity transform — this is the expected usage.

        The assertions below catch the most common contract violations:

        1. **World-scale input** (e.g. raw RGB-D at 2–3 m from camera):
           coordinates will be outside [-2, 2], triggering the bounds check.
        2. **Fewer than 100 points**: SuperDec was trained on 4096-point
           clouds; very sparse inputs produce garbage outputs.
        3. **Zero-variance / degenerate clouds**: std < 0.01 across all
           axes indicates a near-flat or single-point cloud.

        Raises
        ------
        AssertionError  — with a descriptive message for each violated check.
        """
        assert pts.shape[0] >= 100, (
            f"SuperdecFitter: input must have ≥ 100 points, got {pts.shape[0]}. "
            "Ensure the segment was not over-filtered before fitting."
        )
        coord_max = float(np.abs(pts).max())
        assert coord_max <= 2.0, (
            f"SuperdecFitter: input coordinates exceed ±2 (max |x|={coord_max:.3f}). "
            "This usually means world-scale coordinates (metres) were passed "
            "directly — normalise to unit sphere first, or use for_superdec=True "
            "in preprocess_pointcloud() so scale normalisation is skipped and "
            "SuperDec normalises internally."
        )
        coord_std = float(pts.std())
        assert coord_std > 0.01, (
            f"SuperdecFitter: input standard deviation {coord_std:.5f} is near-zero. "
            "The cloud may be degenerate (single point or flat slab)."
        )

    # ------------------------------------------------------------------
    def fit_adaptive(
        self,
        points: np.ndarray,
        shape_hint=None,        # ignored — kept for interface compatibility
        n_restarts: int = 1,    # ignored
        **kwargs,
    ) -> 'MultiSQFit':
        """
        Run SuperDec inference on a single object point cloud.

        Parameters
        ----------
        points : (N, 3) float32 array — object segment in world coordinates.

        Returns
        -------
        MultiSQFit with up to 16 SuperquadricFit primitives (filtered by exist).
        """
        pts = np.asarray(points, dtype=np.float64)
        if len(pts) < 10:
            return MultiSQFit(primitives=[], n_points=0)

        # Validate input before normalisation (catches double-normalisation,
        # world-scale coordinates, and degenerate clouds early).
        self._check_input_contract(pts)

        # sample to fixed resolution
        n = len(pts)
        idx = np.random.choice(n, self.n_points, replace=n < self.n_points)
        pts_sampled = pts[idx]

        # normalise to unit sphere (SuperDec convention)
        pts_norm, translation, scale = self._normalize_points(pts_sampled)
        # translation: (3,)   scale: scalar
        # wrap in batch dim
        x = torch.from_numpy(pts_norm).unsqueeze(0).float().to(self.device)

        with torch.no_grad():
            out = self.model(x)
            raw_out = out

        # denormalise back to world coordinates
        # denormalize_outdict expects translation: (B,3), scale: (B,)
        t_batch = torch.from_numpy(translation).unsqueeze(0).float().to(self.device)
        s_batch = torch.tensor([scale], dtype=torch.float32).to(self.device)
        out = self._denormalize_outdict(out, t_batch, s_batch)

        # extract per-primitive parameters — all (B=1, P=16, ...)
        scales      = out['scale'][0].cpu().numpy()       # (16, 3)
        rotations   = out['rotate'][0].cpu().numpy()      # (16, 3, 3)
        translations= out['trans'][0].cpu().numpy()       # (16, 3)
        exponents   = out['shape'][0].cpu().numpy()       # (16, 2)
        exist_score = out['exist'][0].cpu().numpy().ravel()  # (N,) regardless of trailing dims

        # exist may be raw logits — apply sigmoid if range is not [0,1]
        if exist_score.min() < -0.5 or exist_score.max() > 1.5:
            exist_score = 1.0 / (1.0 + np.exp(-exist_score))

        primitives = []
        for p_idx in range(scales.shape[0]):
            if float(exist_score[p_idx]) < self.exist_threshold:
                continue

            e1 = float(np.clip(exponents[p_idx, 0], SQ_EXPONENT_MIN, SQ_EXPONENT_CONVEX_MAX))
            e2 = float(np.clip(exponents[p_idx, 1], SQ_EXPONENT_MIN, SQ_EXPONENT_CONVEX_MAX))
            sx, sy, sz = (float(v) for v in np.abs(scales[p_idx]))
            tx, ty, tz = (float(v) for v in translations[p_idx])
            rx, ry, rz = _rotmat_to_euler_xyz(rotations[p_idx])

            fit = SuperquadricFit(
                sx=sx, sy=sy, sz=sz,
                e1=e1, e2=e2,
                tx=tx, ty=ty, tz=tz,
                rx=rx, ry=ry, rz=rz,
                shape_type=sq_type_from_exponents(e1, e2),
                shape_conf=float(exist_score[p_idx]),
                converged=True,
                chamfer_l2=0.0,
            )
            primitives.append(fit)

        _clamp_exponents(primitives)
        primitives = _filter_degenerate_primitives(primitives)

        # compute per-primitive Chamfer L2 via surface sampling
        if primitives and len(pts) > 0:
            for fit in primitives:
                fit.chamfer_l2 = _chamfer_l2_from_surface(pts, fit)

        return MultiSQFit(primitives=primitives, n_points=len(pts))

    def fit_batch(self, points_list: List[np.ndarray]) -> List['MultiSQFit']:
        """
        Batch inference — all objects in one forward pass.
        Dramatically faster than calling fit_adaptive N times.
        """
        if not points_list:
            return []

        valid_idx = [i for i, pts in enumerate(points_list) if len(pts) >= 10]
        if not valid_idx:
            return [MultiSQFit(primitives=[], n_points=0) for _ in points_list]

        pts_raw_list   = []
        translations   = []
        scales         = []
        pts_norm_batch = []

        for i in valid_idx:
            pts = np.asarray(points_list[i], dtype=np.float64)
            pts_raw_list.append(pts)
            n = len(pts)
            idx = np.random.choice(n, self.n_points, replace=n < self.n_points)
            pts_sampled = pts[idx]
            pts_norm, t, s = self._normalize_points(pts_sampled)
            pts_norm_batch.append(pts_norm)
            translations.append(t)
            scales.append(s)

        x = torch.from_numpy(np.stack(pts_norm_batch, axis=0)).float().to(self.device)

        with torch.no_grad():
            out = self.model(x)
            raw_out = out

        t_batch = torch.from_numpy(np.stack(translations, axis=0)).float().to(self.device)
        s_batch = torch.tensor(scales, dtype=torch.float32).to(self.device)
        out = self._denormalize_outdict(out, t_batch, s_batch)

        scales_out  = out['scale'].cpu().numpy()
        rotations   = out['rotate'].cpu().numpy()
        trans_out   = out['trans'].cpu().numpy()
        exponents   = out['shape'].cpu().numpy()
        exist_score = out['exist'].cpu().numpy()
        exist_score = exist_score.reshape(exist_score.shape[0], -1)  # (B, N) — drop trailing dim
        assign      = raw_out.get('assign_matrix')

        if exist_score.min() < -0.5 or exist_score.max() > 1.5:
            exist_score = 1.0 / (1.0 + np.exp(-exist_score))

        results = [MultiSQFit(primitives=[], n_points=0) for _ in points_list]

        for b, orig_i in enumerate(valid_idx):
            pts_b = pts_raw_list[b]

            primitives = []
            for p_idx in range(scales_out.shape[1]):
                if float(exist_score[b, p_idx]) < self.exist_threshold:
                    continue

                e1 = float(np.clip(exponents[b, p_idx, 0], SQ_EXPONENT_MIN, SQ_EXPONENT_CONVEX_MAX))
                e2 = float(np.clip(exponents[b, p_idx, 1], SQ_EXPONENT_MIN, SQ_EXPONENT_CONVEX_MAX))
                sx, sy, sz = (float(v) for v in np.abs(scales_out[b, p_idx]))
                tx, ty, tz = (float(v) for v in trans_out[b, p_idx])
                rx, ry, rz = _rotmat_to_euler_xyz(rotations[b, p_idx])

                fit = SuperquadricFit(
                    sx=sx, sy=sy, sz=sz,
                    e1=e1, e2=e2,
                    tx=tx, ty=ty, tz=tz,
                    rx=rx, ry=ry, rz=rz,
                    shape_type=sq_type_from_exponents(e1, e2),
                    shape_conf=float(exist_score[b, p_idx]),
                    converged=True,
                    chamfer_l2=0.0,
                )
                primitives.append(fit)

            _clamp_exponents(primitives)
            primitives = _filter_degenerate_primitives(primitives)

            if primitives and len(pts_b) > 0:
                for fit in primitives:
                    fit.chamfer_l2 = _chamfer_l2_from_surface(pts_b, fit)

            results[orig_i] = MultiSQFit(primitives=primitives, n_points=len(pts_b))

        return results

