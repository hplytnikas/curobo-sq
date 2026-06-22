"""
pipeline.py — tabletop RGB-D perception pipeline.

The pipeline follows four explicit stages:

  Stage 1 — RGB-D input
      depth_to_pointcloud(depth, K) converts a depth image + camera intrinsics
      to an (N, 3) point cloud.  If you already have a point cloud, skip this.

  Stage 2 — Table / background removal
      remove_table(pts, ...) runs RANSAC plane fitting to locate the work
      surface, then applies three filters:
        a) depth gate — removes background walls visible in angled views
           (ARID20 "bottom" camera orientation)
        b) height filter — keeps only points physically above the table
        c) XY radius crop — discards points far from the camera origin

  Stage 3 — Instance segmentation
      segment_instances(obj_pts, ...) clusters the foreground cloud with
      DBSCAN then applies up to four splitting passes to separate touching
      or stacked objects:
        Pass 1: horizontal density saddle (XY PCA projection)
        Pass 2: vertical density saddle (Y / height axis)
        Pass 3B: 2-D concavity / narrow neck (XZ projection)
        Pass 3C: height-layer gaps (near-empty Y histogram bands)
      (Pass 3A — normal discontinuity — is disabled: open3d
       estimate_normals segfaults on macOS after DBSCAN.)

  Stage 4 — Shape classification / SQ hint
      Each cluster gets a lightweight geometry-heuristic label
      (Ellipsoid / Cylinder / Cuboid / Other) that biases the subsequent
      Levenberg-Marquardt or SuperDec superquadric fit.

Design notes
  • _clean() forces a numpy copy to break open3d shared-memory views.
  • Stage 2 uses camera-space Z for the depth gate (not world height) so
    that background walls in tilted views are correctly rejected.
  • All stage functions are importable standalone for testing.
"""

import copy
import logging
import os
import time
import numpy as np
import open3d as o3d
from dataclasses import dataclass, field
from typing import List, Optional, Tuple


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class ObjectSegment:
    """One segmented object returned by the perception pipeline.

    Consumed downstream by SuperquadricFitter / SuperdecFitter and by
    GraspSelector.  All fields are read-only after construction.
    """
    id:         int
    points:     np.ndarray   # (N, 3) float32, world coordinates
    centroid:   np.ndarray   # (3,) float32
    bbox_min:   np.ndarray   # (3,) float32
    bbox_max:   np.ndarray   # (3,) float32
    n_points:   int   = 0
    shape_type: str   = "Other"   # Ellipsoid | Cylinder | Cuboid | Other
    shape_conf: float = 0.0       # heuristic confidence in [0, 1]

    def __post_init__(self):
        self.n_points = len(self.points)

    @property
    def bbox_extent(self) -> np.ndarray:
        """Bounding-box side lengths in metres."""
        return self.bbox_max - self.bbox_min

    def to_dict(self) -> dict:
        return dict(id=self.id, points=self.points, centroid=self.centroid,
                    bbox_min=self.bbox_min, bbox_max=self.bbox_max,
                    shape_type=self.shape_type, shape_conf=self.shape_conf)


@dataclass
class PerceptionResult:
    """Full output of TabletopPerception.run()."""
    objects:                List[ObjectSegment] = field(default_factory=list)
    table_normal:           np.ndarray = field(default_factory=lambda: np.array([0.,0.,1.]))
    table_height:           float = 0.0
    n_points_input:         int = 0
    n_points_after_denoise: int = 0
    n_points_table:         int = 0

    def summary(self) -> str:
        lines = [
            f"Perception result — {len(self.objects)} object(s) found",
            f"  Input points   : {self.n_points_input:,}",
            f"  After denoise  : {self.n_points_after_denoise:,}",
            f"  Table pts      : {self.n_points_table:,}",
            f"  Table height   : {self.table_height:.3f} m",
            f"  Table normal   : {self.table_normal.round(3)}",
        ]
        for obj in self.objects:
            e = obj.bbox_extent
            lines.append(
                f"  Object {obj.id}: {obj.n_points} pts | "
                f"centroid {obj.centroid.round(3)} | "
                f"extent [{e[0]:.3f}, {e[1]:.3f}, {e[2]:.3f}] m | "
                f"shape={obj.shape_type} ({obj.shape_conf:.2f})"
            )
        return "\n".join(lines)


class PerceptionTimer:
    """Wall-clock profiler for perception pipeline stages.

    Usage::

        timer = PerceptionTimer()
        timer.start('unproject')
        pts = pointcloud_from_depth(depth, K)
        timer.stop('unproject')
        print(timer.to_dict())   # {'unproject': 0.003}
        print(f"total: {timer.total*1000:.1f} ms")

    Each stage accumulates across repeated start/stop calls, so it is safe
    to call inside a per-segment loop.
    """

    def __init__(self) -> None:
        self._times: dict  = {}   # stage → total elapsed seconds
        self._starts: dict = {}   # stage → last start timestamp

    def start(self, stage: str) -> None:
        """Record the start of *stage*."""
        self._starts[stage] = time.perf_counter()

    def stop(self, stage: str) -> float:
        """Record the end of *stage*; return elapsed seconds."""
        t0 = self._starts.pop(stage, None)
        if t0 is None:
            _log.debug("PerceptionTimer.stop('%s') without matching start()", stage)
            return 0.0
        dt = time.perf_counter() - t0
        self._times[stage] = self._times.get(stage, 0.0) + dt
        return dt

    def to_dict(self) -> dict:
        """Return ``{stage_name: elapsed_seconds}`` for every stopped stage."""
        return dict(self._times)

    @property
    def total(self) -> float:
        """Sum of all recorded stage durations (seconds)."""
        return sum(self._times.values())


@dataclass
class SQWorldModel:
    """Compact superquadric scene model for a single RGB-D observation.

    Returned by :func:`single_frame_pipeline` and serves as the canonical
    interface for downstream manipulation modules:

    * **Collision geometry**  — :meth:`to_curobo_obstacles` (CuRobo motions)
    * **Signed-distance SDF** — ``Scene.from_fits(world_model.all_primitives)``
    * **Grasp candidates**    — ``GraspSelector(world_model.all_primitives)``

    Parameters
    ----------
    fits : list[MultiSQFit]
        One :class:`MultiSQFit` per detected object, poses in world frame.
    """
    fits: list   # List[MultiSQFit] — typed as plain list to avoid circular import

    @property
    def n_objects(self) -> int:
        """Number of segmented object instances."""
        return len(self.fits)

    @property
    def n_primitives(self) -> int:
        """Total superquadric primitives across all objects."""
        return sum(len(m.primitives) for m in self.fits)

    @property
    def all_primitives(self) -> list:
        """Flat list of every :class:`SuperquadricFit` in the scene."""
        return [p for m in self.fits for p in m.primitives]

    def to_curobo_obstacles(self, margin: float = 0.005) -> list:
        """Convert all primitives to CuRobo obstacle dicts.

        Parameters
        ----------
        margin : float
            Uniform safety margin inflating every primitive (metres).
        """
        try:
            from .superquadric import fits_to_curobo_obstacles
        except ImportError:
            from superquadric import fits_to_curobo_obstacles
        return fits_to_curobo_obstacles(self.all_primitives, margin=margin)


# ---------------------------------------------------------------------------
# Internal utilities
# ---------------------------------------------------------------------------

def _clean(pts: np.ndarray, max_coord: float = 4.0) -> np.ndarray:
    """Remove NaN/inf and out-of-range points; force a numpy copy.

    The copy breaks open3d shared-memory views so that downstream filters
    cannot inadvertently corrupt the open3d point cloud buffer.
    """
    pts = np.array(pts, dtype=np.float64, copy=True)
    good = np.isfinite(pts).all(axis=1) & (np.abs(pts) < max_coord).all(axis=1)
    return pts[good]


def _to_o3d(pts: np.ndarray) -> o3d.geometry.PointCloud:
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts.astype(np.float64))
    return pcd


# ---------------------------------------------------------------------------
# Preprocessing — point cloud cleaning, normalisation, PCA canonicalisation
# ---------------------------------------------------------------------------

_log = logging.getLogger(__name__)


def _r_x(deg: float) -> np.ndarray:
    """3×3 rotation matrix around the X axis by *deg* degrees."""
    rad = float(np.deg2rad(deg))
    c, s = float(np.cos(rad)), float(np.sin(rad))
    return np.array([[1., 0., 0.],
                     [0.,  c, -s],
                     [0.,  s,  c]], dtype=np.float64)


# Per-category canonical rotation applied AFTER preprocess_pointcloud() and
# BEFORE SuperDec inference.
#
# Ablation results (SuperDec fine-tuned, 3 samples/category, orig-L2):
#   Category  | with rotation | without rotation | verdict
#   ----------|---------------|------------------|--------
#   bottle    |  ~0.15 (bad)  |  ~0.074          | NO rotation — model fine-tuned on native Y-up
#   mug       |  ~0.18 (bad)  |  ~0.088          | NO rotation — same reason
#   bowl      |  0.105        |  0.115            | KEEP rotation — opening faces up correctly
#   knife     |  (not tested) |  —               | keep R_x(90) provisionally
#
# Bottles and mugs are left at identity: SuperDec was fine-tuned on ShapeNet's
# native Y-up orientation; rotating away from it breaks the model for those categories.
SHAPENET_CATEGORY_ROTATIONS: dict = {
    "02876657": _r_x(0),    # bottle  — identity; rotation hurts (ablation)
    "03797390": _r_x(0),    # mug     — identity; rotation hurts (ablation)
    "03624134": _r_x(90),   # knife   — long axis Y → Z (not ablated)
    "03642806": _r_x(0),    # laptop  — flat, no rotation needed
    "02880940": _r_x(90),   # bowl    — rotation helps: 0.105 vs 0.115 (ablation)
    "gso":      _r_x(0),    # GSO    — unknown, leave as-is
}


def _fps_numpy(pts: np.ndarray, n_samples: int) -> np.ndarray:
    """Farthest point sampling; returns index array of length n_samples."""
    N = len(pts)
    if n_samples >= N:
        return np.arange(N)
    # Pre-subsample for efficiency when N >> n_samples
    if N > 4 * n_samples:
        pre = np.random.choice(N, 4 * n_samples, replace=False)
        sub_idx = _fps_numpy(pts[pre], n_samples)
        return pre[sub_idx]
    selected = np.zeros(n_samples, dtype=np.int64)
    dists = np.full(N, np.inf)
    selected[0] = 0
    for i in range(1, n_samples):
        last = pts[selected[i - 1]]
        d = np.sum((pts - last) ** 2, axis=1)
        dists = np.minimum(dists, d)
        selected[i] = int(np.argmax(dists))
    return selected


def preprocess_pointcloud(
    pts: np.ndarray,
    normals: Optional[np.ndarray] = None,
    target_n: int = 4096,
    outlier_std: float = 2.5,
    rotate: bool = False,
    table_normal: Optional[np.ndarray] = None,
    table_height: float = 0.0,
    for_superdec: bool = False,
) -> Tuple[np.ndarray, Optional[np.ndarray], dict]:
    """Clean and resample an object point cloud for SQ fitting.

    Steps
    -----
    0. (Optional) Table-frame canonicalisation — when *table_normal* is
       provided, the cloud is brought into a coordinate frame where the table
       surface is the XY plane.  This is applied **before** outlier removal.
       Two sub-steps:
         A. Translate along the table normal so that the segment centroid's
            height above the table becomes zero (removes the height component
            of the centroid, leaving XY position intact).
         B. Rotate so that *table_normal* maps to +Z (Rodriguez formula,
            no scipy).  When the normal is already [0,0,1] this is the
            identity.  PCA canonicalisation (rotate=True) is skipped when
            table_normal is provided.
    1. Statistical outlier removal — drop points outside
       centroid ± outlier_std * std on **any** axis (axis-aligned box test).
    2. Optional PCA orientation canonicalisation via SVD (``rotate=True``) —
       rotate so the longest principal axis maps to Z and the second to Y.
       Disabled by default; also skipped when table_normal is provided
       (table frame is already a principled canonical frame).
    3. Uniform resampling to exactly *target_n* points:
       - N > target_n : Farthest Point Sampling.
       - N < target_n : random duplication + Gaussian jitter (σ = 0.001).
       - N == target_n: no-op.

    Note: scale normalisation (divide by max radius) is intentionally omitted
    because SuperdecFitter already normalises to unit sphere internally via
    ``normalize_points()``.  Adding a second normalisation here caused double
    scaling and degraded Chamfer L2.

    Parameters
    ----------
    pts          : (N, 3) input point cloud
    normals      : (N, 3) optional normals; same outlier mask and rotation
                   applied (rotation only — not scaled or translated).
    target_n     : target point count (default 4096)
    outlier_std  : axis-aligned std multiplier for outlier removal
    rotate       : if True, apply PCA canonicalisation (default False).
                   Ignored when table_normal is not None.
    table_normal : (3,) unit normal of the table plane in the input frame.
                   When provided, Step 0 (table-frame canonicalisation) is
                   applied and PCA rotation is skipped.
    table_height : signed distance of the table plane from the origin along
                   table_normal.  Used with table_normal to compute heights.
    for_superdec : bool (default False)
                   When True, indicates the output will be fed to
                   SuperdecFitter rather than SuperquadricFitter (LM).
                   Scale normalisation is skipped — SuperDec normalises
                   internally via normalize_points() in fit_adaptive().
                   Double normalisation would shift the input distribution
                   away from the ShapeNet training regime.
                   (Liu et al. CVPR 2022 Sec. 3.4; Supplementary Sec. 2)
                   Outlier removal, FPS resampling, and table-frame rotation
                   are still applied when for_superdec=True.

    Returns
    -------
    pts_out     : (target_n, 3) preprocessed points
    normals_out : (target_n, 3) or None
    meta        : dict with keys
                    'scale'              float — always 1.0 (no-op)
                    'centroid'           (3,)  — always zeros (no-op)
                    'rotation'           (3,3) — table or PCA rotation, or eye
                    'n_outliers_removed' int
                    'table_centroid'     (3,)  — only present when table_normal
                                                 was provided; the translation
                                                 subtracted in Step 0A
                    'table_rotation'     (3,3) — only present when table_normal
                                                 was provided; the rotation
                                                 applied in Step 0B
    """
    pts = np.asarray(pts, dtype=np.float64)
    if normals is not None:
        normals = np.asarray(normals, dtype=np.float64)

    meta_extra: dict = {}

    # ── 0. Table-frame canonicalisation (optional) ────────────────────────────
    if table_normal is not None:
        n = np.asarray(table_normal, dtype=np.float64)
        n = n / (np.linalg.norm(n) + 1e-12)

        # Step 0A — translate: remove height component of segment centroid
        seg_h = float(pts.mean(axis=0) @ n - table_height)
        table_centroid = seg_h * n                     # vector along table normal
        pts = pts - table_centroid
        if normals is not None:
            pass  # normals are directions; translation does not affect them
        meta_extra["table_centroid"] = table_centroid

        # Step 0B — rotate: map table_normal → [0, 0, 1] via Rodriguez
        z = np.array([0.0, 0.0, 1.0])
        v = np.cross(n, z)
        s = float(np.linalg.norm(v))
        c = float(np.dot(n, z))
        if s < 1e-9:
            # Normal is already ±Z
            R_table = np.eye(3) if c > 0 else np.diag([1.0, -1.0, -1.0])
        else:
            vx = np.array([
                [ 0.0,  -v[2],  v[1]],
                [ v[2],  0.0,  -v[0]],
                [-v[1],  v[0],  0.0],
            ])
            R_table = np.eye(3) + vx + vx @ vx * (1.0 - c) / (s * s)
        pts = (R_table @ pts.T).T
        if normals is not None:
            normals = (R_table @ normals.T).T
        meta_extra["table_rotation"] = R_table

    # ── 1. Outlier removal (per-axis std, axis-aligned box) ──────────────────
    centroid0 = pts.mean(axis=0)
    std0 = pts.std(axis=0)
    std0[std0 < 1e-9] = 1e-9
    mask = np.all(np.abs(pts - centroid0) <= outlier_std * std0, axis=1)
    n_outliers = int((~mask).sum())
    if n_outliers > 0:
        _log.debug("preprocess_pointcloud: removed %d outliers (>%.1fσ per axis)",
                   n_outliers, outlier_std)
    pts = pts[mask]
    if normals is not None:
        normals = normals[mask]

    if len(pts) < 10:
        raise ValueError(
            f"preprocess_pointcloud: only {len(pts)} points remain after "
            "outlier removal — cannot continue."
        )

    pts_out = pts

    # ── 2. Optional PCA canonicalisation via SVD ──────────────────────────────
    # Skipped when table_normal is provided (table frame is already canonical).
    if rotate and table_normal is None:
        pts_c = pts_out - pts_out.mean(axis=0)
        _, _, Vt = np.linalg.svd(pts_c, full_matrices=False)   # Vt: (3, 3)
        # Vt[0] = longest axis → map to Z; Vt[1] → Y; Vt[2] → X
        rotation = np.array([Vt[2], Vt[1], Vt[0]], dtype=np.float64)
        if np.linalg.det(rotation) < 0:
            rotation[0] = -rotation[0]   # ensure right-handed
        pts_out = (rotation @ pts_c.T).T
        if normals is not None:
            normals = (rotation @ normals.T).T
    else:
        rotation = np.eye(3, dtype=np.float64)

    # ── 3. Uniform resampling ─────────────────────────────────────────────────
    N = len(pts_out)
    if N > target_n:
        idx = _fps_numpy(pts_out, target_n)
        pts_out = pts_out[idx]
        if normals is not None:
            normals = normals[idx]
    elif N < target_n:
        n_add = target_n - N
        idx = np.random.choice(N, n_add, replace=True)
        jitter = np.random.normal(0.0, 0.001, (n_add, 3))
        pts_out = np.vstack([pts_out, pts_out[idx] + jitter])
        if normals is not None:
            normals = np.vstack([normals, normals[idx]])

    meta = {
        "scale":              1.0,
        "centroid":           np.zeros(3, dtype=np.float64),
        "rotation":           rotation,
        "n_outliers_removed": n_outliers,
        **meta_extra,
    }
    return pts_out, normals, meta


def postprocess_fits(
    fits: list,
    meta: dict,
) -> list:
    """Invert preprocess_pointcloud() transforms on fitted SQ poses.

    The preprocessing applied::

        pts_clean = rotation @ ((pts_orig - centroid) / scale)

    The inverse is::

        t_orig = rotation.T @ t_clean * scale + centroid
        R_orig = rotation.T @ R_clean
        sx/sy/sz *= scale

    Parameters
    ----------
    fits : List[MultiSQFit]
        Output of fit_superquadrics(), poses in the preprocessed frame.
    meta : dict
        As returned by preprocess_pointcloud().

    Returns
    -------
    List[MultiSQFit] — same structure, but poses in the original frame.
    """
    import scipy.spatial.transform as _sst

    R_pca = meta["rotation"]     # (3, 3)
    R_inv = R_pca.T              # orthogonal inverse
    scale = float(meta["scale"])
    centroid = np.asarray(meta["centroid"], dtype=np.float64)

    try:
        from .superquadric import MultiSQFit
    except ImportError:
        from superquadric import MultiSQFit  # bare-script mode

    # Table-frame inversion (applied last — outermost in the transform chain).
    # These keys are only present when preprocess_pointcloud was called with
    # table_normal != None.
    R_table    = np.asarray(meta["table_rotation"], dtype=np.float64) \
                 if "table_rotation" in meta else None
    tbl_centroid = np.asarray(meta["table_centroid"], dtype=np.float64) \
                   if "table_centroid" in meta else None

    result = []
    for multi in fits:
        new_prims = []
        for prim in multi.primitives:
            p = copy.copy(prim)
            # Scale semi-axes back to world units
            p.sx = prim.sx * scale
            p.sy = prim.sy * scale
            p.sz = prim.sz * scale
            # Invert PCA / identity rotation and scale
            t_clean = prim.translation           # (3,)
            t_orig = R_inv @ t_clean * scale + centroid
            R_orig = R_inv @ prim.rotation_matrix
            # Invert table-frame transform (outermost — applied last)
            if R_table is not None:
                R_tbl_inv = R_table.T             # orthogonal inverse
                t_orig = R_tbl_inv @ t_orig + tbl_centroid
                R_orig = R_tbl_inv @ R_orig
            p.tx, p.ty, p.tz = float(t_orig[0]), float(t_orig[1]), float(t_orig[2])
            rx, ry, rz = _sst.Rotation.from_matrix(R_orig).as_euler("xyz")
            p.rx, p.ry, p.rz = float(rx), float(ry), float(rz)
            new_prims.append(p)
        result.append(MultiSQFit(primitives=new_prims, n_points=multi.n_points))
    return result


# ---------------------------------------------------------------------------
# Stage 1 — RGB-D input
# ---------------------------------------------------------------------------

def depth_to_pointcloud(depth: np.ndarray, K: np.ndarray,
                         depth_scale: float = 1000.0,
                         max_depth: float = 4.0) -> np.ndarray:
    """Back-project a depth image to a 3-D point cloud.

    Parameters
    ----------
    depth       : (H, W) uint16 or float depth image
    K           : (3, 3) camera intrinsic matrix (fx, fy, cx, cy)
    depth_scale : divisor that converts raw depth units to metres
                  (1000 for PNG millimetre depths, 1 if already in metres)
    max_depth   : discard points beyond this distance (metres)

    Returns
    -------
    (N, 3) float64 array of XYZ points in camera space (Z forward, Y down)
    """
    H, W = depth.shape
    fx, fy = float(K[0, 0]), float(K[1, 1])
    cx, cy = float(K[0, 2]), float(K[1, 2])

    zz = depth.astype(np.float64) / depth_scale
    v, u = np.mgrid[0:H, 0:W]

    x = (u - cx) * zz / fx
    y = (v - cy) * zz / fy
    z = zz

    pts = np.stack([x, y, z], axis=-1).reshape(-1, 3)
    valid = (pts[:, 2] > 0) & (pts[:, 2] < max_depth)
    return pts[valid]


def pointcloud_from_depth(
    depth: np.ndarray,
    K: np.ndarray,
    extrinsic: Optional[np.ndarray] = None,
    max_depth: float = 3.0,
) -> np.ndarray:
    """Unproject a float-metre depth image to a point cloud.

    This is the primary entry point for single-view RGB-D input as described
    in the project proposal (Sec. 2.1: "Starting from a depth image…
    single-view reconstruction as the primary operating mode").

    Unlike :func:`depth_to_pointcloud`, this function expects depth already
    in **metres** (float32/float64, not uint16 millimetres) and supports an
    optional camera-to-world extrinsic transform.

    Parameters
    ----------
    depth     : (H, W) float32 or float64 depth image in metres.
                Pixels with ``depth <= 0`` or ``depth >= max_depth`` are
                treated as invalid and excluded from the output.
    K         : (3, 3) camera intrinsic matrix ``[[fx,0,cx],[0,fy,cy],[0,0,1]]``.
    extrinsic : (4, 4) camera-to-world transform (SE(3)).  If ``None``,
                points are returned in the camera frame (Z forward, Y down).
    max_depth : maximum valid depth in metres.  Default 3.0 m.

    Returns
    -------
    (N, 3) float32 XYZ array.  World frame when *extrinsic* is provided,
    camera frame otherwise.  Invalid pixels are excluded; N may be 0.
    """
    H, W = depth.shape
    u, v = np.meshgrid(np.arange(W, dtype=np.float32),
                       np.arange(H, dtype=np.float32))
    fx, fy = float(K[0, 0]), float(K[1, 1])
    cx, cy = float(K[0, 2]), float(K[1, 2])

    valid = (depth > 0) & (depth < max_depth)
    z = depth[valid].astype(np.float32)
    x = (u[valid] - cx) * z / fx
    y = (v[valid] - cy) * z / fy
    pts = np.stack([x, y, z], axis=1)   # (N, 3) float32

    if extrinsic is not None:
        E = np.asarray(extrinsic, dtype=np.float32)
        pts_h = np.concatenate([pts, np.ones((len(pts), 1), dtype=np.float32)],
                               axis=1)
        pts = (E @ pts_h.T).T[:, :3]
    return pts


# ---------------------------------------------------------------------------
# Stage 2 — Table / background removal
# ---------------------------------------------------------------------------

def remove_table(
    pts: np.ndarray,
    plane_dist_threshold:   float = 0.012,
    plane_ransac_n:         int   = 3,
    plane_num_iterations:   int   = 1000,
    min_height_above_table: float = 0.005,
    max_height_above_table: float = 0.25,
    depth_margin:           float = 0.25,
    xy_radius:              float = 0.80,
    plane_hint:             Optional[Tuple[np.ndarray, float]] = None,
) -> Tuple[np.ndarray, np.ndarray, float, np.ndarray, int]:
    """Locate the table plane and return only the foreground object points.

    Parameters
    ----------
    pts                    : (N, 3) point cloud (already denoised)
    plane_dist_threshold   : RANSAC inlier distance in metres
    plane_ransac_n         : minimum points per RANSAC hypothesis
    plane_num_iterations   : RANSAC iterations
    min_height_above_table : lower height cutoff above table (metres)
    max_height_above_table : upper height cutoff (metres) — rejects ceiling
    depth_margin           : camera-Z slack beyond table max-Z (metres)
    xy_radius              : XY radius crop centred on camera origin (metres)
    plane_hint             : (normal, height) from a previous frame to skip
                             RANSAC (saves ~200 ms/frame in sequences)

    Returns
    -------
    obj_pts      : (M, 3) foreground points above the table
    table_normal : (3,) unit normal of the fitted plane
    table_height : signed distance of the plane from origin (metres)
    table_pts    : (K, 3) points on the table surface (RANSAC inliers)
    n_table_pts  : number of table inlier points
    """
    if plane_hint is not None:
        table_normal, table_height = plane_hint
        table_normal = np.array(table_normal, dtype=np.float64)
        table_height = float(table_height)
    else:
        pcd = _to_o3d(pts)
        plane_model, _ = pcd.segment_plane(
            distance_threshold=plane_dist_threshold,
            ransac_n=plane_ransac_n,
            num_iterations=plane_num_iterations,
        )
        a, b, c, d = [float(x) for x in plane_model]
        n_raw    = np.array([a, b, c], dtype=np.float64)
        norm_len = float(np.linalg.norm(n_raw))
        if norm_len < 1e-9:
            raise ValueError("RANSAC returned a degenerate plane (zero normal)")

        table_normal = n_raw / norm_len
        table_height = float(-d / norm_len)
        # canonical convention: normal points upward (toward camera)
        if table_height > 0:
            table_normal = -table_normal
            table_height = -table_height

    # recompute inlier mask for this frame's point cloud
    signed_dists = pts @ table_normal - table_height
    table_mask   = np.abs(signed_dists) < plane_dist_threshold * 2
    table_pts    = pts[table_mask]
    n_table_pts  = int(table_mask.sum())

    # --- depth gate: remove background behind table in angled views --------
    table_z_max = float(table_pts[:, 2].max()) if len(table_pts) > 0 else 0.0
    obj_mask    = ~table_mask
    obj_pts     = _clean(pts[obj_mask])

    z_mask  = obj_pts[:, 2] < (table_z_max + depth_margin)
    obj_pts = obj_pts[z_mask]

    # --- height filter: keep points physically above the table -------------
    with np.errstate(divide='ignore', over='ignore', invalid='ignore'):
        heights = obj_pts @ table_normal - table_height
    finite = np.isfinite(heights)
    h_mask  = (finite &
               (heights > min_height_above_table) &
               (heights < max_height_above_table))
    obj_pts = obj_pts[h_mask]

    # --- XY radius crop: discard points far from camera origin -------------
    # Use camera projection origin [0,0] not table centroid — table centroid
    # can be far from objects near scene edges.  OCID cameras point along +Z
    # so XY=[0,0] is always in-view.
    xy_dist = np.linalg.norm(obj_pts[:, :2], axis=1)
    obj_pts = obj_pts[xy_dist < xy_radius]

    return obj_pts, table_normal, table_height, table_pts, n_table_pts


# ---------------------------------------------------------------------------
# Stage 3 helpers — splitting heuristics for touching / stacked objects
# ---------------------------------------------------------------------------

def _split_cluster(pts: np.ndarray, min_pts: int = 30,
                   saddle_depth: float = 0.25, n_bins: int = 40,
                   max_depth: int = 3) -> List[np.ndarray]:
    """Recursively split a cluster at horizontal density saddle points.

    Algorithm:
      1. PCA → project onto the dominant horizontal (XY) axis
      2. Build a 1-D density histogram along that axis
      3. Find local minima where density drops > saddle_depth relative to
         the lower of the two neighbouring peaks
      4. Split at the deepest saddle; recurse on each half

    saddle_depth: 0.25 means a 25 % drop from the local peak triggers a split.
    """
    if len(pts) < min_pts * 2 or max_depth == 0:
        return [pts]

    xy = pts[:, :2].astype(np.float64)
    centered = xy - xy.mean(axis=0)
    cov = (centered.T @ centered) / len(centered)
    eigvals, vecs = np.linalg.eigh(cov)

    if eigvals[-1] < 1e-6:
        return [pts]

    axis = vecs[:, -1]
    with np.errstate(divide='ignore', over='ignore', invalid='ignore'):
        proj = centered @ axis
    if not np.isfinite(proj).all():
        return [pts]

    hist, edges = np.histogram(proj, bins=n_bins)
    hist   = hist.astype(float)
    kernel = np.array([0.25, 0.5, 0.25])
    hist_s = np.convolve(hist, kernel, mode='same')

    best_split_proj = None
    best_saddle_score = 0.0

    for i in range(1, n_bins - 1):
        if hist_s[i] >= hist_s[i-1] or hist_s[i] >= hist_s[i+1]:
            continue
        left_peak  = hist_s[:i].max() if i > 0 else 0.0
        right_peak = hist_s[i+1:].max() if i < n_bins - 1 else 0.0
        lower_peak = min(left_peak, right_peak)
        if lower_peak < 1e-6:
            continue
        drop = 1.0 - hist_s[i] / lower_peak
        if drop > saddle_depth and drop > best_saddle_score:
            best_saddle_score = drop
            best_split_proj   = (edges[i] + edges[i+1]) / 2.0

    if best_split_proj is None:
        return [pts]

    left_pts  = pts[proj <= best_split_proj]
    right_pts = pts[proj >  best_split_proj]

    if len(left_pts) < min_pts or len(right_pts) < min_pts:
        return [pts]

    return (_split_cluster(left_pts,  min_pts, saddle_depth, n_bins, max_depth-1) +
            _split_cluster(right_pts, min_pts, saddle_depth, n_bins, max_depth-1))


def _split_cluster_vertical(pts: np.ndarray, min_pts: int = 30,
                             saddle_depth: float = 0.20,
                             n_bins: int = 30) -> List[np.ndarray]:
    """Split a cluster along the vertical (Y / height-above-table) axis.

    Only triggers when the vertical extent is >= 40 % of the horizontal
    extent, to avoid spurious splits on tall-but-single objects.
    Handles objects stacked on top of each other.
    """
    if len(pts) < min_pts * 2:
        return [pts]

    y_ext  = pts[:, 1].max() - pts[:, 1].min()
    xz_ext = max(pts[:, 0].max() - pts[:, 0].min(),
                 pts[:, 2].max() - pts[:, 2].min())

    if y_ext < 0.40 * xz_ext:
        return [pts]

    hist, edges = np.histogram(pts[:, 1], bins=n_bins)
    hist   = hist.astype(float)
    hist_s = np.convolve(hist, np.array([0.25, 0.5, 0.25]), mode='same')

    best_split, best_score = None, 0.0
    for i in range(1, n_bins - 1):
        if hist_s[i] >= hist_s[i-1] or hist_s[i] >= hist_s[i+1]:
            continue
        left_peak  = hist_s[:i].max() if i > 0 else 0.0
        right_peak = hist_s[i+1:].max() if i < n_bins-1 else 0.0
        lower_peak = min(left_peak, right_peak)
        if lower_peak < 1e-6:
            continue
        drop = 1.0 - hist_s[i] / lower_peak
        if drop > saddle_depth and drop > best_score:
            best_score = drop
            best_split = (edges[i] + edges[i+1]) / 2.0

    if best_split is None:
        return [pts]

    bottom = pts[pts[:, 1] <= best_split]
    top    = pts[pts[:, 1] >  best_split]

    if len(bottom) < min_pts or len(top) < min_pts:
        return [pts]

    return [bottom, top]


def _split_by_concavity(pts: np.ndarray,
                         min_pts: int = 30,
                         concavity_threshold: float = 0.015) -> List[np.ndarray]:
    """Split touching side-by-side objects via a 2-D neck in the XZ projection.

    Side-by-side objects create an hourglass cross-section.  The split fires
    at the narrowest column if it is < 30 % of the overall XZ width.
    """
    if len(pts) < min_pts * 2:
        return [pts]

    try:
        xz = pts[:, [0, 2]].astype(np.float64)
        centered = xz - xz.mean(axis=0)
        cov = (centered.T @ centered) / len(centered)
        eigvals, vecs = np.linalg.eigh(cov)

        if eigvals[-1] < 1e-8:
            return [pts]

        axis = vecs[:, -1]
        perp = vecs[:, 0]
        with np.errstate(divide='ignore', over='ignore', invalid='ignore'):
            proj  = centered @ axis
            width = centered @ perp
        if not (np.isfinite(proj).all() and np.isfinite(width).all()):
            return [pts]

        n_bins = 50
        edges  = np.linspace(proj.min(), proj.max(), n_bins + 1)
        spans, mids = [], []

        for i in range(n_bins):
            mask = (proj >= edges[i]) & (proj < edges[i+1])
            cnt  = mask.sum()
            spans.append(width[mask].max() - width[mask].min() if cnt >= 3 else 0.0)
            mids.append((edges[i] + edges[i+1]) / 2)

        spans = np.array(spans)
        overall_width = spans.max()

        if overall_width < 1e-6:
            return [pts]

        interior  = range(3, n_bins - 3)
        min_idx   = min(interior, key=lambda i: spans[i])
        min_width = spans[min_idx]

        if min_width > overall_width * 0.30:
            return [pts]

        split_pos = mids[min_idx]
        if min(int((proj <= split_pos).sum()),
               int((proj > split_pos).sum())) < len(pts) * 0.20:
            return [pts]

        left  = pts[proj <= split_pos]
        right = pts[proj >  split_pos]

        if len(left) < min_pts or len(right) < min_pts:
            return [pts]

        return [left, right]

    except Exception:
        return [pts]


def _split_by_height_layers(pts: np.ndarray,
                             min_pts: int = 30,
                             gap_threshold: float = 0.008) -> List[np.ndarray]:
    """Recursively split clusters at genuine near-empty height gaps.

    Finds height bands (Y axis) where point density drops to ≤ 0.5 % of
    total and the gap is at least gap_threshold metres wide.  More reliable
    than the saddle splitter for stacked objects.
    """
    if len(pts) < min_pts * 2:
        return [pts]

    y       = pts[:, 1]
    y_range = y.max() - y.min()

    if y_range < gap_threshold * 2:
        return [pts]

    n_bins = max(20, int(y_range / 0.003))
    hist, edges = np.histogram(y, bins=n_bins)

    empty          = hist <= max(1, len(pts) * 0.005)
    best_gap_mid   = None
    best_gap_width = 0.0

    i = 0
    while i < n_bins:
        if empty[i]:
            j = i
            while j < n_bins and empty[j]:
                j += 1
            gap_w = (edges[j] if j < n_bins else edges[-1]) - edges[i]
            if gap_w >= gap_threshold and gap_w > best_gap_width:
                best_gap_width = gap_w
                best_gap_mid   = (edges[i] + (edges[j] if j < n_bins else edges[-1])) / 2
            i = j
        else:
            i += 1

    if best_gap_mid is None:
        return [pts]

    bottom = pts[y <= best_gap_mid]
    top    = pts[y >  best_gap_mid]

    if len(bottom) < min_pts or len(top) < min_pts:
        return [pts]

    return (_split_by_height_layers(bottom, min_pts, gap_threshold) +
            _split_by_height_layers(top,    min_pts, gap_threshold))


# ---------------------------------------------------------------------------
# Stage 3 — Instance segmentation
# ---------------------------------------------------------------------------

def adaptive_cluster_eps(
    pts: np.ndarray,
    multiplier: float = 3.0,
    eps_max: float = None,
) -> float:
    """Estimate a good DBSCAN neighbourhood radius from nearest-neighbour distances.

    Subsamples up to 1 000 points, computes each point's distance to its
    nearest neighbour, and returns ``multiplier × median(nn_distances)``.

    This adapts automatically to the point density of the cloud, which is
    critical when the same pipeline is used on both normalised single-object
    clouds (ShapeNet, ~0.1–0.2 m scale) and real-world multi-object scenes
    (RGB-D Scenes v2, distances in metres).

    Parameters
    ----------
    pts        : (N, 3) point cloud
    multiplier : scale factor applied to the median NN distance.
                 3.0 is a reasonable default for most scenes; use 2.0 for
                 denser clouds where objects are closely spaced.
    eps_max    : optional hard upper cap (metres).  When the computed eps
                 would exceed this value it is clamped to eps_max.  Useful
                 as a safety net for scenes where table removal leaves too
                 few points and the NN distances blow up.

    Returns
    -------
    Estimated eps in the same length units as pts, clamped to eps_max.
    """
    from sklearn.neighbors import NearestNeighbors
    n_sub = min(1000, len(pts))
    idx   = np.random.choice(len(pts), n_sub, replace=False)
    sub   = pts[idx].astype(np.float64)
    nbrs  = NearestNeighbors(n_neighbors=2, algorithm="ball_tree").fit(sub)
    dists, _ = nbrs.kneighbors(sub)
    eps = float(multiplier * np.median(dists[:, 1]))
    if eps_max is not None and eps > eps_max:
        _log.warning(
            "adaptive_cluster_eps: computed eps %.4f m > eps_max %.4f m; "
            "clamping to eps_max.",
            eps, eps_max,
        )
        eps = float(eps_max)
    return eps


def segment_instances(
    obj_pts: np.ndarray,
    obj_col: Optional[np.ndarray] = None,
    cluster_eps:        float = 0.018,
    cluster_min_points: int   = 20,
    split_min_points:   int   = 600,
    saddle_depth:       float = 0.25,
    split_n_bins:       int   = 40,
    concavity_threshold:  float = 0.015,
    height_gap_threshold: float = 0.008,
    rgb_weight:           float = 0.0,
    adaptive_eps:         bool  = False,
    eps_multiplier:       float = 3.0,
    eps_max:              Optional[float] = None,
) -> List[np.ndarray]:
    """Cluster foreground points into per-object groups.

    Applies DBSCAN followed by four splitting passes to separate objects
    that are touching horizontally, stacked vertically, or joined at a neck.

    Parameters
    ----------
    obj_pts               : (N, 3) foreground points above the table
    obj_col               : (N, 3) float32 RGB colours in [0, 1], optional
    cluster_eps           : DBSCAN neighbourhood radius (metres); overridden
                            when adaptive_eps=True
    cluster_min_points    : minimum DBSCAN core-point neighbourhood size
    split_min_points      : cluster size threshold to trigger splitting passes
    saddle_depth          : relative density drop to trigger a saddle split
    split_n_bins          : histogram bins for saddle detection
    concavity_threshold   : width ratio threshold for neck split
    height_gap_threshold  : minimum empty-band height to trigger layer split
    rgb_weight            : colour feature weight for colour-aware DBSCAN
                            (0 = geometry-only; useful for strongly coloured scenes)
    adaptive_eps          : if True, compute cluster_eps automatically via
                            adaptive_cluster_eps() and ignore the cluster_eps arg
    eps_multiplier        : multiplier passed to adaptive_cluster_eps(); only
                            used when adaptive_eps=True

    Returns
    -------
    List of (N_i, 3) point arrays, one per detected instance.
    """
    if len(obj_pts) == 0:
        return []

    if adaptive_eps:
        cluster_eps = adaptive_cluster_eps(obj_pts, multiplier=eps_multiplier, eps_max=eps_max)
        _log.info("segment_instances: adaptive cluster_eps = %.4f m", cluster_eps)

    # DBSCAN clustering
    if obj_col is not None and rgb_weight > 0:
        from sklearn.cluster import DBSCAN as skDBSCAN
        feats  = np.hstack([obj_pts, obj_col * rgb_weight]).astype(np.float32)
        labels = skDBSCAN(eps=cluster_eps, min_samples=cluster_min_points,
                          algorithm='ball_tree', n_jobs=1).fit(feats).labels_
    else:
        pcd_obj = _to_o3d(obj_pts)
        labels  = np.array(pcd_obj.cluster_dbscan(
            eps=cluster_eps,
            min_points=cluster_min_points,
            print_progress=False,
        ))

    # Pass 1 — horizontal saddle split (XY PCA projection)
    raw_clusters: List[np.ndarray] = []
    for lbl in sorted(set(labels) - {-1}):
        cluster = obj_pts[labels == lbl]
        if len(cluster) >= split_min_points:
            raw_clusters.extend(_split_cluster(
                cluster, min_pts=cluster_min_points,
                saddle_depth=saddle_depth, n_bins=split_n_bins))
        else:
            raw_clusters.append(cluster)

    # Pass 2 — vertical saddle split (stacked objects)
    after_vertical: List[np.ndarray] = []
    for cluster in raw_clusters:
        if len(cluster) >= split_min_points:
            after_vertical.extend(_split_cluster_vertical(
                cluster, min_pts=cluster_min_points, saddle_depth=saddle_depth))
        else:
            after_vertical.append(cluster)

    # Pass 3A — normal discontinuity (disabled: segfault on macOS after DBSCAN)
    after_normal = after_vertical

    # Pass 3B — 2-D concavity / narrow neck
    after_concavity: List[np.ndarray] = []
    for cluster in after_normal:
        if len(cluster) >= split_min_points:
            after_concavity.extend(_split_by_concavity(
                cluster, min_pts=cluster_min_points,
                concavity_threshold=concavity_threshold))
        else:
            after_concavity.append(cluster)

    # Pass 3C — height-layer gaps
    after_height: List[np.ndarray] = []
    for cluster in after_concavity:
        after_height.extend(_split_by_height_layers(
            cluster, min_pts=cluster_min_points,
            gap_threshold=height_gap_threshold))

    return after_height


def segment_instances_dual(
    obj_pts: np.ndarray,
    cluster_eps:        float = 0.018,
    cluster_min_points: int   = 20,
    k_neighbours:       int   = 20,
    iou_dedup_threshold: float = 0.5,
    adaptive_eps:       bool  = False,
    eps_multiplier:     float = 3.0,
    eps_max:            Optional[float] = None,
) -> List[np.ndarray]:
    """PointGroup dual-set DBSCAN segmentation (Jiang et al. CVPR 2020).

    Produces two independent cluster sets and returns their union with IoU
    deduplication to reduce over- and under-segmentation:

      C_p — standard DBSCAN on raw XYZ coordinates.
      C_q — DBSCAN on centroid-offset-shifted coordinates:
              for each point p_i, compute the mean coordinate of its k
              nearest neighbours; the shifted point is
              p_i + (centroid_of_kNN(p_i) − p_i).
              This pulls scattered points toward object centres, making
              DBSCAN more robust to non-uniform density.

    Deduplication: any cluster in C_q that overlaps (point-set IoU >
    iou_dedup_threshold) with an already-kept cluster from C_p is dropped,
    preventing the same object from appearing twice.

    Note: unlike the original PointGroup which uses a learned offset
    prediction network, this implementation uses a simple geometric
    k-NN centroid estimate.  It is fast (CPU, sklearn), parameter-free
    beyond k and the dedup threshold, and does not require normals.

    Parameters
    ----------
    obj_pts               : (N, 3) foreground points above the table.
    cluster_eps           : DBSCAN neighbourhood radius (metres).
    cluster_min_points    : DBSCAN min_samples.
    k_neighbours          : k for the centroid-offset k-NN step.
    iou_dedup_threshold   : minimum point-set IoU to suppress a C_q duplicate.
    adaptive_eps          : compute eps automatically (same as segment_instances).
    eps_multiplier        : multiplier for adaptive_cluster_eps().
    eps_max               : upper bound for adaptive eps.

    Returns
    -------
    List of (N_i, 3) arrays, one per detected instance, deduplicated.
    """
    from sklearn.cluster import DBSCAN as _DBSCAN
    from sklearn.neighbors import NearestNeighbors

    if len(obj_pts) == 0:
        return []

    if adaptive_eps:
        cluster_eps = adaptive_cluster_eps(obj_pts, multiplier=eps_multiplier,
                                           eps_max=eps_max)
        _log.info("segment_instances_dual: adaptive cluster_eps = %.4f m", cluster_eps)

    pts = obj_pts.astype(np.float64)

    # ── Set C_p: standard DBSCAN on XYZ ──────────────────────────────────────
    labels_p = _DBSCAN(
        eps=cluster_eps, min_samples=cluster_min_points,
        algorithm='ball_tree', n_jobs=1,
    ).fit(pts).labels_

    clusters_p: List[np.ndarray] = [
        pts[labels_p == lbl]
        for lbl in sorted(set(labels_p) - {-1})
    ]

    # ── Centroid-offset shift ─────────────────────────────────────────────────
    k = min(k_neighbours, len(pts) - 1)
    if k < 1:
        return clusters_p   # degenerate cloud

    nbrs = NearestNeighbors(n_neighbors=k + 1, algorithm='ball_tree', n_jobs=1)
    nbrs.fit(pts)
    _, indices = nbrs.kneighbors(pts)          # (N, k+1); col 0 is the point itself
    # centroid of the k nearest neighbours (excluding self)
    knn_centroids = pts[indices[:, 1:]].mean(axis=1)   # (N, 3)
    # shift: each point moves toward the centroid of its neighbours
    pts_shifted = pts + (knn_centroids - pts)          # (N, 3)

    # ── Set C_q: DBSCAN on shifted coordinates ───────────────────────────────
    labels_q = _DBSCAN(
        eps=cluster_eps, min_samples=cluster_min_points,
        algorithm='ball_tree', n_jobs=1,
    ).fit(pts_shifted).labels_

    clusters_q: List[np.ndarray] = [
        pts[labels_q == lbl]                  # recover original coords, not shifted
        for lbl in sorted(set(labels_q) - {-1})
    ]

    # ── Union with IoU deduplication ─────────────────────────────────────────
    # Build index sets for fast IoU: map each point to its position in pts.
    # We use a dict from tuple → row index to identify point overlaps.
    def _cluster_to_set(cluster: np.ndarray) -> set:
        """Return frozenset of row indices of cluster points in pts."""
        # Use rounded coordinates as keys (handles float equality)
        key_map = {tuple(np.round(pts[i], 8)): i for i in range(len(pts))}
        return frozenset(key_map[tuple(np.round(p, 8))] for p in cluster
                         if tuple(np.round(p, 8)) in key_map)

    # Build index sets for C_p
    sets_p = [_cluster_to_set(c) for c in clusters_p]

    result_clusters = list(clusters_p)
    result_sets     = list(sets_p)

    for c_q, s_q in zip(clusters_q, [_cluster_to_set(c) for c in clusters_q]):
        duplicate = False
        for s_kept in result_sets:
            if len(s_kept) == 0 or len(s_q) == 0:
                continue
            inter = len(s_kept & s_q)
            union = len(s_kept | s_q)
            if union > 0 and inter / union > iou_dedup_threshold:
                duplicate = True
                break
        if not duplicate:
            result_clusters.append(c_q)
            result_sets.append(s_q)

    return result_clusters


def merge_nearby_segments(
    segments: List[np.ndarray],
    merge_dist: float = 0.15,
) -> List[np.ndarray]:
    """Merge segments whose centroids are within *merge_dist* of each other.

    Addresses over-segmentation where DBSCAN splits a single object's point
    cloud into 2–3 fragments because FPS subsampling leaves spatial gaps within
    a single object's surface.

    Algorithm
    ---------
    1. Compute the centroid of each segment.
    2. Build a pairwise centroid distance matrix.
    3. Greedy merge starting from the largest segment: absorb all segments
       within *merge_dist* of the current segment's centroid, update the
       centroid, and repeat until no further merges occur.
    4. Return the merged segment list (largest first).

    Parameters
    ----------
    segments   : list of (N_i, 3) point arrays from segment_instances[_dual].
    merge_dist : centroid-to-centroid distance threshold (metres).
                 0.15 m works well for tabletop scenes; increase if fragments
                 of the same object are further apart.

    Returns
    -------
    List of (N_i, 3) arrays, merged and sorted largest-first.
    """
    if len(segments) <= 1:
        return list(segments)

    # Sort largest-first so greedy pass starts from the most reliable centroid.
    order    = sorted(range(len(segments)), key=lambda i: -len(segments[i]))
    segs     = [segments[i] for i in order]
    merged   = [False] * len(segs)
    result: List[np.ndarray] = []

    centroids = np.array([s.mean(axis=0) for s in segs])   # (K, 3)

    for i in range(len(segs)):
        if merged[i]:
            continue
        # Collect all unmerged segments within merge_dist of segment i.
        group = [i]
        merged[i] = True
        # Recompute centroid after each absorption until stable.
        changed = True
        while changed:
            changed = False
            cur_centroid = np.vstack([segs[j] for j in group]).mean(axis=0)
            for j in range(len(segs)):
                if merged[j]:
                    continue
                if np.linalg.norm(centroids[j] - cur_centroid) <= merge_dist:
                    group.append(j)
                    merged[j] = True
                    changed    = True
        result.append(np.vstack([segs[j] for j in group]))

    return result


# ---------------------------------------------------------------------------
# Stage 4 — Shape classification / SQ hint
# ---------------------------------------------------------------------------

def classify_shape_hint(cluster: np.ndarray) -> Tuple[str, float]:
    """Extent-ratio heuristic for SQ initialisation hint.

    Does NOT require a trained classifier — works on partial views.
    Returns (shape_type, confidence) where shape_type is used only as an
    LM warm-start; the authoritative shape_type comes from the fitted exponents.
    """
    if len(cluster) < 10:
        return "Other", 0.0
    lo, hi   = cluster.min(0), cluster.max(0)
    exts     = np.sort(hi - lo)              # [min, mid, max]
    r_flat   = exts[0] / (exts[2] + 1e-6)   # low  → flat (bowl/cap)
    r_elon   = exts[1] / (exts[2] + 1e-6)   # low  → elongated (box)
    xy_ext   = hi[:2] - lo[:2]
    r_circ   = min(xy_ext) / (max(xy_ext) + 1e-6)  # high → circular XY

    if r_flat < 0.25:                          # very flat → bowl/disc
        return "Ellipsoid", 0.6
    elif r_flat > 0.45 and r_circ > 0.65:     # squat + circular → can
        return "Cylinder", 0.6
    elif r_elon < 0.45:                        # elongated one axis → box
        return "Cuboid", 0.55
    else:
        return "Other", 0.4


# ---------------------------------------------------------------------------
# Stage 4b — Superquadric fitting (bridge to superquadric.py / superdec_fitter.py)
# ---------------------------------------------------------------------------

def fit_superquadrics(
    segments: list,
    fitter=None,
    l2_threshold: float = 0.008,
    max_primitives: int = 3,
) -> list:
    """Fit superquadric primitives to each segmented object.

    This is Stage 4b of the pipeline — it bridges the instance segmentation
    output (ObjectSegment list) to the SQ fitting layer (SuperquadricFitter
    or SuperdecFitter).

    Parameters
    ----------
    segments : List[ObjectSegment]
        Output of TabletopPerception.run() or segment_instances().
    fitter : SuperquadricFitter or SuperdecFitter, optional
        The fitter instance to use.  If None, creates a default
        SuperquadricFitter (LM optimisation).  Pass a SuperdecFitter instance
        to use the fine-tuned neural fitter instead.
    l2_threshold : float
        Chamfer L2 threshold for adaptive splitting (LM fitter only).
    max_primitives : int
        Maximum primitives per object (LM fitter only).

    Returns
    -------
    List[MultiSQFit] — one MultiSQFit per ObjectSegment, same order.
    The list feeds directly into:
        • fits_to_curobo_obstacles(flat_fits) — collision avoidance
        • Scene.from_fits(flat_fits)           — path planning SDF
        • GraspSelector.grasp_candidates()     — grasp planning

    Example
    -------
    ::
        result  = TabletopPerception().run(pts)
        sq_fits = fit_superquadrics(result.objects)
        flat    = [sq for m in sq_fits for sq in m.primitives]
        scene   = Scene.from_fits(flat)
        sd      = scene.get_signed_distance(query_pts)
    """
    if fitter is None:
        try:
            from .superquadric import SuperquadricFitter
        except ImportError:
            from superquadric import SuperquadricFitter
        fitter = SuperquadricFitter(n_restarts=3, n_lm_rounds=15, subsample=512)

    results = []
    for seg in segments:
        multi = fitter.fit_adaptive(
            seg.points,
            shape_hint=seg.shape_type,
            l2_threshold=l2_threshold,
            max_primitives=max_primitives,
        )
        # propagate shape_type / shape_conf from perception onto each primitive
        for prim in multi.primitives:
            if not prim.shape_type or prim.shape_type == "Other":
                prim.shape_type = seg.shape_type
            prim.shape_conf = seg.shape_conf
        results.append(multi)

    return results


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

class TabletopPerception:
    """Full tabletop RGB-D perception pipeline.

    Orchestrates the four stages (preprocessing → table removal → instance
    segmentation → shape classification) and returns a PerceptionResult
    containing ObjectSegment objects ready for superquadric fitting.

    Parameters
    ----------
    voxel_size              : voxel down-sample resolution (metres)
    nb_neighbors / std_ratio: statistical outlier removal parameters
    plane_dist_threshold    : RANSAC inlier distance for table fitting
    plane_ransac_n          : minimum RANSAC hypothesis size
    plane_num_iterations    : RANSAC iterations
    min_height_above_table  : lower height cutoff above table (metres)
    max_height_above_table  : upper height cutoff (metres)
    depth_margin            : camera-Z slack beyond table back edge (metres).
                              Removes background walls in angled views.
    cluster_eps             : DBSCAN eps (metres)
    cluster_min_points      : DBSCAN min_samples
    min_object_points       : reject segments smaller than this
    max_object_points       : reject segments larger than this
    min_object_extent       : reject segments whose longest bbox side < this
    max_object_extent       : reject segments whose longest bbox side > this
    xy_radius               : discard points > this distance from XY origin
    split_min_points        : trigger splitting passes only above this size
    saddle_depth            : relative saddle depth for horizontal/vertical split
    split_n_bins            : histogram bins for saddle detection
    concavity_threshold     : neck-width ratio for concavity split
    height_gap_threshold    : minimum gap height for layer split (metres)
    geo_min_points          : minimum size to run geometric splits (3B/3C)
    rgb_weight              : colour feature weight for colour-aware DBSCAN
    """

    def __init__(
        self,
        voxel_size:              float = 0.005,
        nb_neighbors:            int   = 20,
        std_ratio:               float = 2.0,
        plane_dist_threshold:    float = 0.012,
        plane_ransac_n:          int   = 3,
        plane_num_iterations:    int   = 1000,
        min_height_above_table:  float = 0.005,
        max_height_above_table:  float = 0.25,
        depth_margin:            float = 0.25,
        cluster_eps:             float = 0.018,
        cluster_min_points:      int   = 20,
        min_object_points:       int   = 50,
        max_object_points:       int   = 5000,
        min_object_extent:       float = 0.02,
        max_object_extent:       float = 0.40,
        xy_radius:               float = 0.80,
        split_min_points:        int   = 600,
        saddle_depth:            float = 0.25,
        split_n_bins:            int   = 40,
        normal_angle_threshold:  float = 60.0,
        concavity_threshold:     float = 0.015,
        height_gap_threshold:    float = 0.008,
        geo_min_points:          int   = 800,
        rgb_weight:              float = 0.0,
    ):
        self.voxel_size             = voxel_size
        self.nb_neighbors           = nb_neighbors
        self.std_ratio              = std_ratio
        self.plane_dist_threshold   = plane_dist_threshold
        self.plane_ransac_n         = plane_ransac_n
        self.plane_num_iterations   = plane_num_iterations
        self.min_height_above_table = min_height_above_table
        self.max_height_above_table = max_height_above_table
        self.depth_margin           = depth_margin
        self.cluster_eps            = cluster_eps
        self.cluster_min_points     = cluster_min_points
        self.min_object_points      = min_object_points
        self.max_object_points      = max_object_points
        self.min_object_extent      = min_object_extent
        self.max_object_extent      = max_object_extent
        self.xy_radius              = xy_radius
        self.split_min_points       = split_min_points
        self.saddle_depth           = saddle_depth
        self.split_n_bins           = split_n_bins
        self.normal_angle_threshold = normal_angle_threshold
        self.concavity_threshold    = concavity_threshold
        self.height_gap_threshold   = height_gap_threshold
        self.geo_min_points         = geo_min_points
        self.rgb_weight             = rgb_weight

    # keep old name as alias so test scripts don't break
    def _classify(self, cluster: np.ndarray) -> Tuple[str, float]:
        return self._shape_hint(cluster)

    def _shape_hint(self, cluster: np.ndarray) -> Tuple[str, float]:
        """Module-level classify_shape_hint(), bound to this instance."""
        return classify_shape_hint(cluster)

    def _valid(self, pts: np.ndarray) -> bool:
        """Return True if a cluster passes size and extent filters."""
        if not (self.min_object_points <= len(pts) <= self.max_object_points):
            return False
        extent = pts.max(axis=0) - pts.min(axis=0)
        return self.min_object_extent <= extent.max() <= self.max_object_extent

    def run(self, points: np.ndarray,
            rgb:        Optional[np.ndarray] = None,
            plane_hint: Optional[tuple]      = None) -> PerceptionResult:
        """Run the full perception pipeline on a point cloud.

        Parameters
        ----------
        points     : (N, 3) XYZ point cloud in metres
        rgb        : (N, 3) float32 / uint8 colours, same row order as points
        plane_hint : (normal, height) from a previous run on the same sequence.
                     If provided, skips RANSAC (saves ~200 ms/frame).

        Returns
        -------
        PerceptionResult with ObjectSegment list and table geometry.
        """
        result = PerceptionResult(n_points_input=len(points))

        # ── preprocess ──────────────────────────────────────────────────────
        pts = _clean(points)
        if len(pts) < 100:
            return result
        result.n_points_input = len(pts)

        good_mask = np.isfinite(points).all(axis=1) & (np.abs(points) < 4.0).all(axis=1)
        if rgb is not None and len(rgb) == len(points):
            col = np.array(rgb, dtype=np.float32)[good_mask]
            if col.max() > 1.5:
                col = col / 255.0
            col = np.clip(col, 0.0, 1.0)
        else:
            col = None

        # voxel down-sample
        pcd = _to_o3d(pts)
        if col is not None:
            pcd.colors = o3d.utility.Vector3dVector(col.astype(np.float64))
        pcd = pcd.voxel_down_sample(self.voxel_size)

        # statistical outlier removal
        pcd, _ = pcd.remove_statistical_outlier(self.nb_neighbors, self.std_ratio)
        pts    = _clean(np.asarray(pcd.points))
        col_ds = np.asarray(pcd.colors).astype(np.float32) if (col is not None and pcd.has_colors()) else None
        if len(pts) < 50:
            return result
        result.n_points_after_denoise = len(pts)

        # ── Stage 2: table / background removal ─────────────────────────────
        try:
            obj_pts, table_normal, table_height, table_pts, n_table_pts = remove_table(
                pts,
                plane_dist_threshold   = self.plane_dist_threshold,
                plane_ransac_n         = self.plane_ransac_n,
                plane_num_iterations   = self.plane_num_iterations,
                min_height_above_table = self.min_height_above_table,
                max_height_above_table = self.max_height_above_table,
                depth_margin           = self.depth_margin,
                xy_radius              = self.xy_radius,
                plane_hint             = plane_hint,
            )
        except Exception as e:
            logging.warning("[pipeline] table removal failed: %s", e)
            return result

        result.table_normal   = table_normal
        result.table_height   = table_height
        result.n_points_table = n_table_pts

        if len(obj_pts) < self.cluster_min_points:
            return result

        # propagate colour mask through depth + height filters if present
        # (only needed for colour-aware DBSCAN — geometry path ignores col_ds)
        obj_col = None
        if col_ds is not None and self.rgb_weight > 0:
            # reconstruct which rows of pts survived into obj_pts;
            # use a KD-tree match since obj_pts went through _clean()
            try:
                import open3d as _o3d
                pcd_all = _to_o3d(pts)
                pcd_obj = _to_o3d(obj_pts)
                tree = _o3d.geometry.KDTreeFlann(pcd_all)
                col_obj = []
                for p in obj_pts:
                    _, idx, _ = tree.search_knn_vector_3d(p, 1)
                    col_obj.append(col_ds[idx[0]])
                obj_col = np.array(col_obj, dtype=np.float32)
            except Exception:
                logging.debug("[pipeline] colour KD-tree lookup failed; running geometry-only DBSCAN")
                obj_col = None

        # ── Stage 3: instance segmentation ──────────────────────────────────
        clusters = segment_instances(
            obj_pts,
            obj_col              = obj_col,
            cluster_eps          = self.cluster_eps,
            cluster_min_points   = self.cluster_min_points,
            split_min_points     = self.split_min_points,
            saddle_depth         = self.saddle_depth,
            split_n_bins         = self.split_n_bins,
            concavity_threshold  = self.concavity_threshold,
            height_gap_threshold = self.height_gap_threshold,
            rgb_weight           = self.rgb_weight,
        )

        # ── Stage 4: shape hint + validity filter ────────────────────────────
        obj_id = 0
        for cluster in clusters:
            if not self._valid(cluster):
                continue
            shape_type, shape_conf = self._shape_hint(cluster)
            result.objects.append(ObjectSegment(
                id=obj_id,
                points=cluster.astype(np.float32),
                centroid=cluster.mean(axis=0),
                bbox_min=cluster.min(axis=0),
                bbox_max=cluster.max(axis=0),
                shape_type=shape_type,
                shape_conf=shape_conf,
            ))
            obj_id += 1

        return result

    def run_from_file(self, path: str) -> PerceptionResult:
        """Convenience wrapper — load a PLY file and run the pipeline."""
        pcd = o3d.io.read_point_cloud(path)
        return self.run(np.asarray(pcd.points))


# ---------------------------------------------------------------------------
# Minimum foreground points threshold (shared by pipeline entry points)
# ---------------------------------------------------------------------------
_MIN_FOREGROUND_PTS: int = 50


# ---------------------------------------------------------------------------
# Single-frame pipeline entry point
# ---------------------------------------------------------------------------

def single_frame_pipeline(
    depth: np.ndarray,
    K: np.ndarray,
    extrinsic: Optional[np.ndarray] = None,
    fitter: object = 'superdec',
    timer: Optional[PerceptionTimer] = None,
) -> Tuple[list, SQWorldModel, dict]:
    """Full perception pipeline from a single depth frame.

    This is the primary entry point for the Franka Panda deployment — it
    runs every stage in sequence and returns both the raw fits and a compact
    world model suitable for downstream planning.

    Pipeline order::

        depth image
          → pointcloud_from_depth()     Stage 1 — unproject to (N,3) cloud
          → remove_table()              Stage 2 — RANSAC plane removal
          → segment_instances_dual()    Stage 3 — PointGroup dual-set DBSCAN
          → preprocess_pointcloud()     Stage 4a — outlier removal + FPS
          → fitter.fit_adaptive()       Stage 4b — SQ fitting per segment
          → postprocess_fits()          Stage 5  — invert preprocessing

    Table removal gracefully falls back to the full cloud when RANSAC fails
    (e.g., flat surfaces absent in the scene or too few foreground points).

    Parameters
    ----------
    depth    : (H, W) float32 depth image in metres; 0 = invalid pixel.
    K        : (3, 3) camera intrinsic matrix ``[[fx,0,cx],[0,fy,cy],[0,0,1]]``.
    extrinsic: (4, 4) camera-to-world SE(3) transform.  ``None`` → camera frame.
    fitter   : ``'superdec'``, ``'lm'``, or a **pre-built fitter instance**.
               Passing a pre-built instance avoids model-loading overhead on
               every call (recommended for repeated inference on the robot).
               When ``'superdec'`` is requested but no GPU / checkpoint is
               available the function transparently falls back to the LM fitter.
    timer    : optional :class:`PerceptionTimer`; a fresh one is created if
               ``None``.

    Returns
    -------
    fits        : List[MultiSQFit] — one per detected object, in world frame.
    world_model : :class:`SQWorldModel` wrapping *fits*.
    timing_dict : ``{stage_name: elapsed_seconds}`` profiling dict.
    """
    if timer is None:
        timer = PerceptionTimer()

    # ── Stage 1: unproject ────────────────────────────────────────────────────
    timer.start('unproject')
    pts = pointcloud_from_depth(depth, K, extrinsic=extrinsic)
    timer.stop('unproject')

    if len(pts) < _MIN_FOREGROUND_PTS:
        _log.warning("single_frame_pipeline: only %d valid depth pixels", len(pts))
        return [], SQWorldModel([]), timer.to_dict()

    # ── Stage 2: table removal (graceful fallback to full cloud) ──────────────
    timer.start('table')
    tbl_normal: Optional[np.ndarray] = None
    tbl_height: float = 0.0
    try:
        obj_pts, tbl_normal, tbl_height, _, _ = remove_table(
            pts.astype(np.float64)
        )
        if len(obj_pts) < _MIN_FOREGROUND_PTS:
            raise ValueError(
                f"only {len(obj_pts)} foreground pts after table removal"
            )
    except Exception as exc:
        _log.debug(
            "single_frame_pipeline: table removal failed (%s); using full cloud", exc
        )
        obj_pts = pts.astype(np.float64)
        tbl_normal = None
        tbl_height = 0.0
    timer.stop('table')

    # ── Stage 3: dual-set DBSCAN segmentation ────────────────────────────────
    timer.start('segment')
    segments = segment_instances_dual(obj_pts, adaptive_eps=True)
    if not segments:
        segments = [obj_pts]
    timer.stop('segment')

    # ── Resolve fitter ────────────────────────────────────────────────────────
    if isinstance(fitter, str):
        if fitter == 'lm':
            try:
                from .superquadric import SuperquadricFitter as _SQF
            except ImportError:
                from superquadric import SuperquadricFitter as _SQF
            _fitter = _SQF(n_restarts=3, n_lm_rounds=15, subsample=512)
            _use_superdec = False
        else:   # 'superdec' — try GPU path, fall back to LM on failure
            try:
                import torch as _torch
                if not _torch.cuda.is_available():
                    raise RuntimeError("no CUDA GPU available")
                from superdec_fitter import SuperdecFitter as _SDF
                try:
                    from .paths import superdec_dir, superdec_checkpoint_dir
                except ImportError:
                    from paths import superdec_dir, superdec_checkpoint_dir
                _fitter = _SDF(superdec_dir=superdec_dir(),
                               checkpoint_dir=superdec_checkpoint_dir())
                _use_superdec = True
            except Exception as exc:
                _log.warning(
                    "single_frame_pipeline: SuperDec unavailable (%s); "
                    "falling back to LM fitter.", exc
                )
                try:
                    from .superquadric import SuperquadricFitter as _SQF
                except ImportError:
                    from superquadric import SuperquadricFitter as _SQF
                _fitter = _SQF(n_restarts=3, n_lm_rounds=15, subsample=512)
                _use_superdec = False
    else:
        _fitter = fitter
        _use_superdec = 'SuperdecFitter' in type(_fitter).__name__

    # ── Stages 4–5: preprocess → fit → postprocess (per segment) ─────────────
    timer.start('fit')
    sq_fits_world: list = []

    # SuperDec was trained on ShapeNet (Y-up). The tabletop dataloader rotates
    # z-up scenes by -90° around X before training; inference must mirror this.
    # R_z2y: camera frame (z-forward, y-down) → SuperDec canonical (y-up).
    # R_y2z: inverse, applied to each predicted primitive to restore camera frame.
    if _use_superdec:
        import scipy.spatial.transform as _sst
        _R_z2y = np.array([[1, 0, 0],
                           [0, 0, 1],
                           [0,-1, 0]], dtype=np.float64)
        _R_y2z = _R_z2y.T

    print(f"[sfp-dbg] {len(segments)} segment(s); "
          f"sizes={[len(s) for s in segments]}")

    for seg_pts in segments:
        if len(seg_pts) < 20:
            print(f"[sfp-dbg] skipping segment with only {len(seg_pts)} pts")
            continue
        try:
            # Center at the segment centroid so |coord| << 2 for SuperDec's
            # input contract; centroid is added back after postprocess_fits.
            centroid = seg_pts.mean(axis=0)
            pts_centered = seg_pts.astype(np.float64) - centroid

            pts_pre, _, meta = preprocess_pointcloud(
                pts_centered,
                for_superdec=_use_superdec,
            )

            print(f"[sfp-dbg] seg: {len(seg_pts)} pts → "
                  f"coord_max={np.abs(pts_pre).max():.3f}  "
                  f"std={pts_pre.std():.4f}")

            if _use_superdec:
                pts_in = pts_pre @ _R_z2y.T   # z-up → y-up before SuperDec
            else:
                pts_in = pts_pre

            multi = _fitter.fit_adaptive(pts_in)

            if _use_superdec:
                # Invert y-up → z-up on each predicted primitive pose.
                for prim in multi.primitives:
                    t_z = _R_y2z @ np.array([prim.tx, prim.ty, prim.tz])
                    prim.tx, prim.ty, prim.tz = float(t_z[0]), float(t_z[1]), float(t_z[2])
                    R_yup = _sst.Rotation.from_euler(
                        "xyz", [prim.rx, prim.ry, prim.rz]).as_matrix()
                    rx, ry, rz = _sst.Rotation.from_matrix(
                        _R_y2z @ R_yup).as_euler("xyz")
                    prim.rx, prim.ry, prim.rz = float(rx), float(ry), float(rz)

            print(f"[sfp-dbg] {len(multi.primitives)} primitive(s) passed "
                  f"exist_threshold")

            # Invert preprocessing transforms (scale=1, table rotation, PCA).
            postprocessed = postprocess_fits([multi], meta)
            # Re-add centroid that was subtracted before preprocess_pointcloud
            # (meta['centroid'] is always zeros — centroid is not tracked there).
            for pp_multi in postprocessed:
                for prim in pp_multi.primitives:
                    prim.tx += float(centroid[0])
                    prim.ty += float(centroid[1])
                    prim.tz += float(centroid[2])
            sq_fits_world.extend(postprocessed)

        except Exception as exc:
            _log.debug(
                "single_frame_pipeline: segment fit failed (%s); skipping", exc
            )
            print(f"[sfp-dbg] segment fit raised: {exc}")
    timer.stop('fit')

    world_model = SQWorldModel(sq_fits_world)
    return sq_fits_world, world_model, timer.to_dict()


# ── Two-stage pipeline ────────────────────────────────────────────────────────
# Stage 1: image(s) → world-frame point cloud
# Stage 2: world point cloud → superquadrics
#
# Set MULTIVIEW = True  to accumulate all frames in Stage 1.
# Set MULTIVIEW = False to process only the first frame (single-view fallback).
# Both produce an identical world-frame output contract.

MULTIVIEW: bool = False   # True = all frames; False = first frame only


@dataclass
class Frame:
    """One RGB-D observation with calibration and pose."""
    depth:     np.ndarray            # (H, W) float32/float64, metres; 0/nan = invalid
    K:         np.ndarray            # (3, 3) camera intrinsics [[fx,0,cx],[0,fy,cy],[0,0,1]]
    rgb:       Optional[np.ndarray] = None   # (H, W, 3) uint8 or float, optional
    extrinsic: Optional[np.ndarray] = None   # (4, 4) camera-to-world SE(3); None = identity


@dataclass
class SuperquadricWorld:
    """One fitted superquadric with pose expressed in world frame."""
    a1:         float          # semi-axis along local x (metres)
    a2:         float          # semi-axis along local y (metres)
    a3:         float          # semi-axis along local z (metres)
    e1:         float          # shape exponent (latitude profile)
    e2:         float          # shape exponent (longitude cross-section)
    position:   np.ndarray     # (3,) world-frame translation
    rotation:   np.ndarray     # (3, 3) world-frame rotation matrix
    shape_type: str   = "Other"
    shape_conf: float = 1.0
    chamfer_l2: float = 0.0

    def quaternion_wxyz(self) -> np.ndarray:
        """Return rotation as [w, x, y, z] unit quaternion (cuRoBO convention)."""
        import scipy.spatial.transform as sst
        return sst.Rotation.from_matrix(self.rotation).as_quat()[[3, 0, 1, 2]]


def _unproject_frame(frame: "Frame", max_depth: float) -> np.ndarray:
    """Back-project one depth frame to world-frame XYZ. Returns (N, 3) float32."""
    depth = np.asarray(frame.depth, dtype=np.float32)
    K     = np.asarray(frame.K, dtype=np.float64)
    H, W  = depth.shape

    u, v  = np.meshgrid(np.arange(W, dtype=np.float32),
                        np.arange(H, dtype=np.float32))
    valid = (depth > 0) & np.isfinite(depth) & (depth < max_depth)
    z     = depth[valid].astype(np.float64)
    x     = (u[valid] - K[0, 2]) * z / K[0, 0]
    y     = (v[valid] - K[1, 2]) * z / K[1, 1]
    pts   = np.stack([x, y, z], axis=1)   # (N, 3) camera frame

    E = np.eye(4, dtype=np.float64) if frame.extrinsic is None \
        else np.asarray(frame.extrinsic, dtype=np.float64)
    pts_h = np.concatenate([pts, np.ones((len(pts), 1))], axis=1)
    return ((E @ pts_h.T).T[:, :3]).astype(np.float32)


def _unproject_colors(frame: "Frame", max_depth: float) -> Optional[np.ndarray]:
    """Return (N, 3) float32 RGB in [0,1] aligned with _unproject_frame output."""
    if frame.rgb is None:
        return None
    depth = np.asarray(frame.depth, dtype=np.float32)
    valid = (depth > 0) & np.isfinite(depth) & (depth < max_depth)
    rgb   = np.asarray(frame.rgb, dtype=np.float32)
    if rgb.max() > 1.5:
        rgb = rgb / 255.0
    H, W  = depth.shape
    return np.clip(rgb.reshape(H * W, 3)[valid.ravel()], 0.0, 1.0)


def get_world_pointcloud(
    frames:     List["Frame"],
    voxel_size: float = 0.005,
    max_depth:  float = 3.0,
) -> o3d.geometry.PointCloud:
    """Stage 1 — unproject RGB-D frame(s) into a world-frame point cloud.

    Parameters
    ----------
    frames      : one or more Frame objects.  When MULTIVIEW=False, only
                  frames[0] is processed but its extrinsic is still applied —
                  output is always in world frame regardless of MULTIVIEW.
    voxel_size  : voxel grid down-sample resolution in metres.  0 = no downsampling.
    max_depth   : pixels beyond this depth (metres) are discarded.

    Returns
    -------
    o3d.geometry.PointCloud with points in world frame.
    """
    if not frames:
        return o3d.geometry.PointCloud()

    active = frames if MULTIVIEW else frames[:1]

    all_pts    = []
    all_colors = []

    for frame in active:
        pts    = _unproject_frame(frame, max_depth)
        colors = _unproject_colors(frame, max_depth)
        if len(pts) == 0:
            continue
        all_pts.append(pts)
        if colors is not None and len(colors) == len(pts):
            all_colors.append(colors)

    if not all_pts:
        return o3d.geometry.PointCloud()

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(
        np.concatenate(all_pts, axis=0).astype(np.float64))

    has_colors = len(all_colors) == len(all_pts)
    if has_colors:
        pcd.colors = o3d.utility.Vector3dVector(
            np.concatenate(all_colors, axis=0).astype(np.float64))

    if voxel_size > 0:
        pcd = pcd.voxel_down_sample(voxel_size)

    return pcd


def fit_superquadrics_world(
    world_pcd:          o3d.geometry.PointCloud,
    fitter              = None,
    min_object_points:  int   = 50,
    max_object_points:  int   = 5000,
    min_object_extent:  float = 0.02,
    max_object_extent:  float = 0.40,
    cluster_eps:        float = 0.018,
    cluster_min_points: int   = 20,
    adaptive_eps:       bool  = True,
    min_height:         float = 0.005,
    max_height:         float = 0.25,
) -> List["SuperquadricWorld"]:
    """Stage 2 — segment a world-frame cloud and fit superquadrics.

    Fitting is performed in each segment's centroid-centered local frame.
    All returned SuperquadricWorld instances carry poses in world frame.

    Parameters
    ----------
    world_pcd           : o3d.PointCloud in world frame (Stage 1 output).
    fitter              : SuperquadricFitter or SuperdecFitter instance.
                          Defaults to SuperquadricFitter (LM, CPU).
    min/max_object_*    : cluster size and extent filters.
    cluster_eps         : DBSCAN neighbourhood radius (metres).
    cluster_min_points  : DBSCAN min_samples.
    adaptive_eps        : estimate eps automatically from point density.
    min/max_height      : foreground height range above fitted table (metres).

    Returns
    -------
    List[SuperquadricWorld] — one or more primitives per object, world-frame poses.
    """
    try:
        from .superquadric import SuperquadricFitter
    except ImportError:
        from superquadric import SuperquadricFitter

    pts = np.asarray(world_pcd.points, dtype=np.float64)
    if len(pts) < 50:
        return []

    # table removal — disable camera-specific filters (depth gate uses camera Z,
    # XY radius assumes camera at origin; neither is valid for world-frame clouds)
    try:
        obj_pts, *_ = remove_table(
            pts,
            min_height_above_table = min_height,
            max_height_above_table = max_height,
            depth_margin           = np.inf,
            xy_radius              = np.inf,
        )
    except Exception:
        obj_pts = pts

    if len(obj_pts) < cluster_min_points:
        return []

    clusters = segment_instances(
        obj_pts,
        cluster_eps         = cluster_eps,
        cluster_min_points  = cluster_min_points,
        adaptive_eps        = adaptive_eps,
        eps_multiplier      = 3.0,
        eps_max             = 0.08,
    )

    valid_segs = [
        seg for seg in clusters
        if (min_object_points <= len(seg) <= max_object_points and
            min_object_extent <= (seg.max(0) - seg.min(0)).max() <= max_object_extent)
    ]

    if not valid_segs:
        return []

    if fitter is None:
        fitter = SuperquadricFitter(n_restarts=2, n_lm_rounds=10, subsample=256)

    results: List[SuperquadricWorld] = []

    for seg in valid_segs:
        centroid  = seg.mean(axis=0)
        local_pts = (seg - centroid).astype(np.float64)

        try:
            multi = fitter.fit_adaptive(local_pts)
        except Exception:
            continue

        for prim in multi.primitives:
            local_t = np.array([prim.tx, prim.ty, prim.tz], dtype=np.float64)
            world_t = local_t + centroid

            results.append(SuperquadricWorld(
                a1         = float(prim.sx),
                a2         = float(prim.sy),
                a3         = float(prim.sz),
                e1         = float(prim.e1),
                e2         = float(prim.e2),
                position   = world_t,
                rotation   = np.array(prim.rotation_matrix, dtype=np.float64),
                shape_type = prim.shape_type,
                shape_conf = float(prim.shape_conf),
                chamfer_l2 = float(prim.chamfer_l2),
            ))

    return results


def superquadrics_to_curobo_world(
    sqs:          List["SuperquadricWorld"],
    e1_min:       float = 0.2,
    e1_max:       float = 1.8,
    e2_min:       float = 0.2,
    e2_max:       float = 1.8,
):
    """Convert List[SuperquadricWorld] → cuRoBO WorldConfig.

    Parameters
    ----------
    sqs             : output of fit_superquadrics_world()
    e1_min/e1_max   : exponent clamp range ([0.2, 1.8] keeps cuRoBO convex)
    e2_min/e2_max   : same for e2

    Returns
    -------
    curobo.geom.sdf.world.WorldConfig, or raises ImportError if cuRoBO absent.
    """
    from curobo.geom.sdf.world import WorldConfig
    from curobo.geom.types import Superquadric

    obstacles = []
    for sq in sqs:
        e1 = float(np.clip(sq.e1, e1_min, e1_max))
        e2 = float(np.clip(sq.e2, e2_min, e2_max))
        q  = sq.quaternion_wxyz()   # [w, x, y, z]
        p  = sq.position

        obstacles.append(Superquadric(
            pose=[float(p[0]), float(p[1]), float(p[2]),
                  float(q[0]), float(q[1]), float(q[2]), float(q[3])],
            scale=[float(sq.a1), float(sq.a2), float(sq.a3)],
            shape=[e1, e2],
            name=f"sq_{id(sq)}",
        ))

    return WorldConfig(superquadric=obstacles)
