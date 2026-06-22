"""
superquadric.py — Superquadric fitting for Person 2.

Given a point cloud segment from the perception pipeline, fits one or more
superquadric primitives using Levenberg-Marquardt optimization.

This is the bridge between Person 3 (perception) and Person 1 (CuRobo planning).
The fitted superquadrics provide:
  - Closed-form signed distance (for CuRobo collision checking)
  - Compact representation (~11 params/object vs thousands of points)
  - Grasp pose estimation via superquadric geometry

Changes from original:
  - SuperquadricFitter.fit() now accepts shape_hint (str) from pipeline classifier
  - shape_hint biases e1/e2 initialisation toward the predicted shape type,
    reducing LM iterations needed and avoiding degenerate local minima
  - SuperquadricFit carries shape_type field for downstream cuRobo interface

Pipeline:
    ObjectSegment (from pipeline.py)
        → shape_type from classifier (Ellipsoid/Cylinder/Cuboid/Other)
        → normalize + PCA align
        → init from bbox, biased by shape_hint
        → LM optimize
        → SuperquadricFit
        → CuRobo SDF / grasp pose

Reference: SUPERDEC (Fedele et al., 2025)
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Tuple, List
import warnings
warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# Shape hint → SQ exponent initialisations
# Ellipsoid: e1=1, e2=1   (smooth, round)
# Cylinder:  e1=0.2, e2=1 (flat top/bottom, circular cross-section)
# Cuboid:    e1=0.2, e2=0.2 (sharp edges)
# Other:     None          (let init_from_bbox decide from aspect ratio)
# ---------------------------------------------------------------------------

_HINT_EXPONENTS = {
    "Ellipsoid": (1.0, 1.0),
    "Cylinder":  (0.2, 1.0),
    "Cuboid":    (0.2, 0.2),
    "Other":     None,
}


# ---------------------------------------------------------------------------
# Superquadric parameterization
# ---------------------------------------------------------------------------

PARAM_NAMES = ['sx', 'sy', 'sz', 'e1', 'e2', 'tx', 'ty', 'tz', 'rx', 'ry', 'rz']

def sq_type_from_exponents(e1: float, e2: float) -> str:
    """
    Derive shape type from fitted SQ exponents.
    This is the AUTHORITATIVE shape_type — determined from the actual fit,
    not from the pre-fit point cloud heuristic.

    SQ exponent semantics:
      e1 controls latitude  (profile shape: 0.2→flat top/bottom, 1→round, 2→concave)
      e2 controls longitude (cross-section: 0.2→square, 1→circle, 2→hyperbolic)

    Regions:
      Ellipsoid : e1>0.6, e2>0.6   (round in all directions)
      Cylinder  : e1<0.45, e2>0.6  (flat top+bottom, circular cross-section)
      Cuboid    : e1<0.45, e2<0.45 (flat top+bottom, square cross-section)
      Other     : mixed / transitional
    """
    if   e1 > 0.6  and e2 > 0.6:   return "Ellipsoid"
    elif e1 < 0.45 and e2 > 0.6:   return "Cylinder"
    elif e1 < 0.45 and e2 < 0.45:  return "Cuboid"
    else:                           return "Other"

N_PARAMS = 11

BOUNDS_LO = np.array([0.01, 0.01, 0.01, 0.1, 0.1, -5., -5., -5., -np.pi, -np.pi, -np.pi])
BOUNDS_HI = np.array([1.00, 1.00, 1.00, 2.0, 1.5,  5.,  5.,  5.,  np.pi,  np.pi,  np.pi])


@dataclass
class SuperquadricFit:
    """
    Single fitted superquadric. Output of SuperquadricFitter.fit().
    Consumed by CuRobo for collision avoidance and grasp planning.
    """
    sx: float; sy: float; sz: float
    e1: float; e2: float
    tx: float; ty: float; tz: float
    rx: float; ry: float; rz: float

    chamfer_l2:     float = 0.0
    n_points:       int   = 0
    n_iterations:   int   = 0
    converged:      bool  = False
    merged_cluster: bool  = False
    shape_type:     str   = "Other"   # carried from pipeline classifier
    shape_conf:     float = 0.0       # classifier confidence

    L2_GOOD     = 0.010
    L2_MARGINAL = 0.020
    L2_POOR     = 0.020

    @property
    def quality_ok(self) -> bool:
        return self.chamfer_l2 < self.L2_MARGINAL

    @property
    def collision_margin(self) -> float:
        if self.chamfer_l2 < 0.010:
            return 0.02
        elif self.chamfer_l2 < 0.020:
            return 0.04
        else:
            return 0.07

    @property
    def params(self) -> np.ndarray:
        return np.array([self.sx, self.sy, self.sz,
                         self.e1, self.e2,
                         self.tx, self.ty, self.tz,
                         self.rx, self.ry, self.rz])

    @property
    def scale(self) -> np.ndarray:
        return np.array([self.sx, self.sy, self.sz])

    @property
    def translation(self) -> np.ndarray:
        return np.array([self.tx, self.ty, self.tz])

    @property
    def rotation_matrix(self) -> np.ndarray:
        return _euler_to_rot(np.array([self.rx, self.ry, self.rz]))

    def signed_distance(self, points: np.ndarray) -> np.ndarray:
        return sq_signed_distance_batch(points, self.params)

    def surface_points(self, n_u: int = 50, n_v: int = 50) -> np.ndarray:
        """Sample points on the superquadric surface.

        Parameters
        ----------
        n_u : int
            Grid resolution along polar angle u ∈ [-π/2, π/2].  Default 50.
        n_v : int
            Grid resolution along azimuthal angle v ∈ [-π, π].  Default 50.

        Returns
        -------
        (n_u * n_v, 3) float64 array in world coordinates.

        Notes
        -----
        Higher resolution improves Chamfer L2 accuracy at quadratic cost.
        Recommended: n_u=n_v=50 for real-time use, n_u=n_v=100 for
        offline evaluation.
        """
        return sq_sample_surface(self.params, n_u=n_u, n_v=n_v)

    def is_point_inside(self, points: np.ndarray) -> np.ndarray:
        return self.signed_distance(points) < 0

    def volume(self) -> float:
        return (4/3) * np.pi * self.sx * self.sy * self.sz

    def __repr__(self):
        return (f"SQ(scale=[{self.sx:.3f},{self.sy:.3f},{self.sz:.3f}] "
                f"shape=[{self.e1:.2f},{self.e2:.2f}] "
                f"type={self.shape_type} "
                f"t=[{self.tx:.3f},{self.ty:.3f},{self.tz:.3f}] "
                f"L2={self.chamfer_l2:.4f} conv={self.converged})")


@dataclass
class MultiSQFit:
    """Set of superquadrics fitted to one object segment."""
    primitives: List[SuperquadricFit] = field(default_factory=list)
    n_points:   int = 0

    def signed_distance(self, points: np.ndarray) -> np.ndarray:
        if not self.primitives:
            return np.full(len(points), np.inf)
        dists = np.stack([p.signed_distance(points) for p in self.primitives])
        return dists.min(axis=0)

    def all_surface_points(self, n_u_per: int = 50, n_v_per: int = 50) -> np.ndarray:
        return np.concatenate([p.surface_points(n_u=n_u_per, n_v=n_v_per)
                               for p in self.primitives])

    def __len__(self):
        return len(self.primitives)


# ---------------------------------------------------------------------------
# Core math
# ---------------------------------------------------------------------------

def _euler_to_rot(euler: np.ndarray) -> np.ndarray:
    rx, ry, rz = euler
    cx, sx = np.cos(rx), np.sin(rx)
    cy, sy = np.cos(ry), np.sin(ry)
    cz, sz = np.cos(rz), np.sin(rz)
    Rx = np.array([[1,0,0],[0,cx,-sx],[0,sx,cx]])
    Ry = np.array([[cy,0,sy],[0,1,0],[-sy,0,cy]])
    Rz = np.array([[cz,-sz,0],[sz,cz,0],[0,0,1]])
    return Rz @ Ry @ Rx


def _transform_to_canonical(points: np.ndarray, params: np.ndarray) -> np.ndarray:
    t = params[5:8]
    R = _euler_to_rot(params[8:11])
    return (R.T @ (points - t).T).T


def sq_implicit(points_canon: np.ndarray, sx, sy, sz, e1, e2) -> np.ndarray:
    e1 = np.clip(e1, 0.1, 2.0)
    e2 = np.clip(e2, 0.1, 2.0)
    sx = np.clip(sx, 1e-4, None)
    sy = np.clip(sy, 1e-4, None)
    sz = np.clip(sz, 1e-4, None)

    x, y, z = points_canon[:, 0], points_canon[:, 1], points_canon[:, 2]
    term_xy = (np.clip(np.abs(x/sx), 1e-10, None)**(2/e2) +
               np.clip(np.abs(y/sy), 1e-10, None)**(2/e2))
    term_xy = np.clip(term_xy, 1e-10, None)**(e2/e1)
    term_z  = np.clip(np.abs(z/sz),  1e-10, None)**(2/e1)
    return term_xy + term_z


def sq_radial_distance(points_canon: np.ndarray, params: np.ndarray) -> np.ndarray:
    sx, sy, sz, e1, e2 = params[:5]
    f     = sq_implicit(points_canon, sx, sy, sz, e1, e2)
    norms = np.linalg.norm(points_canon, axis=1)
    e1    = np.clip(e1, 0.1, 2.0)
    dr    = norms * np.abs(1.0 - np.clip(f, 1e-10, None)**(-e1/2))
    return dr


def sq_signed_distance_batch(points: np.ndarray, params: np.ndarray) -> np.ndarray:
    sx, sy, sz, e1, e2 = params[:5]
    pts_c = _transform_to_canonical(points, params)
    f     = sq_implicit(pts_c, sx, sy, sz, e1, e2)
    dr    = sq_radial_distance(pts_c, params)
    return np.sign(f - 1.0) * dr


def sq_sample_surface(params: np.ndarray, n_u: int = 50, n_v: int = 50) -> np.ndarray:
    """Sample points on the surface of a superquadric using a regular UV grid.

    Parameters
    ----------
    params : (11,) array
        [sx, sy, sz, e1, e2, tx, ty, tz, rx, ry, rz]
    n_u : int
        Number of grid points along the polar angle u ∈ [-π/2, π/2].
        Default 50.  Higher values improve Chamfer L2 accuracy at a
        quadratic cost in point count (n_u × n_v total samples).
    n_v : int
        Number of grid points along the azimuthal angle v ∈ [-π, π].
        Default 50.

    Returns
    -------
    (n_u * n_v, 3) float64 surface-point array in world coordinates.

    Notes
    -----
    A grid-based parametric sample provides uniform angular coverage and
    deterministic output, unlike random sampling.  Recommended values:
      • Real-time use (Chamfer quality estimate): n_u=n_v=50  (2 500 pts)
      • Offline evaluation (higher accuracy):     n_u=n_v=100 (10 000 pts)
    """
    sx, sy, sz, e1, e2 = params[:5]
    t = params[5:8]
    R = _euler_to_rot(params[8:11])

    u = np.linspace(-np.pi/2, np.pi/2, n_u)
    v = np.linspace(-np.pi,   np.pi,   n_v)
    uu, vv = np.meshgrid(u, v)
    uu = uu.ravel()
    vv = vv.ravel()

    def _sign_pow(x, p):
        return np.sign(x) * (np.abs(x) ** p)

    x = sx * _sign_pow(np.cos(uu), e1) * _sign_pow(np.cos(vv), e2)
    y = sy * _sign_pow(np.cos(uu), e1) * _sign_pow(np.sin(vv), e2)
    z = sz * _sign_pow(np.sin(uu), e1)

    pts_canon = np.stack([x, y, z], axis=1)
    return (R @ pts_canon.T).T + t


# ---------------------------------------------------------------------------
# Initialization
# ---------------------------------------------------------------------------

def init_from_bbox(points: np.ndarray,
                   shape_hint: Optional[str] = None) -> np.ndarray:
    """
    Initialize SQ parameters from point cloud bounding box + PCA.

    shape_hint: if provided (from pipeline classifier), overrides the
                e1/e2 starting values on the first hypothesis, giving LM
                a better starting point and reducing iterations needed.
    """
    centroid = points.mean(axis=0)
    centered = points - centroid

    cov = (centered.T @ centered) / len(centered)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    idx = np.argsort(eigenvalues)[::-1]
    R = eigenvectors[:, idx]
    if np.linalg.det(R) < 0:
        R[:, 0] = -R[:, 0]

    pts_pca = (R.T @ centered.T).T
    extents = pts_pca.max(axis=0) - pts_pca.min(axis=0)
    scales  = np.clip(extents / 2, 0.01, 0.5)
    euler   = _rot_to_euler(R)
    t       = centroid

    aspect = float(extents.max() / (extents.min() + 1e-6))
    if aspect > 3.0:
        shape_hypotheses = [(1.0, 0.3), (0.5, 0.5), (0.2, 0.2), (1.0, 1.0), (0.3, 0.5)]
    elif aspect < 1.5:
        shape_hypotheses = [(0.2, 0.2), (0.5, 0.5), (1.0, 1.0), (0.3, 0.3), (1.0, 0.5)]
    else:
        shape_hypotheses = [(1.0, 1.0), (0.5, 0.5), (0.2, 0.2), (1.0, 0.5), (0.3, 0.8)]

    # inject classifier hint as first hypothesis (overrides aspect-ratio guess)
    if shape_hint and shape_hint in _HINT_EXPONENTS:
        hint_eps = _HINT_EXPONENTS[shape_hint]
        if hint_eps is not None:
            shape_hypotheses = [hint_eps] + [h for h in shape_hypotheses if h != hint_eps]

    best_params, best_score = None, np.inf
    for e1, e2 in shape_hypotheses:
        params = np.array([
            scales[0], scales[1], scales[2],
            e1, e2,
            t[0], t[1], t[2],
            euler[0], euler[1], euler[2],
        ])
        params = np.clip(params, BOUNDS_LO, BOUNDS_HI)
        idx_s = np.random.choice(len(points), min(100, len(points)), replace=False)
        pts_c = _transform_to_canonical(points[idx_s], params)
        r     = sq_radial_distance(pts_c, params)
        score = float(np.mean(r))
        if score < best_score:
            best_score  = score
            best_params = params

    return best_params


def _rot_to_euler(R: np.ndarray) -> np.ndarray:
    sy = np.sqrt(R[0,0]**2 + R[1,0]**2)
    singular = sy < 1e-6
    if not singular:
        x = np.arctan2( R[2,1], R[2,2])
        y = np.arctan2(-R[2,0], sy)
        z = np.arctan2( R[1,0], R[0,0])
    else:
        x = np.arctan2(-R[1,2], R[1,1])
        y = np.arctan2(-R[2,0], sy)
        z = 0.0
    return np.array([x, y, z])


# ---------------------------------------------------------------------------
# LM Optimization
# ---------------------------------------------------------------------------

def _residuals(params: np.ndarray, points: np.ndarray) -> np.ndarray:
    params = np.clip(params, BOUNDS_LO, BOUNDS_HI)
    pts_c  = _transform_to_canonical(points, params)
    return sq_radial_distance(pts_c, params)


def _jacobian_fd(params: np.ndarray, points: np.ndarray,
                 eps: float = 1e-5) -> np.ndarray:
    r0 = _residuals(params, points)
    J  = np.zeros((len(r0), len(params)))
    for i in range(len(params)):
        p_hi = params.copy(); p_hi[i] += eps
        p_lo = params.copy(); p_lo[i] -= eps
        J[:, i] = (_residuals(p_hi, points) - _residuals(p_lo, points)) / (2*eps)
    return J


def lm_optimize(
    points:      np.ndarray,
    params_init: np.ndarray,
    n_rounds:    int   = 10,
    lam_init:    float = 0.01,
    lam_factor:  float = 10.0,
    subsample:   int   = 512,
) -> Tuple[np.ndarray, float, bool]:
    params = params_init.copy()
    lam    = lam_init

    if len(points) > subsample:
        idx = np.random.choice(len(points), subsample, replace=False)
        pts = points[idx]
    else:
        pts = points

    converged = False

    for _ in range(n_rounds):
        r    = _residuals(params, pts)
        loss = float(np.mean(r**2))
        J    = _jacobian_fd(params, pts)
        JtJ  = J.T @ J
        Jtr  = J.T @ r

        A = JtJ + lam * np.eye(len(params))
        try:
            delta = np.linalg.solve(A, -Jtr)
        except np.linalg.LinAlgError:
            break

        params_new = np.clip(params + delta, BOUNDS_LO, BOUNDS_HI)
        loss_new   = float(np.mean(_residuals(params_new, pts)**2))

        if loss_new < loss:
            params = params_new
            lam    = lam / lam_factor
            if abs(loss - loss_new) < 1e-7:
                converged = True
                break
        else:
            lam = lam * lam_factor

    return params, float(np.mean(_residuals(params, pts)**2)), converged


# ---------------------------------------------------------------------------
# Chamfer distance
# ---------------------------------------------------------------------------

def _arclength_resample(t_dense: np.ndarray, xy: np.ndarray,
                        n_out: int) -> Tuple[np.ndarray, np.ndarray]:
    """Resample a 2-D parametric curve at equal arc-length intervals.

    Parameters
    ----------
    t_dense : (N,) parameter values.
    xy      : (N, 2) curve points.
    n_out   : number of output samples.

    Returns
    -------
    t_out  : (n_out,) resampled parameter values.
    xy_out : (n_out, 2) resampled curve points.
    """
    diffs  = np.diff(xy, axis=0)
    seg_len = np.sqrt((diffs ** 2).sum(axis=1))
    cumlen  = np.concatenate([[0.0], np.cumsum(seg_len)])
    total   = float(cumlen[-1])
    if total < 1e-12:
        idx = np.round(np.linspace(0, len(t_dense) - 1, n_out)).astype(int)
        return t_dense[idx], xy[idx]
    targets = np.linspace(0.0, total, n_out)
    t_out   = np.interp(targets, cumlen, t_dense)
    x_out   = np.interp(targets, cumlen, xy[:, 0])
    y_out   = np.interp(targets, cumlen, xy[:, 1])
    return t_out, np.stack([x_out, y_out], axis=1)


def sample_surface_equal_distance(sq: 'SuperquadricFit',
                                  n_points: int = 500,
                                  n_dense:  int = 1000) -> np.ndarray:
    """Sample SQ surface with approximately equal arc-length spacing.

    Uniform (eta, omega) sampling clusters points near the z-poles, biasing
    the Chamfer L2 metric.  This function uses the equal-distance algorithm
    from Liu et al. CVPR 2022, Supplementary Sec. 4 (extending Vaskevicius &
    Birk 2019):

      1. Sample the principal superellipse {e1, 1, sz} with equal arc-length
         spacing → latitude angles eta_i and ring scale factors cos^e1(eta_i).
      2. For each latitude, sample the orthogonal ring
         {e2, sx·r_i, sy·r_i} with the same arc-length delta.
      3. Combine via the SQ parametric equation to get 3-D points.
      4. Apply the SQ pose (R, t).
      5. Sub/upsample to exactly n_points.

    Parameters
    ----------
    sq       : SuperquadricFit
    n_points : target number of surface points (default 500).
    n_dense  : dense parameter grid size used for arc-length estimation
               (default 1000; higher = more accurate distribution).

    Returns
    -------
    (n_points, 3) float64 array in world coordinates.
    """
    def _sp(x, e):
        return np.sign(x) * (np.abs(x) ** e)

    e1 = float(np.clip(sq.e1, 1e-6, 2.0))
    e2 = float(np.clip(sq.e2, 1e-6, 2.0))
    sx, sy, sz = float(sq.sx), float(sq.sy), float(sq.sz)

    # ── Step 1: equal arc-length latitude samples ──────────────────────────
    eta_dense = np.linspace(-np.pi / 2, np.pi / 2, n_dense)
    lat_r  = _sp(np.cos(eta_dense), e1)          # cos^e1(eta) — ring scale [0,1]
    lat_z  = sz * _sp(np.sin(eta_dense), e1)     # z on surface [−sz, sz]

    # Use physical (scaled) coordinates for arc-length computation so that
    # the pole transition (Δx ≈ sx) and the barrel height (Δz ≈ sz) are
    # weighted by their actual physical extents.  Without this, for a cylinder
    # the unscaled Δr≈1 dominates Δz≈0.2, placing too many latitude samples
    # near the poles.
    # (Liu et al. CVPR 2022 Supplementary Sec. 4, Pilu & Fisher 1995)
    lat_curve_phys = np.stack([sx * lat_r, lat_z], axis=1)  # physical units

    diffs_lat       = np.diff(lat_curve_phys, axis=0)
    seg_lat         = np.sqrt((diffs_lat ** 2).sum(axis=1))
    perim_principal = float(seg_lat.sum())

    # n_lat ≈ sqrt(n_points); delta in physical length units
    n_lat = max(int(round(np.sqrt(n_points))), 4)
    delta = perim_principal / n_lat if perim_principal > 1e-10 else 1.0

    # Resample at equal physical arc-length; recover unscaled ring scale
    _, lat_phys_resampled = _arclength_resample(eta_dense, lat_curve_phys, n_lat)
    lat_r_vals = lat_phys_resampled[:, 0] / (sx + 1e-12)  # back to [0, 1]
    lat_z_vals = lat_phys_resampled[:, 1]

    omega_dense = np.linspace(-np.pi, np.pi, n_dense + 1)[:-1]  # n_dense pts

    # ── Step 3: collect surface points ────────────────────────────────────
    all_pts: List[np.ndarray] = []

    for r_scale, z_val in zip(lat_r_vals, lat_z_vals):
        r = float(abs(r_scale))

        ring_x = sx * r * _sp(np.cos(omega_dense), e2)
        ring_y = sy * r * _sp(np.sin(omega_dense), e2)
        # Close the ring for arc-length computation
        ring_cx = np.append(ring_x, ring_x[0])
        ring_cy = np.append(ring_y, ring_y[0])
        diffs   = np.diff(np.stack([ring_cx, ring_cy], axis=1), axis=0)
        seg_len = np.sqrt((diffs ** 2).sum(axis=1))
        perimeter = float(seg_len.sum())

        # Skip rings too small to contribute a full sample at this delta.
        # This avoids accumulating many near-zero polar rings that would
        # crowd the up-sampling pool and distort the z-distribution.
        n_lon = int(round(perimeter / delta))
        if n_lon == 0:
            continue

        cumlen = np.concatenate([[0.0], np.cumsum(seg_len)])  # n_dense+1 vals
        # targets: n_lon equally-spaced arc positions on [0, perimeter)
        targets = np.linspace(0.0, perimeter, n_lon + 1)[:-1]
        # cumlen[:-1] (n_dense values) aligns with ring_x/ring_y
        rx_eq = np.interp(targets, cumlen[:-1], ring_x)
        ry_eq = np.interp(targets, cumlen[:-1], ring_y)
        z_arr = np.full(n_lon, float(z_val))
        all_pts.append(np.stack([rx_eq, ry_eq, z_arr], axis=1))

    if not all_pts:
        return np.zeros((n_points, 3), dtype=np.float64)

    pts_local = np.vstack(all_pts).astype(np.float64)

    # ── Step 4: sub/upsample to exactly n_points ───────────────────────────
    if len(pts_local) > n_points:
        idx = np.random.choice(len(pts_local), n_points, replace=False)
        pts_local = pts_local[idx]
    elif len(pts_local) < n_points:
        extra = np.random.choice(len(pts_local),
                                 n_points - len(pts_local), replace=True)
        pts_local = np.vstack([pts_local, pts_local[extra]])

    # ── Step 5: apply SQ pose ──────────────────────────────────────────────
    R = np.array(sq.rotation_matrix, dtype=np.float64)
    t = np.array(sq.translation,     dtype=np.float64)
    return (R @ pts_local.T).T + t


def chamfer_l2(points: np.ndarray, sq_params: np.ndarray,
               n_u: int = 50, n_v: int = 50) -> float:
    surf    = sq_sample_surface(sq_params, n_u=n_u, n_v=n_v)
    diff_ps = points[:, None, :] - surf[None, :, :]
    d_ps    = np.min(np.sum(diff_ps**2, axis=2), axis=1)
    diff_sp = surf[:, None, :] - points[None, :, :]
    d_sp    = np.min(np.sum(diff_sp**2, axis=2), axis=1)
    return float(np.mean(d_ps) + np.mean(d_sp))


# ---------------------------------------------------------------------------
# Main fitter class
# ---------------------------------------------------------------------------

class SuperquadricFitter:
    """
    Fits one or more superquadrics to an ObjectSegment.

    Usage:
        fitter = SuperquadricFitter()

        # with shape hint from pipeline classifier:
        sq = fitter.fit(seg.points, shape_hint=seg.shape_type)

        # adaptive multi-SQ:
        multi = fitter.fit_adaptive(seg.points, shape_hint=seg.shape_type)

        sd = sq.signed_distance(query_points)  # for CuRobo
    """

    def __init__(
        self,
        n_primitives: int   = 1,
        n_lm_rounds:  int   = 15,
        subsample:    int   = 512,
        n_restarts:   int   = 3,
    ):
        self.n_primitives = n_primitives
        self.n_lm_rounds  = n_lm_rounds
        self.subsample    = subsample
        self.n_restarts   = n_restarts

    def fit_adaptive(self, points: np.ndarray,
                     l2_threshold:  float = 0.007,
                     max_primitives: int  = 4,
                     shape_hint:    str   = None,
                     _depth:        int   = 0) -> MultiSQFit:
        """
        Fit one SQ; if Chamfer L2 > threshold, split and recurse.
        shape_hint flows into the initial fit only (first primitive).
        """
        sq = self.fit(points, shape_hint=shape_hint)

        if sq.chamfer_l2 <= l2_threshold or _depth >= 2 or max_primitives <= 1:
            return MultiSQFit(primitives=[sq], n_points=len(points))

        remaining      = points.copy()
        all_primitives = []
        all_inlier_pts = []

        for _ in range(min(max_primitives, 3)):
            if len(remaining) < 30:
                break
            sq_i        = self.fit(remaining)
            sd          = sq_i.signed_distance(remaining)
            inlier_mask = np.abs(sd) <= 0.015
            all_primitives.append(sq_i)
            all_inlier_pts.append(remaining[inlier_mask])
            remaining = remaining[~inlier_mask]

        if not all_primitives:
            return MultiSQFit(primitives=[sq], n_points=len(points))

        mean_l2_split = float(np.mean([p.chamfer_l2 for p in all_primitives]))

        if mean_l2_split < sq.chamfer_l2 * 0.80:
            final = []
            for sq_i, pts_i in zip(all_primitives, all_inlier_pts):
                if sq_i.chamfer_l2 > l2_threshold and _depth < 1 and len(pts_i) > 30:
                    sub = self.fit_adaptive(pts_i, l2_threshold=l2_threshold,
                                            max_primitives=2, _depth=_depth + 1)
                    final.extend(sub.primitives)
                else:
                    final.append(sq_i)
            return MultiSQFit(primitives=final[:max_primitives], n_points=len(points))

        return MultiSQFit(primitives=[sq], n_points=len(points))

    def fit(self, points: np.ndarray,
            shape_hint: Optional[str] = None) -> SuperquadricFit:
        """
        Fit a single superquadric to a point cloud segment.

        shape_hint : str, optional
            Shape type predicted by the pipeline classifier
            ('Ellipsoid', 'Cylinder', 'Cuboid', 'Other').
            If provided, biases e1/e2 initialisation toward the predicted
            type, reducing LM iterations and avoiding degenerate minima.
        """
        if len(points) < 10:
            return self._degenerate(points, shape_hint)

        centroid = points.mean(axis=0)
        scale    = np.linalg.norm(points - centroid, axis=1).max()
        scale    = max(scale, 0.01)
        pts_norm = (points - centroid) / scale

        best_params, best_loss, best_conv = None, np.inf, False

        for restart_i in range(self.n_restarts):
            # pass shape_hint into init on every restart
            p0 = init_from_bbox(pts_norm, shape_hint=shape_hint)

            # add small noise on subsequent restarts to escape local minima
            if restart_i > 0:
                p0 += np.random.randn(N_PARAMS) * 0.05
                p0  = np.clip(p0, BOUNDS_LO, BOUNDS_HI)

            params, loss, conv = lm_optimize(
                pts_norm, p0,
                n_rounds=self.n_lm_rounds,
                subsample=self.subsample,
            )
            if loss < best_loss:
                best_loss   = loss
                best_params = params
                best_conv   = conv

        # denormalize
        best_params       = best_params.copy()
        best_params[0:3] *= scale
        best_params[5:8]  = best_params[5:8] * scale + centroid

        ch = chamfer_l2(points, best_params)

        fitted_e1   = float(best_params[3])
        fitted_e2   = float(best_params[4])
        fitted_type = sq_type_from_exponents(fitted_e1, fitted_e2)

        return SuperquadricFit(
            sx=best_params[0], sy=best_params[1], sz=best_params[2],
            e1=fitted_e1, e2=fitted_e2,
            tx=best_params[5], ty=best_params[6], tz=best_params[7],
            rx=best_params[8], ry=best_params[9], rz=best_params[10],
            chamfer_l2=ch, n_points=len(points),
            n_iterations=self.n_lm_rounds, converged=best_conv,
            shape_type=fitted_type,   # ← from exponents, not from hint
            shape_conf=1.0 if best_conv else 0.7,
        )

    def fit_multi(self, points: np.ndarray,
                  n: Optional[int] = None,
                  shape_hint: Optional[str] = None) -> MultiSQFit:
        n         = n or self.n_primitives
        remaining = points.copy()
        primitives = []

        for i in range(n):
            if len(remaining) < 20:
                break
            # only pass hint for first primitive
            sq = self.fit(remaining, shape_hint=shape_hint if i == 0 else None)
            primitives.append(sq)
            sd   = sq.signed_distance(remaining)
            mask = np.abs(sd) > 0.02
            remaining = remaining[mask]

        return MultiSQFit(primitives=primitives, n_points=len(points))

    def _degenerate(self, points: np.ndarray,
                    shape_hint: Optional[str] = None) -> SuperquadricFit:
        c = points.mean(axis=0) if len(points) > 0 else np.zeros(3)
        return SuperquadricFit(
            sx=0.05, sy=0.05, sz=0.05, e1=1.0, e2=1.0,
            tx=c[0], ty=c[1], tz=c[2], rx=0., ry=0., rz=0.,
            chamfer_l2=np.inf, n_points=len(points), converged=False,
            shape_type=shape_hint or "Other",
        )


# ---------------------------------------------------------------------------
# CuRobo interface
# ---------------------------------------------------------------------------

def fits_to_curobo_obstacles(fits: List[SuperquadricFit],
                              margin: float = None) -> List[dict]:
    """
    Convert SuperquadricFit list to CuRobo obstacle format.
    shape_type is carried through so cuRobo can apply type-specific margins.
    """
    obstacles = []
    for i, fit in enumerate(fits):
        m = margin if margin is not None else fit.collision_margin
        obstacles.append({
            'type':       'superquadric',
            'id':         i,
            'shape_type': fit.shape_type,    # from pipeline classifier
            'shape_conf': fit.shape_conf,
            'quality_ok': fit.quality_ok,
            'params': {
                'scale':       fit.scale.tolist(),
                'shape':       [fit.e1, fit.e2],
                'translation': fit.translation.tolist(),
                'rotation':    fit.rotation_matrix.tolist(),
                'margin':      m,
            },
            'sdf_fn': fit.signed_distance,
        })
    return obstacles


# ---------------------------------------------------------------------------
# Quick test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("SuperquadricFitter — self test")
    np.random.seed(42)

    pts = np.random.uniform(-1, 1, (500, 3))
    pts[:, 0] *= 0.15
    pts[:, 1] *= 0.08
    pts[:, 2] *= 0.05
    pts += np.array([0.3, 0.0, 0.8])

    fitter = SuperquadricFitter(n_restarts=3, n_lm_rounds=20)

    # test with shape hint
    fit = fitter.fit(pts, shape_hint="Cuboid")
    print(f"  Fitted (hint=Cuboid): {fit}")
    print(f"  Chamfer L2 (×1e3): {fit.chamfer_l2 * 1e3:.4f}")

    inside  = pts[:5]
    outside = pts[:5] + np.array([1, 1, 1])
    print(f"  SD inside  (should be ≤0): {fit.signed_distance(inside).round(4)}")
    print(f"  SD outside (should be >0): {fit.signed_distance(outside).round(4)}")

    multi = fitter.fit_multi(pts, n=2, shape_hint="Cuboid")
    print(f"  Multi-SQ: {len(multi)} primitives")
    print("Self test passed.")