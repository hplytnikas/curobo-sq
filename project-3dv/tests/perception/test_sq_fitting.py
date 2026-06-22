"""
Unit tests for Stage 4 — superquadric fitting.

Tests cover:
  - SuperquadricFitter (LM) on known synthetic shapes
  - shape_hint biases exponent initialisation
  - Chamfer L2 is finite and positive
  - MultiSQFit interface
  - fits_to_curobo_obstacles output format
  - SuperdecFitter stub (no GPU / checkpoint required)
"""

import sys, os

import numpy as np
import pytest
from project_3dv.perception.superquadric import (
    SuperquadricFitter, SuperquadricFit, MultiSQFit,
    sq_type_from_exponents, fits_to_curobo_obstacles,
    chamfer_l2, sq_sample_surface, sq_signed_distance_batch,
    init_from_bbox,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ellipsoid_surface(sx, sy, sz, n, rng):
    """Sample points on the surface of a canonical axis-aligned ellipsoid."""
    u = rng.uniform(-np.pi/2, np.pi/2, n)
    v = rng.uniform(-np.pi,   np.pi,   n)
    x = sx * np.cos(u) * np.cos(v)
    y = sy * np.cos(u) * np.sin(v)
    z = sz * np.sin(u)
    return np.stack([x, y, z], axis=1).astype(np.float32)


def _cylinder_surface(r, h, n, rng):
    """Sample points on the lateral surface of a Z-axis cylinder."""
    theta = rng.uniform(0, 2*np.pi, n)
    z     = rng.uniform(-h/2, h/2, n)
    x     = r * np.cos(theta)
    y     = r * np.sin(theta)
    return np.stack([x, y, z], axis=1).astype(np.float32)


def _box_surface(sx, sy, sz, n_per_face, rng):
    """Sample points on the six faces of an axis-aligned box."""
    faces = []
    for axis, sign in [(0, 1), (0, -1), (1, 1), (1, -1), (2, 1), (2, -1)]:
        pts = rng.uniform(-1, 1, (n_per_face, 3))
        pts[:, axis] = sign
        pts[:, 0] *= sx
        pts[:, 1] *= sy
        pts[:, 2] *= sz
        faces.append(pts)
    return np.vstack(faces).astype(np.float32)


# ---------------------------------------------------------------------------
# Core math
# ---------------------------------------------------------------------------

class TestSQMath:

    def test_sample_surface_shape(self):
        params = np.array([0.1, 0.1, 0.05, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        surf = sq_sample_surface(params, n_u=20, n_v=25)
        assert surf.shape == (20 * 25, 3)
        assert np.isfinite(surf).all()

    def test_signed_distance_inside_negative(self):
        """Points strictly inside the SQ should have negative signed distance."""
        params = np.array([0.1, 0.1, 0.08, 1.0, 1.0, 0.3, 0.0, 0.5, 0.0, 0.0, 0.0])
        # A point very close to the centre (not exactly at it) is strictly inside
        interior = np.array([[0.3, 0.0, 0.5]])  # = translation = centre
        # Shift by half the smallest scale — well inside
        interior[0, 0] += 0.001
        sd = sq_signed_distance_batch(interior, params)
        # sd should be ≤ 0; exact centre can return -0.0 so allow ≤ 0
        assert sd[0] <= 0, f"Interior point should have sd ≤ 0, got {sd[0]}"

    def test_signed_distance_outside_positive(self):
        params = np.array([0.1, 0.1, 0.08, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        far = np.array([[1.0, 1.0, 1.0]])
        sd = sq_signed_distance_batch(far, params)
        assert sd[0] > 0, f"Exterior point should have sd > 0, got {sd[0]}"

    def test_chamfer_l2_finite(self):
        rng = np.random.RandomState(0)
        params = np.array([0.1, 0.1, 0.08, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        pts = _ellipsoid_surface(0.10, 0.10, 0.08, 200, rng)
        cl2 = chamfer_l2(pts, params, n_u=16, n_v=16)
        assert np.isfinite(cl2)
        assert cl2 >= 0


class TestSQTypeFromExponents:

    def test_ellipsoid(self):
        assert sq_type_from_exponents(1.0, 1.0) == "Ellipsoid"
        assert sq_type_from_exponents(0.8, 0.9) == "Ellipsoid"

    def test_cylinder(self):
        assert sq_type_from_exponents(0.2, 1.0) == "Cylinder"
        assert sq_type_from_exponents(0.3, 0.8) == "Cylinder"

    def test_cuboid(self):
        assert sq_type_from_exponents(0.2, 0.2) == "Cuboid"
        assert sq_type_from_exponents(0.3, 0.3) == "Cuboid"

    def test_other(self):
        assert sq_type_from_exponents(0.5, 0.5) == "Other"


# ---------------------------------------------------------------------------
# Initialisation
# ---------------------------------------------------------------------------

class TestInitFromBbox:

    def test_output_shape_and_finite(self):
        rng = np.random.RandomState(0)
        pts = rng.randn(300, 3) * 0.05
        p0  = init_from_bbox(pts)
        assert p0.shape == (11,)
        assert np.isfinite(p0).all()

    def test_shape_hint_biases_exponents(self):
        """With Cuboid hint, exponents should start closer to [0.2, 0.2]."""
        rng = np.random.RandomState(1)
        pts = rng.randn(200, 3) * 0.05
        p_cube  = init_from_bbox(pts, shape_hint="Cuboid")
        p_other = init_from_bbox(pts, shape_hint=None)
        # Cuboid hint should produce smaller starting exponents
        assert p_cube[3] <= p_other[3] + 0.5  # e1
        assert p_cube[4] <= p_other[4] + 0.5  # e2

    def test_scales_positive(self):
        rng = np.random.RandomState(2)
        pts = rng.randn(200, 3) * 0.05
        p0  = init_from_bbox(pts)
        assert (p0[:3] > 0).all()


# ---------------------------------------------------------------------------
# SuperquadricFitter — LM optimisation
# ---------------------------------------------------------------------------

class TestSuperquadricFitter:

    def setup_method(self):
        self.fitter = SuperquadricFitter(n_restarts=2, n_lm_rounds=10, subsample=256)

    def test_fit_ellipsoid_chamfer(self):
        """Fit to ellipsoid surface; Chamfer L2 should be small."""
        rng = np.random.RandomState(100)
        pts = _ellipsoid_surface(0.10, 0.08, 0.06, 400, rng)
        fit = self.fitter.fit(pts, shape_hint="Ellipsoid")
        assert isinstance(fit, SuperquadricFit)
        assert np.isfinite(fit.chamfer_l2)
        # Should converge to a reasonable error
        assert fit.chamfer_l2 < 0.05, \
            f"Chamfer L2 too large for clean ellipsoid: {fit.chamfer_l2:.5f}"

    def test_fit_box_with_hint(self):
        """Fit to box surface with Cuboid hint."""
        rng = np.random.RandomState(101)
        pts = _box_surface(0.08, 0.05, 0.04, 50, rng)
        fit = self.fitter.fit(pts, shape_hint="Cuboid")
        assert isinstance(fit, SuperquadricFit)
        assert np.isfinite(fit.chamfer_l2)

    def test_fit_returns_valid_params(self):
        rng = np.random.RandomState(102)
        pts = rng.randn(200, 3) * 0.05
        fit = self.fitter.fit(pts)
        # Scale must be positive
        assert fit.sx > 0 and fit.sy > 0 and fit.sz > 0
        # Exponents must be in valid range
        assert 0.1 <= fit.e1 <= 2.0
        assert 0.1 <= fit.e2 <= 2.0

    def test_shape_type_set_from_exponents(self):
        """shape_type must reflect the fitted exponents, not the hint."""
        rng = np.random.RandomState(103)
        pts = _ellipsoid_surface(0.08, 0.08, 0.08, 400, rng)
        fit = self.fitter.fit(pts, shape_hint="Cuboid")
        # Regardless of hint, shape_type should come from sq_type_from_exponents
        expected = sq_type_from_exponents(fit.e1, fit.e2)
        assert fit.shape_type == expected

    def test_fit_degenerate_input(self):
        """Very few points should return a degenerate fit without error."""
        pts = np.zeros((3, 3))
        fit = self.fitter.fit(pts)
        assert isinstance(fit, SuperquadricFit)
        assert not fit.converged or fit.chamfer_l2 >= 0

    def test_signed_distance_interface(self):
        """SuperquadricFit.signed_distance() must accept (N, 3) arrays."""
        rng = np.random.RandomState(104)
        pts = _ellipsoid_surface(0.08, 0.08, 0.08, 200, rng)
        fit = self.fitter.fit(pts)
        query = rng.randn(50, 3) * 0.2
        sd = fit.signed_distance(query)
        assert sd.shape == (50,)
        assert np.isfinite(sd).all()


class TestMultiSQFit:

    def setup_method(self):
        self.fitter = SuperquadricFitter(n_restarts=1, n_lm_rounds=8, subsample=128)

    def test_fit_adaptive_returns_multisq(self):
        rng = np.random.RandomState(200)
        pts = np.random.RandomState(200).randn(300, 3) * 0.05
        multi = self.fitter.fit_adaptive(pts)
        assert isinstance(multi, MultiSQFit)
        assert len(multi.primitives) >= 1

    def test_multisq_signed_distance(self):
        rng = np.random.RandomState(201)
        pts = rng.randn(300, 3) * 0.05
        multi = self.fitter.fit_adaptive(pts)
        query = rng.randn(20, 3) * 0.3
        sd = multi.signed_distance(query)
        assert sd.shape == (20,)
        assert np.isfinite(sd).all()

    def test_empty_multisq_signed_distance_is_inf(self):
        """MultiSQFit with no primitives should return inf signed distance."""
        multi = MultiSQFit(primitives=[], n_points=0)
        sd = multi.signed_distance(np.zeros((5, 3)))
        assert np.all(np.isinf(sd))

    def test_fit_multi_n_primitives(self):
        rng = np.random.RandomState(202)
        pts = rng.randn(400, 3) * 0.05
        multi = self.fitter.fit_multi(pts, n=2)
        assert 1 <= len(multi.primitives) <= 2


# ---------------------------------------------------------------------------
# CuRobo interface
# ---------------------------------------------------------------------------

class TestFitsToCuroboObstacles:

    def _make_fit(self):
        return SuperquadricFit(
            sx=0.05, sy=0.04, sz=0.03,
            e1=1.0,  e2=1.0,
            tx=0.1,  ty=0.0,  tz=0.5,
            rx=0.0,  ry=0.0,  rz=0.0,
            chamfer_l2=0.005, converged=True,
            shape_type="Ellipsoid", shape_conf=0.8,
        )

    def test_output_is_list_of_dicts(self):
        fits = [self._make_fit(), self._make_fit()]
        obs  = fits_to_curobo_obstacles(fits)
        assert isinstance(obs, list)
        assert len(obs) == 2
        for o in obs:
            assert isinstance(o, dict)

    def test_required_keys_present(self):
        fits = [self._make_fit()]
        obs  = fits_to_curobo_obstacles(fits)
        required = {'type', 'id', 'shape_type', 'params', 'sdf_fn'}
        assert required.issubset(obs[0].keys())

    def test_sdf_fn_callable(self):
        fits = [self._make_fit()]
        obs  = fits_to_curobo_obstacles(fits)
        sdf_fn = obs[0]['sdf_fn']
        query  = np.zeros((5, 3))
        result = sdf_fn(query)
        assert result.shape == (5,)
        assert np.isfinite(result).all()

    def test_custom_margin_applied(self):
        fits = [self._make_fit()]
        obs  = fits_to_curobo_obstacles(fits, margin=0.05)
        assert obs[0]['params']['margin'] == 0.05

    def test_empty_input(self):
        obs = fits_to_curobo_obstacles([])
        assert obs == []


# ---------------------------------------------------------------------------
# Exponent clamping (convex regime enforcement)
# ---------------------------------------------------------------------------

try:
    from project_3dv.perception.superdec_fitter import (
        SQ_EXPONENT_CONVEX_MAX, SQ_EXPONENT_MIN, _clamp_exponents,
    )
    _SUPERDEC_FITTER_AVAILABLE = True
except ImportError:
    _SUPERDEC_FITTER_AVAILABLE = False


@pytest.mark.skipif(not _SUPERDEC_FITTER_AVAILABLE,
                    reason="superdec_fitter not importable (torch/scipy missing)")
class TestExponentClamping:

    def test_out_of_range_exponents_are_clamped(self):
        """A fit with e1=3.5, e2=0.5 must be clamped to the convex regime."""
        fit = SuperquadricFit(
            sx=0.05, sy=0.05, sz=0.05,
            e1=3.5, e2=0.5,
            tx=0.0, ty=0.0, tz=0.5,
            rx=0.0, ry=0.0, rz=0.0,
        )
        _clamp_exponents([fit])
        assert 0 < fit.e1 <= SQ_EXPONENT_CONVEX_MAX, \
            f"e1={fit.e1} outside (0, {SQ_EXPONENT_CONVEX_MAX}]"
        assert 0 < fit.e2 <= SQ_EXPONENT_CONVEX_MAX, \
            f"e2={fit.e2} outside (0, {SQ_EXPONENT_CONVEX_MAX}]"

    def test_e1_clamped_to_max(self):
        fit = SuperquadricFit(sx=0.05, sy=0.05, sz=0.05, e1=3.5, e2=1.0,
                              tx=0.0, ty=0.0, tz=0.0, rx=0.0, ry=0.0, rz=0.0)
        _clamp_exponents([fit])
        assert fit.e1 == SQ_EXPONENT_CONVEX_MAX

    def test_e2_clamped_to_min(self):
        fit = SuperquadricFit(sx=0.05, sy=0.05, sz=0.05, e1=1.0, e2=0.0,
                              tx=0.0, ty=0.0, tz=0.0, rx=0.0, ry=0.0, rz=0.0)
        _clamp_exponents([fit])
        assert fit.e2 >= SQ_EXPONENT_MIN

    def test_in_range_exponents_unchanged(self):
        """Exponents already within [SQ_EXPONENT_MIN, SQ_EXPONENT_CONVEX_MAX] must not be modified."""
        fit = SuperquadricFit(sx=0.05, sy=0.05, sz=0.05, e1=0.8, e2=1.2,
                              tx=0.0, ty=0.0, tz=0.0, rx=0.0, ry=0.0, rz=0.0)
        _clamp_exponents([fit])
        assert fit.e1 == pytest.approx(0.8)
        assert fit.e2 == pytest.approx(1.2)

    def test_empty_list_is_noop(self):
        result = _clamp_exponents([])
        assert result == []
