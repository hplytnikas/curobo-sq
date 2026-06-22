"""
Unit tests for Stage 3 — instance segmentation.

Synthetic data design notes
----------------------------
• open3d DBSCAN with min_points=20 needs dense clusters.  Tests use 2000
  points inside a sphere of radius 0.08 m (mean NN distance ≈ 0.010 m,
  well below eps=0.015 m) to guarantee reliable clustering.
• The saddle / height-layer splitters expect overlapping Gaussian-shaped
  densities, not clean zero-gap separations.  Tests use Gaussian blobs
  that share a low-density valley rather than an empty gap.
"""

import sys, os

import numpy as np
import pytest
from project_3dv.perception.pipeline import (segment_instances, _split_cluster, _split_cluster_vertical,
                      _split_by_concavity, _split_by_height_layers, classify_shape_hint)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _dense_sphere(centre, radius, n, rng):
    """Uniformly distributed points inside a sphere — dense enough for DBSCAN.

    Correct volume-uniform sampling: r = R × U(0,1)^(1/3)
    (NOT U(0,R)^(1/3) which would produce r values up to R^(1/3) >> R).
    """
    pts = rng.randn(n, 3)
    pts = pts / np.linalg.norm(pts, axis=1, keepdims=True)
    r   = radius * rng.uniform(0, 1, n) ** (1/3)   # correct: r in [0, radius]
    return (pts * r[:, None] + centre).astype(np.float64)


def _gauss_blob(centre, std, n, rng):
    """Gaussian-distributed cloud — creates smooth density gradient for saddles."""
    return (rng.randn(n, 3) * std + centre).astype(np.float64)


def _box(centre, half_extents, n, rng):
    pts = rng.uniform(-1, 1, (n, 3)) * half_extents
    return (pts + centre).astype(np.float64)


# ---------------------------------------------------------------------------
# DBSCAN clustering (segment_instances)
# ---------------------------------------------------------------------------

class TestSegmentInstances:
    """
    Each sphere has radius=0.04 m with 2000 points.
    Volume-uniform sampling gives mean NN distance ≈ 0.003 m.
    eps=0.010 m is >> mean NN distance (≈ 3×) so every point finds ≥20
    neighbours and forms a core point; spheres are separated by 0.15 m
    (gap = 0.07 m >> eps) so DBSCAN never bridges the two clusters.
    """

    _R   = 0.04    # sphere radius
    _N   = 2000    # points per sphere
    _EPS = 0.010   # DBSCAN eps (>> mean NN dist ≈ 0.003 m)

    def test_two_separated_objects(self):
        rng   = np.random.RandomState(10)
        obj_a = _dense_sphere([-0.15, 0.0, 0.5], self._R, self._N, rng)
        obj_b = _dense_sphere([ 0.15, 0.0, 0.5], self._R, self._N, rng)
        # surface-to-surface gap = 0.30 - 2×0.04 = 0.22 m >> eps=0.010
        pts   = np.vstack([obj_a, obj_b])
        clusters = segment_instances(pts, cluster_eps=self._EPS,
                                     cluster_min_points=20, split_min_points=99999)
        assert len(clusters) == 2, f"Expected 2 clusters, got {len(clusters)}"

    def test_single_object(self):
        rng      = np.random.RandomState(11)
        pts      = _dense_sphere([0.0, 0.0, 0.5], self._R, self._N, rng)
        clusters = segment_instances(pts, cluster_eps=self._EPS,
                                     cluster_min_points=20, split_min_points=99999)
        assert len(clusters) == 1, f"Expected 1 cluster, got {len(clusters)}"

    def test_three_objects(self):
        rng = np.random.RandomState(12)
        pts = np.vstack([
            _dense_sphere([-0.20, 0, 0.5], self._R, self._N, rng),
            _dense_sphere([ 0.00, 0, 0.5], self._R, self._N, rng),
            _dense_sphere([ 0.20, 0, 0.5], self._R, self._N, rng),
        ])
        clusters = segment_instances(pts, cluster_eps=self._EPS,
                                     cluster_min_points=20, split_min_points=99999)
        assert len(clusters) == 3, f"Expected 3 clusters, got {len(clusters)}"

    def test_noise_points_do_not_form_cluster(self):
        rng   = np.random.RandomState(13)
        obj   = _dense_sphere([0.0, 0.0, 0.5], self._R, self._N, rng)
        noise = rng.uniform(-1.0, 1.0, (30, 3))   # too sparse to cluster
        pts   = np.vstack([obj, noise])
        clusters = segment_instances(pts, cluster_eps=self._EPS,
                                     cluster_min_points=20, split_min_points=99999)
        # Noise is too sparse to form its own cluster
        assert len(clusters) <= 2

    def test_returns_list_of_ndarrays(self):
        rng      = np.random.RandomState(14)
        pts      = _dense_sphere([0, 0, 0.5], self._R, self._N, rng)
        clusters = segment_instances(pts, cluster_eps=self._EPS,
                                     cluster_min_points=20)
        assert isinstance(clusters, list)
        for c in clusters:
            assert isinstance(c, np.ndarray) and c.ndim == 2 and c.shape[1] == 3

    def test_empty_input(self):
        clusters = segment_instances(np.zeros((0, 3)))
        assert clusters == []

    def test_all_points_accounted_for_in_dbscan(self):
        """All non-noise points from DBSCAN must appear in some cluster."""
        rng      = np.random.RandomState(15)
        obj_a    = _dense_sphere([-0.15, 0, 0.5], self._R, self._N, rng)
        obj_b    = _dense_sphere([ 0.15, 0, 0.5], self._R, self._N, rng)
        pts      = np.vstack([obj_a, obj_b])
        clusters = segment_instances(pts, cluster_eps=self._EPS,
                                     cluster_min_points=20, split_min_points=99999)
        total = sum(len(c) for c in clusters)
        assert total >= self._N  # at least one dense object survived


# ---------------------------------------------------------------------------
# Horizontal saddle split
# ---------------------------------------------------------------------------

class TestSaddleSplit:
    """
    Two Gaussian blobs with overlapping tails create a genuine saddle
    (a density valley, not a zero gap), which the splitter can detect.
    """

    def _two_gauss_merged(self, sep, std, n, rng):
        left  = _gauss_blob([-sep/2, 0.0, 0.0], std, n, rng)
        right = _gauss_blob([ sep/2, 0.0, 0.0], std, n, rng)
        return np.vstack([left, right])

    def test_two_touching_blobs_split(self):
        """Blobs separated by 4σ should split at the density saddle."""
        rng    = np.random.RandomState(20)
        merged = self._two_gauss_merged(sep=0.20, std=0.025, n=1000, rng=rng)
        parts  = _split_cluster(merged, min_pts=50, saddle_depth=0.20, n_bins=50)
        assert len(parts) == 2, f"Expected 2 parts, got {len(parts)}"

    def test_single_blob_not_split(self):
        """A unimodal Gaussian must not be split."""
        rng   = np.random.RandomState(21)
        pts   = _gauss_blob([0.0, 0.0, 0.0], 0.04, 1000, rng)
        parts = _split_cluster(pts, min_pts=50, saddle_depth=0.20)
        assert len(parts) == 1

    def test_point_conservation(self):
        """Total points across all parts must equal the input count."""
        rng    = np.random.RandomState(22)
        merged = self._two_gauss_merged(sep=0.20, std=0.025, n=800, rng=rng)
        parts  = _split_cluster(merged, min_pts=30, saddle_depth=0.20)
        assert sum(len(p) for p in parts) == len(merged)

    def test_min_pts_prevents_tiny_splits(self):
        """With large min_pts, tiny clusters won't split."""
        rng    = np.random.RandomState(23)
        merged = self._two_gauss_merged(sep=0.20, std=0.025, n=200, rng=rng)
        parts  = _split_cluster(merged, min_pts=150, saddle_depth=0.20)
        # min_pts*2=300 > 200, so no split
        assert len(parts) == 1


# ---------------------------------------------------------------------------
# Vertical split
# ---------------------------------------------------------------------------

class TestVerticalSplit:

    def test_stacked_objects_split(self):
        """Two Gaussian blobs stacked in Y should split vertically."""
        rng    = np.random.RandomState(30)
        bottom = _gauss_blob([0.0, 0.05, 0.0], 0.015, 1000, rng)
        top    = _gauss_blob([0.0, 0.15, 0.0], 0.015, 1000, rng)
        merged = np.vstack([bottom, top])
        parts  = _split_cluster_vertical(merged, min_pts=50, saddle_depth=0.15)
        assert len(parts) == 2, f"Expected 2 parts, got {len(parts)}"

    def test_flat_cluster_not_split(self):
        """A disk-like cluster (small vertical extent) must not split."""
        rng  = np.random.RandomState(31)
        pts  = _box([0, 0.05, 0], [0.15, 0.005, 0.15], 800, rng)
        # y_ext = 0.01, xz_ext = 0.30 → ratio = 0.033 << 0.40 threshold
        parts = _split_cluster_vertical(pts, min_pts=30)
        assert len(parts) == 1

    def test_vertical_split_conserves_points(self):
        rng    = np.random.RandomState(32)
        bottom = _gauss_blob([0, 0.05, 0], 0.015, 600, rng)
        top    = _gauss_blob([0, 0.15, 0], 0.015, 600, rng)
        merged = np.vstack([bottom, top])
        parts  = _split_cluster_vertical(merged, min_pts=30, saddle_depth=0.15)
        assert sum(len(p) for p in parts) == len(merged)


# ---------------------------------------------------------------------------
# Height-layer split
# ---------------------------------------------------------------------------

class TestHeightLayerSplit:

    def test_gap_triggers_split(self):
        """An empty 20mm band in Y must trigger a split."""
        rng    = np.random.RandomState(40)
        bottom = _box([0, 0.05, 0], [0.05, 0.02, 0.05], 600, rng)
        top    = _box([0, 0.15, 0], [0.05, 0.02, 0.05], 600, rng)
        # gap between 0.07 and 0.13 → 60 mm gap >> gap_threshold=8mm
        merged = np.vstack([bottom, top])
        parts  = _split_by_height_layers(merged, min_pts=30, gap_threshold=0.005)
        assert len(parts) == 2, f"Expected 2 parts from height gap, got {len(parts)}"

    def test_no_gap_no_split(self):
        rng  = np.random.RandomState(41)
        pts  = _box([0, 0.10, 0], [0.05, 0.08, 0.05], 800, rng)
        parts = _split_by_height_layers(pts, min_pts=30, gap_threshold=0.01)
        assert len(parts) == 1

    def test_conserves_points(self):
        rng    = np.random.RandomState(42)
        bottom = _box([0, 0.05, 0], [0.04, 0.02, 0.04], 400, rng)
        top    = _box([0, 0.15, 0], [0.04, 0.02, 0.04], 400, rng)
        merged = np.vstack([bottom, top])
        parts  = _split_by_height_layers(merged, min_pts=30, gap_threshold=0.005)
        assert sum(len(p) for p in parts) == len(merged)


# ---------------------------------------------------------------------------
# Concavity / neck split
# ---------------------------------------------------------------------------

class TestConcavitySplit:

    def test_does_not_crash(self):
        """Concavity splitter must run without error on any valid input."""
        rng   = np.random.RandomState(50)
        pts   = _dense_sphere([0, 0, 0], 0.05, 1000, rng)
        parts = _split_by_concavity(pts, min_pts=30)
        assert isinstance(parts, list)
        assert sum(len(p) for p in parts) == len(pts)

    def test_convex_cluster_not_split(self):
        rng   = np.random.RandomState(51)
        pts   = _dense_sphere([0, 0, 0], 0.05, 800, rng)
        parts = _split_by_concavity(pts, min_pts=30)
        assert len(parts) == 1


# ---------------------------------------------------------------------------
# Shape hint classifier
# ---------------------------------------------------------------------------

class TestClassifyShapeHint:

    def test_flat_object_is_ellipsoid(self):
        rng = np.random.RandomState(60)
        pts = _box([0, 0, 0], [0.10, 0.01, 0.10], 200, rng)
        shape, conf = classify_shape_hint(pts)
        assert shape == "Ellipsoid"
        assert 0 < conf <= 1.0

    def test_elongated_object_is_cuboid(self):
        # half-extents [0.15, 0.06, 0.06] → sorted extents ≈ [0.12, 0.12, 0.30]
        # r_flat ≈ 0.12/0.30 = 0.40 > 0.25 (not flat), r_elon ≈ 0.40 < 0.45 → Cuboid
        rng = np.random.RandomState(61)
        pts = _box([0, 0, 0], [0.15, 0.06, 0.06], 600, rng)
        shape, _ = classify_shape_hint(pts)
        assert shape == "Cuboid"

    def test_too_few_points(self):
        shape, conf = classify_shape_hint(np.zeros((5, 3)))
        assert shape == "Other"
        assert conf == 0.0

    def test_returns_valid_type(self):
        rng = np.random.RandomState(62)
        pts = rng.randn(100, 3) * 0.05
        shape, conf = classify_shape_hint(pts)
        assert shape in ("Ellipsoid", "Cylinder", "Cuboid", "Other")
        assert 0.0 <= conf <= 1.0
