"""
Unit tests for Stage 2 — table / background removal.

Coordinate convention used throughout
--------------------------------------
Camera at origin, looking in +Z direction (standard RGB-D / OpenCV frame).
Objects sit on a table at Z = table_z.  Objects "above" the table = closer
to the camera = smaller Z than table_z.  This matches OCID and the
remove_table() sign convention after RANSAC normal flipping.

All tests use synthetic point clouds; no dataset files are required.
"""

import sys, os

import numpy as np
import pytest
from project_3dv.perception.pipeline import remove_table, depth_to_pointcloud


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _tabletop_scene(
    n_table: int = 2000,
    n_obj:   int = 500,
    table_z: float = 0.80,
    obj_dz:  float = 0.10,     # how far in FRONT of the table (smaller Z)
    noise_std: float = 0.002,
    rng: np.random.RandomState = None,
) -> np.ndarray:
    """
    Camera at origin, table at Z = table_z.
    Objects sit in front of the table (Z = table_z - obj_dz).
    XY coordinates scattered around origin.
    """
    if rng is None:
        rng = np.random.RandomState(42)

    table_pts = rng.uniform(-0.4, 0.4, (n_table, 3))
    table_pts[:, 2] = table_z + rng.randn(n_table) * noise_std

    obj_pts = rng.uniform(-0.05, 0.05, (n_obj, 3))
    obj_pts[:, 2] = table_z - obj_dz + rng.randn(n_obj) * noise_std

    return np.vstack([table_pts, obj_pts]).astype(np.float64)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestRemoveTable:

    def test_finds_table_plane(self):
        """RANSAC must find the synthetic plane; at least 1000 inliers expected."""
        rng  = np.random.RandomState(0)
        pts  = _tabletop_scene(rng=rng)
        _, normal, height, _, n_table = remove_table(pts)
        assert n_table >= 1000, f"Expected ≥1000 table inliers, got {n_table}"

    def test_table_normal_points_in_z(self):
        """After normalisation the dominant component should be Z (camera forward)."""
        rng  = np.random.RandomState(1)
        pts  = _tabletop_scene(rng=rng)
        _, normal, _, _, _ = remove_table(pts)
        assert abs(normal[2]) > 0.9, f"Expected Z-dominant normal, got {normal}"

    def test_object_points_returned(self):
        """Objects in front of the table (closer to camera) should be returned."""
        rng   = np.random.RandomState(2)
        n_obj = 500
        pts   = _tabletop_scene(n_obj=n_obj, obj_dz=0.10, rng=rng)
        obj_pts, _, _, _, _ = remove_table(
            pts,
            min_height_above_table=0.005,
            max_height_above_table=0.30,
        )
        assert len(obj_pts) >= n_obj * 0.50, \
            f"Expected ≥{n_obj*0.50:.0f} object pts, got {len(obj_pts)}"

    def test_table_inliers_excluded(self):
        """Table-plane points must not appear in the returned object cloud."""
        rng   = np.random.RandomState(3)
        pts   = _tabletop_scene(n_obj=200, rng=rng)
        obj_pts, normal, height, _, _ = remove_table(
            pts, min_height_above_table=0.01, max_height_above_table=0.30)
        if len(obj_pts) == 0:
            return
        signed = obj_pts @ normal - height
        assert signed.min() > 0.005, \
            f"Some returned pts at/below table: min signed dist = {signed.min():.4f}"

    def test_depth_gate_removes_background(self):
        """Background points behind the table (large Z) must be filtered out."""
        rng = np.random.RandomState(4)
        pts = _tabletop_scene(rng=rng)
        # inject background: Z much larger than table_z=0.8
        bg          = rng.uniform(-0.2, 0.2, (200, 3))
        bg[:, 2]    = 3.5
        pts_with_bg = np.vstack([pts, bg])

        obj_pts, _, _, _, _ = remove_table(
            pts_with_bg, depth_margin=0.30,
            min_height_above_table=0.005, max_height_above_table=0.30)
        if len(obj_pts) > 0:
            assert obj_pts[:, 2].max() < 2.0, \
                "Background points (Z>3 m) survived the depth gate"

    def test_plane_hint_matches_ransac(self):
        """plane_hint should give the same foreground point count as RANSAC."""
        rng   = np.random.RandomState(5)
        pts   = _tabletop_scene(rng=rng)
        obj1, normal, height, _, _ = remove_table(pts)
        obj2, _, _, _, _            = remove_table(pts, plane_hint=(normal, height))
        # Allow ±5 points difference (numerical precision)
        assert abs(len(obj1) - len(obj2)) <= 5, \
            f"plane_hint gave {len(obj2)} pts vs RANSAC {len(obj1)}"

    def test_xy_radius_crop(self):
        """Points with XY radius > xy_radius must be discarded."""
        rng = np.random.RandomState(6)
        pts = _tabletop_scene(n_obj=100, rng=rng)
        # inject far-away points
        far       = rng.uniform(0.01, 0.05, (100, 3))
        far[:, :2] += 2.0    # XY ≫ 0.8 m limit
        far[:, 2]   = 0.70   # in front of table
        pts_far   = np.vstack([pts, far])

        obj_pts, _, _, _, _ = remove_table(pts_far, xy_radius=0.80)
        if len(obj_pts) > 0:
            xy_dist = np.linalg.norm(obj_pts[:, :2], axis=1)
            assert xy_dist.max() <= 0.80 + 1e-6, \
                f"Point beyond xy_radius survived: max={xy_dist.max():.3f}"

    def test_height_filter_bounds(self):
        """Points too far above the table must be clipped by max_height_above_table."""
        rng = np.random.RandomState(7)
        pts = _tabletop_scene(n_obj=0, rng=rng)  # table only
        # inject points very far in front of table (e.g., camera body)
        near = rng.uniform(-0.1, 0.1, (200, 3))
        near[:, 2] = 0.10   # 0.70 m in front of table at 0.80 — exceeds 0.30 max
        pts_near = np.vstack([pts, near])

        obj_pts, _, _, _, _ = remove_table(
            pts_near, min_height_above_table=0.005, max_height_above_table=0.30)
        # The points at z=0.10 are 0.70 m in front of the table → filtered
        if len(obj_pts) > 0:
            # all returned points must be within max_height_above_table
            # (approximate check — just verify no Z < 0.5 for table at Z=0.8)
            assert obj_pts[:, 2].min() >= 0.50, \
                "Far-in-front points survived max_height filter"


class TestDepthToPointcloud:

    def _K(self, fx=500.0, fy=500.0, cx=32.0, cy=24.0):
        return np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)

    def test_central_pixel_unprojects_correctly(self):
        """Central pixel at known depth should unproject to (0, 0, depth_m)."""
        K     = self._K(fx=500, fy=500, cx=4, cy=4)
        depth = np.zeros((9, 9), dtype=np.uint16)
        depth[4, 4] = 1000  # 1000 mm = 1.0 m

        pts = depth_to_pointcloud(depth, K, depth_scale=1000.0)
        assert len(pts) == 1
        np.testing.assert_allclose(pts[0], [0.0, 0.0, 1.0], atol=1e-6)

    def test_output_shape(self):
        H, W = 32, 32
        K    = self._K(fx=200, fy=200, cx=W/2, cy=H/2)
        depth = (np.ones((H, W)) * 500).astype(np.uint16)  # 0.5 m
        pts   = depth_to_pointcloud(depth, K, depth_scale=1000.0)
        assert pts.shape == (H * W, 3)

    def test_zero_depth_excluded(self):
        """Zero-depth pixels must produce no 3-D points."""
        K     = self._K()
        depth = np.zeros((10, 10), dtype=np.uint16)
        pts   = depth_to_pointcloud(depth, K)
        assert len(pts) == 0

    def test_max_depth_cutoff(self):
        """Points beyond max_depth (in metres) must be filtered out."""
        K     = self._K(fx=500, fy=500, cx=4, cy=4)
        depth = np.zeros((9, 9), dtype=np.uint16)
        depth[4, 4] = 5000  # 5.0 m — beyond default max_depth=4
        pts   = depth_to_pointcloud(depth, K, depth_scale=1000.0, max_depth=4.0)
        assert len(pts) == 0

    def test_off_centre_pixel(self):
        """Off-centre pixel should have correct x = (u - cx) / fx * z."""
        K  = self._K(fx=500, fy=500, cx=4, cy=4)
        H, W = 9, 9
        depth = np.zeros((H, W), dtype=np.uint16)
        depth[4, 6] = 1000  # u=6, v=4, depth=1.0 m
        pts = depth_to_pointcloud(depth, K, depth_scale=1000.0)
        assert len(pts) == 1
        expected_x = (6 - 4) / 500 * 1.0   # = 0.004
        np.testing.assert_allclose(pts[0, 0], expected_x, atol=1e-6)
        np.testing.assert_allclose(pts[0, 2], 1.0, atol=1e-6)
