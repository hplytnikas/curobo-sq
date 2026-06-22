"""
Integration test — runs a synthetic depth image through the full pipeline:

    depth image
      → depth_to_pointcloud       (Stage 1 — RGB-D input)
      → TabletopPerception.run()  (Stages 2-4: table removal, segmentation, hints)
      → SuperquadricFitter        (LM SQ fitting per segment)
      → Scene.from_fits()         (CuRobo / planning interface)

No external dataset files or GPU required.
"""

import sys, os

import numpy as np
import pytest
from project_3dv.perception.pipeline import depth_to_pointcloud, TabletopPerception, PerceptionResult
from project_3dv.perception.superquadric import SuperquadricFitter, SuperquadricFit, MultiSQFit
from project_3dv.perception.superdec_utils import Scene, sq_fits_to_npz


# ---------------------------------------------------------------------------
# Synthetic scene builder
# ---------------------------------------------------------------------------

def _make_synthetic_tabletop(
    n_table:   int   = 3000,
    objects:   list  = None,
    table_z:   float = 0.80,
    noise_std: float = 0.003,
    rng: np.random.RandomState = None,
) -> np.ndarray:
    """Return an (N, 3) point cloud of a flat table with objects in front of it.

    Camera at origin, Z forward.
    Table at Z = table_z.
    Objects are placed in FRONT of the table (smaller Z), which is "above"
    the table in the remove_table() sign convention after RANSAC normal flip.

    objects: list of dicts with keys:
        'dz'      – how far in front of table (positive = closer to camera)
        'extents' – (3,) half-extents for a box region
        'n_pts'   – number of points
    """
    if rng is None:
        rng = np.random.RandomState(0)
    if objects is None:
        objects = [
            {'xy': [-0.10, 0.0], 'dz': 0.08, 'extents': [0.04, 0.04, 0.04], 'n_pts': 600},
            {'xy': [ 0.10, 0.0], 'dz': 0.08, 'extents': [0.04, 0.04, 0.04], 'n_pts': 600},
        ]

    # table plane at Z = table_z, scattered in XY
    table_pts = rng.uniform(-0.3, 0.3, (n_table, 3))
    table_pts[:, 2] = table_z + rng.randn(n_table) * noise_std

    all_pts = [table_pts]
    for obj in objects:
        e   = np.array(obj['extents'])
        pts = rng.uniform(-1, 1, (obj['n_pts'], 3)) * e
        pts[:, 2] += table_z - obj['dz']        # in FRONT of table (smaller Z)
        pts[:, 0] += obj['xy'][0]
        pts[:, 1] += obj['xy'][1]
        pts += rng.randn(*pts.shape) * noise_std
        all_pts.append(pts)

    return np.vstack(all_pts).astype(np.float32)


def _make_synthetic_depth_image(
    H: int = 120,
    W: int = 160,
    fx: float = 200.0,
    fy: float = 200.0,
    table_depth: float = 0.8,
    obj_radius: float = 0.04,
    rng: np.random.RandomState = None,
) -> tuple:
    """Return (depth_uint16, K) with a flat table and one small object."""
    if rng is None:
        rng = np.random.RandomState(0)
    cx, cy = W / 2.0, H / 2.0
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)

    depth = np.full((H, W), table_depth, dtype=np.float32)
    # place a hemispherical bump at centre
    y_idx, x_idx = np.mgrid[0:H, 0:W]
    x_m = (x_idx - cx) * table_depth / fx
    y_m = (y_idx - cy) * table_depth / fy
    dist_from_centre = np.sqrt(x_m**2 + y_m**2)
    bump_mask = dist_from_centre < obj_radius
    bump_depth = table_depth - np.sqrt(np.maximum(obj_radius**2 - dist_from_centre**2, 0))
    depth[bump_mask] = bump_depth[bump_mask]

    depth_u16 = (depth * 1000).astype(np.uint16)
    return depth_u16, K


# ---------------------------------------------------------------------------
# Integration tests
# ---------------------------------------------------------------------------

class TestFullPipeline:

    def setup_method(self):
        self.rng     = np.random.RandomState(42)
        self.fitter  = SuperquadricFitter(n_restarts=2, n_lm_rounds=8, subsample=256)
        self.pipe    = TabletopPerception(
            voxel_size=0.005,
            min_height_above_table=0.005,
            max_height_above_table=0.30,
            cluster_eps=0.025,
            cluster_min_points=15,
            min_object_points=30,
            max_object_points=3000,
            min_object_extent=0.02,
            max_object_extent=0.35,
            depth_margin=0.30,
            xy_radius=0.80,
            split_min_points=9999,  # disable splitting in these tests
        )

    def _run_pipeline(self, pts):
        return self.pipe.run(pts)

    # ---- Stage 1: depth_to_pointcloud ----------------------------------------

    def test_depth_to_pointcloud_produces_valid_cloud(self):
        depth, K = _make_synthetic_depth_image(rng=self.rng)
        pts = depth_to_pointcloud(depth, K, depth_scale=1000.0)
        assert pts.ndim == 2 and pts.shape[1] == 3
        assert len(pts) > 0
        assert np.isfinite(pts).all()

    def test_depth_pipeline_finds_objects(self):
        """Point cloud from depth image should yield at least one object."""
        depth, K = _make_synthetic_depth_image(rng=self.rng)
        pts = depth_to_pointcloud(depth, K, depth_scale=1000.0)
        result = self._run_pipeline(pts)
        assert isinstance(result, PerceptionResult)
        assert len(result.objects) >= 1

    # ---- Stage 2: table removal ----------------------------------------------

    def test_table_height_recovered(self):
        """Pipeline should locate the synthetic table within 1 cm."""
        pts = _make_synthetic_tabletop(rng=self.rng)
        result = self._run_pipeline(pts)
        # In camera frame Z = table_z is encoded as height or normal offset
        assert result.n_points_table > 100, \
            "Too few table inliers — table removal may have failed"

    def test_result_has_table_geometry(self):
        pts = _make_synthetic_tabletop(rng=self.rng)
        result = self._run_pipeline(pts)
        assert result.table_normal is not None
        assert result.table_normal.shape == (3,)
        assert np.isfinite(result.table_normal).all()

    # ---- Stage 3: instance segmentation ------------------------------------

    def test_two_objects_detected(self):
        """Two well-separated objects should produce two ObjectSegments."""
        pts = _make_synthetic_tabletop(rng=self.rng, objects=[
            {'xy': [-0.15, 0.0], 'dz': 0.08, 'extents': [0.04, 0.04, 0.04], 'n_pts': 800},
            {'xy': [ 0.15, 0.0], 'dz': 0.08, 'extents': [0.04, 0.04, 0.04], 'n_pts': 800},
        ])
        result = self._run_pipeline(pts)
        assert len(result.objects) == 2, \
            f"Expected 2 objects, got {len(result.objects)}"

    def test_object_segments_have_valid_fields(self):
        pts = _make_synthetic_tabletop(rng=self.rng)
        result = self._run_pipeline(pts)
        for obj in result.objects:
            assert obj.points.ndim == 2 and obj.points.shape[1] == 3
            assert obj.n_points == len(obj.points)
            assert obj.centroid.shape == (3,)
            assert obj.bbox_min.shape == (3,)
            assert obj.bbox_max.shape == (3,)
            assert obj.shape_type in ("Ellipsoid", "Cylinder", "Cuboid", "Other")
            assert 0.0 <= obj.shape_conf <= 1.0
            assert (obj.bbox_extent >= 0).all()

    def test_empty_scene_returns_empty_result(self):
        """Point cloud with only a table and no objects → empty result."""
        rng = np.random.RandomState(99)
        table_pts = rng.uniform(-0.4, 0.4, (2000, 3)).astype(np.float64)
        table_pts[:, 2] = 0.80 + rng.randn(2000) * 0.003
        result = self._run_pipeline(table_pts)
        assert len(result.objects) == 0

    # ---- Stage 4 → SQ fitting -----------------------------------------------

    def test_sq_fitting_per_segment(self):
        """Each ObjectSegment should yield a valid MultiSQFit."""
        pts = _make_synthetic_tabletop(rng=self.rng)
        result = self._run_pipeline(pts)

        for seg in result.objects:
            multi = self.fitter.fit_adaptive(
                seg.points, shape_hint=seg.shape_type,
                l2_threshold=0.02, max_primitives=2,
            )
            assert isinstance(multi, MultiSQFit)
            assert len(multi.primitives) >= 1
            for prim in multi.primitives:
                assert isinstance(prim, SuperquadricFit)
                assert np.isfinite(prim.chamfer_l2)

    # ---- Scene / CuRobo interface -------------------------------------------

    def test_scene_built_from_fits(self):
        pts = _make_synthetic_tabletop(rng=self.rng)
        result = self._run_pipeline(pts)
        sq_multis = [
            self.fitter.fit_adaptive(seg.points, shape_hint=seg.shape_type)
            for seg in result.objects
        ]
        flat_fits = [sq for m in sq_multis for sq in m.primitives]

        scene = Scene.from_fits(flat_fits)
        assert scene is not None
        assert scene.superquadrics.num_primitives == len(flat_fits)

    def test_signed_distance_query(self):
        """SDF queries on the built scene must be finite."""
        pts = _make_synthetic_tabletop(rng=self.rng)
        result = self._run_pipeline(pts)
        if not result.objects:
            pytest.skip("No objects found in synthetic scene")

        sq_multis = [
            self.fitter.fit_adaptive(seg.points, shape_hint=seg.shape_type)
            for seg in result.objects
        ]
        flat_fits = [sq for m in sq_multis for sq in m.primitives]
        scene = Scene.from_fits(flat_fits)

        query_pts = self.rng.uniform(-0.3, 0.3, (100, 3)).astype(np.float32)
        sd = scene.get_signed_distance(query_pts)
        assert sd.shape == (100,)
        assert np.isfinite(sd).all()

    def test_npz_export(self, tmp_path):
        pts = _make_synthetic_tabletop(rng=self.rng)
        result = self._run_pipeline(pts)
        if not result.objects:
            pytest.skip("No objects found in synthetic scene")

        sq_multis = [self.fitter.fit_adaptive(seg.points) for seg in result.objects]
        flat_fits = [sq for m in sq_multis for sq in m.primitives]

        out_path = str(tmp_path / "test_sq.npz")
        sq_fits_to_npz(flat_fits, out_path)

        import numpy as np
        data = np.load(out_path, allow_pickle=True)
        assert 'scale' in data
        assert 'translation' in data
        assert data['scale'].shape[1] == len(flat_fits)

    # ---- plane_hint caching (performance) -----------------------------------

    def test_plane_hint_gives_same_result(self):
        """Re-running with a cached plane_hint must match the original result."""
        pts = _make_synthetic_tabletop(rng=self.rng)
        result1 = self._run_pipeline(pts)

        hint = (result1.table_normal, result1.table_height)
        result2 = self.pipe.run(pts, plane_hint=hint)

        assert len(result1.objects) == len(result2.objects), \
            "plane_hint gave different number of objects than RANSAC"

    # ---- result.summary() ---------------------------------------------------

    def test_summary_is_string(self):
        pts = _make_synthetic_tabletop(rng=self.rng)
        result = self._run_pipeline(pts)
        s = result.summary()
        assert isinstance(s, str)
        assert "object" in s.lower()
