"""
superdec_utils.py — SUPERDEC inference utilities, adapted for our pipeline.

Original source: Fedele et al., "SUPERDEC: 3D Scene Decomposition with
Superquadric Primitives", ETH Zurich / Stanford, 2025.
https://super-dec.github.io

Changes from original:
  - Added sq_fits_to_npz(): converts our SuperquadricFit list → .npz format
  - Added load_from_fits(): constructs Superquadrics directly from fit list
  - Removed trimesh dependency (save_ply uses plyfile only)
  - Added get_sdf(): returns signed distance (negative inside) for CuRobo
  - Fixed rotation convention to match our euler→matrix output
  - shape_type and shape_conf from pipeline classifier carried through
    to obstacle dicts consumed by CuRobo

Usage:
    # From our pipeline output:
    from superdec_utils import sq_fits_to_npz, Superquadrics, Scene
    from pipeline import TabletopPerception
    from superquadric import SuperquadricFitter

    perception = TabletopPerception(classifier_path="sq_shape_classifier.pkl")
    fitter     = SuperquadricFitter()

    result     = perception.run(points, rgb=colors)
    sq_multis  = [fitter.fit_adaptive(seg.points, shape_hint=seg.shape_type)
                  for seg in result.objects]
    flat_fits  = [sq for m in sq_multis for sq in m.primitives]

    scene = Scene.from_fits(flat_fits)
    sd    = scene.get_signed_distance(query_pts)   # CuRobo entry point
    scene.save_superquadrics_vis("scene.ply")
"""

import numpy as np
from plyfile import PlyData, PlyElement
import random
import colorsys
from typing import List


# ---------------------------------------------------------------------------
# Colour generation (original SUPERDEC)
# ---------------------------------------------------------------------------

def generate_ncolors(num):
    def get_n_hls_colors(num):
        hls_colors = []
        i    = 0
        step = 360.0 / num if num > 0 else 360.0
        while i < 360:
            h = i
            s = 90 + random.random() * 10
            l = 50 + random.random() * 10
            hls_colors.append([h / 360.0, l / 100.0, s / 100.0])
            i += step
        return hls_colors

    rgb_colors = np.zeros((0, 3), dtype=np.uint8)
    if num < 1:
        return rgb_colors
    for hlsc in get_n_hls_colors(num):
        r, g, b = colorsys.hls_to_rgb(hlsc[0], hlsc[1], hlsc[2])
        rgb_colors = np.concatenate(
            (rgb_colors, np.array([[int(r*255), int(g*255), int(b*255)]], dtype=np.uint8))
        )
    return rgb_colors


# ---------------------------------------------------------------------------
# Shape-type colours (override random colours for visualisation)
# Makes it easy to see what the classifier decided in MeshLab/open3d
# ---------------------------------------------------------------------------

SQ_TYPE_COLORS = {
    "Ellipsoid": np.array([[ 80, 200,  80]], dtype=np.uint8),   # green
    "Cylinder":  np.array([[255, 140,   0]], dtype=np.uint8),   # orange
    "Cuboid":    np.array([[ 30, 144, 255]], dtype=np.uint8),   # blue
    "Other":     np.array([[200,  50,  50]], dtype=np.uint8),   # red
}


# ---------------------------------------------------------------------------
# Bridge: our SuperquadricFit list → SUPERDEC .npz format
# ---------------------------------------------------------------------------

def sq_fits_to_npz(fits, path: str):
    """
    Convert a flat list of SuperquadricFit objects to SUPERDEC's .npz format.

    Args:
        fits: List[SuperquadricFit] — flattened (all primitives from all objects)
        path: output .npz path
    """
    N     = len(fits)
    exist = np.ones((1, N, 1),  dtype=np.float32)
    scale = np.array([[f.scale           for f in fits]], dtype=np.float32)
    exps  = np.array([[[f.e1, f.e2]      for f in fits]], dtype=np.float32)
    rots  = np.array([[f.rotation_matrix for f in fits]], dtype=np.float32)
    trans = np.array([[f.translation     for f in fits]], dtype=np.float32)
    types = np.array([[f.shape_type      for f in fits]])   # (1, N) str
    confs = np.array([[f.shape_conf      for f in fits]], dtype=np.float32)

    np.savez(path, exist=exist, scale=scale, exponents=exps,
             rotation=rots, translation=trans,
             shape_type=types, shape_conf=confs)
    print(f"Saved {N} superquadrics → {path}")


# ---------------------------------------------------------------------------
# Superquadrics class (original SUPERDEC, extended)
# ---------------------------------------------------------------------------

class Superquadrics:
    """
    Loads and operates on a set of superquadric primitives.

    Supports:
      - Loading from SUPERDEC .npz (original format)
      - Loading directly from our SuperquadricFit list
      - Radial distance / closest point computation (for CuRobo SDF)
      - Signed distance (negative inside, for collision checking)
      - Surface vertex sampling (for visualization)
      - PLY mesh export coloured by shape type
    """

    def __init__(self, path_to_sq_parameters: str):
        print(f"Loading superquadrics from {path_to_sq_parameters}")
        parameters = np.load(path_to_sq_parameters, allow_pickle=True)
        self._load_from_arrays(
            exist  = parameters['exist'],
            scales = parameters['scale'],
            shapes = parameters['exponents'],
            rots   = parameters['rotation'],
            trans  = parameters['translation'],
            types  = parameters.get('shape_type', None),
            confs  = parameters.get('shape_conf', None),
        )

    @classmethod
    def from_fits(cls, fits) -> 'Superquadrics':
        """
        Construct directly from a list of SuperquadricFit objects.
        No file I/O needed.

        Example:
            sq_multis  = [fitter.fit_adaptive(seg.points, shape_hint=seg.shape_type)
                          for seg in result.objects]
            flat_fits  = [sq for m in sq_multis for sq in m.primitives]
            sq         = Superquadrics.from_fits(flat_fits)
            sq.save_ply('scene.ply')
        """
        obj = cls.__new__(cls)
        N   = len(fits)
        obj._load_from_arrays(
            exist  = np.ones((1, N, 1), dtype=np.float32),
            scales = np.array([[f.scale           for f in fits]], dtype=np.float32),
            shapes = np.array([[[f.e1, f.e2]      for f in fits]], dtype=np.float32),
            rots   = np.array([[f.rotation_matrix for f in fits]], dtype=np.float32),
            trans  = np.array([[f.translation     for f in fits]], dtype=np.float32),
            types  = np.array([[f.shape_type      for f in fits]]),
            confs  = np.array([[f.shape_conf      for f in fits]], dtype=np.float32),
        )
        return obj

    def _load_from_arrays(self, exist, scales, shapes, rots, trans,
                          types=None, confs=None):
        N, K, _ = exist.shape
        total   = N * K
        mask    = (exist[..., 0] >= 0.5).reshape(total)

        def flatten(x):
            return x.reshape(total, -1) if x.ndim > 2 else x.reshape(total, 1)

        self.scales         = flatten(scales)[mask]
        self.shapes         = flatten(shapes)[mask]
        self.rotations      = flatten(rots)[mask].reshape(-1, 3, 3)
        self.translations   = flatten(trans)[mask]
        self.num_primitives = self.scales.shape[0]

        if types is not None:
            flat_types       = types.reshape(total)[mask]
            self.shape_types = flat_types.tolist()
        else:
            self.shape_types = ["Other"] * self.num_primitives

        if confs is not None:
            self.shape_confs = flatten(confs)[mask].ravel().tolist()
        else:
            self.shape_confs = [0.0] * self.num_primitives

        # colour by shape type for visualisation
        self.colors = np.vstack([
            SQ_TYPE_COLORS.get(t, SQ_TYPE_COLORS["Other"])
            for t in self.shape_types
        ])

    # ── Core geometry ────────────────────────────────────────────────────────

    def move_to_sq_frame(self, points: np.ndarray) -> np.ndarray:
        """Transform world points into each SQ's canonical frame. → (M, P, 3)"""
        pc_inver = points[None, ...] - self.translations[:, None, :]
        pc_inver = np.einsum(
            'abc,acd->abd',
            self.rotations.transpose(0, 2, 1),
            pc_inver.transpose(0, 2, 1)
        ).transpose(0, 2, 1)
        return pc_inver

    def _implicit(self, pc_canon: np.ndarray) -> np.ndarray:
        """Evaluate SQ implicit f(x,y,z). f<1: inside, f=1: surface, f>1: outside."""
        e1 = np.clip(self.shapes[:, 0:1], 0.1, 2.0)
        e2 = np.clip(self.shapes[:, 1:2], 0.1, 2.0)
        sx = np.clip(self.scales[:, 0:1], 1e-4, None)
        sy = np.clip(self.scales[:, 1:2], 1e-4, None)
        sz = np.clip(self.scales[:, 2:3], 1e-4, None)

        x = pc_canon[..., 0]
        y = pc_canon[..., 1]
        z = pc_canon[..., 2]

        xy = (np.clip(np.abs(x / sx), 1e-10, None) ** (2 / e2) +
              np.clip(np.abs(y / sy), 1e-10, None) ** (2 / e2))
        xy = np.clip(xy, 1e-10, None) ** (e2 / e1)
        zz = np.clip(np.abs(z / sz), 1e-10, None) ** (2 / e1)
        return xy + zz

    def get_radial_distance_and_closest_points(self, points: np.ndarray):
        """
        Radial distance from each point to its nearest SQ surface.
        Original SUPERDEC interface — used by CuRobo / Scene class.
        """
        def get_directions_to_centers(indices):
            vec  = self.translations[indices] - points
            norm = np.linalg.norm(vec, axis=1, keepdims=True)
            return vec / (norm + 1e-9)

        pc_inver = self.move_to_sq_frame(points)
        r_norm   = np.sqrt(np.sum(pc_inver ** 2, axis=-1))

        e1 = self.shapes[:, 0:1]
        f  = self._implicit(pc_inver)
        e  = np.clip(f, 1e-10, None) ** (-e1 / 2) - 1

        rad_res = r_norm * np.abs(e)
        min_idx = np.argmin(rad_res, axis=0)
        rad_min = rad_res[min_idx, np.arange(len(points))]

        vec            = get_directions_to_centers(min_idx)
        closest_points = points + vec * rad_min[:, None]
        return rad_min, closest_points

    def get_signed_distance(self, points: np.ndarray) -> np.ndarray:
        """
        Signed distance to nearest SQ surface.
        Negative inside, positive outside.
        CuRobo collision checker entry point.
        """
        pc_inver = self.move_to_sq_frame(points)
        f        = self._implicit(pc_inver)
        e1       = self.shapes[:, 0:1]

        r_norm  = np.sqrt(np.sum(pc_inver ** 2, axis=-1))
        e       = np.clip(f, 1e-10, None) ** (-e1 / 2) - 1
        rad_res = r_norm * np.abs(e)
        signs   = np.sign(f - 1.0)
        sd      = signs * rad_res
        return sd.min(axis=0)

    # ── Surface sampling ──────────────────────────────────────────────────────

    def get_vertices(self, N: int = 20) -> np.ndarray:
        """Sample N×N surface points per primitive. → (M, N², 3)"""
        def f(o, m):
            return np.sign(np.sin(o)) * np.abs(np.sin(o)) ** m[..., None]
        def g(o, m):
            return np.sign(np.cos(o)) * np.abs(np.cos(o)) ** m[..., None]

        eps = np.pi / (4 * N)
        u = np.tile(np.linspace(-np.pi, np.pi, N, endpoint=False), N)
        v = np.repeat(np.linspace(-np.pi/2 + eps, np.pi/2 - eps, N, endpoint=True), N)

        x = self.scales[..., 0, None] * g(u, self.shapes[..., 0]) * g(v, self.shapes[..., 1])
        y = self.scales[..., 1, None] * g(u, self.shapes[..., 0]) * f(v, self.shapes[..., 1])
        z = self.scales[..., 2, None] * f(u, self.shapes[..., 0])

        verts = np.stack([x, y, z], axis=-1)
        verts = np.einsum('...jk,...ik->...ij', self.rotations, verts) + self.translations[:, None, :]
        return verts

    # ── PLY export ────────────────────────────────────────────────────────────

    def save_ply(self, output_path: str, resolution: int = 20):
        """
        Export superquadric meshes as a coloured PLY file.
        Primitives are coloured by shape type:
          green=Ellipsoid, orange=Cylinder, blue=Cuboid, red=Other
        """
        vertices = self.get_vertices(resolution)
        n_pts    = resolution ** 2
        R        = resolution

        tmp_vertex = np.zeros(
            self.num_primitives * n_pts,
            dtype=[('x','f4'),('y','f4'),('z','f4'),
                   ('red','u1'),('green','u1'),('blue','u1')]
        )
        for v1 in range(self.num_primitives):
            for v2 in range(n_pts):
                idx = v1 * n_pts + v2
                tmp_vertex[idx] = (
                    vertices[v1, v2, 0],
                    vertices[v1, v2, 1],
                    vertices[v1, v2, 2],
                    self.colors[v1, 0],
                    self.colors[v1, 1],
                    self.colors[v1, 2],
                )

        triangles, tri_colors = [], []
        for k in range(self.num_primitives):
            base = k * n_pts
            for i in range(R - 1):
                for j in range(R - 1):
                    a = base + i * R + j
                    b = base + i * R + (j + 1) % R
                    c = base + (i + 1) * R + j
                    d = base + (i + 1) * R + (j + 1) % R
                    triangles  += [[a, b, c], [b, d, c]]
                    tri_colors += [self.colors[k], self.colors[k]]
            for i in range(R - 1):
                a = base + i * R + (R - 1)
                b = base + i * R + 0
                c = base + (i + 1) * R + (R - 1)
                d = base + (i + 1) * R + 0
                triangles  += [[a, b, c], [b, d, c]]
                tri_colors += [self.colors[k], self.colors[k]]

        tmp_tri = np.zeros(
            len(triangles),
            dtype=[('vertex_indices', 'i4', (3,)),
                   ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]
        )
        for i, (tri, col) in enumerate(zip(triangles, tri_colors)):
            tmp_tri[i] = (tri, col[0], col[1], col[2])

        ply_out = PlyData([
            PlyElement.describe(tmp_vertex, 'vertex', comments=['vertices']),
            PlyElement.describe(tmp_tri, 'face'),
        ], text=True)
        ply_out.write(output_path)
        print(f"Saved PLY → {output_path}  "
              f"({self.num_primitives} primitives, res={resolution})")
        for t in ["Ellipsoid", "Cylinder", "Cuboid", "Other"]:
            n = self.shape_types.count(t)
            if n:
                print(f"  {t}: {n}")


# ---------------------------------------------------------------------------
# Scene wrapper (original SUPERDEC interface, for CuRobo)
# ---------------------------------------------------------------------------

class Scene:
    """
    Wraps a set of superquadrics for path planning.
    Person 1 uses this as the world model in CuRobo.

    Usage:
        scene = Scene.from_fits(flat_sq_fits)
        sd    = scene.get_signed_distance(query_pts)   # CuRobo collision check
        scene.save_superquadrics_vis("scene.ply")
    """

    def __init__(self, path_to_sq_parameters: str):
        self.superquadrics = Superquadrics(path_to_sq_parameters)

    @classmethod
    def from_fits(cls, fits) -> 'Scene':
        obj = cls.__new__(cls)
        obj.superquadrics = Superquadrics.from_fits(fits)
        return obj

    def get_distances_and_closest_points(self, points: np.ndarray):
        return self.superquadrics.get_radial_distance_and_closest_points(points)

    def get_signed_distance(self, points: np.ndarray) -> np.ndarray:
        """CuRobo collision checker entry point. Negative = inside obstacle."""
        return self.superquadrics.get_signed_distance(points)

    def save_superquadrics_vis(self, path: str, resolution: int = 15):
        self.superquadrics.save_ply(path, resolution=resolution)