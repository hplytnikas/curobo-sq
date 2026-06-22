"""
datasets/rgbd_scenes.py

Loader for the RGB-D Scenes v2 dataset.

Data format (per scene NN, files live in a flat directory):
  NN.ply    — binary big-endian PLY with per-point fields:
                x  y  z  diffuse_red  diffuse_green  diffuse_blue
  NN.label  — first line is the point count N (integer), then N lines
               each containing one integer instance label
               (0 = background, 1..K = object instances)
  NN.pose   — camera-pose matrices (not used by this loader)

The RGB-D Scenes v2 dataset contains 14 tabletop scenes captured with a
Kinect sensor.  Point clouds are in camera coordinates.

Usage
-----
    from datasets.rgbd_scenes import RGBDScenesV2

    scene  = RGBDScenesV2("/path/to/rgbd-scenes-v2/pc", scene_id=1)
    pts, rgb, labels = scene.load()      # (N,3) float32, (N,3) uint8, (N,) int32
    clouds = scene.get_object_clouds()   # {label_id: (M,3) float32}, label 0 excluded
"""

import numpy as np
from pathlib import Path
from typing import Dict, Optional, Tuple


class RGBDScenesV2:
    """Loader for a single scene from the RGB-D Scenes v2 dataset.

    Parameters
    ----------
    data_root : str | Path
        Directory containing the NN.ply / NN.label / NN.pose files.
        Typically `<project>/data/rgbd-scenes-v2/pc/`.
    scene_id  : int
        Scene index in [1, 14].  The file names are zero-padded to two
        digits: 01.ply, 02.ply, …, 14.ply.
    """

    NUM_SCENES = 14

    def __init__(self, data_root: str, scene_id: int):
        self.data_root = Path(data_root)
        self.scene_id  = int(scene_id)
        if not (1 <= self.scene_id <= self.NUM_SCENES):
            raise ValueError(
                f"scene_id must be in [1, {self.NUM_SCENES}], got {scene_id}"
            )
        prefix = f"{self.scene_id:02d}"
        self.ply_path   = self.data_root / f"{prefix}.ply"
        self.label_path = self.data_root / f"{prefix}.label"

        # Cached arrays (populated by load())
        self._pts:    Optional[np.ndarray] = None   # (N, 3) float32
        self._rgb:    Optional[np.ndarray] = None   # (N, 3) uint8
        self._labels: Optional[np.ndarray] = None   # (N,) int32

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Load the scene (cached after the first call).

        Returns
        -------
        pts    : (N, 3) float32  — XYZ camera-frame coordinates (metres)
        rgb    : (N, 3) uint8    — diffuse RGB colours, values 0–255
        labels : (N,) int32      — per-point instance label
                                   0 = background, 1..K = object instances
        """
        if self._pts is None:
            self._pts, self._rgb, self._labels = self._load_all()
        return self._pts, self._rgb, self._labels

    def get_object_clouds(self) -> Dict[int, np.ndarray]:
        """Split the cloud by instance label, excluding background (label 0).

        Returns
        -------
        dict mapping label_id (int, > 0) → (M, 3) float32 point array.
        The background label (0) is never included.
        """
        pts, _, labels = self.load()
        result: Dict[int, np.ndarray] = {}
        for lbl in np.unique(labels):
            if lbl == 0:
                continue
            result[int(lbl)] = pts[labels == lbl]
        return result

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _load_all(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        pts, rgb = self._load_ply()
        labels   = self._load_labels(expected_n=len(pts))
        return pts, rgb, labels

    def _load_ply(self) -> Tuple[np.ndarray, np.ndarray]:
        """Read xyz and diffuse RGB from the binary PLY file via plyfile."""
        import plyfile

        ply = plyfile.PlyData.read(str(self.ply_path))
        v   = ply["vertex"]

        pts = np.column_stack([
            v["x"].astype(np.float32),
            v["y"].astype(np.float32),
            v["z"].astype(np.float32),
        ])
        rgb = np.column_stack([
            v["diffuse_red"].astype(np.uint8),
            v["diffuse_green"].astype(np.uint8),
            v["diffuse_blue"].astype(np.uint8),
        ])
        return pts, rgb

    def _load_labels(self, expected_n: int) -> np.ndarray:
        """Parse the .label file.

        Format: line 1 = N (point count), lines 2..N+1 = one integer each.
        """
        labels = np.loadtxt(str(self.label_path), dtype=np.int32, skiprows=1)
        if labels.shape[0] != expected_n:
            raise ValueError(
                f"Scene {self.scene_id:02d}: label count {labels.shape[0]} "
                f"!= PLY point count {expected_n}"
            )
        return labels
