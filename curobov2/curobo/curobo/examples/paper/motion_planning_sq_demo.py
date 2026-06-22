"""Tabletop motion planning demo with ShapeNet objects and SuperDec decomposition.

Five deterministic scenes are generated from random ShapeNet objects placed on
the robot's tabletop workspace. Select scenes and representation (SQ / mesh) from
the Viser web UI.

All downloaded assets live in one folder at the repo root, ``data/paper/`` (see
the repository README "Reproducing the Paper Results"). A ShapeNet subset must be
present before running. Only the **test split** of the tabletop categories is
needed (not the full dataset): the loader reads each ``{synset}/test.lst`` and
uses only the models whose ``{model}/pointcloud.npz`` is actually present, so a
handful of test samples per category is enough. Layout (ONet/ConvONet format)::

    data/paper/ShapeNet_test/{synset}/test.lst
    data/paper/ShapeNet_test/{synset}/{model}/pointcloud.npz

Alternatively, fetch the PyG part-annotation dataset into that folder::

    conda run -n 3dv python -c "
        from torch_geometric.datasets import ShapeNet
        ShapeNet(root='data/paper/ShapeNet_test', split='test')
    "

The SuperDec checkpoint defaults to ``data/paper/tabletop_finetuned`` (override
with ``--checkpoint_folder``). Then launch::

    conda run -n 3dv python motion_planning_sq_demo.py
"""

from __future__ import annotations

import argparse
import os
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import trimesh
import warp as wp
from omegaconf import OmegaConf
from scipy.spatial.transform import Rotation as SciRotation

from curobo._src.geom.collision.buffer_collision import CollisionBuffer
from curobo._src.geom.collision.collision_scene import SceneCollisionCfg
from curobo._src.geom.types import Cuboid, Mesh, SceneCfg, Superquadric
from curobo._src.motion.motion_planner_cfg import MotionPlannerCfg
from curobo._src.types.device_cfg import DeviceCfg
from curobo.motion_planner import MotionPlanner
from curobo.types import ContentPath, GoalToolPose, JointState, Pose
from curobo.viewer import ViserVisualizer

WORKSPACE_ROOT = Path(__file__).resolve().parents[5]
SUPERDEC_ROOT = WORKSPACE_ROOT / "superdec"
# Single root folder for all downloaded paper assets (checkpoint, ShapeNet subset,
# chair.ply, scenes_cache.pkl). See the repo README "Reproducing the Paper Results".
ASSETS_ROOT = WORKSPACE_ROOT / "data" / "paper"
if str(SUPERDEC_ROOT) not in sys.path:
    sys.path.append(str(SUPERDEC_ROOT))

from superdec.superdec import SuperDec
from superdec.utils.predictions_handler import PredictionHandler
from superdec.data.dataloader import normalize_points, denormalize_outdict
from superdec.data.transform import rotate_around_axis

# Optional fast GPU FPS via torch_cluster
try:
    from torch_cluster import fps as _tc_fps
    _HAS_TORCH_CLUSTER = True
except ImportError:
    _HAS_TORCH_CLUSTER = False


# ── constants ─────────────────────────────────────────────────────────────────

CHECKPOINT_FOLDER = str(ASSETS_ROOT / "tabletop_finetuned")
CKPT_FILE = "ckpt.pt"

TABLE = Cuboid(
    name="table",
    pose=[0.0, 0.0, -0.05, 1.0, 0.0, 0.0, 0.0],
    dims=[1.4, 1.4, 0.05],
)

# Objects are placed directly into robot z-up frame — no additional scene offset.
SCENE_TRANSLATION = np.array([0.0, 0.0, 0.0], dtype=np.float32)
SCENE_QUAT_WXYZ   = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)

TABLETOP_INNER_R = 0.28     # min distance from robot base (clears the base cylinder)
TABLETOP_OUTER_R = 0.65     # max distance from robot base (stays within 1.4 m table)
OBJECT_SIZE_M    = 0.20     # default object half-extent for placement
OBJECT_MIN_SEP   = 0.04     # minimum edge-to-edge gap between objects

TABLETOP_CATEGORIES = ["Bottle", "Bowl", "Knife", "Laptop", "Mug"]
DEFAULT_SUPERDEC_SAMPLE_POINTS = 4096

_OBJECT_COLORS: List[List[float]] = [
    [0.85, 0.55, 0.55],
    [0.55, 0.80, 0.55],
    [0.55, 0.65, 0.90],
    [0.90, 0.85, 0.50],
    [0.85, 0.60, 0.90],
    [0.55, 0.85, 0.85],
]


# ── dataclasses ───────────────────────────────────────────────────────────────

@dataclass
class SuperDecPrediction:
    iid: int
    outdict: dict
    mesh: trimesh.Trimesh | None


@dataclass
class SceneSpec:
    """One tabletop scene: raw per-object point clouds plus inference cache."""
    name: str
    object_pts: List[np.ndarray]    # per-object z-up robot-frame points [N, 3]
    object_colors: List[np.ndarray] # per-point RGB in [0, 1], same length as object_pts
    # Populated after first SuperDec inference:
    predictions: Optional[List[SuperDecPrediction]] = field(default=None)
    xyz: Optional[np.ndarray] = field(default=None)  # combined scene point cloud
    rgb: Optional[np.ndarray] = field(default=None)  # combined scene colors


@dataclass
class _SceneBuildCfg:
    name: str
    n_objects: int
    seed: int
    dataset_type: str = "shapenet"  # "shapenet" or "gso"
    object_size_m: float = OBJECT_SIZE_M


SCENE_BUILD_CONFIGS: List[_SceneBuildCfg] = [
    _SceneBuildCfg("Scene 1", 10, 42),
    _SceneBuildCfg("Scene 2", 15, 137),
    _SceneBuildCfg("Scene 3", 20, 256),
    _SceneBuildCfg("Scene 4", 20, 512,  object_size_m=0.25),
    _SceneBuildCfg("Scene 5", 20, 1024, object_size_m=0.25),
    _SceneBuildCfg("Scene 6 (GSO)", 10, 2048, dataset_type="gso"),
    _SceneBuildCfg("Scene 7 (GSO)", 10, 4096, dataset_type="gso"),
]


# ── utilities ─────────────────────────────────────────────────────────────────

def _count_items(items) -> int:
    return len(items) if items is not None else 0


def _get_outdict_scalar(outdict: dict, key: str, idx: int) -> float:
    v = outdict.get(key)
    if isinstance(v, torch.Tensor):
        v = v.cpu().numpy()
    a = np.asarray(v)
    try:
        val = a[0, idx]
    except Exception:
        try:
            val = a.reshape(-1)[idx]
        except Exception:
            val = a
    val = np.asarray(val)
    if val.size == 1:
        return float(val.item())
    return float(val.ravel()[0])


def _rotation_matrix_to_wxyz(rotation_matrix: np.ndarray) -> List[float]:
    quat_xyzw = SciRotation.from_matrix(rotation_matrix).as_quat()
    return [float(quat_xyzw[3]), float(quat_xyzw[0]), float(quat_xyzw[1]), float(quat_xyzw[2])]


def _apply_scene_transform(
    translation: Sequence[float] | np.ndarray,
    rotation_matrix: np.ndarray,
    scene_translation: np.ndarray,
    scene_quat_wxyz: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    scene_quat_xyzw = np.array(
        [scene_quat_wxyz[1], scene_quat_wxyz[2], scene_quat_wxyz[3], scene_quat_wxyz[0]],
        dtype=np.float32,
    )
    scene_rot = SciRotation.from_quat(scene_quat_xyzw)
    new_t = scene_rot.apply(np.asarray(translation, dtype=np.float32)) + scene_translation
    new_r = scene_rot.as_matrix() @ np.asarray(rotation_matrix, dtype=np.float32)
    return new_t, new_r


def _transform_trimesh_mesh(
    mesh_tm: trimesh.Trimesh,
    rotation_matrix: np.ndarray,
    translation: Sequence[float],
    scale_factor: float = 1.0,
) -> trimesh.Trimesh:
    v = np.asarray(mesh_tm.vertices, dtype=np.float32) * float(scale_factor)
    v = (rotation_matrix @ v.T).T + np.asarray(translation, dtype=np.float32)
    return trimesh.Trimesh(vertices=v, faces=np.asarray(mesh_tm.faces), process=False)


def _transform_pointcloud(
    points: np.ndarray,
    scene_translation: Sequence[float],
    scene_quat_wxyz: Sequence[float],
) -> np.ndarray:
    scene_quat_xyzw = np.array(
        [scene_quat_wxyz[1], scene_quat_wxyz[2], scene_quat_wxyz[3], scene_quat_wxyz[0]],
        dtype=np.float32,
    )
    return (
        SciRotation.from_quat(scene_quat_xyzw).apply(np.asarray(points, dtype=np.float32))
        + np.asarray(scene_translation, dtype=np.float32)
    )


def _inv_pose_to_world_center(inv_pose_row: np.ndarray) -> np.ndarray:
    t = inv_pose_row[:3].astype(np.float64)
    q = inv_pose_row[3:7].astype(np.float64)
    R = SciRotation.from_quat([q[1], q[2], q[3], q[0]]).as_matrix()
    return -(R.T @ t)


# ── farthest-point sampling ───────────────────────────────────────────────────

def _fps_sample(pts: np.ndarray, n: int) -> np.ndarray:
    """Return n well-distributed points via farthest-point sampling."""
    if len(pts) <= n:
        return pts
    if _HAS_TORCH_CLUSTER:
        pts_t = torch.from_numpy(pts.astype(np.float32))
        ratio = n / len(pts)
        idx = _tc_fps(pts_t, ratio=ratio, random_start=True).cpu().numpy()
        idx = idx[:n]
        if len(idx) < n:
            extra = np.random.choice(len(pts), n - len(idx), replace=False)
            idx = np.concatenate([idx, extra])
        return pts[idx]
    # Pure-torch fallback (O(n·N), fast enough at N~100k, n=4096)
    pts_t = torch.from_numpy(pts.astype(np.float32))
    dists = torch.full((len(pts),), float("inf"))
    selected = [0]
    for _ in range(n - 1):
        d = torch.sum((pts_t - pts_t[selected[-1]]) ** 2, dim=1)
        dists = torch.minimum(dists, d)
        selected.append(int(torch.argmax(dists).item()))
    return pts[selected]


# ── model loading ─────────────────────────────────────────────────────────────

def _load_model(ckpt_dir: str, device: str, ckpt_file: str = CKPT_FILE) -> SuperDec:
    cfg = OmegaConf.load(os.path.join(ckpt_dir, "config.yaml"))
    model = SuperDec(cfg.superdec).to(device)
    model.lm_optimization = False
    ckpt = torch.load(os.path.join(ckpt_dir, ckpt_file), map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"Loaded SuperDec from {ckpt_dir}/{ckpt_file}")
    return model


# ── ShapeNet scene generation ─────────────────────────────────────────────────

# Mapping from human-readable category names to ShapeNet synset IDs
_CATEGORY_TO_SYNSET: Dict[str, str] = {
    "Bag": "02773838", "Bottle": "02876657", "Bowl": "02880940",
    "Cap": "02954340", "Car": "02958343", "Chair": "03001627",
    "Earphone": "03261776", "Guitar": "03467517", "Knife": "03624134",
    "Lamp": "03636649", "Laptop": "03642806", "Motorbike": "03790512",
    "Mug": "03797390", "Pistol": "03948459", "Rocket": "04099429",
    "Skateboard": "04225987", "Table": "04379243",
}
_SYNSET_TO_CATEGORY: Dict[str, str] = {v: k for k, v in _CATEGORY_TO_SYNSET.items()}


class _LocalShapeNetItem:
    __slots__ = ("pos",)

    def __init__(self, pts: np.ndarray):
        self.pos = torch.from_numpy(pts)


class _LocalShapeNetDataset:
    """Reads the ONet/ConvONet ShapeNet format: {root}/{synset}/{model}/pointcloud.npz.

    Uses {synset}/test.lst when available; falls back to all model subdirectories.
    Works as a drop-in for the PyG ShapeNet API (len + __getitem__ returning .pos).
    """

    def __init__(self, root: str, categories: Optional[List[str]] = None, split: Optional[str] = None):
        requested_synsets = {_CATEGORY_TO_SYNSET[c] for c in (categories or []) if c in _CATEGORY_TO_SYNSET}
        available_synsets = {
            d for d in os.listdir(root)
            if os.path.isdir(os.path.join(root, d)) and d not in ("gso", "raw") and not d.startswith(".")
        }
        synsets = (available_synsets & requested_synsets) if requested_synsets else available_synsets
        if not synsets:
            synsets = available_synsets

        self._items: List[str] = []  # full paths to pointcloud.npz files
        self._item_categories: List[str] = []  # category name per item index
        for synset in sorted(synsets):
            synset_dir = os.path.join(root, synset)
            cat_name = _SYNSET_TO_CATEGORY.get(synset, synset)
            if split is not None:
                lst_path = os.path.join(synset_dir, f"{split}.lst")
                if os.path.isfile(lst_path):
                    with open(lst_path) as fh:
                        model_ids = [l.strip() for l in fh if l.strip()]
                else:
                    model_ids = [d for d in os.listdir(synset_dir) if os.path.isdir(os.path.join(synset_dir, d))]
            else:
                model_ids = [d for d in os.listdir(synset_dir) if os.path.isdir(os.path.join(synset_dir, d))]
            for mid in model_ids:
                p = os.path.join(synset_dir, mid, "pointcloud.npz")
                if os.path.isfile(p):
                    self._items.append(p)
                    self._item_categories.append(cat_name)

        self._items_by_category: Dict[str, List[int]] = {}
        for i, cat in enumerate(self._item_categories):
            self._items_by_category.setdefault(cat, []).append(i)

        if not self._items:
            raise RuntimeError(
                f"No ShapeNet models found under '{root}' for split '{split or 'all'}'.\n"
                "The split list(s) were read but no matching '{model}/pointcloud.npz' files "
                "exist on disk. Download at least a few of the listed test-split models into "
                "{root}/{synset}/{model}/pointcloud.npz (you do not need the full dataset)."
            )

        cat_names = sorted(self._items_by_category.keys())
        print(f"  LocalShapeNet: {len(self._items)} models from {cat_names} (split={split or 'all'})")

    def __len__(self) -> int:
        return len(self._items)

    def __getitem__(self, idx: int) -> _LocalShapeNetItem:
        pts = np.load(self._items[idx])["points"].astype(np.float32)
        if len(pts) > 16384:
            pts = pts[np.random.choice(len(pts), 16384, replace=False)]
        return _LocalShapeNetItem(pts)


class _GSODataset:
    """Google Scanned Objects: flat {root}/{object_name}/pointcloud.npz structure."""

    def __init__(self, root: str):
        self._items: List[str] = sorted(
            os.path.join(root, d, "pointcloud.npz")
            for d in os.listdir(root)
            if os.path.isfile(os.path.join(root, d, "pointcloud.npz"))
        )
        print(f"  GSODataset: {len(self._items)} models")

    def __len__(self) -> int:
        return len(self._items)

    def __getitem__(self, idx: int) -> _LocalShapeNetItem:
        pts = np.load(self._items[idx])["points"].astype(np.float32)
        if len(pts) > 16384:
            pts = pts[np.random.choice(len(pts), 16384, replace=False)]
        return _LocalShapeNetItem(pts)


def _load_shapenet(root: str, categories: List[str], split: Optional[str] = "test"):
    """Load ShapeNet dataset.

    Prefers the local ONet/ConvONet format ({root}/{synset}/{model}/pointcloud.npz).
    Falls back to PyG ShapeNet if the local format is not present.

    Only the requested ``split`` (default "test") is loaded: each synset's
    ``{split}.lst`` file lists the model ids, and only those whose
    ``pointcloud.npz`` is actually present on disk are used.  This means you can
    ship just a subset of ShapeNet -- e.g. the test-split models for the tabletop
    categories -- instead of the full dataset.  Pass ``split=None`` to load every
    model directory found on disk regardless of the split files.
    """
    # Detect the local ONet format: a synset directory either exposes a split
    # list ({split}.lst) or already contains at least one {model}/pointcloud.npz.
    local_synsets = [
        d for d in os.listdir(root)
        if os.path.isdir(os.path.join(root, d)) and d not in ("gso", "raw") and not d.startswith(".")
    ] if os.path.isdir(root) else []

    def _synset_is_local(s: str) -> bool:
        synset_dir = os.path.join(root, s)
        if split is not None and os.path.isfile(os.path.join(synset_dir, f"{split}.lst")):
            return True
        return any(
            os.path.isfile(os.path.join(synset_dir, m, "pointcloud.npz"))
            for m in os.listdir(synset_dir)
            if os.path.isdir(os.path.join(synset_dir, m))
        )

    if any(_synset_is_local(s) for s in local_synsets):
        return _LocalShapeNetDataset(root, categories, split=split)

    from torch_geometric.datasets import ShapeNet
    raw_dir = os.path.join(root, "raw")
    if not os.path.isdir(raw_dir) or len(os.listdir(raw_dir)) == 0:
        raise RuntimeError(
            f"ShapeNet data not found at '{root}'.\n"
            "Either place ONet-format data ({synset}/{model}/pointcloud.npz) directly in that "
            "folder (only the test-split models are needed), or download the PyG "
            "part-annotation dataset with:\n"
            f"  python -c \"from torch_geometric.datasets import ShapeNet; "
            f"ShapeNet(root='{root}', split='test')\""
        )
    return ShapeNet(root=root, categories=categories, split=split or "test")


def _shapenet_item_to_zup(item, size_m: float, color_idx: int, dataset_type: str = "shapenet") -> Tuple[np.ndarray, np.ndarray]:
    """Convert a ShapeNet PyG Data item (y-up, normalized) to z-up robot frame.

    ShapeNet data.pos is in y-up coordinates normalized to ~[-0.5, 0.5].
    Scaling and rotating +90° around x converts to z-up with bottom at z=0.
    The caller then translates to the desired table position.

    Returns (pts [N,3] in z-up robot frame, colors [N,3] in [0,1]).
    """
    pts = item.pos.numpy().astype(np.float32) * size_m
    # y-up → z-up: rotate +90° around x-axis (inverse of the -90° the inference pipeline applies)
    pts = rotate_around_axis(pts, axis=(1, 0, 0), angle=np.pi / 2, center_point=np.zeros(3))
    if dataset_type == "gso":
        pts = rotate_around_axis(pts, axis=(1, 0, 0), angle=-np.pi / 2, center_point=np.zeros(3))
    pts[:, 2] -= pts[:, 2].min()  # floor at z = 0
    color = np.asarray(_OBJECT_COLORS[color_idx % len(_OBJECT_COLORS)], dtype=np.float32)
    colors = np.tile(color, (len(pts), 1))
    return pts, colors


def _place_on_table(
    n: int,
    rng: np.random.RandomState,
    inner_r: float,
    outer_r: float,
    min_sep: float,
    max_tries: int = 5000,
) -> List[np.ndarray]:
    """Place n centers in an annulus around the robot base (360° coverage).

    Samples radius from a uniform-in-area distribution (r² uniform) and angle
    uniformly in [0, 2π).  Uses Poisson-disk rejection to guarantee min_sep.
    On the rare fallback, picks the candidate with the largest minimum distance
    to existing objects and logs a warning.
    """
    centers: List[np.ndarray] = []
    for slot in range(n):
        best_c: Optional[np.ndarray] = None
        best_gap = -1.0
        placed = False
        for _ in range(max_tries):
            r = float(np.sqrt(rng.uniform(inner_r ** 2, outer_r ** 2)))
            theta = rng.uniform(0.0, 2.0 * np.pi)
            c = np.array([r * np.cos(theta), r * np.sin(theta)], dtype=np.float32)
            if not centers:
                centers.append(c)
                placed = True
                break
            gap = min(float(np.linalg.norm(c - p)) for p in centers)
            if gap >= min_sep:
                centers.append(c)
                placed = True
                break
            if gap > best_gap:
                best_gap = gap
                best_c = c
        if not placed:
            centers.append(best_c)
            print(
                f"  [placement] WARNING slot {slot + 1}: "
                f"best gap {best_gap:.3f} m < required {min_sep:.3f} m — objects may overlap"
            )

    # Verification: log actual minimum gap across all pairs
    if len(centers) > 1:
        gaps = [
            float(np.linalg.norm(centers[i] - centers[j]))
            for i in range(len(centers))
            for j in range(i + 1, len(centers))
        ]
        print(f"  [placement] {n} objects placed, min gap = {min(gaps):.3f} m (required ≥ {min_sep:.3f} m)")

    return centers


# ── SuperDec inference ────────────────────────────────────────────────────────

def _infer_single_object(
    pts_scene: np.ndarray,
    model: "SuperDec",
    device: str,
    mesh_resolution: int,
    iid: int,
) -> Optional[SuperDecPrediction]:
    """Run SuperDec on one object (z-up robot-frame pts).

    Returns None if any active primitive is needle-thin (< 3 mm on its thinnest axis)
    or if no mesh can be generated — the caller should replace the object and retry.
    """
    obj_pts = _fps_sample(pts_scene, DEFAULT_SUPERDEC_SAMPLE_POINTS)
    obj_pts_yup = rotate_around_axis(obj_pts, axis=(1, 0, 0), angle=-np.pi / 2,
                                      center_point=np.zeros(3))
    pts_norm, translation, scale = normalize_points(obj_pts_yup)
    pts_t = torch.from_numpy(pts_norm).unsqueeze(0).to(device).float()

    with torch.no_grad():
        out = model(pts_t)
    out = {key: (v.cpu() if isinstance(v, torch.Tensor) else v) for key, v in out.items()}
    out = denormalize_outdict(out, np.array([translation]), np.array([scale]), z_up=True)

    _MIN_R = 0.003
    _s = np.asarray(out["scale"]).astype(np.float32)
    _e = np.asarray(out["exist"])
    _n = _s.shape[1]
    _needle = (_e.reshape(_n) > 0.5) & (_s[0].min(axis=1) < _MIN_R)
    if _needle.any():
        return None  # reject — caller picks a different object

    pts_back = torch.from_numpy(pts_scene[None].astype(np.float32))
    handler = PredictionHandler.from_outdict(
        out, pts_back[:, :DEFAULT_SUPERDEC_SAMPLE_POINTS], [str(iid)]
    )
    mesh = handler.get_meshes(resolution=mesh_resolution)[0]
    if mesh is None:
        return None

    return SuperDecPrediction(iid=iid, outdict=out, mesh=mesh)


def _build_all_scenes(
    shapenet_dataset,
    gso_dataset,
    checkpoint_folder: str,
    mesh_resolution: int,
    scene_configs: Optional[List[_SceneBuildCfg]] = None,
) -> List[SceneSpec]:
    """Build and pre-compute all scenes upfront.

    For each object slot, keeps picking random models until SuperDec produces no
    needle-thin primitives.  The model is loaded once and reused across all scenes.
    """
    if scene_configs is None:
        scene_configs = SCENE_BUILD_CONFIGS

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = _load_model(checkpoint_folder, device)

    # globally_bad: indices confirmed needle-thin for a given dataset type — never retry them.
    # Persists across all scenes so bad objects are only run through SuperDec once.
    globally_bad: Dict[str, set] = {"shapenet": set(), "gso": set()}

    scenes: List[SceneSpec] = []
    for cfg in scene_configs:
        dataset = gso_dataset if cfg.dataset_type == "gso" else shapenet_dataset
        bad = globally_bad[cfg.dataset_type]
        print(f"\nBuilding {cfg.name} ({cfg.n_objects} objects, {cfg.dataset_type})…")
        rng = np.random.RandomState(cfg.seed)

        centers = _place_on_table(
            cfg.n_objects, rng,
            inner_r=TABLETOP_INNER_R,
            outer_r=TABLETOP_OUTER_R,
            min_sep=cfg.object_size_m * 2.0 + OBJECT_MIN_SEP,
        )

        all_pts: List[np.ndarray] = []
        all_colors: List[np.ndarray] = []
        predictions: List[SuperDecPrediction] = []
        used_in_scene: set = set()  # successfully placed in this scene — don't reuse

        # For ShapeNet scenes guarantee at least one object from each available category.
        # Shuffle the category list per-scene (seeded) so the guaranteed slots vary.
        slot_pool: List[Optional[List[int]]] = [None] * cfg.n_objects
        if cfg.dataset_type == "shapenet" and hasattr(dataset, "_items_by_category"):
            cats = list(dataset._items_by_category.keys())
            rng.shuffle(cats)
            for i, cat in enumerate(cats[: cfg.n_objects]):
                slot_pool[i] = dataset._items_by_category[cat]

        for slot, center in enumerate(centers):
            pool = slot_pool[slot]  # None → whole dataset
            exclude = used_in_scene | bad
            attempt = 0
            while True:
                attempt += 1
                if pool is not None:
                    candidates = [i for i in pool if i not in exclude]
                    if not candidates:
                        # All items in this category are bad or used — fall back to whole dataset
                        pool = None
                        candidates = [i for i in range(len(dataset)) if i not in exclude]
                else:
                    candidates = [i for i in range(len(dataset)) if i not in exclude]
                if not candidates:
                    # All objects exhausted — reset bad set and warn
                    print(f"  WARNING: all {len(dataset)} objects tried; resetting globally_bad for {cfg.dataset_type}")
                    bad.clear()
                    exclude = used_in_scene
                    candidates = [i for i in range(len(dataset)) if i not in exclude]
                idx = int(rng.choice(candidates))

                pts, colors = _shapenet_item_to_zup(dataset[idx], cfg.object_size_m, slot, cfg.dataset_type)
                pts[:, 0] += center[0]
                pts[:, 1] += center[1]

                pred = _infer_single_object(pts, model, device, mesh_resolution, iid=slot)
                if pred is not None:
                    used_in_scene.add(idx)
                    exclude = used_in_scene | bad
                    all_pts.append(pts)
                    all_colors.append(colors[: len(pts)])
                    predictions.append(pred)
                    cat_label = dataset._item_categories[idx] if hasattr(dataset, "_item_categories") else ""
                    status = f"attempt {attempt}" if attempt > 1 else "ok"
                    print(f"  slot {slot + 1}/{cfg.n_objects} [{cat_label}]: {status}")
                    break
                bad.add(idx)
                exclude = used_in_scene | bad
                print(f"  slot {slot + 1}/{cfg.n_objects}: needle-thin (attempt {attempt}, {len(bad)} bad known), retrying…")

        spec = SceneSpec(
            name=cfg.name,
            object_pts=all_pts,
            object_colors=all_colors,
            predictions=predictions,
        )
        spec.xyz = np.concatenate(all_pts) if all_pts else np.zeros((0, 3), dtype=np.float32)
        spec.rgb = np.concatenate(all_colors) if all_colors else np.zeros((0, 3), dtype=np.float32)
        scenes.append(spec)
        print(f"  {cfg.name} ready: {len(predictions)} object(s)")

    return scenes


def _infer_scene_spec(spec: SceneSpec, checkpoint_folder: str, mesh_resolution: int) -> None:
    """Fallback: run SuperDec on a spec that has no predictions yet (no retry logic)."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = _load_model(checkpoint_folder, device)
    predictions: List[SuperDecPrediction] = []
    all_xyz: List[np.ndarray] = []
    all_rgb: List[np.ndarray] = []
    for k, (obj_pts_scene, obj_colors) in enumerate(zip(spec.object_pts, spec.object_colors)):
        pred = _infer_single_object(obj_pts_scene, model, device, mesh_resolution, iid=k)
        if pred is None:
            print(f"  object {k}: skipped (needle-thin or no mesh)")
            continue
        predictions.append(pred)
        all_xyz.append(obj_pts_scene)
        all_rgb.append(obj_colors[: len(obj_pts_scene)])
        print(f"  object {k}: fitted ({k + 1}/{len(spec.object_pts)})")
    spec.predictions = predictions
    spec.xyz = np.concatenate(all_xyz) if all_xyz else np.zeros((0, 3), dtype=np.float32)
    spec.rgb = np.concatenate(all_rgb) if all_rgb else np.zeros((0, 3), dtype=np.float32)


# ── scene building ────────────────────────────────────────────────────────────

def _prediction_to_scene_cfg(
    predictions: List[SuperDecPrediction],
    world_representation: str,
    scale_factor: float,
    scene_translation: Sequence[float],
    scene_quat_wxyz: Sequence[float],
    points_tensor: torch.Tensor,
) -> SceneCfg:
    scene_t_np = np.asarray(scene_translation, dtype=np.float32)
    scene_q_np = np.asarray(scene_quat_wxyz, dtype=np.float32)
    scene_q_xyzw = np.array([scene_q_np[1], scene_q_np[2], scene_q_np[3], scene_q_np[0]], dtype=np.float32)

    superquadrics: List[Superquadric] = []
    meshes: List[Mesh] = []

    for pred in predictions:
        outdict = pred.outdict
        n_prim = int(outdict["scale"].shape[1])
        for idx in range(n_prim):
            if _get_outdict_scalar(outdict, "exist", idx) <= 0.5:
                continue
            sc = np.asarray(outdict["scale"][0, idx], dtype=np.float32) * scale_factor
            ex = np.asarray(outdict["shape"][0, idx], dtype=np.float32)
            rot = np.asarray(outdict["rotate"][0, idx], dtype=np.float32)
            tr  = np.asarray(outdict["trans"][0, idx], dtype=np.float32)
            new_t, new_r = _apply_scene_transform(tr.tolist(), rot, scene_t_np, scene_q_np)
            pose = [float(new_t[0]), float(new_t[1]), float(new_t[2]),
                    *_rotation_matrix_to_wxyz(new_r)]
            superquadrics.append(Superquadric(
                name=f"obj_{pred.iid}_sq_{idx}",
                pose=pose,
                radii=sc.tolist(),
                shape=ex.tolist(),
            ))

    if world_representation == "mesh":
        rot_mat = SciRotation.from_quat(scene_q_xyzw).as_matrix()
        for pred in predictions:
            if pred.mesh is None:
                continue
            tm = _transform_trimesh_mesh(pred.mesh, rot_mat, scene_t_np, scale_factor)
            meshes.append(Mesh(
                name=f"obj_{pred.iid}",
                vertices=tm.vertices.tolist(),
                faces=tm.faces.tolist(),
                pose=[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
            ))

    if world_representation == "superquadrics":
        scene = SceneCfg(cuboid=[TABLE], superquadric=superquadrics)
    elif world_representation == "mesh":
        scene = SceneCfg(cuboid=[TABLE], mesh=meshes)
    else:
        raise ValueError(f"Unknown world_representation: {world_representation!r}")

    n_prims = len(superquadrics) if world_representation == "superquadrics" else len(meshes)
    print(f"Built scene: {n_prims} {world_representation} primitive(s)")
    return scene


def _superdec_display_scene_cfg(
    predictions: List[SuperDecPrediction],
    cuboids,
    scene_translation: Sequence[float],
    scene_quat_wxyz: Sequence[float],
    scale_factor: float,
) -> Tuple[SceneCfg, Dict[int, np.ndarray]]:
    """Build a display-only SceneCfg.

    Each mesh is expressed in centroid-relative local coordinates with the centroid
    stored as the mesh pose. This places the Viser control-frame gizmo on the object
    and lets point-cloud / overlay-mesh children inherit the gizmo transform.

    Returns (SceneCfg, centroids) where centroids maps pred.iid → centroid xyz.
    """
    scene_q_xyzw = np.array(
        [scene_quat_wxyz[1], scene_quat_wxyz[2], scene_quat_wxyz[3], scene_quat_wxyz[0]],
        dtype=np.float32,
    )
    rot = SciRotation.from_quat(scene_q_xyzw).as_matrix()
    display_meshes = []
    centroids: Dict[int, np.ndarray] = {}
    for pred in predictions:
        if pred.mesh is None:
            continue
        tm = _transform_trimesh_mesh(pred.mesh, rot, scene_translation, scale_factor)
        verts = np.array(tm.vertices, dtype=np.float32)
        centroid = verts.mean(axis=0)
        centroids[pred.iid] = centroid
        display_meshes.append(Mesh(
            name=f"obj_{pred.iid}",
            vertices=(verts - centroid).tolist(),
            faces=tm.faces.tolist(),
            pose=[float(centroid[0]), float(centroid[1]), float(centroid[2]), 1.0, 0.0, 0.0, 0.0],
        ))
    return SceneCfg(cuboid=list(cuboids) if cuboids else [], mesh=display_meshes), centroids


# ── Viser overlays ────────────────────────────────────────────────────────────

def _add_superdec_overlays(
    viser_viz,
    object_pts: List[np.ndarray],
    object_colors: List[np.ndarray],
    predictions: List[SuperDecPrediction],
    scene_translation: Sequence[float],
    scene_quat_wxyz: Sequence[float],
    scale_factor: float,
    centroids: Dict[int, np.ndarray],
    add_meshes: bool = True,
) -> list:
    """Add point cloud and fitted meshes to the Viser scene; returns handles.

    Point cloud: one combined cloud at /scene (single draw call, no per-object overhead).
    Overlay meshes: added as children of /obstacles/obj_{iid}/ so they follow the gizmo.
    """
    server = viser_viz._server
    handles = []

    all_pts = [_transform_pointcloud(pts, scene_translation, scene_quat_wxyz) for pts in object_pts]
    all_colors_clipped = [c[: len(p)] for p, c in zip(all_pts, object_colors)]
    if all_pts:
        handles.append(server.scene.add_point_cloud(
            name="/scene",
            points=np.concatenate(all_pts).astype(np.float32),
            colors=np.concatenate(all_colors_clipped).astype(np.float32),
            point_size=0.003,
            visible=True,
        ))

    if not add_meshes:
        return handles

    scene_q_xyzw = np.array(
        [scene_quat_wxyz[1], scene_quat_wxyz[2], scene_quat_wxyz[3], scene_quat_wxyz[0]],
        dtype=np.float32,
    )
    rot = SciRotation.from_quat(scene_q_xyzw).as_matrix()
    for pred in predictions:
        if pred.mesh is None:
            continue
        centroid = centroids.get(pred.iid, np.zeros(3, dtype=np.float32))
        tm = _transform_trimesh_mesh(pred.mesh, rot, scene_translation, scale_factor)
        verts_local = np.array(tm.vertices, dtype=np.float32) - centroid
        handles.append(server.scene.add_mesh_trimesh(
            f"/obstacles/obj_{pred.iid}/fit",
            mesh=trimesh.Trimesh(vertices=verts_local, faces=np.asarray(tm.faces), process=False),
            visible=True,
        ))
    return handles


# ── collision diagnostics ─────────────────────────────────────────────────────

def _find_colliding_obstacles(planner: MotionPlanner, active_js, viser_server=None) -> None:
    scene_collision = None
    try:
        scene_collision = planner.graph_planner.feasibility_rollout.scene_collision_checker
    except AttributeError:
        pass
    if scene_collision is None:
        scene_collision = getattr(planner, "scene_collision_checker", None)
    if scene_collision is None:
        return

    try:
        kin_state = planner.kinematics.compute_kinematics(active_js)
        if kin_state.robot_spheres is None:
            return
        spheres_np = kin_state.robot_spheres[0, 0].detach().cpu().numpy()
    except Exception as exc:
        print(f"  [collision diag] FK failed: {exc}")
        return

    data = scene_collision.data
    n_cub  = int(data.cuboids.count[0].item())       if data.cuboids       is not None else 0
    n_sq   = int(data.superquadrics.count[0].item()) if data.superquadrics is not None else 0
    n_mesh = int(data.meshes.count[0].item())         if data.meshes        is not None else 0
    print(f"  [collision diag] {n_cub} cuboids, {n_sq} SQs, {n_mesh} meshes")

    results: list = []

    def _check(name: str, center: np.ndarray, r: float) -> None:
        d = min(float(np.linalg.norm(s[:3] - center)) - float(s[3]) - r for s in spheres_np)
        results.append((name, d))

    try:
        if data.superquadrics is not None:
            for i in range(n_sq):
                nm = data.superquadrics.names[0][i] or f"sq_{i}"
                ctr = _inv_pose_to_world_center(data.superquadrics.inv_pose[0, i].detach().cpu().numpy())
                r   = float(np.max(np.asarray(data.superquadrics.params[0, i].detach().cpu().numpy())[:3])) * 1.732
                _check(nm, ctr, r)
        if data.cuboids is not None:
            for i in range(n_cub):
                nm  = data.cuboids.names[0][i] or f"cub_{i}"
                ctr = _inv_pose_to_world_center(data.cuboids.inv_pose[0, i].detach().cpu().numpy())
                r   = float(np.max(data.cuboids.dims[0, i].detach().cpu().numpy()[:3])) / 2 * 1.732
                _check(nm, ctr, r)
    except Exception as exc:
        print(f"  [collision diag] scan failed: {exc}")
        return

    results.sort(key=lambda x: x[1])
    for nm, d in [(n, d) for n, d in results if d < 0.0][:5]:
        print(f"  COLLISION {nm}  dist={d:.4f} m")

    try:
        col_buf = CollisionBuffer.from_shape(kin_state.robot_spheres.shape, planner.device_cfg)
        w   = torch.ones(1,  dtype=torch.float32, device=planner.device_cfg.device)
        act = torch.zeros(1, dtype=torch.float32, device=planner.device_cfg.device)
        with torch.no_grad():
            cost = scene_collision.get_sphere_distance_raw(
                query_spheres=kin_state.robot_spheres,
                collision_buffer=col_buf, weight=w, activation_distance=act,
            )
        cost_np = cost[0, 0].cpu().numpy()
        kin_cfg = planner.kinematics.config.kinematics_config
        idx_to_name = {v: k for k, v in kin_cfg.link_name_to_idx_map.items()}
        sphere_links = [
            idx_to_name.get(int(kin_cfg.link_sphere_idx_map[i].item()), f"sphere_{i}")
            for i in range(len(kin_cfg.link_sphere_idx_map))
        ]
        colliding = [
            (sphere_links[i], spheres_np[i], float(cost_np[i]))
            for i in range(len(cost_np)) if cost_np[i] > 0.0
        ]
        colliding.sort(key=lambda x: -x[2])
        if colliding:
            print(f"  {len(colliding)} robot sphere(s) in collision")
            if viser_server is not None:
                try:
                    viser_server.scene.remove_by_name("/collision_spheres")
                except Exception:
                    pass
                for i, (lname, sxyzr, _c) in enumerate(colliding):
                    m = trimesh.creation.icosphere(subdivisions=3, radius=float(sxyzr[3]) * 2.0)
                    m.apply_translation([float(sxyzr[0]), float(sxyzr[1]), float(sxyzr[2])])
                    viser_server.scene.add_mesh_trimesh(
                        name=f"/collision_spheres/{lname}_{i}", mesh=m,
                        wxyz=(1.0, 0.0, 0.0, 0.0), position=(0.0, 0.0, 0.0),
                    )
    except Exception as exc:
        print(f"  [sphere collision] {exc}")


# ── planner helper ────────────────────────────────────────────────────────────

def _rebuild_planner(
    scene_cfg: SceneCfg,
    world_representation: str,
    device_cfg: DeviceCfg,
) -> MotionPlanner:
    use_cg = world_representation != "superquadrics"
    pcfg = MotionPlannerCfg.create(
        robot="franka.yml",
        scene_model="collision_test.yml",
        device_cfg=device_cfg,
        use_cuda_graph=use_cg,
        max_goalset=10,
    )
    pcfg.scene_collision_cfg = SceneCollisionCfg(
        device_cfg=device_cfg,
        scene_model=scene_cfg,
        cache={
            "cuboid":       _count_items(scene_cfg.cuboid),
            "mesh":         _count_items(scene_cfg.mesh),
            "superquadric": _count_items(scene_cfg.superquadric),
        },
    )
    p = MotionPlanner(pcfg)
    p.warmup(enable_graph=use_cg, num_warmup_iterations=5)
    return p


# ── interactive UI ────────────────────────────────────────────────────────────

def interactive_motion_planning(
    planner: MotionPlanner,
    scenes: List[SceneSpec],
    initial_scene_idx: int,
    world_representation: str,
    checkpoint_folder: str,
    mesh_resolution: int,
    port: int = 8080,
) -> None:
    device_cfg = planner.device_cfg

    viser_viz = ViserVisualizer(
        content_path=ContentPath(robot_config_file="franka.yml"),
        connect_ip="0.0.0.0",
        connect_port=port,
        add_control_frames=True,
        visualize_robot_spheres=False,
    )
    server = viser_viz._server

    # ── mutable state (plain variables; callbacks use nonlocal) ────────────────
    is_moving    = False
    objects_movable = True
    cur_rep      = world_representation
    cur_spec     = scenes[initial_scene_idx]
    current_state: JointState = planner.default_joint_state.clone().unsqueeze(0)
    _initial_js  = planner.default_joint_state.clone()
    _initial_cf  = {
        name: (tuple(f.position), tuple(f.wxyz))
        for name, f in viser_viz._control_frames.items()
    }

    # ── initial obstacle display ───────────────────────────────────────────────
    _disp_cfg, cur_centroids = _superdec_display_scene_cfg(
        cur_spec.predictions or [], [TABLE], SCENE_TRANSLATION, SCENE_QUAT_WXYZ, 1.0
    )
    obstacle_frames: dict = viser_viz.add_scene(_disp_cfg, add_control_frames=True)

    overlay_handles: list = []
    if cur_spec.object_pts:
        overlay_handles = _add_superdec_overlays(
            viser_viz, cur_spec.object_pts, cur_spec.object_colors,
            cur_spec.predictions or [],
            SCENE_TRANSLATION, SCENE_QUAT_WXYZ, 1.0,
            centroids=cur_centroids,
            add_meshes=(cur_rep != "mesh"),
        )

    # ── tracked obstacle frames for pose update ────────────────────────────────
    _col_names = set(planner.scene_collision_checker.get_obstacle_names())
    tracked_frames: dict = {n: f for n, f in obstacle_frames.items() if n in _col_names}
    old_obstacle_poses: dict = {
        n: Pose.from_numpy(f.position, f.wxyz) for n, f in tracked_frames.items()
    }

    planner.warmup(enable_graph=(cur_rep != "superquadrics"), num_warmup_iterations=5)

    # ── GUI widgets ────────────────────────────────────────────────────────────
    status_md   = server.gui.add_markdown("**Status:** Ready")
    scene_dd    = server.gui.add_dropdown(
        "Scene", options=[s.name for s in scenes],
        initial_value=scenes[initial_scene_idx].name,
    )
    repr_dd     = server.gui.add_dropdown(
        "Representation", options=["superquadrics", "mesh"],
        initial_value=world_representation,
    )
    movable_chk = server.gui.add_checkbox("Objects movable", initial_value=True)
    pc_chk      = server.gui.add_checkbox("Show point clouds", initial_value=True)
    move_btn    = server.gui.add_button("Move",  color="green")
    grasp_btn   = server.gui.add_button("Grasp", color="blue")
    reset_btn   = server.gui.add_button("Reset", color="red")

    # ── internal helpers ───────────────────────────────────────────────────────

    def _rebuild_obstacle_display(spec: SceneSpec, add_cf: bool) -> Dict[int, np.ndarray]:
        nonlocal obstacle_frames, tracked_frames, old_obstacle_poses
        try:
            server.scene.remove_by_name("/obstacles")
        except Exception:
            pass
        disp, new_centroids = _superdec_display_scene_cfg(
            spec.predictions or [], [TABLE], SCENE_TRANSLATION, SCENE_QUAT_WXYZ, 1.0
        )
        obstacle_frames = viser_viz.add_scene(disp, add_control_frames=add_cf)
        col_ns = set(planner.scene_collision_checker.get_obstacle_names())
        tracked_frames = {n: f for n, f in obstacle_frames.items() if n in col_ns}
        old_obstacle_poses = {n: Pose.from_numpy(f.position, f.wxyz) for n, f in tracked_frames.items()}
        return new_centroids

    def _clear_overlays() -> None:
        for h in overlay_handles:
            try:
                h.remove()
            except Exception:
                pass
        overlay_handles.clear()
        try:
            server.scene.remove_by_name("/scene")
        except Exception:
            pass

    def update_obstacles() -> None:
        if not objects_movable:
            return
        for name, frame in tracked_frames.items():
            new_pose = Pose.from_numpy(frame.position, frame.wxyz)
            if new_pose != old_obstacle_poses[name]:
                planner.scene_collision_checker.update_obstacle_pose(name, new_pose)
                old_obstacle_poses[name] = new_pose.clone()

    def execute_trajectory(trajectory) -> None:
        nonlocal current_state
        traj = trajectory.squeeze(0)
        for i in range(traj.position.shape[-2]):
            if not is_moving:
                return
            viser_viz.set_joint_state(
                JointState.from_position(
                    traj.position[0, i, :].unsqueeze(0),
                    joint_names=traj.joint_names,
                ).squeeze(0)
            )
            time.sleep(0.02)
        current_state = JointState.from_position(
            traj.position[0, -1, :].unsqueeze(0),
            joint_names=traj.joint_names,
        )

    # ── callbacks ──────────────────────────────────────────────────────────────

    @movable_chk.on_update
    def _(_event) -> None:
        nonlocal objects_movable, cur_centroids
        objects_movable = movable_chk.value
        _clear_overlays()
        cur_centroids = _rebuild_obstacle_display(cur_spec, objects_movable)
        if pc_chk.value and cur_spec.object_pts:
            overlay_handles.extend(_add_superdec_overlays(
                viser_viz, cur_spec.object_pts, cur_spec.object_colors,
                cur_spec.predictions or [],
                SCENE_TRANSLATION, SCENE_QUAT_WXYZ, 1.0,
                centroids=cur_centroids,
                add_meshes=(cur_rep != "mesh"),
            ))

    @pc_chk.on_update
    def _(_event) -> None:
        for h in overlay_handles:
            try:
                h.visible = pc_chk.value
            except Exception:
                pass

    def _switch_representation(new_rep: str) -> None:
        nonlocal planner, is_moving, cur_rep, tracked_frames, old_obstacle_poses
        if not cur_spec.predictions:
            return
        is_moving = True
        status_md.content = f"**Status:** Switching to {new_rep}…"
        try:
            new_cfg = _prediction_to_scene_cfg(
                cur_spec.predictions, new_rep, 1.0,
                SCENE_TRANSLATION, SCENE_QUAT_WXYZ, torch.zeros(1),
            )
            planner = _rebuild_planner(new_cfg, new_rep, device_cfg)
            cur_rep = new_rep
            col_ns = set(planner.scene_collision_checker.get_obstacle_names())
            tracked_frames = {n: f for n, f in obstacle_frames.items() if n in col_ns}
            old_obstacle_poses = {
                n: Pose.from_numpy(f.position, f.wxyz) for n, f in tracked_frames.items()
            }
            status_md.content = "**Status:** Ready"
        except Exception as exc:
            print(f"[repr] error: {exc}")
            import traceback; traceback.print_exc()
            status_md.content = "**Status:** Error switching representation"
        finally:
            is_moving = False

    @repr_dd.on_update
    def _(_event) -> None:
        if is_moving:
            return
        threading.Thread(target=_switch_representation, args=(repr_dd.value,), daemon=True).start()

    def _switch_scene(new_name: str) -> None:
        nonlocal planner, is_moving, cur_spec, tracked_frames, old_obstacle_poses, cur_centroids
        is_moving = True
        status_md.content = f"**Status:** Loading {new_name}…"
        try:
            spec = next(s for s in scenes if s.name == new_name)
            if spec.predictions is None:
                status_md.content = f"**Status:** Running SuperDec for {new_name}…"
                _infer_scene_spec(spec, checkpoint_folder, mesh_resolution)
            new_cfg = _prediction_to_scene_cfg(
                spec.predictions or [], cur_rep, 1.0,
                SCENE_TRANSLATION, SCENE_QUAT_WXYZ, torch.zeros(1),
            )
            planner = _rebuild_planner(new_cfg, cur_rep, device_cfg)
            cur_spec = spec
            _clear_overlays()
            cur_centroids = _rebuild_obstacle_display(spec, objects_movable)
            if pc_chk.value and spec.object_pts:
                overlay_handles.extend(_add_superdec_overlays(
                    viser_viz, spec.object_pts, spec.object_colors,
                    spec.predictions or [],
                    SCENE_TRANSLATION, SCENE_QUAT_WXYZ, 1.0,
                    centroids=cur_centroids,
                    add_meshes=(cur_rep != "mesh"),
                ))
            status_md.content = "**Status:** Ready"
        except Exception as exc:
            print(f"[scene] error: {exc}")
            import traceback; traceback.print_exc()
            status_md.content = f"**Status:** Error loading {new_name}"
        finally:
            is_moving = False

    @scene_dd.on_update
    def _(_event) -> None:
        if is_moving:
            return
        threading.Thread(target=_switch_scene, args=(scene_dd.value,), daemon=True).start()

    @move_btn.on_click
    def on_move(_) -> None:
        nonlocal is_moving
        if is_moving:
            return

        def _plan() -> None:
            nonlocal is_moving
            is_moving = True
            update_obstacles()
            target_poses = viser_viz.get_control_frame_pose()
            active_js = planner.kinematics.get_active_js(current_state.clone())
            result = planner.plan_pose(
                GoalToolPose.from_poses(target_poses, num_goalset=1),
                active_js,
                use_implicit_goal=True,
                max_attempts=3,
            )
            if result is not None and result.success.any():
                status_md.content = "**Status:** Moving…"
                execute_trajectory(result.get_interpolated_plan())
                status_md.content = "**Status:** Ready"
            else:
                _find_colliding_obstacles(planner, active_js, server)
                status_md.content = "**Status:** Target unreachable"
            is_moving = False

        threading.Thread(target=_plan, daemon=True).start()

    @grasp_btn.on_click
    def on_grasp(_) -> None:
        nonlocal is_moving
        if is_moving:
            return

        def _plan() -> None:
            nonlocal is_moving
            is_moving = True
            update_obstacles()
            target_poses = viser_viz.get_control_frame_pose()
            active_js = planner.kinematics.get_active_js(current_state.clone())
            offset = Pose.from_list([0.0, 0.0, -0.15, 1.0, 0.0, 0.0, 0.0])
            approach_poses = {f: p.multiply(offset) for f, p in target_poses.items()}

            approach = planner.plan_pose(
                GoalToolPose.from_poses(approach_poses, num_goalset=1),
                active_js, max_attempts=5,
            )
            if approach is None or not approach.success.any():
                status_md.content = "**Status:** Grasp approach unreachable"
                is_moving = False
                return

            approach_end = planner.kinematics.get_active_js(
                JointState.from_position(
                    approach.js_solution.position[0, 0, -1, :].unsqueeze(0),
                    joint_names=approach.js_solution.joint_names,
                )
            )
            grasp = planner.plan_pose(
                GoalToolPose.from_poses(target_poses, num_goalset=1),
                approach_end, max_attempts=5,
            )
            if grasp is None or not grasp.success.any():
                status_md.content = "**Status:** Grasp target unreachable"
                is_moving = False
                return

            grasp_end = planner.kinematics.get_active_js(
                JointState.from_position(
                    grasp.js_solution.position[0, 0, -1, :].unsqueeze(0),
                    joint_names=grasp.js_solution.joint_names,
                )
            )
            lift = planner.plan_pose(
                GoalToolPose.from_poses(approach_poses, num_goalset=1),
                grasp_end, max_attempts=5,
            )
            status_md.content = "**Status:** Approaching…"
            execute_trajectory(approach.get_interpolated_plan())
            status_md.content = "**Status:** Grasping…"
            execute_trajectory(grasp.get_interpolated_plan())
            if lift is not None and lift.success.any():
                status_md.content = "**Status:** Lifting…"
                execute_trajectory(lift.get_interpolated_plan())
            else:
                print("Lift planning failed, skipping")
            status_md.content = "**Status:** Ready"
            is_moving = False

        threading.Thread(target=_plan, daemon=True).start()

    @reset_btn.on_click
    def _reset(_) -> None:
        nonlocal is_moving, current_state
        is_moving = False
        current_state = _initial_js.clone().unsqueeze(0)
        viser_viz.set_joint_state(_initial_js)
        for name, (pos, wxyz) in _initial_cf.items():
            if name in viser_viz._control_frames:
                viser_viz._control_frames[name].position = pos
                viser_viz._control_frames[name].wxyz = wxyz
        try:
            server.scene.remove_by_name("/collision_spheres")
        except Exception:
            pass
        status_md.content = "**Status:** Ready"

    print(f"\nTabletop Motion Planning Demo → http://localhost:{port}")
    print("  Scene 1 loaded.  Select scenes from the dropdown to run SuperDec on demand.")
    print("  Drag the goal frame, then click Move or Grasp.")
    print("Press Ctrl+C to exit.\n")

    try:
        while True:
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("\nShutting down.")


# ── argument parsing ──────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Tabletop motion planning demo with ShapeNet objects and SuperDec"
    )
    parser.add_argument(
        "--shapenet_root",
        type=str,
        default=str(ASSETS_ROOT / "ShapeNet_test"),
        help="Root directory of the ShapeNet dataset (ONet .npz format or PyG part-annotation)",
    )
    parser.add_argument(
        "--shapenet_split",
        type=str,
        default="test",
        help="ShapeNet split to load from {synset}/{split}.lst (default: test). "
             "Use 'all' to load every model directory present on disk.",
    )
    parser.add_argument(
        "--checkpoint_folder",
        type=str,
        default=CHECKPOINT_FOLDER,
        help="SuperDec checkpoint folder (must contain ckpt.pt and config.yaml)",
    )
    parser.add_argument(
        "--world_representation",
        type=str,
        choices=("superquadrics", "mesh"),
        default="superquadrics",
        help="Initial collision representation",
    )
    parser.add_argument("--mesh_resolution", type=int, default=48,
                        help="SuperDec mesh resolution for display and mesh collision mode")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Compute device: cuda or cpu")
    parser.add_argument("--port", type=int, default=8080,
                        help="Viser web server port")
    parser.add_argument("--gso_only", action="store_true",
                        help="Generate scenes using only GSO objects (ignores ShapeNet)")
    parser.add_argument("--num_scenes", type=int, default=None,
                        help="Number of scenes to generate (default: all in SCENE_BUILD_CONFIGS, or 1 for --gso_only)")
    parser.add_argument("--object_size", type=float, default=None,
                        help="Object half-extent in metres (default: per-scene value from SCENE_BUILD_CONFIGS, or 0.15 for --gso_only)")
    return parser.parse_args()


# ── entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    args = _parse_args()
    wp.init()

    print("Loading GSO dataset…")
    gso_dataset = _GSODataset(os.path.join(args.shapenet_root, "gso"))

    if args.gso_only:
        n_scenes = args.num_scenes if args.num_scenes is not None else 1
        obj_size = args.object_size if args.object_size is not None else 0.15
        scene_configs = [
            _SceneBuildCfg(f"GSO Scene {i + 1}", 10, 2048 + i * 111, dataset_type="gso", object_size_m=obj_size)
            for i in range(n_scenes)
        ]
        shapenet_dataset = None
    else:
        print("Loading ShapeNet dataset…")
        split = None if args.shapenet_split.lower() == "all" else args.shapenet_split
        shapenet_dataset = _load_shapenet(args.shapenet_root, TABLETOP_CATEGORIES, split=split)
        print(f"ShapeNet loaded: {len(shapenet_dataset)} models")
        base_configs = SCENE_BUILD_CONFIGS
        if args.num_scenes is not None:
            base_configs = base_configs[: args.num_scenes]
        if args.object_size is not None:
            base_configs = [
                _SceneBuildCfg(c.name, c.n_objects, c.seed, c.dataset_type, args.object_size)
                for c in base_configs
            ]
        scene_configs = base_configs

    n_scenes = len(scene_configs)
    print(f"\nBuilding {n_scenes} scene(s) (SuperDec runs now; needle-thin objects are replaced)…")
    scenes = _build_all_scenes(shapenet_dataset, gso_dataset, args.checkpoint_folder, args.mesh_resolution, scene_configs=scene_configs)

    scene_cfg = _prediction_to_scene_cfg(
        scenes[0].predictions or [], args.world_representation, 1.0,
        SCENE_TRANSLATION, SCENE_QUAT_WXYZ, torch.zeros(1),
    )
    device_cfg = DeviceCfg(device=args.device)
    planner = _rebuild_planner(scene_cfg, args.world_representation, device_cfg)

    interactive_motion_planning(
        planner=planner,
        scenes=scenes,
        initial_scene_idx=0,
        world_representation=args.world_representation,
        checkpoint_folder=args.checkpoint_folder,
        mesh_resolution=args.mesh_resolution,
        port=args.port,
    )


if __name__ == "__main__":
    main()
