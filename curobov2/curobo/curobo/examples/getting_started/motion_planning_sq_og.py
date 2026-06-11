"""Motion planning example that uses a SuperDec chair scene.

This is the curobo v2 counterpart to ``motion_planning.py``. It loads a
SuperDec point cloud, runs inference, and then builds either:

* a native superquadric scene, or
* a mesh scene generated from the same SuperDec predictions.

The SuperDec bridge in this workspace exposes the predicted primitive
parameters through ``PredictionHandler``, so this script keeps the SQ and mesh
representations in sync.

Timing logs are written to /home/haroldas/3DV/logs/curobov2/timing/ as JSON.
Run with --auto_cube_targets to sequence through target positions automatically.
Compare runs with compare_timings.py.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import threading
import time
from pathlib import Path
from typing import List, Sequence, Tuple

import numpy as np
import torch
import warp as wp
from omegaconf import OmegaConf
from scipy.spatial.transform import Rotation as SciRotation
import trimesh
import yaml

from curobo._src.geom.collision.collision_scene import SceneCollisionCfg
from curobo._src.geom.types import Cuboid, Mesh, SceneCfg, Superquadric
from curobo._src.motion.motion_planner_cfg import MotionPlannerCfg
from curobo._src.types.device_cfg import DeviceCfg
from curobo.motion_planner import MotionPlanner
from curobo.types import ContentPath, GoalToolPose, JointState, Pose
from curobo.viewer import ViserVisualizer


WORKSPACE_ROOT = Path(__file__).resolve().parents[5]
SUPERDEC_ROOT = WORKSPACE_ROOT / "superdec"
if str(SUPERDEC_ROOT) not in sys.path:
    sys.path.append(str(SUPERDEC_ROOT))

from superdec.superdec import SuperDec
from superdec.data.dataloader import normalize_points as normalize_points_superdec, denormalize_outdict as denormalize_outdict_superdec
from superdec.data.transform import rotate_around_axis as rotate_around_axis_superdec
from superdec.utils.predictions_handler import PredictionHandler
# from superdec.utils.visualizations import generate_ncolors

LOG_DIR = Path("/home/haroldas/3DV/logs/curobov2/timing")

TABLE = Cuboid(
    name="table",
    pose=[0.0, 0.0, -0.05, 1.0, 0.0, 0.0, 0.0],
    dims=[1.4, 1.4, 0.05],
)

DEFAULT_SCENE_TRANSLATION = np.array([-0.29955, -0.68389, 0.13559], dtype=np.float32)
DEFAULT_SCENE_QUAT_WXYZ = np.array([0.70711, 0.70711, 0.0, 0.0], dtype=np.float32)
DEFAULT_PLY_ROTATE_DEG_X = 90.0
DEFAULT_PLY_TRANSLATION = np.array([0.3, 0.8, -0.2], dtype=np.float32)

DEFAULT_NPZ_SCENE_TRANSLATION = np.array([-0.3, -0.2, 0.03], dtype=np.float32)
# Stored normalization centers are already in robot base frame (z-up), so no rotation needed.
DEFAULT_NPZ_SCENE_QUAT_WXYZ = np.array([0.70711, -0.70711, 0.0, 0.0], dtype=np.float32)

DEFAULT_SUPERDEC_SAMPLE_POINTS = 8192
DEFAULT_SQ_DISPLAY_MESH_RESOLUTION = 1000
SKIP_INSTANCES = {0}  
N_POINTS = 4096
RESOLUTION = 30
CKPT_FILE = "ckpt.pt"

SCALING_TEST_SIZES = [0, 2, 5, 20, 50]
DEFAULT_SCALING_TARGETS = [
    [0.7, 0.17, 0.5],
    [-0.1, -0.368, 0.5],
    [0.4, 0.4, 0.6],
    [0.5, -0.3, 0.4],
]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ply_path", type=str, default="/home/haroldas/3DV/superdec/examples/chair.ply", help="Input point cloud (.ply) for SuperDec")
    parser.add_argument("--npz_path", type=str, default="/home/haroldas/3DV/superdec/examples/scene.npz", help="Input scene point cloud (.npz) for SuperDec")

    parser.add_argument(
        "--checkpoint_folder",
        type=str,
        default="/home/haroldas/3DV/superdec/checkpoints/finetuned",
        help="SuperDec checkpoint folder containing ckpt.pt and config.yaml",
    )
    parser.add_argument(
        "--world_representation",
        type=str,
        choices=("superquadrics", "mesh"),
        default="superquadrics",
        help="Which scene representation to plan against",
    )
    parser.add_argument("--mesh_resolution", type=int, default=48, help="SuperDec mesh resolution")
    parser.add_argument(
        "--sq_display_mesh_resolution",
        type=int,
        default=DEFAULT_SQ_DISPLAY_MESH_RESOLUTION,
        help="Display-only SQ mesh resolution in superquadrics mode",
    )
    parser.add_argument("--scale_factor", type=float, default=1.0, help="Uniform scale applied to the SuperDec object")
    parser.add_argument(
        "--scene_translation",
        type=float,
        nargs=3,
        default=DEFAULT_SCENE_TRANSLATION.tolist(),
        help="Rigid translation applied to every SuperDec primitive",
    )
    parser.add_argument(
        "--scene_quat_wxyz",
        type=float,
        nargs=4,
        default=DEFAULT_SCENE_QUAT_WXYZ.tolist(),
        help="Rigid quaternion [qw, qx, qy, qz] applied to every SuperDec primitive",
    )
    parser.add_argument("--device", type=str, default="cuda", help="Planner device: cuda or cpu")
    parser.add_argument("--visualize", action="store_true", help="Launch the same Viser viewer as motion_planning.py")
    parser.add_argument(
        "--auto_cube_targets",
        type=str,
        default=None,
        help=(
            "JSON list of [x,y,z] target positions to plan to in sequence, "
            'e.g. \'[[0.7, 0.17, 0.5], [-0.1, -0.368, 0.5]]\'. '
            "Runs without user interaction and logs per-target planning times."
        ),
    )
    parser.add_argument(
        "--no_log_timing",
        action="store_true",
        help="Disable writing timing JSON to the log directory",
    )
    parser.add_argument(
        "--sofas",
        type=int,
        default=0,
        metavar="N",
        help=(
            "Add N additional instances of the input pointcloud arranged in a grid "
            "around the robot arm, ~1.5 m apart, each processed through SuperDec."
        ),
    )
    parser.add_argument(
        "--scaling_test",
        action="store_true",
        help=(
            f"Run a scaling benchmark: iterate through {SCALING_TEST_SIZES} additional sofa "
            "instances, plan to target poses at each size, and plot planning time vs obstacle count. "
            "Uses --auto_cube_targets if provided, otherwise falls back to DEFAULT_SCALING_TARGETS."
        ),
    )
    return parser.parse_args()


def _count_items(items) -> int:
    return len(items) if items is not None else 0


def _sofa_grid_positions(n: int, spacing: float = 1.5, min_dist: float = 1.1) -> List[Tuple[float, float]]:
    """Return (x, y) ground-plane positions for n sofas arranged around the origin.

    Candidates are generated on a regular grid with the given spacing.  Those
    within min_dist of the robot base are skipped so sofas never overlap the
    arm mount.  The remaining candidates are sorted nearest-first so the ring
    fills in naturally as n grows.
    """
    half = math.ceil(math.sqrt(n * 4)) + 2
    candidates: List[Tuple[float, float, float]] = []
    for row in range(-half, half + 1):
        for col in range(-half, half + 1):
            x = col * spacing
            y = row * spacing
            d = math.sqrt(x ** 2 + y ** 2)
            if d >= min_dist:
                candidates.append((x, y, d))
    candidates.sort(key=lambda p: p[2])
    return [(p[0], p[1]) for p in candidates[:n]]


def _make_pointcloud_instances(
    n: int,
    outdict: dict,
    points_tensor: torch.Tensor,
    world_representation: str,
    mesh_resolution: int,
    scale_factor: float,
    base_scene_translation: np.ndarray,
    base_scene_quat_wxyz: np.ndarray,
    native_meshes: List[trimesh.Trimesh] | None = None,
    spacing: float = 1.5,
) -> Tuple[List[Superquadric], List[Mesh]]:
    """Return SQ/mesh primitives for N copies of the SuperDec prediction placed around the robot.

    Each instance is placed at a grid position from _sofa_grid_positions and rotated
    around Z so it faces toward the robot, composing with base_scene_quat_wxyz.
    """
    if n <= 0:
        return [], []

    positions = _sofa_grid_positions(n, spacing=spacing)
    base_rot_xyzw = np.array([
        base_scene_quat_wxyz[1], base_scene_quat_wxyz[2],
        base_scene_quat_wxyz[3], base_scene_quat_wxyz[0],
    ], dtype=np.float32)
    base_rot = SciRotation.from_quat(base_rot_xyzw)

    primitive_count = int(outdict["scale"].shape[1])
    if world_representation == "mesh" and native_meshes is None:
        native_meshes = _superdec_native_meshes(outdict, points_tensor, mesh_resolution)

    all_sqs: List[Superquadric] = []
    all_meshes: List[Mesh] = []

    for inst_idx, (cx, cy) in enumerate(positions):
        dist = math.sqrt(cx ** 2 + cy ** 2)
        theta = math.atan2(cy, cx) if dist > 1e-6 else 0.0

        # Rz(theta) rotates the chair to face the robot; compose with the base orientation
        inst_rot = SciRotation.from_euler("z", theta) * base_rot
        inst_quat_xyzw = inst_rot.as_quat()
        inst_quat_wxyz = np.array([
            inst_quat_xyzw[3], inst_quat_xyzw[0], inst_quat_xyzw[1], inst_quat_xyzw[2],
        ], dtype=np.float32)
        inst_translation = np.array([cx, cy, float(base_scene_translation[2])], dtype=np.float32)

        if world_representation == "mesh" and native_meshes is not None:
            inst_rot = SciRotation.from_euler("z", theta) * base_rot
            inst_rot_matrix = inst_rot.as_matrix()
            for mesh_idx, mesh_tm in enumerate(native_meshes):
                transformed_mesh = _transform_trimesh_mesh(
                    mesh_tm,
                    inst_rot_matrix,
                    inst_translation,
                    scale_factor=scale_factor,
                )
                all_meshes.append(Mesh(
                    name=f"inst_{inst_idx}_mesh_{mesh_idx}",
                    vertices=transformed_mesh.vertices.tolist(),
                    faces=transformed_mesh.faces.tolist(),
                    pose=[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                ))
            continue

        for idx in range(primitive_count):
            if float(outdict["exist"][0, idx]) <= 0.5:
                continue

            scale = np.asarray(outdict["scale"][0, idx], dtype=np.float32) * scale_factor
            exponents = np.asarray(outdict["shape"][0, idx], dtype=np.float32)
            rotation = np.asarray(outdict["rotate"][0, idx], dtype=np.float32)
            translation = np.asarray(outdict["trans"][0, idx], dtype=np.float32)

            t_trans, t_rot = _apply_scene_transform(
                translation.tolist(), rotation, inst_translation, inst_quat_wxyz
            )
            pose = [
                float(t_trans[0]), float(t_trans[1]), float(t_trans[2]),
                *_rotation_matrix_to_wxyz(t_rot),
            ]
            
            if world_representation == "superquadrics":
                all_sqs.append(Superquadric(
                    name=f"inst_{inst_idx}_sq_{idx}",
                    pose=pose,
                    radii=scale.tolist(),
                    shape=exponents.tolist(),
                ))

    return all_sqs, all_meshes


def _normalize_points(points: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    translation = points.mean(axis=0)
    centered = points - translation
    scale = float(2.0 * np.max(np.abs(centered)))
    normalized = centered / scale
    return normalized, translation, scale


def _denormalize_outdict(outdict, translation: np.ndarray, scale: float, z_up: bool = False):
    scale_arr = np.asarray([[scale]], dtype=np.float32)
    translation_arr = translation.reshape(1, 1, 3)
    outdict["scale"] = outdict["scale"] * scale_arr[:, :, None]
    outdict["trans"] = outdict["trans"] * scale_arr[:, :, None] + translation_arr
    return outdict


def _denormalize_points(points: torch.Tensor, translation: np.ndarray, scale: float, z_up: bool = False):
    scale_t = torch.tensor(scale, dtype=points.dtype, device=points.device).view(1, 1, 1)
    translation_t = torch.tensor(translation, dtype=points.dtype, device=points.device).view(1, 1, 3)
    return points * scale_t + translation_t

def subsample(points, n):
    if points.shape[0] == n:
        return points
    idx = np.random.choice(points.shape[0], n, replace=points.shape[0] < n)
    return points[idx]


def _load_superdec_outputs_npz(
    npz_path: str,
    checkpoint_folder: str,
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    checkpoint = torch.load(os.path.join(checkpoint_folder, "ckpt.pt"), map_location=device, weights_only=False)
    configs = OmegaConf.load(os.path.join(checkpoint_folder, "config.yaml"))
    print("Loading SuperDec model from checkpoint ...")
    model = SuperDec(configs.superdec).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    data = np.load(npz_path, allow_pickle=True)
    xyz = data["xyz"].astype(np.float32)
    rgb = data["color"].astype(np.float32)
    if rgb.max() > 1.0:
        rgb = rgb / 255.0
    inst = data["instance_label"].astype(np.int64)

    instance_ids = [i for i in np.unique(inst) if i not in SKIP_INSTANCES]
    print(f"{len(instance_ids)} objects to fit")

    Handler = PredictionHandler

    outdicts = []
    points_tensors = []


    # +90° around X restores z-up from y-up (Rx(+90°) = [[1,0,0],[0,0,-1],[0,1,0]])
    _rot_x_pos90 = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], dtype=np.float32)
    # -90° around X converts z-up input to y-up (SuperDec's training frame)
    _rot_x_neg90 = SciRotation.from_euler("x", -90.0, degrees=True)

    for k, iid in enumerate(instance_ids):
        obj_pts_scene = xyz[inst == iid]               # z-up scene coords
        obj_pts = subsample(obj_pts_scene, N_POINTS)

        # Convert to y-up before inference (SuperDec was trained on y-up data)
        obj_pts_yup = _rot_x_neg90.apply(obj_pts.copy())
        pts_norm, translation, scale = _normalize_points(obj_pts_yup)
        pts_t = torch.from_numpy(pts_norm).unsqueeze(0).to(device).float()

        with torch.no_grad():
            out = model(pts_t)

        out = {key: (v.cpu() if isinstance(v, torch.Tensor) else v) for key, v in out.items()}

        # Denormalize back into y-up coordinates
        out = _denormalize_outdict(out, np.asarray(translation, dtype=np.float32), scale, False)

        # Rotate predictions from y-up back to z-up (scene world frame)
        trans_np = out["trans"].numpy() if isinstance(out["trans"], torch.Tensor) else np.asarray(out["trans"], dtype=np.float32)
        out["trans"] = torch.from_numpy((_rot_x_pos90 @ trans_np[0].T).T[None].astype(np.float32))
        rot_np = (out["rotate"].numpy() if isinstance(out["rotate"], torch.Tensor) else np.asarray(out["rotate"], dtype=np.float32)).copy()
        for p_idx in range(rot_np.shape[1]):
            rot_np[0, p_idx] = _rot_x_pos90 @ rot_np[0, p_idx]
        out["rotate"] = torch.from_numpy(rot_np)

        # Use original z-up scene points for PredictionHandler mesh generation
        pts_scene_t = torch.from_numpy(obj_pts_scene[None].astype(np.float32))

        outdicts.append(out)
        points_tensors.append(pts_scene_t)
        # server.scene.add_mesh_trimesh(f"/fit/obj_{iid}", mesh=mesh, visible=True)
        # print(f"  instance {iid}: fitted ({k + 1}/{len(instance_ids)})")

        # describe outdict
        print(f"[.npz] Instance {iid}: scale: {out['scale']}")
        print(f"[.npz] Summary of out: {out.keys()}, scale shape: {out['scale'].shape}, trans shape: {out['trans'].shape}")
    
    return outdicts, points_tensors


def _load_superdec_outputs_no_norm(
        ply_path: str,
        checkpoint_folder: str,
) -> tuple[dict, torch.Tensor]:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    checkpoint = torch.load(os.path.join(checkpoint_folder, "ckpt.pt"), map_location=device, weights_only=False)
    configs = OmegaConf.load(os.path.join(checkpoint_folder, "config.yaml"))
    print("Loading SuperDec model from checkpoint ...")
    model = SuperDec(configs.superdec).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    print("Loaded SuperDec model from checkpoint.")

    pc = trimesh.load(ply_path, process=False)

    if isinstance(pc, trimesh.Scene):
        point_sets = [
            np.asarray(geometry.vertices)
            for geometry in pc.geometry.values()
            if hasattr(geometry, "vertices")
        ]
        if not point_sets:
            raise ValueError(f"No point data found in point cloud: {ply_path}")
        points_np = np.concatenate(point_sets, axis=0)
    else:
        vertices = getattr(pc, "vertices", None)
        if vertices is None:
            raise ValueError(f"No point data found in point cloud: {ply_path}")
        points_np = np.asarray(vertices)

    if points_np.size == 0:
        raise ValueError(f"No points found in point cloud: {ply_path}")

    # Align imported point clouds to the expected frame before SuperDec inference.
    # rot_x = SciRotation.from_euler("x", DEFAULT_PLY_ROTATE_DEG_X, degrees=True)
    # points_np = rot_x.apply(points_np)
    # points_np = points_np + DEFAULT_PLY_TRANSLATION

    points_tensor = torch.from_numpy(points_np).unsqueeze(0).to(device).float()

    print("Running SuperDec inference ...")
    with torch.no_grad():
        outdict = model(points_tensor)
        for key, value in outdict.items():
            if isinstance(value, torch.Tensor):
                outdict[key] = value.cpu()
    
    print("SuperDec inference complete.")

    print(f"[.ply] Summary of outdict keys: {outdict.keys()}, scale shape: {outdict['scale'].shape}, trans shape: {outdict['trans'].shape}")

    return outdict, points_tensor


def _load_superdec_outputs(
    ply_path: str,
    checkpoint_folder: str,
) -> tuple[dict, torch.Tensor]:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    checkpoint = torch.load(os.path.join(checkpoint_folder, "ckpt.pt"), map_location=device, weights_only=False)
    configs = OmegaConf.load(os.path.join(checkpoint_folder, "config.yaml"))
    print("Loading SuperDec model from checkpoint ...")
    model = SuperDec(configs.superdec).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    print("Loaded SuperDec model from checkpoint.")

    pc = trimesh.load(ply_path, process=False)

    if isinstance(pc, trimesh.Scene):
        point_sets = [
            np.asarray(geometry.vertices)
            for geometry in pc.geometry.values()
            if hasattr(geometry, "vertices")
        ]
        if not point_sets:
            raise ValueError(f"No point data found in point cloud: {ply_path}")
        points_np = np.concatenate(point_sets, axis=0)
    else:
        vertices = getattr(pc, "vertices", None)
        if vertices is None:
            raise ValueError(f"No point data found in point cloud: {ply_path}")
        points_np = np.asarray(vertices)

    if points_np.size == 0:
        raise ValueError(f"No points found in point cloud: {ply_path}")

    # Align imported point clouds to the expected frame before SuperDec inference.
    rot_x = SciRotation.from_euler("x", DEFAULT_PLY_ROTATE_DEG_X, degrees=True)
    points_np = rot_x.apply(points_np)
    points_np = points_np + DEFAULT_PLY_TRANSLATION

    sample_size = min(DEFAULT_SUPERDEC_SAMPLE_POINTS, len(points_np))
    sample_idx = np.random.choice(len(points_np), sample_size, replace=len(points_np) < sample_size)
    points = points_np[sample_idx]
    points, translation, scale = _normalize_points(points)
    points_tensor = torch.from_numpy(points).unsqueeze(0).to(device).float()

    print("Running SuperDec inference ...")
    with torch.no_grad():
        outdict = model(points_tensor)
        for key, value in outdict.items():
            if isinstance(value, torch.Tensor):
                outdict[key] = value.cpu()
        outdict = _denormalize_outdict(outdict, np.asarray(translation, dtype=np.float32), scale, False)
        points_tensor = _denormalize_points(
            points_tensor.cpu(), np.asarray(translation, dtype=np.float32), scale, False
        )
    print("SuperDec inference complete.")

    print(f"[.ply] Instance 0: scale: {outdict['scale']}")
    print(f"[.ply] Shape shape: {outdict['shape']}")

    return outdict, points_tensor


def _load_superdec_outputs_norm(
    folder_path: str,
    checkpoint_folder: str,
) -> tuple[list[dict], list[torch.Tensor]]:
    """Load pre-normalized PLY objects and run SuperDec inference.

    Reads normalization.npz for per-object center/scale, then for each
    obj_XX.ply runs inference on the already-normalized points and denormalizes
    the outputs back to world coordinates.  Returns parallel lists matching the
    convention of _load_superdec_outputs_npz.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    checkpoint = torch.load(os.path.join(checkpoint_folder, "ckpt.pt"), map_location=device, weights_only=False)
    configs = OmegaConf.load(os.path.join(checkpoint_folder, "config.yaml"))
    print("Loading SuperDec model from checkpoint ...")
    model = SuperDec(configs.superdec).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # Load per-object normalization parameters
    norm_path = os.path.join(folder_path, "normalization.npz")
    norm_data = np.load(norm_path, allow_pickle=True)
    norm_lookup: dict[int, tuple[np.ndarray, float]] = {
        int(norm_data["obj_id"][i]): (
            np.asarray(norm_data["center"][i], dtype=np.float32),
            float(norm_data["scale"][i]),
        )
        for i in range(len(norm_data["obj_id"]))
    }
    print(f"Normalization params loaded for obj_ids: {sorted(norm_lookup.keys())}")

    ply_files = sorted(f for f in os.listdir(folder_path) if f.endswith(".ply"))
    print(f"{len(ply_files)} PLY files found in {folder_path}")

    outdicts: list[dict] = []
    points_tensors: list[torch.Tensor] = []

    for filename in ply_files:
        # Parse obj_id from filename pattern "obj_XX.ply"
        try:
            obj_id = int(filename.split("_")[1].split(".")[0])
        except (IndexError, ValueError):
            print(f"Skipping {filename}: cannot parse obj_id from filename")
            continue

        if obj_id not in norm_lookup:
            print(f"Skipping {filename}: no normalization entry for obj_id {obj_id}")
            continue

        center, scale = norm_lookup[obj_id]

        ply_path = os.path.join(folder_path, filename)
        pc = trimesh.load(ply_path, process=False)
        if isinstance(pc, trimesh.Scene):
            point_sets = [
                np.asarray(g.vertices)
                for g in pc.geometry.values()
                if hasattr(g, "vertices")
            ]
            if not point_sets:
                print(f"Skipping {filename}: no point data in scene")
                continue
            points_np = np.concatenate(point_sets, axis=0).astype(np.float32)
        else:
            vertices = getattr(pc, "vertices", None)
            if vertices is None:
                print(f"Skipping {filename}: no vertices")
                continue
            points_np = np.asarray(vertices, dtype=np.float32)

        points_np = subsample(points_np, N_POINTS)

        # Points are already normalized — feed directly to the model
        pts_t = torch.from_numpy(points_np).unsqueeze(0).to(device).float()
        with torch.no_grad():
            out = model(pts_t)
        out = {k: (v.cpu() if isinstance(v, torch.Tensor) else v) for k, v in out.items()}

        # Denormalize predictions back to world coordinates
        out = _denormalize_outdict(out, center, scale)

        # World-frame points for PredictionHandler mesh generation
        world_pts = (points_np * scale + center).astype(np.float32)
        points_tensor = torch.from_numpy(world_pts[None])

        outdicts.append(out)
        points_tensors.append(points_tensor)
        print(f"[norm] {filename} (obj_id={obj_id}): scale={out['scale']}")

    return outdicts, points_tensors


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
        [scene_quat_wxyz[1], scene_quat_wxyz[2], scene_quat_wxyz[3], scene_quat_wxyz[0]], dtype=np.float32
    )
    scene_rot = SciRotation.from_quat(scene_quat_xyzw)
    primitive_rot = SciRotation.from_matrix(rotation_matrix)
    new_translation = scene_rot.apply(np.asarray(translation, dtype=np.float32)) + scene_translation
    new_rotation = (scene_rot * primitive_rot).as_matrix()
    return new_translation, new_rotation


def _primitive_mesh(
    scale: Sequence[float] | np.ndarray,
    exponents: Sequence[float] | np.ndarray,
    rotation_matrix: np.ndarray,
    translation: Sequence[float] | np.ndarray,
    resolution: int,
):
    def f(angle, exponent):
        return np.sign(np.sin(angle)) * np.abs(np.sin(angle)) ** exponent

    def g(angle, exponent):
        return np.sign(np.cos(angle)) * np.abs(np.cos(angle)) ** exponent

    u = np.linspace(-np.pi, np.pi, resolution, endpoint=True)
    v = np.linspace(-np.pi / 2.0, np.pi / 2.0, resolution, endpoint=True)
    u = np.tile(u, resolution)
    v = np.repeat(v, resolution)

    if np.linalg.det(rotation_matrix) < 0:
        u = u[::-1]

    x = scale[0] * g(v, exponents[0]) * g(u, exponents[1])
    y = scale[1] * g(v, exponents[0]) * f(u, exponents[1])
    z = scale[2] * f(v, exponents[0])

    x[:resolution] = 0.0
    x[-resolution:] = 0.0

    vertices = np.concatenate(
        [np.expand_dims(x, 1), np.expand_dims(y, 1), np.expand_dims(z, 1)], axis=1
    )
    vertices = (rotation_matrix @ vertices.T).T + np.asarray(translation, dtype=np.float32)

    triangles = []
    for i in range(resolution - 1):
        for j in range(resolution - 1):
            triangles.append([i * resolution + j, i * resolution + j + 1, (i + 1) * resolution + j])
            triangles.append([(i + 1) * resolution + j, i * resolution + j + 1, (i + 1) * resolution + (j + 1)])
    for i in range(resolution - 1):
        triangles.append([i * resolution + (resolution - 1), i * resolution, (i + 1) * resolution + (resolution - 1)])
        triangles.append([(i + 1) * resolution + (resolution - 1), i * resolution, (i + 1) * resolution])

    triangles.append([(resolution - 1) * resolution + (resolution - 1), (resolution - 1) * resolution, (resolution - 1)])
    triangles.append([(resolution - 1), (resolution - 1) * resolution, 0])

    return trimesh.Trimesh(vertices=np.array(vertices), faces=np.array(triangles))


def _transform_trimesh_mesh(
    mesh_tm: trimesh.Trimesh,
    rotation_matrix: np.ndarray,
    translation: Sequence[float],
    scale_factor: float = 1.0,
) -> trimesh.Trimesh:
    vertices = np.asarray(mesh_tm.vertices, dtype=np.float32) * float(scale_factor)
    vertices = (rotation_matrix @ vertices.T).T + np.asarray(translation, dtype=np.float32)
    return trimesh.Trimesh(vertices=vertices, faces=np.asarray(mesh_tm.faces), process=False)


def _close_mesh(mesh_tm: trimesh.Trimesh) -> trimesh.Trimesh:
    """Make a mesh watertight so Warp BVH signed-distance is reliable.

    Fills boundary holes and fixes normal orientation.  Falls back to the
    convex hull if repair leaves the mesh non-watertight (hull is always
    closed and gives a conservative collision volume).
    """
    # import trimesh.repair
    # m = mesh_tm.copy()
    # trimesh.repair.fill_holes(m)
    # trimesh.repair.fix_normals(m, multibody=True)
    # if not m.is_watertight:
    #     m = mesh_tm.convex_hull
    return mesh_tm.copy()


def _superdec_native_meshes(outdict: dict, points_tensor: torch.Tensor, resolution: int) -> List[trimesh.Trimesh]:
    """Build the native SuperDec meshes for a single inferred object."""
    pred_handler = PredictionHandler.from_outdict(outdict, points_tensor, ["object"])
    combined_mesh = pred_handler.get_meshes(resolution=resolution, colors=False)[0]
    if combined_mesh is None:
        return []

    individual_meshes = combined_mesh.split()
    if len(individual_meshes) == 0:
        individual_meshes = [combined_mesh]

    return [trimesh.Trimesh(vertices=mesh.vertices, faces=mesh.faces, process=False) for mesh in individual_meshes]


def _superdec_display_scene_cfg(
    outdict: dict,
    points_tensor: torch.Tensor,
    resolution: int,
    scene_translation: Sequence[float],
    scene_quat_wxyz: Sequence[float],
) -> SceneCfg:
    """Build high-resolution display meshes from SuperDec predictions.

    This is visualization-only and does not affect collision checking.
    """
    native_meshes = _superdec_native_meshes(outdict, points_tensor, resolution)
    if len(native_meshes) == 0:
        return SceneCfg(mesh=[])

    scene_quat_xyzw = np.array(
        [scene_quat_wxyz[1], scene_quat_wxyz[2], scene_quat_wxyz[3], scene_quat_wxyz[0]], dtype=np.float32
    )
    scene_rot = SciRotation.from_quat(scene_quat_xyzw)

    meshes: List[Mesh] = []
    for i, mesh_tm in enumerate(native_meshes):
        transformed = _transform_trimesh_mesh(mesh_tm, scene_rot.as_matrix(), scene_translation)
        meshes.append(
            Mesh(
                name=f"sq_display_{i}",
                vertices=transformed.vertices.tolist(),
                faces=transformed.faces.tolist(),
                pose=[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
            )
        )
    return SceneCfg(mesh=meshes)


def _prediction_to_scene_cfg(
    outdict: dict,
    points_tensor: torch.Tensor,
    world_representation: str,
    mesh_resolution: int,
    scale_factor: float,
    scene_translation: Sequence[float],
    scene_quat_wxyz: Sequence[float],
) -> SceneCfg:
    
    print(f"Summary of outdict: keys={list(outdict.keys())}, scale_shape={outdict['scale'].shape}, exist_shape={outdict['exist'].shape}")

    scene_translation_np = np.asarray(scene_translation, dtype=np.float32)
    scene_quat_np = np.asarray(scene_quat_wxyz, dtype=np.float32)
    scene_quat_xyzw = np.array(
        [scene_quat_np[1], scene_quat_np[2], scene_quat_np[3], scene_quat_np[0]], dtype=np.float32
    )
    scene_rot = SciRotation.from_quat(scene_quat_xyzw)

    superquadrics: List[Superquadric] = []
    meshes: List[Mesh] = []

    primitive_count = int(outdict["scale"].shape[1])
    for idx in range(primitive_count):
        if float(outdict["exist"][0, idx]) <= 0.5:
            continue

        scale = np.asarray(outdict["scale"][0, idx], dtype=np.float32) * scale_factor
        exponents = np.asarray(outdict["shape"][0, idx], dtype=np.float32)
        rotation = np.asarray(outdict["rotate"][0, idx], dtype=np.float32)
        translation = np.asarray(outdict["trans"][0, idx], dtype=np.float32)

        transformed_translation, transformed_rotation = _apply_scene_transform(
            translation.tolist(), rotation, scene_translation_np, scene_quat_np
        )
        pose = [
            float(transformed_translation[0]),
            float(transformed_translation[1]),
            float(transformed_translation[2]),
            *_rotation_matrix_to_wxyz(transformed_rotation),
        ]

        if world_representation == "superquadrics":
            superquadrics.append(
                Superquadric(
                    name=f"chair_sq_{idx}",
                    pose=pose,
                    radii=scale.tolist(),
                    shape=exponents.tolist(),
                )
            )

    if world_representation == "mesh":
        native_meshes = _superdec_native_meshes(outdict, points_tensor, mesh_resolution)
        for idx, mesh_tm in enumerate(native_meshes):
            transformed_mesh = _transform_trimesh_mesh(
                mesh_tm,
                scene_rot.as_matrix(),
                scene_translation_np,
                scale_factor=scale_factor,
            )
            meshes.append(
                Mesh(
                    name=f"chair_mesh_{idx}",
                    vertices=transformed_mesh.vertices.tolist(),
                    faces=transformed_mesh.faces.tolist(),
                    pose=[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                )
            )

    if world_representation == "superquadrics":
        scene = SceneCfg(cuboid=[TABLE], superquadric=superquadrics)
    elif world_representation == "mesh":
        scene = SceneCfg(cuboid=[TABLE], mesh=meshes)
    else:
        raise ValueError(f"Unsupported world_representation: {world_representation}")

    n_prims = len(superquadrics) if world_representation == "superquadrics" else len(meshes)
    print(f"Built SuperDec scene with {n_prims} {world_representation} primitives")
    return scene

def _prediction_to_scene_cfg_npz(
    outdicts: List[dict],
    points_tensors: List[torch.Tensor],
    world_representation: str,
    mesh_resolution: int,
    scale_factor: float,
    scene_translation: Sequence[float],
    scene_quat_wxyz: Sequence[float],
) -> SceneCfg:
    scene_translation_np = np.asarray(scene_translation, dtype=np.float32)
    scene_quat_np = np.asarray(scene_quat_wxyz, dtype=np.float32)
    scene_quat_xyzw = np.array(
        [scene_quat_np[1], scene_quat_np[2], scene_quat_np[3], scene_quat_np[0]], dtype=np.float32
    )
    scene_rot = SciRotation.from_quat(scene_quat_xyzw)

    superquadrics: List[Superquadric] = []
    meshes: List[Mesh] = []

    MIN_RADIUS_M = 0.005  # skip needle-thin primitives that destabilise the Warp BVH

    sq_i = 0
    for outdict in outdicts:
        primitive_count = int(outdict["scale"].shape[1])
        for idx in range(primitive_count):
            if float(outdict["exist"][0, idx]) <= 0.5:
                continue

            scale = np.asarray(outdict["scale"][0, idx], dtype=np.float32) * scale_factor
            if float(scale.min()) < MIN_RADIUS_M:
                continue
            exponents = np.asarray(outdict["shape"][0, idx], dtype=np.float32)
            rotation = np.asarray(outdict["rotate"][0, idx], dtype=np.float32)
            translation = np.asarray(outdict["trans"][0, idx], dtype=np.float32)

            transformed_translation, transformed_rotation = _apply_scene_transform(
                translation.tolist(), rotation, scene_translation_np, scene_quat_np
            )
            pose = [
                float(transformed_translation[0]),
                float(transformed_translation[1]),
                float(transformed_translation[2]),
                *_rotation_matrix_to_wxyz(transformed_rotation),
            ]

            # Always build SQs — mesh mode converts them via sq.get_mesh() below
            superquadrics.append(
                Superquadric(
                    name=f"sq_{sq_i}",
                    pose=pose,
                    radii=scale.tolist(),
                    shape=exponents.tolist(),
                )
            )
            sq_i += 1

    if world_representation == "mesh":
        mesh_i = 0
        for outdict, points_tensor in zip(outdicts, points_tensors):
            native_meshes = _superdec_native_meshes(outdict, points_tensor, mesh_resolution)
            for mesh_tm in native_meshes:
                transformed_mesh = _transform_trimesh_mesh(
                    mesh_tm, scene_rot.as_matrix(), scene_translation_np, scale_factor=scale_factor
                )
                transformed_mesh = _close_mesh(transformed_mesh)
                meshes.append(
                    Mesh(
                        name=f"mesh_{mesh_i}",
                        vertices=transformed_mesh.vertices.tolist(),
                        faces=transformed_mesh.faces.tolist(),
                        pose=[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                    )
                )
                mesh_i += 1

    if world_representation == "superquadrics":
        scene = SceneCfg(cuboid=[TABLE], superquadric=superquadrics)
    elif world_representation == "mesh":
        scene = SceneCfg(cuboid=[TABLE], mesh=meshes)
    else:
        raise ValueError(f"Unsupported world_representation: {world_representation}")

    n_prims = len(superquadrics) if world_representation == "superquadrics" else len(meshes)
    print(f"Built SuperDec scene with {n_prims} {world_representation} primitives")
    if superquadrics:
        zs = [sq.pose[2] for sq in superquadrics]
        rs = [max(sq.radii) for sq in superquadrics]
        print(f"  SQ z-range in robot frame: [{min(zs):.3f}, {max(zs):.3f}] m")
        print(f"  SQ max-radius range: [{min(rs):.3f}, {max(rs):.3f}] m")
        large = [sq for sq in superquadrics if max(sq.radii) > 0.4]
        if large:
            print(f"  WARNING: {len(large)} SQ(s) with radius > 0.4 m (may cause false collisions):")
            for sq in large[:5]:
                print(f"    {sq.name}: pose_z={sq.pose[2]:.3f}, radii={[f'{r:.3f}' for r in sq.radii]}")
    return scene



def _make_goal_pose(planner: MotionPlanner) -> GoalToolPose:
    positions = torch.tensor([[[[[0.55, 0.0, 0.65]]]]], device=planner.device_cfg.device, dtype=torch.float32)
    quaternions = torch.tensor([[[[[1.0, 0.0, 0.0, 0.0]]]]], device=planner.device_cfg.device, dtype=torch.float32)
    return GoalToolPose(tool_frames=planner.tool_frames, position=positions, quaternion=quaternions)


def _goal_pose_for_target(planner: MotionPlanner, xyz: Sequence[float]) -> GoalToolPose:
    positions = torch.tensor(
        [[[[[xyz[0], xyz[1], xyz[2]]]]]],
        device=planner.device_cfg.device,
        dtype=torch.float32,
    )
    quaternions = torch.tensor(
        [[[[[1.0, 0.0, 0.0, 0.0]]]]],
        device=planner.device_cfg.device,
        dtype=torch.float32,
    )
    return GoalToolPose(tool_frames=planner.tool_frames, position=positions, quaternion=quaternions)


def _save_timing_log(timing_data: dict) -> Path:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    rep = timing_data.get("representation", "unknown")
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    path = LOG_DIR / f"{rep}_{timestamp}.json"
    with open(path, "w") as fh:
        json.dump(timing_data, fh, indent=2)
    print(f"Timing log saved → {path}")
    return path


def pose_planning_example(planner: MotionPlanner) -> None:
    q_start = JointState.from_position(
        torch.as_tensor(planner.default_joint_state.position, device=planner.device_cfg.device, dtype=torch.float32).unsqueeze(0),
        joint_names=planner.joint_names,
    )
    goal_pose = _make_goal_pose(planner)
    result = planner.plan_pose(goal_pose, q_start)
    if result is not None and result.success.any():
        print("Pose planning succeeded")
    else:
        print("Pose planning failed")


def grasp_planning_example(planner: MotionPlanner) -> None:
    q_start = JointState.from_position(
        torch.as_tensor(planner.default_joint_state.position, device=planner.device_cfg.device, dtype=torch.float32).unsqueeze(0),
        joint_names=planner.joint_names,
    )

    n_grasps = 3
    positions = torch.zeros(1, 1, 1, n_grasps, 3, device=planner.device_cfg.device, dtype=torch.float32)
    positions[..., 0] = 0.55
    positions[0, 0, 0, :, 1] = torch.linspace(-0.15, 0.15, n_grasps, device=planner.device_cfg.device)
    positions[..., 2] = 0.65

    quaternions = torch.zeros(1, 1, 1, n_grasps, 4, device=planner.device_cfg.device, dtype=torch.float32)
    quaternions[..., 0] = 1.0

    grasp_poses = GoalToolPose(tool_frames=planner.tool_frames, position=positions, quaternion=quaternions)
    result = planner.plan_grasp(
        current_state=q_start,
        grasp_poses=grasp_poses,
        grasp_approach_offset=0.1,
        grasp_lift_offset=0.1,
        plan_approach_to_grasp=True,
        plan_grasp_to_lift=True,
        grasp_lift_in_tool_frame=True,
    )

    if result.success is not None and result.success.any():
        print("Grasp planning succeeded")
    else:
        status = getattr(result, "status", "unknown")
        print(f"Grasp planning failed: {status}")


def run_auto_targets(
    planner: MotionPlanner,
    targets: List[List[float]],
) -> List[dict]:
    """Plan to each target position in sequence and return per-target timing records."""
    records = []
    current_state = JointState.from_position(
        torch.as_tensor(
            planner.default_joint_state.position,
            device=planner.device_cfg.device,
            dtype=torch.float32,
        ).unsqueeze(0),
        joint_names=planner.joint_names,
    )

    for i, xyz in enumerate(targets):
        goal_pose = _goal_pose_for_target(planner, xyz)
        print(f"  [{i+1}/{len(targets)}] Planning to target {xyz} ...", end=" ", flush=True)

        t0 = time.perf_counter()
        result = planner.plan_pose(goal_pose, current_state, max_attempts=3)
        elapsed = time.perf_counter() - t0

        success = bool(result is not None and result.success.any())
        status_str = "OK" if success else "FAIL"
        print(f"{elapsed:.4f}s  [{status_str}]")

        records.append({
            "target_index": i,
            "target_xyz": xyz,
            "plan_s": elapsed,
            "success": success,
        })

        # Advance start state to end of last successful trajectory so the
        # sequence is physically continuous.
        if success and result is not None:
            try:
                js = result.js_solution
                if js is not None:
                    # position: (1, 1, horizon, dof) — take the last waypoint
                    pos_t: torch.Tensor = js.position  # type: ignore[assignment]
                    last_pos = pos_t.reshape(-1, pos_t.shape[-1])[-1:, :]
                    current_state = JointState.from_position(
                        last_pos, joint_names=js.joint_names
                    )
                    current_state = planner.kinematics.get_active_js(current_state)
            except Exception:
                pass  # keep previous state if extraction fails

    return records


def run_auto_targets_visualized(
    planner: MotionPlanner,
    scene_cfg: SceneCfg,
    targets: List[List[float]],
    display_scene_cfg: SceneCfg | None = None,
    port: int = 8080,
) -> Tuple[List[dict], float]:
    """Plan to each target, animate the robot in Viser, then keep the viewer open.

    Sequencing per target i:
      1. Target frame jumps to position i          (visible immediately)
      2. Wait 2 s so the user can see it
      3. Target frame jumps to position i+1        (robot is about to chase i)
      4. Robot plans + animates to position i      (frame already shows next target)
    """
    viser_viz = ViserVisualizer(
        content_path=ContentPath(robot_config_file="franka.yml"),
        connect_ip="0.0.0.0",
        connect_port=port,
        add_control_frames=True,
        visualize_robot_spheres=False,
    )
    viser_viz.add_scene(display_scene_cfg if display_scene_cfg is not None else scene_cfg, add_control_frames=True)

    print(f"\nOpen http://localhost:{port} to watch the robot.")
    print("Waiting 3 s for browser to connect before starting...")
    time.sleep(3.0)

    current_state = JointState.from_position(
        torch.as_tensor(
            planner.default_joint_state.position,
            device=planner.device_cfg.device,
            dtype=torch.float32,
        ).unsqueeze(0),
        joint_names=planner.joint_names,
    )

    is_moving = False

    def execute_trajectory(trajectory) -> None:
        nonlocal current_state, is_moving
        traj = trajectory.squeeze(0)
        for i in range(traj.position.shape[-2]):
            if not is_moving:
                return
            waypoint = JointState.from_position(
                traj.position[0, i, :].unsqueeze(0),
                joint_names=traj.joint_names,
            )
            viser_viz.set_joint_state(waypoint.squeeze(0))
            time.sleep(0.02)
        current_state = JointState.from_position(
            traj.position[0, -1, :].unsqueeze(0),
            joint_names=traj.joint_names,
        )
    
    viz_pause_total_s = 3.0  # initial browser connection wait

    records = []
    for i, xyz in enumerate(targets):
        print(f"  [{i+1}/{len(targets)}] Planning to target {xyz} ...", end=" ", flush=True)
        viser_viz._control_frames["panda_hand"].position = (xyz[0], xyz[1], xyz[2])

        def plan_and_execute(_i=i, _xyz=xyz) -> None:
            nonlocal is_moving
            is_moving = True
            target_poses = viser_viz.get_control_frame_pose()

            active_js = planner.kinematics.get_active_js(current_state.clone())
            t0 = time.perf_counter()
            result = planner.plan_pose(
                GoalToolPose.from_poses(target_poses, num_goalset=1),
                active_js,
                use_implicit_goal=True,
                max_attempts=3,
            )
            elapsed = time.perf_counter() - t0

            success = bool(result is not None and result.success.any())
            status_str = "OK" if success else "FAIL"
            print(f"{elapsed:.4f}s  [{status_str}]")

            records.append({
                "target_index": _i,
                "target_xyz": _xyz,
                "plan_s": elapsed,
                "success": success,
            })

            if result is not None and result.success.any():
                execute_trajectory(result.get_interpolated_plan())
            else:
                print("Motion planning failed")
            is_moving = False

        t = threading.Thread(target=plan_and_execute, daemon=True)
        t.start()
        t.join()

    print("\nAll targets done. Viewer stays open — press Ctrl+C to exit.")
    try:
        while True:
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("\nShutting down...")

    return records, viz_pause_total_s


def interactive_motion_planning(
    planner: MotionPlanner,
    scene_cfg: SceneCfg,
    use_cuda_graph: bool = True,
    port: int = 8080,
    display_scene_cfg: SceneCfg | None = None,
) -> None:
    """Launch the same Viser-based interaction model as the standard tutorial."""
    viser_viz = ViserVisualizer(
        content_path=ContentPath(robot_config_file="franka.yml"),
        connect_ip="0.0.0.0",
        connect_port=port,
        add_control_frames=True,
        visualize_robot_spheres=False,
    )

    obstacle_frames = viser_viz.add_scene(scene_cfg, add_control_frames=True)
    # Add optional high-resolution display meshes as a visual overlay.
    if display_scene_cfg is not None and display_scene_cfg.mesh is not None:
        _add_scene_handles(viser_viz, display_scene_cfg)
    old_obstacle_poses = {
        name: Pose.from_numpy(frame.position, frame.wxyz) for name, frame in obstacle_frames.items()
    }
    scene_collision_checker = planner.scene_collision_checker
    assert scene_collision_checker is not None

    current_state = planner.default_joint_state.clone().unsqueeze(0)
    planner.warmup(enable_graph=True, num_warmup_iterations=5)
    is_moving = False

    def update_obstacles() -> None:
        for name in obstacle_frames.keys():
            new_pose = Pose.from_numpy(obstacle_frames[name].position, obstacle_frames[name].wxyz)
            if new_pose != old_obstacle_poses[name]:
                scene_collision_checker.update_obstacle_pose(name, new_pose)
                old_obstacle_poses[name] = new_pose.clone()

    def execute_trajectory(trajectory) -> None:
        nonlocal current_state, is_moving
        traj = trajectory.squeeze(0)
        for i in range(traj.position.shape[-2]):
            if not is_moving:
                return
            waypoint = JointState.from_position(
                traj.position[0, i, :].unsqueeze(0),
                joint_names=traj.joint_names,
            )
            viser_viz.set_joint_state(waypoint.squeeze(0))
            time.sleep(0.02)
        current_state = JointState.from_position(
            traj.position[0, -1, :].unsqueeze(0),
            joint_names=traj.joint_names,
        )

    def on_move(_):
        nonlocal is_moving
        if is_moving:
            return

        def plan_and_execute() -> None:
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
                execute_trajectory(result.get_interpolated_plan())
            else:
                print("Motion planning failed")
            is_moving = False

        threading.Thread(target=plan_and_execute, daemon=True).start()

    def on_grasp(_):
        nonlocal is_moving
        if is_moving:
            return

        def plan_grasp_and_execute() -> None:
            nonlocal is_moving
            is_moving = True
            update_obstacles()
            target_poses = viser_viz.get_control_frame_pose()
            active_js = planner.kinematics.get_active_js(current_state.clone())

            offset = Pose.from_list([0.0, 0.0, -0.15, 1.0, 0.0, 0.0, 0.0])
            approach_poses = {f: p.multiply(offset) for f, p in target_poses.items()}
            lift_poses = approach_poses

            approach_result = planner.plan_pose(
                GoalToolPose.from_poses(approach_poses, num_goalset=1),
                active_js,
                max_attempts=5,
            )
            if approach_result is None or not approach_result.success.any():
                print("Grasp planning failed: approach pose unreachable")
                is_moving = False
                return

            approach_end = JointState.from_position(
                approach_result.js_solution.position[0, 0, -1, :].unsqueeze(0),
                joint_names=approach_result.js_solution.joint_names,
            )
            approach_end = planner.kinematics.get_active_js(approach_end)

            grasp_result = planner.plan_pose(
                GoalToolPose.from_poses(target_poses, num_goalset=1),
                approach_end,
                max_attempts=5,
            )
            if grasp_result is None or not grasp_result.success.any():
                print("Grasp planning failed: grasp pose unreachable from approach")
                is_moving = False
                return

            grasp_end = JointState.from_position(
                grasp_result.js_solution.position[0, 0, -1, :].unsqueeze(0),
                joint_names=grasp_result.js_solution.joint_names,
            )
            grasp_end = planner.kinematics.get_active_js(grasp_end)

            lift_result = planner.plan_pose(
                GoalToolPose.from_poses(lift_poses, num_goalset=1),
                grasp_end,
                max_attempts=5,
            )

            execute_trajectory(approach_result.get_interpolated_plan())
            execute_trajectory(grasp_result.get_interpolated_plan())
            if lift_result is not None and lift_result.success.any():
                execute_trajectory(lift_result.get_interpolated_plan())
            else:
                print("Lift planning failed, skipping")
            is_moving = False

        threading.Thread(target=plan_grasp_and_execute, daemon=True).start()

    move_btn = viser_viz._server.gui.add_button("Move", color="green")
    move_btn.on_click(on_move)
    grasp_btn = viser_viz._server.gui.add_button("Grasp", color="blue")
    grasp_btn.on_click(on_grasp)

    print(f"\nInteractive Motion Planner running at http://localhost:{port}")
    print("  - Drag the target frame to set goal pose")
    print("  - Drag obstacles to reposition them")
    print("  - Click 'Move' for pose-to-pose planning")
    print("  - Click 'Grasp' for approach-grasp-lift planning")
    print("Press Ctrl+C to exit.\n")

    try:
        while True:
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("\nShutting down...")


def _add_scene_handles(viser_viz, scene_cfg: SceneCfg) -> list:
    """Add scene to Viser as meshes and return handles so they can be removed later."""
    mesh_scene = SceneCfg.create_mesh_scene(scene_cfg)
    handles = []
    for mesh in (mesh_scene.mesh or []):
        h = viser_viz.add_mesh(
            mesh.get_trimesh_mesh(transform_with_pose=True),
            name="/obstacles/" + mesh.name + "/mesh",
        )
        handles.append(h)
    return handles


def _run_size_visualized(
    viser_viz,
    planner: MotionPlanner,
    targets: List[List[float]],
) -> List[dict]:
    """Plan and animate through all targets in an existing Viser viewer.

    Mirrors run_auto_targets_visualized but returns immediately after the last
    target instead of blocking — the caller decides whether to keep the viewer open.
    """
    current_state = JointState.from_position(
        torch.as_tensor(
            planner.default_joint_state.position,
            device=planner.device_cfg.device,
            dtype=torch.float32,
        ).unsqueeze(0),
        joint_names=planner.joint_names,
    )
    is_moving = False
    records: List[dict] = []

    def execute_trajectory(trajectory) -> None:
        nonlocal current_state, is_moving
        traj = trajectory.squeeze(0)
        for step in range(traj.position.shape[-2]):
            if not is_moving:
                return
            waypoint = JointState.from_position(
                traj.position[0, step, :].unsqueeze(0),
                joint_names=traj.joint_names,
            )
            viser_viz.set_joint_state(waypoint.squeeze(0))
            time.sleep(0.02)
        current_state = JointState.from_position(
            traj.position[0, -1, :].unsqueeze(0),
            joint_names=traj.joint_names,
        )

    for i, xyz in enumerate(targets):
        print(f"  [{i+1}/{len(targets)}] Planning to {xyz} ...", end=" ", flush=True)
        viser_viz._control_frames["panda_hand"].position = (xyz[0], xyz[1], xyz[2])

        def plan_and_execute(_i=i, _xyz=xyz) -> None:
            nonlocal is_moving
            is_moving = True
            target_poses = viser_viz.get_control_frame_pose()
            active_js = planner.kinematics.get_active_js(current_state.clone())
            t0 = time.perf_counter()
            result = planner.plan_pose(
                GoalToolPose.from_poses(target_poses, num_goalset=1),
                active_js,
                use_implicit_goal=True,
                max_attempts=3,
            )
            elapsed = time.perf_counter() - t0
            success = bool(result is not None and result.success.any())
            print(f"{elapsed:.4f}s  [{'OK' if success else 'FAIL'}]")
            records.append({"target_index": _i, "target_xyz": _xyz, "plan_s": elapsed, "success": success})
            if result is not None and success:
                execute_trajectory(result.get_interpolated_plan())
            else:
                print("Motion planning failed")
            is_moving = False

        t = threading.Thread(target=plan_and_execute, daemon=True)
        t.start()
        t.join()

    return records


def _plot_scaling_results(results: List[dict], representation: str) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    xs = [r["n_collision_objects"] for r in results]
    means = [r["mean_plan_s"] for r in results]
    mins = [r["min_plan_s"] for r in results]
    maxs = [r["max_plan_s"] for r in results]
    err_low = [m - mn for m, mn in zip(means, mins)]
    err_high = [mx - m for mx, m in zip(maxs, means)]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.errorbar(
        xs, means,
        yerr=[err_low, err_high],
        marker="o", capsize=5, linewidth=2, markersize=8,
        label=representation,
    )
    for r in results:
        ax.annotate(
            f"n={r['n_sofas']}\n{r['n_success']}/{r['n_total']}",
            (r["n_collision_objects"], r["mean_plan_s"]),
            textcoords="offset points", xytext=(6, 4), fontsize=8,
        )
    ax.set_xlabel("Collision objects in scene")
    ax.set_ylabel("Mean planning time (s)")
    ax.set_title(f"Planning time vs scene complexity — {representation}")
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.tight_layout()

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    plot_path = LOG_DIR / f"scaling_{representation}_{timestamp}.png"
    plt.savefig(plot_path, dpi=150)
    plt.close(fig)
    print(f"Scaling plot saved → {plot_path}")
    return plot_path


def run_scaling_test(
    args: argparse.Namespace,
    outdict: dict,
    points_tensor: torch.Tensor,
    targets: List[List[float]],
    sizes: List[int] = SCALING_TEST_SIZES,
    visualize: bool = False,
    port: int = 8080,
) -> List[dict]:
    """Plan to targets at each scene size in *sizes* and plot the results.

    When *visualize* is True a single Viser viewer is kept open for the entire
    test; the scene is swapped between sizes so the browser never needs to reload.
    After the last size the viewer stays open until Ctrl-C.
    """
    device_cfg = DeviceCfg(device=args.device)
    results = []

    # ---- One-time Viser setup ------------------------------------------------
    viser_viz = None
    scene_handles: list = []
    if visualize:
        viser_viz = ViserVisualizer(
            content_path=ContentPath(robot_config_file="franka.yml"),
            connect_ip="0.0.0.0",
            connect_port=port,
            add_control_frames=True,
            visualize_robot_spheres=False,
        )
        print(f"\nOpen http://localhost:{port} to watch the scaling test.")
        print("Waiting 3 s for browser to connect before starting...")
        time.sleep(3.0)

    for n_sofas in sizes:
        print(f"\n{'='*55}")
        print(f"Scaling test  |  representation={args.world_representation}  |  n_sofas={n_sofas}")
        print(f"{'='*55}")

        scene_cfg = _prediction_to_scene_cfg(
            outdict=outdict,
            points_tensor=points_tensor,
            world_representation=args.world_representation,
            mesh_resolution=args.mesh_resolution,
            scale_factor=args.scale_factor,
            scene_translation=args.scene_translation,
            scene_quat_wxyz=args.scene_quat_wxyz,
        )

        if n_sofas > 0:
            inst_sqs, inst_meshes = _make_pointcloud_instances(
                n_sofas,
                outdict,
                points_tensor,
                args.world_representation,
                args.mesh_resolution,
                args.scale_factor,
                np.asarray(args.scene_translation, dtype=np.float32),
                np.asarray(args.scene_quat_wxyz, dtype=np.float32),
            )
            scene_cfg = SceneCfg(
                cuboid=scene_cfg.cuboid,
                superquadric=list(scene_cfg.superquadric or []) + inst_sqs,
                mesh=list(scene_cfg.mesh or []) + inst_meshes,
            )

        n_sq = _count_items(scene_cfg.superquadric)
        n_mesh = _count_items(scene_cfg.mesh)
        n_cub = _count_items(scene_cfg.cuboid)
        n_total_obs = n_sq + n_mesh + n_cub
        print(f"Scene: {n_sq} SQs  {n_mesh} meshes  {n_cub} cuboids  = {n_total_obs} total")

        # ---- Swap scene in viewer --------------------------------------------
        if viser_viz is not None:
            display_scene_cfg = scene_cfg
            if args.world_representation == "superquadrics":
                display_scene_cfg = _superdec_display_scene_cfg(
                    outdict=outdict,
                    points_tensor=points_tensor,
                    resolution=args.sq_display_mesh_resolution,
                    scene_translation=args.scene_translation,
                    scene_quat_wxyz=args.scene_quat_wxyz,
                )
            for h in scene_handles:
                try:
                    h.remove()
                except Exception:
                    pass
            scene_handles = _add_scene_handles(viser_viz, display_scene_cfg)

        planner_cfg = MotionPlannerCfg.create(
            robot="franka.yml",
            scene_model="collision_test.yml",
            device_cfg=device_cfg,
            use_cuda_graph=False,
            max_goalset=10,
        )
        planner_cfg.scene_collision_cfg = SceneCollisionCfg(
            device_cfg=device_cfg,
            scene_model=scene_cfg,
            cache={"cuboid": n_cub, "mesh": n_mesh, "superquadric": n_sq},
        )
        planner = MotionPlanner(planner_cfg)
        planner.warmup(enable_graph=False, num_warmup_iterations=5)

        if viser_viz is not None:
            plan_records = _run_size_visualized(viser_viz, planner, targets)
        else:
            plan_records = run_auto_targets(planner, targets)

        plan_times = [r["plan_s"] for r in plan_records]
        n_ok = sum(r["success"] for r in plan_records)

        results.append({
            "n_sofas": n_sofas,
            "n_collision_objects": n_total_obs,
            "n_sq": n_sq,
            "n_mesh": n_mesh,
            "mean_plan_s": float(np.mean(plan_times)) if plan_times else 0.0,
            "min_plan_s": float(np.min(plan_times)) if plan_times else 0.0,
            "max_plan_s": float(np.max(plan_times)) if plan_times else 0.0,
            "n_success": n_ok,
            "n_total": len(plan_records),
        })
        print(
            f"  mean={results[-1]['mean_plan_s']:.4f}s  "
            f"min={results[-1]['min_plan_s']:.4f}s  "
            f"max={results[-1]['max_plan_s']:.4f}s  "
            f"success={n_ok}/{len(plan_records)}"
        )

    _plot_scaling_results(results, args.world_representation)

    if not args.no_log_timing:
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        log_path = LOG_DIR / f"scaling_{args.world_representation}_{timestamp}.json"
        with open(log_path, "w") as fh:
            json.dump({
                "representation": args.world_representation,
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "sizes": sizes,
                "results": results,
            }, fh, indent=2)
        print(f"Scaling log saved → {log_path}")

    if viser_viz is not None:
        print("\nAll sizes done. Final scene stays visible — press Ctrl+C to exit.")
        try:
            while True:
                time.sleep(0.1)
        except KeyboardInterrupt:
            print("\nShutting down...")

    return results

def fit_and_save_ply(
        npz_path: str,
        checkpoint_folder: str,
        output_ply_path: str,
):
    outdicts, points_tensors = _load_superdec_outputs_npz(npz_path, checkpoint_folder)
    meshes = []
    for outdict, points_tensor in zip(outdicts, points_tensors):
        meshes.extend(_superdec_native_meshes(outdict, points_tensor, resolution=300))
    combined_mesh = trimesh.util.concatenate(meshes)
    combined_mesh.export(output_ply_path)
    print(f"Fitted mesh saved to {output_ply_path}")
        


def main() -> None:
    print("Starting SQ motion planning example")
    args = _parse_args()
    wp.init()

    # fit_and_save_ply(args.npz_path, args.checkpoint_folder, "/home/haroldas/3DV/superdec/examples/scene_fit_2.ply")

    t_total_start = time.perf_counter()
    timing: dict = {
        "representation": args.world_representation,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "ply_path": args.ply_path,
    }

    device_cfg = DeviceCfg(device=args.device)

    t0 = time.perf_counter()


    # outdict, points_tensor = _load_superdec_outputs(args.ply_path, args.checkpoint_folder)
    # outdict, points_tensor = _load_superdec_outputs_norm(args.ply_path, args.checkpoint_folder)
    # outdicts, points_tensors = _load_superdec_outputs_npz(args.npz_path, args.checkpoint_folder) 

    # Object red list: obj_03.ply, 

    MULT_PLY_PATH = "/home/haroldas/3DV/superdec/examples/Archive/objects_pc/"#Archive/objects_pc_normalized"
    # outdicts, points_tensors = _load_superdec_outputs_norm(MULT_PLY_PATH, args.checkpoint_folder)
    outdicts = []
    points_tensors = []
    for file in os.listdir(MULT_PLY_PATH):
        if file.endswith(".ply") and not file.startswith("obj_03"):  # Exclude obj_03.ply
            file_path = os.path.join(MULT_PLY_PATH, file)
            outdict, points_tensor= _load_superdec_outputs(file_path, args.checkpoint_folder)
            outdicts.append(outdict)
            points_tensors.append(points_tensor)


    timing["superdec_inference_s"] = time.perf_counter() - t0
    print(f"SuperDec inference: {timing['superdec_inference_s']:.3f}s")

    # ---- Scaling test (iterates over SCALING_TEST_SIZES, then exits) --------
    if args.scaling_test:
        if args.auto_cube_targets is not None:
            try:
                scaling_targets = json.loads(args.auto_cube_targets)
            except json.JSONDecodeError as exc:
                raise ValueError(f"--auto_cube_targets must be valid JSON: {exc}") from exc
        else:
            scaling_targets = DEFAULT_SCALING_TARGETS
            print(f"--scaling_test: no --auto_cube_targets supplied, using {len(scaling_targets)} default targets")
        run_scaling_test(args, outdict, points_tensor, scaling_targets, visualize=args.visualize)
        return

    t0 = time.perf_counter()
    # scene_cfg = _prediction_to_scene_cfg(
    #     outdict=outdict,
    #     points_tensor=points_tensor,
    #     world_representation=args.world_representation,
    #     mesh_resolution=args.mesh_resolution,
    #     scale_factor=args.scale_factor,
    #     scene_translation=args.scene_translation,
    #     scene_quat_wxyz=args.scene_quat_wxyz,
    # )
    scene_cfg = _prediction_to_scene_cfg_npz(
        outdicts=outdicts,
        points_tensors=points_tensors,
        world_representation=args.world_representation,
        mesh_resolution=args.mesh_resolution,
        scale_factor=args.scale_factor,
        scene_translation=DEFAULT_NPZ_SCENE_TRANSLATION.tolist(),
        scene_quat_wxyz=DEFAULT_NPZ_SCENE_QUAT_WXYZ.tolist(),
    )

    timing["scene_build_s"] = time.perf_counter() - t0
    print(f"Scene build: {timing['scene_build_s']:.3f}s")

    

    if args.sofas > 0:
        inst_sqs, inst_meshes = _make_pointcloud_instances(
            args.sofas,
            outdict,
            points_tensor,
            args.world_representation,
            args.mesh_resolution,
            args.scale_factor,
            np.asarray(args.scene_translation, dtype=np.float32),
            np.asarray(args.scene_quat_wxyz, dtype=np.float32),
        )
        scene_cfg = SceneCfg(
            cuboid=scene_cfg.cuboid,
            superquadric=list(scene_cfg.superquadric or []) + inst_sqs,
            mesh=list(scene_cfg.mesh or []) + inst_meshes,
        )
        print(f"Added {args.sofas} pointcloud instances ({len(inst_sqs)} SQs, {len(inst_meshes)} meshes) to the scene")

    timing["n_superquadrics"] = _count_items(scene_cfg.superquadric)
    timing["n_meshes"] = _count_items(scene_cfg.mesh)
    timing["n_cuboids"] = _count_items(scene_cfg.cuboid)
    timing["n_sofas"] = args.sofas

    print(
        f"Scene summary: {timing['n_superquadrics']} superquadrics, "
        f"{timing['n_meshes']} meshes, {timing['n_cuboids']} cuboids "
        f"(incl. {args.sofas} sofas)"
    )

    # use_cuda_graph = args.world_representation != "superquadrics"
    use_cuda_graph = False

    t0 = time.perf_counter()
    planner_cfg = MotionPlannerCfg.create(
        robot="franka.yml",
        scene_model="collision_test.yml",
        device_cfg=device_cfg,
        use_cuda_graph=use_cuda_graph,
        max_goalset=10,
    )
    planner_cfg.scene_collision_cfg = SceneCollisionCfg(
        device_cfg=device_cfg,
        scene_model=scene_cfg,
        cache={
            "cuboid": _count_items(scene_cfg.cuboid),
            "mesh": _count_items(scene_cfg.mesh),
            "superquadric": _count_items(scene_cfg.superquadric),
        },
    )
    planner = MotionPlanner(planner_cfg)
    timing["planner_init_s"] = time.perf_counter() - t0
    print(f"Planner init: {timing['planner_init_s']:.3f}s")

    t0 = time.perf_counter()
    planner.warmup(enable_graph=use_cuda_graph, num_warmup_iterations=5)
    timing["warmup_s"] = time.perf_counter() - t0
    print(f"Warmup: {timing['warmup_s']:.3f}s")

    # ---- Auto-target sequencing -----------------------------------------
    display_scene_cfg: SceneCfg | None = None
    if args.visualize and args.world_representation == "superquadrics":
        # Build display meshes from the NPZ superquadrics (world-frame), same approach
        # as mesh collision mode, so the display matches what the planner sees.
        display_meshes = []
        for i, sq in enumerate(scene_cfg.superquadric or []):
            world_mesh = sq.get_trimesh_mesh(transform_with_pose=True)
            display_meshes.append(
                Mesh(
                    name=f"display_{i}",
                    vertices=world_mesh.vertices.tolist(),
                    faces=world_mesh.faces.tolist(),
                    pose=[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                )
            )
        display_scene_cfg = SceneCfg(mesh=display_meshes)

    if args.auto_cube_targets is not None:
        try:
            targets = json.loads(args.auto_cube_targets)
        except json.JSONDecodeError as exc:
            raise ValueError(f"--auto_cube_targets must be valid JSON: {exc}") from exc
        if not isinstance(targets, list) or not all(len(t) == 3 for t in targets):
            raise ValueError("--auto_cube_targets must be a list of [x, y, z] triples")

        if args.visualize:
            # Animate the robot through each target in the Viser viewer (blocks until Ctrl+C).
            plan_records, viz_pause_total_s = run_auto_targets_visualized(
                planner,
                scene_cfg,
                targets,
                display_scene_cfg=display_scene_cfg,
            )
            timing["viz_pause_total_s"] = viz_pause_total_s
        else:
            print(f"\nAuto-target planning: {len(targets)} targets")
            plan_records = run_auto_targets(planner, targets)

        timing["plans"] = plan_records
        plan_times = [r["plan_s"] for r in plan_records]
        timing["plan_mean_s"] = float(np.mean(plan_times)) if plan_times else None
        timing["plan_min_s"] = float(np.min(plan_times)) if plan_times else None
        timing["plan_max_s"] = float(np.max(plan_times)) if plan_times else None
        timing["plan_total_s"] = float(np.sum(plan_times)) if plan_times else None
        n_success = sum(1 for r in plan_records if r["success"])
        print(
            f"\nPlanning summary: {n_success}/{len(targets)} succeeded  "
            f"mean={timing['plan_mean_s']:.4f}s  "
            f"min={timing['plan_min_s']:.4f}s  "
            f"max={timing['plan_max_s']:.4f}s"
        )
    elif not args.visualize:
        # Fall back to the fixed examples only when no targets and no visualizer
        t0 = time.perf_counter()
        pose_planning_example(planner)
        timing["pose_plan_s"] = time.perf_counter() - t0

        t0 = time.perf_counter()
        grasp_planning_example(planner)
        timing["grasp_plan_s"] = time.perf_counter() - t0

        print(f"Pose planning: {timing['pose_plan_s']:.3f}s")
        print(f"Grasp planning: {timing['grasp_plan_s']:.3f}s")

    timing["total_s"] = time.perf_counter() - t_total_start
    print(f"\nTotal wall time: {timing['total_s']:.3f}s")

    if not args.no_log_timing:
        _save_timing_log(timing)

    # --visualize alone (no --auto_cube_targets) → interactive mode.
    if args.visualize and args.auto_cube_targets is None:
        interactive_motion_planning(
            planner,
            scene_cfg,
            use_cuda_graph=use_cuda_graph,
            display_scene_cfg=display_scene_cfg,
        )


if __name__ == "__main__":
    main()
