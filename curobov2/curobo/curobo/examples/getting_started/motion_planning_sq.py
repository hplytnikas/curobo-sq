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
from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence, Tuple

import numpy as np
import torch
import warp as wp
from omegaconf import OmegaConf
from scipy.spatial.transform import Rotation as SciRotation
import trimesh

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
if str(SUPERDEC_ROOT) not in sys.path:
    sys.path.append(str(SUPERDEC_ROOT))

from superdec.superdec import SuperDec
from superdec.utils.predictions_handler import PredictionHandler
from superdec.data.dataloader import normalize_points, denormalize_outdict
from superdec.data.transform import rotate_around_axis



LOG_DIR = Path("/home/haroldas/3DV/logs/curobov2/timing")

TABLE = Cuboid(
    name="table",
    pose=[0.0, 0.0, -0.05, 1.0, 0.0, 0.0, 0.0],
    dims=[1.4, 1.4, 0.05],
)

DEFAULT_SCENE_TRANSLATION = np.array([-0.1, -0.5, -0.77], dtype=np.float32)
DEFAULT_SCENE_QUAT_WXYZ = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
DEFAULT_PLY_ROTATE_DEG_X = 90.0
DEFAULT_PLY_TRANSLATION = np.array([0.3, 0.8, -0.2], dtype=np.float32)
DEFAULT_SUPERDEC_SAMPLE_POINTS = 8192
RESOLUTION = 30
CKPT_FILE = "ckpt.pt"
SKIP_INSTANCES = {0}      # instance 0 is the table — skip (degenerate flat plane)

SCALING_TEST_SIZES = [0, 2, 5, 20, 50]
DEFAULT_SCALING_TARGETS = [
    [0.7, 0.17, 0.5],
    [-0.1, -0.368, 0.5],
    [0.4, 0.4, 0.6],
    [0.5, -0.3, 0.4],
]

GLOBAL_SCALE = 1.0


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ply_path", type=str, default="/home/haroldas/3DV/superdec/examples/chair.ply", help="Input point cloud (.ply/.npz) for SuperDec")
    parser.add_argument(
        "--scenes_dir",
        type=str,
        default=None,
        help="Directory of NPZ scene files; adds a Scene dropdown in the interactive viewer",
    )
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
    parser.add_argument(
        "--show_pointcloud",
        action="store_true",
        help="Show the input pointcloud in the Viser viewer, transformed into the same scene frame as the fitted objects",
    )
    parser.add_argument(
        "--fix_objects",
        action="store_true",
        help="Add obstacle meshes without control-frame axes in Viser",
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
        "--visualize_sdf",
        action="store_true",
        help="Sample random points around the robot and color them red (in collision) / blue (free) to visualize the scene SDF",
    )
    parser.add_argument(
        "--sdf_n_points",
        type=int,
        default=5000,
        help="Number of random points to sample for SDF visualization (default: 5000)",
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


def _pose_debug_list(poses) -> dict:
    return {name: pose.tolist() for name, pose in poses.items()}


def _inv_pose_to_world_center(inv_pose_row: np.ndarray) -> np.ndarray:
    """Return obstacle world-frame center from an inv_pose row [tx,ty,tz,qw,qx,qy,qz,...]."""
    t = inv_pose_row[:3].astype(np.float64)
    q = inv_pose_row[3:7].astype(np.float64)  # [qw, qx, qy, qz]
    q_xyzw = np.array([q[1], q[2], q[3], q[0]])
    R = SciRotation.from_quat(q_xyzw).as_matrix()
    return -(R.T @ t)


def _find_colliding_obstacles(planner, active_js, viser_server=None) -> None:
    """Print the obstacles that appear to be in collision with the robot at active_js.

    Uses approximate bounding-sphere distances: reports any obstacle whose
    bounding sphere overlaps any robot collision sphere.

    If viser_server is provided, draws red spheres at each colliding robot sphere position.
    """
    # Use the rollout's checker — it is the one actually used for feasibility decisions.
    scene_collision = None
    try:
        scene_collision = planner.graph_planner.feasibility_rollout.scene_collision_checker
    except AttributeError:
        pass
    if scene_collision is None:
        scene_collision = getattr(planner, "scene_collision_checker", None)
    if scene_collision is None:
        print("  [collision diag] no scene_collision_checker found")
        return

    try:
        kin_state = planner.kinematics.compute_kinematics(active_js)
        if kin_state.robot_spheres is None:
            print("  [collision diag] robot_spheres is None (spheres not computed)")
            return
        # [1, 1, n_spheres, 4] → [n_spheres, 4]
        spheres_np = kin_state.robot_spheres[0, 0].detach().cpu().numpy()
    except Exception as exc:
        print(f"  [collision diag] FK failed: {exc}")
        return

    data = scene_collision.data
    n_cub = int(data.cuboids.count[0].item()) if data.cuboids is not None else 0
    n_sq = int(data.superquadrics.count[0].item()) if data.superquadrics is not None else 0
    n_mesh = int(data.meshes.count[0].item()) if data.meshes is not None else 0
    print(f"  [collision diag] scene: {n_cub} cuboids, {n_sq} SQs, {n_mesh} meshes")

    results = []

    def _check_store(name: str, obs_center: np.ndarray, obs_bounding_r: float) -> None:
        min_dist = float("inf")
        for s in spheres_np:
            d = float(np.linalg.norm(s[:3] - obs_center)) - float(s[3]) - obs_bounding_r
            if d < min_dist:
                min_dist = d
        results.append((name, min_dist))

    try:
        if data.superquadrics is not None:
            sq = data.superquadrics
            for i in range(n_sq):
                name = sq.names[0][i] or f"sq_{i}"
                inv_pose = sq.inv_pose[0, i].detach().cpu().numpy()
                params = sq.params[0, i].detach().cpu().numpy()
                center = _inv_pose_to_world_center(inv_pose)
                bounding_r = float(np.max(params[:3])) * 1.732
                _check_store(name, center, bounding_r)

        if data.cuboids is not None:
            cub = data.cuboids
            for i in range(n_cub):
                name = cub.names[0][i] or f"cub_{i}"
                inv_pose = cub.inv_pose[0, i].detach().cpu().numpy()
                dims = cub.dims[0, i].detach().cpu().numpy()
                center = _inv_pose_to_world_center(inv_pose)
                bounding_r = float(np.max(dims[:3])) / 2 * 1.732
                _check_store(name, center, bounding_r)
    except Exception as exc:
        print(f"  [collision diag] obstacle scan failed: {exc}")
        return

    results.sort(key=lambda x: x[1])
    in_collision = [(n, d) for n, d in results if d < 0.0]
    near = [(n, d) for n, d in results if 0.0 <= d < 0.05]

    if in_collision:
        print(f"  Obstacles in collision ({len(in_collision)}):")
        for name, dist in in_collision:
            print(f"    {name}  (bounding-sphere dist={dist:.4f} m)")
    elif near:
        print(f"  No bounding-sphere overlaps, but {len(near)} obstacle(s) within 5 cm:")
        for name, dist in near[:5]:
            print(f"    {name}  (dist={dist:.4f} m)")
    else:
        if results:
            print(f"  No bounding-sphere overlaps (closest: {results[0][0]} dist={results[0][1]:.4f} m)")
        else:
            print("  No obstacles in scene.")

    # --- per-sphere collision check using actual collision cost ---
    try:
        spheres_t = kin_state.robot_spheres  # [1, 1, N, 4]
        col_buf = CollisionBuffer.from_shape(spheres_t.shape, planner.device_cfg)
        w = torch.ones(1, dtype=torch.float32, device=planner.device_cfg.device)
        act = torch.zeros(1, dtype=torch.float32, device=planner.device_cfg.device)
        with torch.no_grad():
            cost = scene_collision.get_sphere_distance_raw(
                query_spheres=spheres_t,
                collision_buffer=col_buf,
                weight=w,
                activation_distance=act,
            )  # [1, 1, N]
        cost_np = cost[0, 0].cpu().numpy()

        # Build sphere index → link name
        kin_cfg = planner.kinematics.config.kinematics_config
        idx_to_name = {v: k for k, v in kin_cfg.link_name_to_idx_map.items()}
        sphere_links = [
            idx_to_name.get(int(kin_cfg.link_sphere_idx_map[i].item()), f"sphere_{i}")
            for i in range(len(kin_cfg.link_sphere_idx_map))
        ]

        colliding_spheres = [
            (sphere_links[i], spheres_np[i], float(cost_np[i]))
            for i in range(len(cost_np))
            if cost_np[i] > 0.0
        ]
        colliding_spheres.sort(key=lambda x: -x[2])

        if colliding_spheres:
            print(f"  Robot spheres in collision ({len(colliding_spheres)}/{len(cost_np)}):")
            for link_name, sxyzr, c in colliding_spheres:
                print(
                    f"    {link_name}  "
                    f"xyz=({sxyzr[0]:.3f},{sxyzr[1]:.3f},{sxyzr[2]:.3f})  "
                    f"r={sxyzr[3]:.3f}  cost={c:.4f}"
                )
            if viser_server is not None:
                # Remove any previously drawn collision spheres then redraw
                try:
                    viser_server.scene.remove_by_name("/collision_spheres")
                except Exception:
                    pass
                for i, (link_name, sxyzr, c) in enumerate(colliding_spheres):
                    mesh = trimesh.creation.icosphere(subdivisions=3, radius=float(sxyzr[3]) * 2.0)
                    mesh.apply_translation([float(sxyzr[0]), float(sxyzr[1]), float(sxyzr[2])])
                    viser_server.scene.add_mesh_trimesh(
                        name=f"/collision_spheres/{link_name}_{i}",
                        mesh=mesh,
                        wxyz=(1.0, 0.0, 0.0, 0.0),
                        position=(0.0, 0.0, 0.0),
                        visible=True,
                    )
        else:
            print(f"  No robot spheres with cost > 0 (checked {len(cost_np)} spheres)")
    except Exception as exc:
        print(f"  [sphere collision] query failed: {exc}")


def _tensor_debug_value(value):
    if value is None:
        return None
    if hasattr(value, "detach"):
        return value.detach().cpu().tolist()
    return value


@dataclass
class SuperDecPrediction:
    iid: int
    outdict: dict
    mesh: trimesh.Trimesh | None


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
    predictions: List[SuperDecPrediction],
    world_representation: str,
    scale_factor: float,
    base_scene_translation: np.ndarray,
    base_scene_quat_wxyz: np.ndarray,
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

        for pred in predictions:
            outdict = pred.outdict
            primitive_count = int(outdict["scale"].shape[1])

            for idx in range(primitive_count):
                if _get_outdict_scalar(outdict, "exist", idx) <= 0.5:
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

                all_sqs.append(Superquadric(
                    name=f"inst_{inst_idx}_sq_{pred.iid}_{idx}",
                    pose=pose,
                    radii=scale.tolist(),
                    shape=exponents.tolist(),
                ))

    if world_representation == "mesh":
        return [], _superquadrics_to_meshes(all_sqs)

    return all_sqs, all_meshes


def _get_outdict_scalar(outdict: dict, key: str, idx: int) -> float:
    """Return a scalar float for outdict[key] at index idx robustly.

    Handles numpy arrays, torch tensors, and singleton arrays with extra dims.
    """
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
    # fall back to first element
    return float(val.ravel()[0])


def _subsample(points, n):
    if points.shape[0] == n:
        return points
    idx = np.random.choice(points.shape[0], n, replace=points.shape[0] < n)
    return points[idx]

def _load_model(ckpt_dir, device, ckpt_file=CKPT_FILE):
    cfg = OmegaConf.load(os.path.join(ckpt_dir, "config.yaml"))
    model = SuperDec(cfg.superdec).to(device)
    model.lm_optimization = False
    ckpt = torch.load(os.path.join(ckpt_dir, ckpt_file), map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"loaded {ckpt_dir}/{ckpt_file}")
    return model


def _load_superdec_outputs(
    ply_path: str,
    checkpoint_folder: str,
    mesh_resolution: int,
) -> Tuple[List[SuperDecPrediction], np.ndarray, np.ndarray]:
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = _load_model(checkpoint_folder, device)
    Handler = PredictionHandler

    print("Loaded SuperDec model from checkpoint.")

    data = np.load(ply_path, allow_pickle=True)
    xyz = data["xyz"].astype(np.float32)
    rgb = data["color"].astype(np.float32)
    if rgb.max() > 1.0:
        rgb = rgb / 255.0
    inst = data["instance_label"].astype(np.int64)
    instance_ids = [i for i in np.unique(inst) if i not in SKIP_INSTANCES]
    print(f"{len(instance_ids)} objects to fit")


    predictions: List[SuperDecPrediction] = []

    for k, iid in enumerate(instance_ids):
        obj_pts_scene = xyz[inst == iid]               # z-up scene coords
        obj_pts = _subsample(obj_pts_scene, DEFAULT_SUPERDEC_SAMPLE_POINTS)

        obj_pts = obj_pts * GLOBAL_SCALE

        obj_pts_yup = rotate_around_axis(obj_pts, axis=(1, 0, 0),
                                         angle=-np.pi / 2, center_point=np.zeros(3))
        
        pts_norm, translation, scale = normalize_points(obj_pts_yup)
        pts_t = torch.from_numpy(pts_norm).unsqueeze(0).to(device).float()

        with torch.no_grad():
            out = model(pts_t)
        out = {key: (v.cpu() if isinstance(v, torch.Tensor) else v) for key, v in out.items()}

        # denormalize back into scene (z-up) coordinates
        out = denormalize_outdict(out, np.array([translation]), np.array([scale]), z_up=True)

        # drop primitives whose thinnest axis is below 3 mm — they produce degenerate
        # needle meshes that cause false positives in the Warp BVH signed-distance query
        _MIN_RADIUS_M = 0.003
        _s = out["scale"]
        _scales_np = (_s.cpu().numpy() if isinstance(_s, torch.Tensor) else np.asarray(_s)).astype(np.float32)
        _exist_np = np.asarray(out["exist"])
        _n = _scales_np.shape[1]
        _active = _exist_np.reshape(_n) > 0.5
        _needle = _scales_np[0].min(axis=1) < _MIN_RADIUS_M
        _drop = _active & _needle
        if _drop.any():
            for _p in np.where(_drop)[0]:
                out["exist"][0, _p] = 0.0
            print(f"  instance {iid}: filtered {int(_drop.sum())} needle-thin primitive(s)")

        pts_back = torch.from_numpy(obj_pts_scene[None].astype(np.float32))
        handler = Handler.from_outdict(out, pts_back[:, :DEFAULT_SUPERDEC_SAMPLE_POINTS], [str(iid)])
        mesh = handler.get_meshes(resolution=mesh_resolution)[0]
        if mesh is None:
            print(f"  instance {iid}: no mesh")
            continue
        predictions.append(SuperDecPrediction(iid=int(iid), outdict=out, mesh=mesh))
        print(f"  instance {iid}: fitted ({k + 1}/{len(instance_ids)})")
        print(f"Mesh type: {type(mesh)}")

    return predictions, xyz, rgb


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


def _superdec_display_scene_cfg(
    predictions: List[SuperDecPrediction],
    cuboids,
    scene_translation: Sequence[float],
    scene_quat_wxyz: Sequence[float],
    scale_factor: float,
) -> "SceneCfg":
    """Build a display-only SceneCfg using SuperDec prediction meshes.

    Uses pred.mesh from handler.get_meshes() (correct winding) instead of
    Superquadric.get_trimesh_mesh(), matching the demo_viser_scene.py approach.
    Vertices are pre-transformed into world frame; pose is identity.
    """
    scene_quat_xyzw = np.array(
        [scene_quat_wxyz[1], scene_quat_wxyz[2], scene_quat_wxyz[3], scene_quat_wxyz[0]],
        dtype=np.float32,
    )
    rot = SciRotation.from_quat(scene_quat_xyzw).as_matrix()
    display_meshes = []
    for pred in predictions:
        if pred.mesh is None:
            continue
        tm = _transform_trimesh_mesh(pred.mesh, rot, scene_translation, scale_factor)
        display_meshes.append(Mesh(
            name=f"obj_{pred.iid}",
            vertices=tm.vertices.tolist(),
            faces=tm.faces.tolist(),
            pose=[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
        ))
    return SceneCfg(cuboid=list(cuboids) if cuboids else [], mesh=display_meshes)


def _transform_pointcloud(
    points: np.ndarray,
    scene_translation: Sequence[float],
    scene_quat_wxyz: Sequence[float],
) -> np.ndarray:
    scene_quat_xyzw = np.array(
        [scene_quat_wxyz[1], scene_quat_wxyz[2], scene_quat_wxyz[3], scene_quat_wxyz[0]],
        dtype=np.float32,
    )
    scene_rot = SciRotation.from_quat(scene_quat_xyzw)
    return scene_rot.apply(np.asarray(points, dtype=np.float32)) + np.asarray(scene_translation, dtype=np.float32)


def _prediction_to_scene_cfg(
    predictions: List[SuperDecPrediction],
    world_representation: str,
    scale_factor: float,
    scene_translation: Sequence[float],
    scene_quat_wxyz: Sequence[float],
    points_tensor: torch.Tensor,
) -> SceneCfg:
    scene_translation_np = np.asarray(scene_translation, dtype=np.float32)
    scene_quat_np = np.asarray(scene_quat_wxyz, dtype=np.float32)
    scene_quat_xyzw = np.array(
        [scene_quat_np[1], scene_quat_np[2], scene_quat_np[3], scene_quat_np[0]], dtype=np.float32
    )
    superquadrics: List[Superquadric] = []
    meshes: List[Mesh] = []

    for pred in predictions:
        outdict = pred.outdict
        primitive_count = int(outdict["scale"].shape[1])
        for idx in range(primitive_count):
            if _get_outdict_scalar(outdict, "exist", idx) <= 0.5:
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

            superquadrics.append(
                Superquadric(
                    name=f"chair_sq_{pred.iid}_{idx}",
                    pose=pose,
                    radii=scale.tolist(),
                    shape=exponents.tolist(),
                )
            )
    scene_rot = SciRotation.from_quat(scene_quat_xyzw)

    if world_representation == "mesh":
        rot = scene_rot.as_matrix()
        for pred in predictions:
            if pred.mesh is None:
                continue
            tm = _transform_trimesh_mesh(pred.mesh, rot, scene_translation_np, scale_factor)
            meshes.append(Mesh(
                name=f"chair_sq_{pred.iid}",
                vertices=tm.vertices.tolist(),
                faces=tm.faces.tolist(),
                pose=[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
            ))


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
            print(f"  WARNING: {len(large)} SQ(s) with radius > 0.4m:")
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
    predictions: List[SuperDecPrediction],
    xyz: np.ndarray,
    rgb: np.ndarray,
    port: int = 8080,
    scene_translation: Sequence[float] | None = None,
    scene_quat_wxyz: Sequence[float] | None = None,
    scale_factor: float = 1.0,
    show_pointcloud: bool = False,
    fix_objects: bool = False,
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
    viser_viz._server.scene.set_up_direction([0.0, 0.0, 1.0])
    _display_cfg = _superdec_display_scene_cfg(
        predictions, scene_cfg.cuboid, scene_translation, scene_quat_wxyz, scale_factor
    )
    viser_viz.add_scene(_display_cfg, add_control_frames=not fix_objects)
    if show_pointcloud and scene_translation is not None and scene_quat_wxyz is not None:
        _add_superdec_overlays(
            viser_viz,
            xyz=xyz,
            rgb=rgb,
            predictions=predictions,
            scene_translation=scene_translation,
            scene_quat_wxyz=scene_quat_wxyz,
            scale_factor=scale_factor,
            add_meshes=_count_items(scene_cfg.mesh) == 0,
        )

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
                _find_colliding_obstacles(planner, active_js, viser_viz._server)
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


def _visualize_sdf(
    viser_server,
    planner: "MotionPlanner",
    n_points: int = 5000,
    name: str = "/sdf_debug",
) -> None:
    """Sample random points around the robot and visualize in-collision vs free-space.

    Red dots: inside an obstacle (collision cost > 0).
    Blue dots: free space (collision cost == 0).

    Samples uniformly in the robot's reachable workspace box:
      x ∈ [-0.9, 0.9], y ∈ [-0.9, 0.9], z ∈ [-0.1, 1.3]
    """
    scene_collision = planner.scene_collision_checker
    if scene_collision is None:
        print("[SDF vis] no scene_collision_checker available, skipping")
        return

    device = planner.device_cfg.device
    dtype = torch.float32

    # Sample random points in the workspace bounding box
    lo = torch.tensor([-0.9, -0.9, -0.1], dtype=dtype, device=device)
    hi = torch.tensor([ 0.9,  0.9,  1.3], dtype=dtype, device=device)
    pts = torch.rand(n_points, 3, dtype=dtype, device=device) * (hi - lo) + lo

    # Build query_spheres: [1, 1, N, 4] with tiny radius (≈ point query)
    radius = 0.005
    radii = torch.full((n_points, 1), radius, dtype=dtype, device=device)
    spheres = torch.cat([pts, radii], dim=-1).unsqueeze(0).unsqueeze(0)  # [1, 1, N, 4]

    col_buf = CollisionBuffer.from_shape(spheres.shape, planner.device_cfg)
    weight = torch.tensor([1.0], dtype=dtype, device=device)
    activation_distance = torch.tensor([0.0], dtype=dtype, device=device)

    with torch.no_grad():
        cost = scene_collision.get_sphere_distance_raw(
            query_spheres=spheres,
            collision_buffer=col_buf,
            weight=weight,
            activation_distance=activation_distance,
        )  # [1, 1, N]

    in_collision = (cost[0, 0] > 0.0).cpu().numpy()
    pts_np = pts.cpu().numpy()

    colors = np.where(
        in_collision[:, None],
        np.array([[255, 30, 30]], dtype=np.uint8),   # red = in collision
        np.array([[30, 100, 255]], dtype=np.uint8),  # blue = free
    )

    n_col = int(in_collision.sum())
    print(f"[SDF vis] {n_col}/{n_points} points in collision")

    handle = viser_server.scene.add_point_cloud(
        name=name,
        points=pts_np,
        colors=colors,
        point_size=0.008,
        visible=True,
    )
    return handle


def interactive_motion_planning(
    planner: MotionPlanner,
    scene_cfg: SceneCfg,
    use_cuda_graph: bool = True,
    port: int = 8080,
    predictions: List[SuperDecPrediction] | None = None,
    xyz: np.ndarray | None = None,
    rgb: np.ndarray | None = None,
    scene_translation: Sequence[float] | None = None,
    scene_quat_wxyz: Sequence[float] | None = None,
    scale_factor: float = 1.0,
    show_pointcloud: bool = False,
    fix_objects: bool = False,
    visualize_sdf: bool = False,
    sdf_n_points: int = 5000,
    world_representation: str = "superquadrics",
    scene_paths: List[str] | None = None,
    checkpoint_folder: str | None = None,
    mesh_resolution: int = 48,
) -> None:
    """Launch the same Viser-based interaction model as the standard tutorial."""
    viser_viz = ViserVisualizer(
        content_path=ContentPath(robot_config_file="franka.yml"),
        connect_ip="0.0.0.0",
        connect_port=port,
        add_control_frames=True,
        visualize_robot_spheres=False,
    )
    # viser_viz._server.scene.set_up_direction([0.0, 0.0, 1.0])

    if predictions is not None and scene_translation is not None and scene_quat_wxyz is not None:
        _display_cfg = _superdec_display_scene_cfg(
            predictions, scene_cfg.cuboid, scene_translation, scene_quat_wxyz, scale_factor
        )
    else:
        _display_cfg = scene_cfg
    obstacle_frames = viser_viz.add_scene(_display_cfg, add_control_frames=not fix_objects)
    overlay_handles: list = []
    if show_pointcloud and predictions is not None and xyz is not None and rgb is not None and scene_translation is not None and scene_quat_wxyz is not None:
        overlay_handles = _add_superdec_overlays(
            viser_viz,
            xyz=xyz,
            rgb=rgb,
            predictions=predictions,
            scene_translation=scene_translation,
            scene_quat_wxyz=scene_quat_wxyz,
            scale_factor=scale_factor,
            add_meshes=_count_items(scene_cfg.mesh) == 0,
        )
    scene_collision_checker = planner.scene_collision_checker
    assert scene_collision_checker is not None

    # Only track frames whose names exist in the collision checker — display-only
    # meshes (e.g. obj_0, obj_1) are in Viser but not in the planner's scene data.
    collision_obstacle_names = set(scene_collision_checker.get_obstacle_names())
    tracked_frames = {
        name: frame
        for name, frame in obstacle_frames.items()
        if name in collision_obstacle_names
    }
    old_obstacle_poses = {
        name: Pose.from_numpy(frame.position, frame.wxyz) for name, frame in tracked_frames.items()
    }

    current_state = planner.default_joint_state.clone().unsqueeze(0)
    _initial_js = planner.default_joint_state.clone()
    _initial_control_frames = {
        name: (tuple(frame.position), tuple(frame.wxyz))
        for name, frame in viser_viz._control_frames.items()
    }
    planner.warmup(enable_graph=True, num_warmup_iterations=5)

    sdf_handle_container: list = []
    if visualize_sdf:
        h = _visualize_sdf(viser_viz._server, planner, n_points=sdf_n_points)
        if h is not None:
            sdf_handle_container.append(h)

    sdf_toggle = viser_viz._server.gui.add_checkbox("Show SDF", initial_value=visualize_sdf)
    sdf_slider = viser_viz._server.gui.add_slider(
        "SDF points", min=100, max=50000, step=100, initial_value=sdf_n_points
    )

    def _refresh_sdf() -> None:
        if sdf_handle_container:
            try:
                sdf_handle_container[0].remove()
            except Exception:
                pass
            sdf_handle_container.clear()
        h = _visualize_sdf(viser_viz._server, planner, n_points=int(sdf_slider.value))
        if h is not None:
            sdf_handle_container.append(h)

    @sdf_toggle.on_update
    def _(_event) -> None:
        if sdf_toggle.value:
            _refresh_sdf()
        else:
            if sdf_handle_container:
                try:
                    sdf_handle_container[0].remove()
                except Exception:
                    pass
                sdf_handle_container.clear()

    @sdf_slider.on_update
    def _(_event) -> None:
        if sdf_toggle.value:
            _refresh_sdf()

    is_moving = False

    # ---- World-representation toggle ----------------------------------------
    repr_dropdown = viser_viz._server.gui.add_dropdown(
        "Representation",
        options=["superquadrics", "mesh"],
        initial_value=world_representation,
    )

    def _switch_representation(new_rep: str) -> None:
        nonlocal planner, scene_collision_checker, tracked_frames, old_obstacle_poses, is_moving
        if predictions is None or scene_translation is None or scene_quat_wxyz is None:
            print("[repr toggle] missing scene params, cannot switch")
            return
        is_moving = True
        print(f"[repr toggle] rebuilding planner for {new_rep} …")
        try:
            new_scene_cfg = _prediction_to_scene_cfg(
                predictions=predictions,
                world_representation=new_rep,
                scale_factor=scale_factor,
                scene_translation=scene_translation,
                scene_quat_wxyz=scene_quat_wxyz,
                points_tensor=torch.zeros(1),
            )
            use_cg = new_rep != "superquadrics"
            new_pcfg = MotionPlannerCfg.create(
                robot="franka.yml",
                scene_model="collision_test.yml",
                device_cfg=planner.device_cfg,
                use_cuda_graph=use_cg,
                max_goalset=10,
            )
            new_pcfg.scene_collision_cfg = SceneCollisionCfg(
                device_cfg=planner.device_cfg,
                scene_model=new_scene_cfg,
                cache={
                    "cuboid": _count_items(new_scene_cfg.cuboid),
                    "mesh": _count_items(new_scene_cfg.mesh),
                    "superquadric": _count_items(new_scene_cfg.superquadric),
                },
            )
            new_planner = MotionPlanner(new_pcfg)
            new_planner.warmup(enable_graph=use_cg, num_warmup_iterations=5)
            planner = new_planner
            scene_collision_checker = planner.scene_collision_checker
            new_collision_names = set(scene_collision_checker.get_obstacle_names())
            tracked_frames = {
                name: frame
                for name, frame in obstacle_frames.items()
                if name in new_collision_names
            }
            old_obstacle_poses = {
                name: Pose.from_numpy(frame.position, frame.wxyz)
                for name, frame in tracked_frames.items()
            }
            if sdf_handle_container and sdf_toggle.value:
                _refresh_sdf()
            print(f"[repr toggle] done → {new_rep}")
        except Exception as exc:
            print(f"[repr toggle] error: {exc}")
            import traceback; traceback.print_exc()
        finally:
            is_moving = False

    @repr_dropdown.on_update
    def _(_event) -> None:
        if is_moving:
            print("[repr toggle] busy, ignoring")
            return
        threading.Thread(
            target=_switch_representation, args=(repr_dropdown.value,), daemon=True
        ).start()

    # ---- Scene selector dropdown ---------------------------------------------
    if scene_paths and len(scene_paths) > 1 and checkpoint_folder is not None:
        scene_stems = [Path(p).stem for p in scene_paths]
        _current_scene_idx = [0]

        scene_dropdown = viser_viz._server.gui.add_dropdown(
            "Scene",
            options=scene_stems,
            initial_value=Path(scene_paths[0]).stem,
        )

        def _switch_scene(new_stem: str) -> None:
            nonlocal planner, scene_collision_checker, tracked_frames, old_obstacle_poses
            nonlocal is_moving, predictions, xyz, rgb, overlay_handles
            idx = scene_stems.index(new_stem)
            if idx == _current_scene_idx[0]:
                return
            is_moving = True
            print(f"[scene] Loading '{new_stem}' — running SuperDec inference …")
            try:
                new_predictions, new_xyz, new_rgb = _load_superdec_outputs(
                    scene_paths[idx], checkpoint_folder, mesh_resolution
                )
                cur_rep = repr_dropdown.value
                new_scene_cfg = _prediction_to_scene_cfg(
                    predictions=new_predictions,
                    world_representation=cur_rep,
                    scale_factor=scale_factor,
                    scene_translation=scene_translation,
                    scene_quat_wxyz=scene_quat_wxyz,
                    points_tensor=torch.zeros(1),
                )
                # Swap Viser overlays
                for h in overlay_handles:
                    try:
                        h.remove()
                    except Exception:
                        pass
                overlay_handles.clear()
                if show_pointcloud and scene_translation is not None and scene_quat_wxyz is not None:
                    overlay_handles.extend(_add_superdec_overlays(
                        viser_viz,
                        xyz=new_xyz,
                        rgb=new_rgb,
                        predictions=new_predictions,
                        scene_translation=scene_translation,
                        scene_quat_wxyz=scene_quat_wxyz,
                        scale_factor=scale_factor,
                        add_meshes=_count_items(new_scene_cfg.mesh) == 0,
                    ))
                # Rebuild planner
                use_cg = cur_rep != "superquadrics"
                new_pcfg = MotionPlannerCfg.create(
                    robot="franka.yml",
                    scene_model="collision_test.yml",
                    device_cfg=planner.device_cfg,
                    use_cuda_graph=use_cg,
                    max_goalset=10,
                )
                new_pcfg.scene_collision_cfg = SceneCollisionCfg(
                    device_cfg=planner.device_cfg,
                    scene_model=new_scene_cfg,
                    cache={
                        "cuboid": _count_items(new_scene_cfg.cuboid),
                        "mesh": _count_items(new_scene_cfg.mesh),
                        "superquadric": _count_items(new_scene_cfg.superquadric),
                    },
                )
                new_planner = MotionPlanner(new_pcfg)
                new_planner.warmup(enable_graph=use_cg, num_warmup_iterations=5)
                planner = new_planner
                scene_collision_checker = planner.scene_collision_checker
                new_collision_names = set(scene_collision_checker.get_obstacle_names())
                tracked_frames = {
                    name: frame
                    for name, frame in obstacle_frames.items()
                    if name in new_collision_names
                }
                old_obstacle_poses = {
                    name: Pose.from_numpy(frame.position, frame.wxyz)
                    for name, frame in tracked_frames.items()
                }
                predictions = new_predictions
                xyz = new_xyz
                rgb = new_rgb
                _current_scene_idx[0] = idx
                if sdf_handle_container and sdf_toggle.value:
                    _refresh_sdf()
                n_sq = _count_items(new_scene_cfg.superquadric)
                print(f"[scene] Loaded '{new_stem}' ({n_sq} SQs)")
            except Exception as exc:
                print(f"[scene] error switching to '{new_stem}': {exc}")
                import traceback; traceback.print_exc()
            finally:
                is_moving = False

        @scene_dropdown.on_update
        def _(_event) -> None:
            if is_moving:
                print("[scene] busy, ignoring")
                return
            threading.Thread(
                target=_switch_scene, args=(scene_dropdown.value,), daemon=True
            ).start()

    def update_obstacles() -> None:
        for name, frame in tracked_frames.items():
            new_pose = Pose.from_numpy(frame.position, frame.wxyz)
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
                _find_colliding_obstacles(planner, active_js, viser_viz._server)
                print("  status:", getattr(result, "status", None))
                print("  current_state:", current_state.position[0].tolist())
                print("  target_poses:", _pose_debug_list(target_poses))
                ik_result = planner.ik_solver.solve_pose(
                    GoalToolPose.from_poses(target_poses, num_goalset=1),
                    active_js,
                )
                print("  ik_success:", _tensor_debug_value(ik_result.success))
                print("  ik_feasible:", _tensor_debug_value(ik_result.feasible))
                print("  ik_position_error:", _tensor_debug_value(ik_result.position_error))
                print("  ik_rotation_error:", _tensor_debug_value(ik_result.rotation_error))
                print("  ik_goalset_index:", _tensor_debug_value(ik_result.goalset_index))
                print("  ik_debug_info:", getattr(ik_result, "debug_info", None))
                # Check whether the start state itself is in collision.
                if planner.graph_planner is not None:
                    start_q = active_js.position.view(1, -1)
                    start_feasible = planner.graph_planner.check_samples_feasibility(start_q)
                    print("  start_state_feasible:", start_feasible.tolist())
                    if ik_result.solution is not None:
                        goal_q = ik_result.solution[:1].view(1, -1)
                        goal_feasible = planner.graph_planner.check_samples_feasibility(goal_q)
                        print("  ik_goal_feasible:", goal_feasible.tolist())
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
                print("  status:", getattr(approach_result, "status", None))
                print("  current_state:", current_state.position[0].tolist())
                print("  target_poses:", _pose_debug_list(target_poses))
                print("  approach_poses:", _pose_debug_list(approach_poses))
                approach_ik_result = planner.ik_solver.solve_pose(
                    GoalToolPose.from_poses(approach_poses, num_goalset=1),
                    active_js,
                )
                print("  approach_ik_success:", _tensor_debug_value(approach_ik_result.success))
                print("  approach_ik_feasible:", _tensor_debug_value(approach_ik_result.feasible))
                print(
                    "  approach_ik_position_error:",
                    _tensor_debug_value(approach_ik_result.position_error),
                )
                print(
                    "  approach_ik_rotation_error:",
                    _tensor_debug_value(approach_ik_result.rotation_error),
                )
                print("  approach_ik_debug_info:", getattr(approach_ik_result, "debug_info", None))
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
                print("  status:", getattr(grasp_result, "status", None))
                print("  approach_end:", approach_end.position[0].tolist())
                print("  target_poses:", _pose_debug_list(target_poses))
                grasp_ik_result = planner.ik_solver.solve_pose(
                    GoalToolPose.from_poses(target_poses, num_goalset=1),
                    approach_end,
                )
                print("  grasp_ik_success:", _tensor_debug_value(grasp_ik_result.success))
                print("  grasp_ik_feasible:", _tensor_debug_value(grasp_ik_result.feasible))
                print(
                    "  grasp_ik_position_error:",
                    _tensor_debug_value(grasp_ik_result.position_error),
                )
                print(
                    "  grasp_ik_rotation_error:",
                    _tensor_debug_value(grasp_ik_result.rotation_error),
                )
                print("  grasp_ik_debug_info:", getattr(grasp_ik_result, "debug_info", None))
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
                print("  status:", getattr(lift_result, "status", None))
                print("  grasp_end:", grasp_end.position[0].tolist())
                print("  lift_poses:", _pose_debug_list(lift_poses))
            is_moving = False

        threading.Thread(target=plan_grasp_and_execute, daemon=True).start()

    def _reset(_) -> None:
        nonlocal current_state, is_moving
        is_moving = False
        current_state = _initial_js.clone().unsqueeze(0)
        viser_viz.set_joint_state(_initial_js)
        for name, (pos, wxyz) in _initial_control_frames.items():
            if name in viser_viz._control_frames:
                viser_viz._control_frames[name].position = pos
                viser_viz._control_frames[name].wxyz = wxyz
        try:
            viser_viz._server.scene.remove_by_name("/collision_spheres")
        except Exception:
            pass

    move_btn = viser_viz._server.gui.add_button("Move", color="green")
    move_btn.on_click(on_move)
    grasp_btn = viser_viz._server.gui.add_button("Grasp", color="blue")
    grasp_btn.on_click(on_grasp)
    reset_btn = viser_viz._server.gui.add_button("Reset", color="red")
    reset_btn.on_click(_reset)

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


def _add_superdec_overlays(
    viser_viz,
    xyz: np.ndarray,
    rgb: np.ndarray,
    predictions: List[SuperDecPrediction],
    scene_translation: Sequence[float],
    scene_quat_wxyz: Sequence[float],
    scale_factor: float,
    add_meshes: bool = True,
) -> list:
    """Add the raw scene point cloud and fitted SuperDec meshes to the Viser server."""
    server = viser_viz._server
    
    handles = []

    transformed_xyz = _transform_pointcloud(xyz, scene_translation, scene_quat_wxyz)

    pc_handle = server.scene.add_point_cloud(
        name="/scene",
        points=transformed_xyz,
        colors=rgb,
        point_size=0.003,
        visible=True,
    )
    handles.append(pc_handle)

    if not add_meshes:
        return handles

    scene_quat_xyzw = np.array(
        [scene_quat_wxyz[1], scene_quat_wxyz[2], scene_quat_wxyz[3], scene_quat_wxyz[0]],
        dtype=np.float32,
    )
    scene_rot = SciRotation.from_quat(scene_quat_xyzw)
    for pred in predictions:
        if pred.mesh is None:
            continue
        transformed_mesh = _transform_trimesh_mesh(
            pred.mesh,
            scene_rot.as_matrix(),
            scene_translation,
            scale_factor=scale_factor,
        )
        handle = server.scene.add_mesh_trimesh(f"/fit/obj_{pred.iid}", mesh=transformed_mesh, visible=True)
        handles.append(handle)

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
                _find_colliding_obstacles(planner, active_js, viser_viz._server)
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
    predictions: List[SuperDecPrediction],
    xyz: np.ndarray,
    rgb: np.ndarray,
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
    pointcloud_handle = None
    if visualize:
        viser_viz = ViserVisualizer(
            content_path=ContentPath(robot_config_file="franka.yml"),
            connect_ip="0.0.0.0",
            connect_port=port,
            add_control_frames=True,
            visualize_robot_spheres=False,
        )
        viser_viz._server.scene.set_up_direction([0.0, 0.0, 1.0])
        pointcloud_handle = viser_viz._server.scene.add_point_cloud(
            name="/scene",
            points=xyz,
            colors=rgb,
            point_size=0.003,
            visible=True,
        )
        print(f"\nOpen http://localhost:{port} to watch the scaling test.")
        print("Waiting 3 s for browser to connect before starting...")
        time.sleep(3.0)

    for n_sofas in sizes:
        print(f"\n{'='*55}")
        print(f"Scaling test  |  representation={args.world_representation}  |  n_sofas={n_sofas}")
        print(f"{'='*55}")

        scene_cfg = _prediction_to_scene_cfg(
            predictions=predictions,
            world_representation=args.world_representation,
            scale_factor=args.scale_factor,
            scene_translation=args.scene_translation,
            scene_quat_wxyz=args.scene_quat_wxyz,
            points_tensor=torch.from_numpy(xyz).to(device_cfg.device),
        )

        if n_sofas > 0:
            inst_sqs, inst_meshes = _make_pointcloud_instances(
                n_sofas,
                predictions,
                args.world_representation,
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
            for h in scene_handles:
                try:
                    h.remove()
                except Exception:
                    pass
            display_scene_cfg = _superdec_display_scene_cfg(
                predictions, scene_cfg.cuboid, args.scene_translation, args.scene_quat_wxyz, args.scale_factor
            )
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


def main() -> None:
    print("Starting SQ motion planning example")
    args = _parse_args()
    wp.init()

    # ---- Collect scene paths for the dropdown --------------------------------
    import glob as _glob
    scene_paths: List[str] = []
    if args.scenes_dir:
        scene_paths = sorted(_glob.glob(os.path.join(args.scenes_dir, "*.npz")))
        if not scene_paths:
            print(f"WARNING: --scenes_dir {args.scenes_dir!r} has no .npz files")
        else:
            print(f"Found {len(scene_paths)} scenes in {args.scenes_dir}")
            default_ply = "/home/haroldas/3DV/superdec/examples/chair.ply"
            if args.ply_path == default_ply:
                args.ply_path = scene_paths[0]
                print(f"Using first scene as initial: {args.ply_path}")
    if not scene_paths:
        scene_paths = [args.ply_path]

    t_total_start = time.perf_counter()
    timing: dict = {
        "representation": args.world_representation,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "ply_path": args.ply_path,
    }

    device_cfg = DeviceCfg(device=args.device)

    t0 = time.perf_counter()
    predictions, xyz, rgb = _load_superdec_outputs(
        args.ply_path,
        args.checkpoint_folder,
        args.mesh_resolution,
    )
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
        run_scaling_test(args, predictions, xyz, rgb, scaling_targets, visualize=args.visualize)
        return

    t0 = time.perf_counter()
    scene_cfg = _prediction_to_scene_cfg(
        predictions=predictions,
        world_representation=args.world_representation,
        scale_factor=args.scale_factor,
        scene_translation=args.scene_translation,
        scene_quat_wxyz=args.scene_quat_wxyz,
        points_tensor=torch.from_numpy(xyz).to(device_cfg.device),
    )
    timing["scene_build_s"] = time.perf_counter() - t0
    print(f"Scene build: {timing['scene_build_s']:.3f}s")

    if args.sofas > 0:
        inst_sqs, inst_meshes = _make_pointcloud_instances(
            args.sofas,
            predictions,
            args.world_representation,
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

    use_cuda_graph = args.world_representation != "superquadrics"

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

    # Print the scene config (all of the obstacles)
    print("\nScene config:")
    if scene_cfg.cuboid is not None:
        print("Cuboids:")
        for cuboid in scene_cfg.cuboid:
            print(f"  {cuboid}")
    if scene_cfg.mesh is not None:
        print("Meshes:")
        for mesh in scene_cfg.mesh:
            print(f"  {mesh.scale}")
    if scene_cfg.superquadric is not None:
        print("Superquadrics:")
        for sq in scene_cfg.superquadric:
            print(f"  {sq}")

    planner = MotionPlanner(planner_cfg)
    timing["planner_init_s"] = time.perf_counter() - t0
    print(f"Planner init: {timing['planner_init_s']:.3f}s")

    t0 = time.perf_counter()
    planner.warmup(enable_graph=use_cuda_graph, num_warmup_iterations=5)
    timing["warmup_s"] = time.perf_counter() - t0
    print(f"Warmup: {timing['warmup_s']:.3f}s")

    # ---- Startup collision check: verify default state is collision-free ----
    if planner.graph_planner is not None:
        default_q = planner.default_joint_state.position.view(1, -1).to(device_cfg.device)
        start_ok = planner.graph_planner.check_samples_feasibility(default_q)
        if not bool(start_ok.all()):
            print("WARNING: default start state is in collision with the scene!")
            print("  Note: robot collision spheres (visualize_robot_spheres=False) are not shown,")
            print("  so collisions may be invisible — the robot's visual mesh can look clear while")
            print("  a collision sphere is inside a scene SQ.")
            print(f"  panda_link0 has collision spheres at world z=0.085 (radius 0.03 -> top z=0.115).")
            print(f"  Current scene_translation z={args.scene_translation[2]:.3f} puts scene floor at")
            print(f"  robot z≈{0.746 + args.scene_translation[2]:.3f}; scene objects must clear z=0.12.")
            print("  Fix: increase scene_translation z (less negative), e.g. --scene_translation -0.1 -0.5 -0.55")
        else:
            print("Default start state is collision-free.")

    # ---- Auto-target sequencing -----------------------------------------
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
                predictions=predictions,
                xyz=xyz,
                rgb=rgb,
                scene_translation=args.scene_translation,
                scene_quat_wxyz=args.scene_quat_wxyz,
                scale_factor=args.scale_factor,
                show_pointcloud=args.show_pointcloud,
                fix_objects=args.fix_objects,
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
            predictions=predictions,
            xyz=xyz,
            rgb=rgb,
            scene_translation=args.scene_translation,
            scene_quat_wxyz=args.scene_quat_wxyz,
            scale_factor=args.scale_factor,
            show_pointcloud=args.show_pointcloud,
            fix_objects=args.fix_objects,
            visualize_sdf=args.visualize_sdf,
            sdf_n_points=args.sdf_n_points,
            world_representation=args.world_representation,
            scene_paths=scene_paths,
            checkpoint_folder=args.checkpoint_folder,
            mesh_resolution=args.mesh_resolution,
        )


if __name__ == "__main__":
    main()
