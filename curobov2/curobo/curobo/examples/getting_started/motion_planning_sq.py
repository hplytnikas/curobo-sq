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

LOG_DIR = Path("/home/haroldas/3DV/logs/curobov2/timing")

TABLE = Cuboid(
    name="table",
    pose=[0.0, 0.0, -0.05, 1.0, 0.0, 0.0, 0.0],
    dims=[1.4, 1.4, 0.05],
)

DEFAULT_SCENE_TRANSLATION = np.array([-0.29955, -0.68389, 0.13559], dtype=np.float32)
DEFAULT_SCENE_QUAT_WXYZ = np.array([0.70711, 0.70711, 0.0, 0.0], dtype=np.float32)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ply_path", type=str, required=True, help="Input point cloud (.ply) for SuperDec")
    parser.add_argument(
        "--checkpoint_folder",
        type=str,
        default="/home/haroldas/3DV/superdec/checkpoints/normalized",
        help="SuperDec checkpoint folder containing ckpt.pt and config.yaml",
    )
    parser.add_argument(
        "--world_representation",
        type=str,
        choices=("superquadrics", "mesh"),
        default="superquadrics",
        help="Which scene representation to plan against",
    )
    parser.add_argument("--mesh_resolution", type=int, default=28, help="SuperDec mesh resolution")
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
            "Add N sofa obstacles (seat + backrest cuboid pairs) arranged in a grid "
            "around the robot arm, ~1 m apart, all on the ground plane."
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


def _make_sofa_obstacles(n_sofas: int, spacing: float = 1.5) -> List[Cuboid]:
    """Return 2*n_sofas Cuboids (seat + backrest) arranged around the robot.

    Each sofa faces toward the robot: the open seat side is toward the origin
    and the backrest is on the far side.  The long axis (0.9 m width) is
    tangential to the circle, keeping roughly 1 m clear between adjacent sofas.

    Dimensions (local frame, x = depth toward robot, y = width, z = height):
      seat      0.45 × 0.90 × 0.18 m
      backrest  0.12 × 0.90 × 0.50 m
    """
    if n_sofas <= 0:
        return []

    positions = _sofa_grid_positions(n_sofas, spacing=spacing)
    cuboids: List[Cuboid] = []

    for i, (cx, cy) in enumerate(positions):
        dist = math.sqrt(cx ** 2 + cy ** 2)
        # θ = angle from origin to sofa → local-x axis points away from robot
        theta = math.atan2(cy, cx) if dist > 1e-6 else 0.0
        qw = math.cos(theta / 2)
        qz = math.sin(theta / 2)
        quat = [qw, 0.0, 0.0, qz]

        # Seat — bottom flush with ground (z = 0)
        cuboids.append(Cuboid(
            name=f"sofa_{i}_seat",
            pose=[cx, cy, 0.09, *quat],
            dims=[0.45, 0.9, 0.18],
        ))

        # Backrest — back face aligned with seat back face (local-x = +0.225),
        # backrest center at local-x = 0.225 − 0.06 = 0.165 from seat centre.
        bx = cx + math.cos(theta) * 0.165
        by = cy + math.sin(theta) * 0.165
        cuboids.append(Cuboid(
            name=f"sofa_{i}_back",
            pose=[bx, by, 0.25, *quat],
            dims=[0.12, 0.9, 0.5],
        ))

    return cuboids


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


def _load_superdec_outputs(
    ply_path: str,
    checkpoint_folder: str,
) -> tuple[dict, torch.Tensor]:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    checkpoint = torch.load(os.path.join(checkpoint_folder, "ckpt.pt"), map_location=device, weights_only=False)
    configs = OmegaConf.load(os.path.join(checkpoint_folder, "config.yaml"))
    model = SuperDec(configs.superdec).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

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

    sample_size = min(4096, len(points_np))
    sample_idx = np.random.choice(len(points_np), sample_size, replace=len(points_np) < sample_size)
    points = points_np[sample_idx]
    points, translation, scale = _normalize_points(points)
    points_tensor = torch.from_numpy(points).unsqueeze(0).to(device).float()

    with torch.no_grad():
        outdict = model(points_tensor)
        for key, value in outdict.items():
            if isinstance(value, torch.Tensor):
                outdict[key] = value.cpu()
        outdict = _denormalize_outdict(outdict, np.asarray(translation, dtype=np.float32), scale, False)
        points_tensor = _denormalize_points(
            points_tensor.cpu(), np.asarray(translation, dtype=np.float32), scale, False
        )

    return outdict, points_tensor


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


def _prediction_to_scene_cfg(
    outdict: dict,
    world_representation: str,
    mesh_resolution: int,
    scale_factor: float,
    scene_translation: Sequence[float],
    scene_quat_wxyz: Sequence[float],
) -> SceneCfg:
    scene_translation_np = np.asarray(scene_translation, dtype=np.float32)
    scene_quat_np = np.asarray(scene_quat_wxyz, dtype=np.float32)

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

        superquadrics.append(
            Superquadric(
                name=f"chair_sq_{idx}",
                pose=pose,
                radii=scale.tolist(),
                shape=exponents.tolist(),
            )
        )

        mesh_tm = _primitive_mesh(
            scale.tolist(),
            exponents.tolist(),
            transformed_rotation,
            transformed_translation.tolist(),
            mesh_resolution,
        )
        meshes.append(
            Mesh(
                name=f"chair_mesh_{idx}",
                vertices=mesh_tm.vertices.tolist(),
                faces=mesh_tm.faces.tolist(),
                pose=[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
            )
        )

    if world_representation == "superquadrics":
        scene = SceneCfg(cuboid=[TABLE], superquadric=superquadrics)
    elif world_representation == "mesh":
        scene = SceneCfg(cuboid=[TABLE], mesh=meshes)
    else:
        raise ValueError(f"Unsupported world_representation: {world_representation}")

    print(
        f"Built SuperDec scene with {len(superquadrics)} superquadrics and {len(meshes)} mesh primitives "
        f"(mode={world_representation})"
    )
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


def interactive_motion_planning(planner: MotionPlanner, scene_cfg: SceneCfg, use_cuda_graph: bool = True, port: int = 8080) -> None:
    """Launch the same Viser-based interaction model as the standard tutorial."""
    viser_viz = ViserVisualizer(
        content_path=ContentPath(robot_config_file="franka.yml"),
        connect_ip="0.0.0.0",
        connect_port=port,
        add_control_frames=True,
        visualize_robot_spheres=False,
    )

    obstacle_frames = viser_viz.add_scene(scene_cfg, add_control_frames=True)
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


def main() -> None:
    args = _parse_args()
    wp.init()

    t_total_start = time.perf_counter()
    timing: dict = {
        "representation": args.world_representation,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "ply_path": args.ply_path,
    }

    device_cfg = DeviceCfg(device=args.device)

    t0 = time.perf_counter()
    outdict, _ = _load_superdec_outputs(args.ply_path, args.checkpoint_folder)
    timing["superdec_inference_s"] = time.perf_counter() - t0
    print(f"SuperDec inference: {timing['superdec_inference_s']:.3f}s")

    t0 = time.perf_counter()
    scene_cfg = _prediction_to_scene_cfg(
        outdict=outdict,
        world_representation=args.world_representation,
        mesh_resolution=args.mesh_resolution,
        scale_factor=args.scale_factor,
        scene_translation=args.scene_translation,
        scene_quat_wxyz=args.scene_quat_wxyz,
    )
    timing["scene_build_s"] = time.perf_counter() - t0
    print(f"Scene build: {timing['scene_build_s']:.3f}s")

    if args.sofas > 0:
        sofa_cuboids = _make_sofa_obstacles(args.sofas)
        scene_cfg = SceneCfg(
            cuboid=list(scene_cfg.cuboid or []) + sofa_cuboids,
            superquadric=scene_cfg.superquadric,
            mesh=scene_cfg.mesh,
        )
        print(f"Added {args.sofas} sofas ({len(sofa_cuboids)} cuboids) to the scene")

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
    planner = MotionPlanner(planner_cfg)
    timing["planner_init_s"] = time.perf_counter() - t0
    print(f"Planner init: {timing['planner_init_s']:.3f}s")

    if args.visualize:
        interactive_motion_planning(planner, scene_cfg, use_cuda_graph=use_cuda_graph)
        return

    t0 = time.perf_counter()
    planner.warmup(enable_graph=use_cuda_graph, num_warmup_iterations=5)
    timing["warmup_s"] = time.perf_counter() - t0
    print(f"Warmup: {timing['warmup_s']:.3f}s")

    # ---- Auto-target sequencing -----------------------------------------
    if args.auto_cube_targets is not None:
        try:
            targets = json.loads(args.auto_cube_targets)
        except json.JSONDecodeError as exc:
            raise ValueError(f"--auto_cube_targets must be valid JSON: {exc}") from exc
        if not isinstance(targets, list) or not all(len(t) == 3 for t in targets):
            raise ValueError("--auto_cube_targets must be a list of [x, y, z] triples")

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
    else:
        # Fall back to the fixed examples when no targets given
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


if __name__ == "__main__":
    main()
