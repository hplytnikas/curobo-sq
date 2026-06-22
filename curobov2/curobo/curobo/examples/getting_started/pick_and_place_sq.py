"""Pick-and-place simulation using the SuperDec chair scene.

Demonstrates that a Superquadric obstacle can be attached to the robot's EE after
picking, so subsequent motion planning automatically avoids collisions caused by the
carried object.  No visual gripper closing is needed — the key feature is collision-
aware transport.

Phases:
  1. plan_pose  → move EE above the pick cube (cube removed from world)
  2. attach     → sphere-fit the SQ directly; cube disabled in world collision
  3. plan_pose  → transport to above the place position (SQ collision active)
  4. plan_pose  → lower to place position (SQ collision active)
  5. plan_pose  → retreat back above place position (SQ still attached)
  6. detach     → cube re-enabled; update world with cube at place position
  7. plan_pose  → return home
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import time
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np
import torch
from omegaconf import OmegaConf
from scipy.spatial.transform import Rotation as SciRotation
import trimesh

from curobo._src.geom.collision.collision_scene import SceneCollisionCfg
from curobo._src.geom.sphere_fit.types import SphereFitType
from curobo._src.geom.types import Cuboid, Mesh, SceneCfg, Superquadric
from curobo._src.motion.motion_planner_cfg import MotionPlannerCfg
from curobo._src.types.device_cfg import DeviceCfg
from curobo._src.types.pose import Pose
from curobo.motion_planner import MotionPlanner
from curobo.types import ContentPath, GoalToolPose, JointState

wp = None

try:
    import warp as wp
    _WARP_AVAILABLE = True
except ImportError:
    _WARP_AVAILABLE = False

WORKSPACE_ROOT = Path(__file__).resolve().parents[5]
SUPERDEC_ROOT = WORKSPACE_ROOT / "superdec"
if str(SUPERDEC_ROOT) not in sys.path:
    sys.path.append(str(SUPERDEC_ROOT))

from superdec.superdec import SuperDec  # noqa: E402

# ---------------------------------------------------------------------------
# Scene constants
# ---------------------------------------------------------------------------

TABLE = Cuboid(
    name="table",
    pose=[0.0, 0.0, -0.05, 1.0, 0.0, 0.0, 0.0],
    dims=[1.4, 1.4, 0.05],
)

# SQ cube to pick: near-box exponents, 6 cm half-sides.
PICK_CUBE = Superquadric(
    name="pick_cube",
    pose=[0.4, 0.1, 0.5, 1.0, 0.0, 0.0, 0.0],
    radii=[0.06, 0.06, 0.06],
    shape=[0.1, 0.1],
)

# Place destination
PLACE_POSITION = [0.4, -0.25, 0.5]

# SuperDec scene placement
DEFAULT_SCENE_TRANSLATION = np.array([-0.29955, -0.68389, 0.13559], dtype=np.float32)
DEFAULT_SCENE_QUAT_WXYZ = np.array([0.70711, 0.70711, 0.0, 0.0], dtype=np.float32)

# Rotate the pick cube by default scene quat
def _rotate_pick_cube(pose, quat_wxyz):
    sq_xyzw = np.array(
        [quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]],
        dtype=np.float32,
    )
    rot = SciRotation.from_quat(sq_xyzw)
    rotated_pos = rot.apply(np.asarray(pose[:3], dtype=np.float32))
    return [float(rotated_pos[0]), float(rotated_pos[1]), float(rotated_pos[2]), *quat_wxyz]
# PICK_CUBE.pose = _rotate_pick_cube(PICK_CUBE.pose, DEFAULT_SCENE_QUAT_WXYZ)


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--ply_path", default="/home/haroldas/3DV/superdec/examples/chair.ply")
    p.add_argument("--checkpoint_folder", default="/home/haroldas/3DV/superdec/checkpoints/normalized")
    p.add_argument("--world_representation", choices=("superquadrics", "mesh"), default="superquadrics")
    p.add_argument("--device", default="cuda")
    p.add_argument("--visualize", action="store_true")
    p.add_argument("--attached_target", action="store_true", help="Drag the transport target interactively in Viser")
    return p.parse_args()


def _count_items(items) -> int:
    return len(items) if items is not None else 0


# ---------------------------------------------------------------------------
# SuperDec helpers (unchanged from motion_planning_sq.py)
# ---------------------------------------------------------------------------

def _normalize_points(points: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    translation = points.mean(axis=0)
    centered = points - translation
    scale = float(2.0 * np.max(np.abs(centered)))
    return centered / scale, translation, scale


def _denormalize_outdict(outdict, translation: np.ndarray, scale: float):
    scale_arr = np.asarray([[scale]], dtype=np.float32)
    translation_arr = translation.reshape(1, 1, 3)
    outdict["scale"] = outdict["scale"] * scale_arr[:, :, None]
    outdict["trans"] = outdict["trans"] * scale_arr[:, :, None] + translation_arr
    return outdict


def _load_superdec_outputs(ply_path: str, checkpoint_folder: str) -> dict:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    checkpoint = torch.load(
        os.path.join(checkpoint_folder, "ckpt.pt"), map_location=device, weights_only=False
    )
    configs = OmegaConf.load(os.path.join(checkpoint_folder, "config.yaml"))
    model = SuperDec(configs.superdec).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    pc = trimesh.load(ply_path, process=False)
    if isinstance(pc, trimesh.Scene):
        points_np = np.concatenate(
            [np.asarray(g.vertices) for g in pc.geometry.values() if hasattr(g, "vertices")], axis=0
        )
    else:
        points_np = np.asarray(pc.vertices)

    idx = np.random.choice(len(points_np), min(4096, len(points_np)), replace=len(points_np) < 4096)
    points, translation, scale = _normalize_points(points_np[idx])
    points_t = torch.from_numpy(points).unsqueeze(0).to(device).float()

    with torch.no_grad():
        outdict = model(points_t)
        for k, v in outdict.items():
            if isinstance(v, torch.Tensor):
                outdict[k] = v.cpu()
        outdict = _denormalize_outdict(outdict, np.asarray(translation, dtype=np.float32), scale)

    return outdict


def _rotation_matrix_to_wxyz(R: np.ndarray) -> List[float]:
    q = SciRotation.from_matrix(R).as_quat()
    return [float(q[3]), float(q[0]), float(q[1]), float(q[2])]


def _apply_scene_transform(
    translation: Sequence[float],
    rotation_matrix: np.ndarray,
    scene_translation: np.ndarray,
    scene_quat_wxyz: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    sq_xyzw = np.array(
        [scene_quat_wxyz[1], scene_quat_wxyz[2], scene_quat_wxyz[3], scene_quat_wxyz[0]],
        dtype=np.float32,
    )
    scene_rot = SciRotation.from_quat(sq_xyzw)
    new_t = scene_rot.apply(np.asarray(translation, dtype=np.float32)) + scene_translation
    new_R = (scene_rot * SciRotation.from_matrix(rotation_matrix)).as_matrix()
    return new_t, new_R


def _primitive_mesh(scale, exponents, R, t, resolution=28):
    def f(a, e): return np.sign(np.sin(a)) * np.abs(np.sin(a)) ** e
    def g(a, e): return np.sign(np.cos(a)) * np.abs(np.cos(a)) ** e

    u = np.tile(np.linspace(-np.pi, np.pi, resolution, endpoint=True), resolution)
    v = np.repeat(np.linspace(-np.pi / 2.0, np.pi / 2.0, resolution, endpoint=True), resolution)
    if np.linalg.det(R) < 0:
        u = u[::-1]

    x = scale[0] * g(v, exponents[0]) * g(u, exponents[1])
    y = scale[1] * g(v, exponents[0]) * f(u, exponents[1])
    z = scale[2] * f(v, exponents[0])
    x[:resolution] = x[-resolution:] = 0.0
    verts = (R @ np.stack([x, y, z], axis=1).T).T + np.asarray(t, dtype=np.float32)

    tris = []
    for i in range(resolution - 1):
        for j in range(resolution - 1):
            tris += [
                [i * resolution + j, i * resolution + j + 1, (i + 1) * resolution + j],
                [(i + 1) * resolution + j, i * resolution + j + 1, (i + 1) * resolution + j + 1],
            ]
    for i in range(resolution - 1):
        tris += [
            [i * resolution + resolution - 1, i * resolution, (i + 1) * resolution + resolution - 1],
            [(i + 1) * resolution + resolution - 1, i * resolution, (i + 1) * resolution],
        ]
    tris += [
        [(resolution - 1) * resolution + resolution - 1, (resolution - 1) * resolution, resolution - 1],
        [resolution - 1, (resolution - 1) * resolution, 0],
    ]
    return trimesh.Trimesh(vertices=verts, faces=np.array(tris))


def _build_scene(
    outdict: dict,
    world_representation: str,
    scene_translation: Sequence[float],
    scene_quat_wxyz: Sequence[float],
    extra_sqs: Optional[List[Superquadric]] = None,
) -> Tuple[SceneCfg, List[Superquadric]]:
    """Build SceneCfg from SuperDec predictions + optional extra SQs.

    Returns (scene_cfg, chair_sqs).
    """
    t_np = np.asarray(scene_translation, dtype=np.float32)
    q_np = np.asarray(scene_quat_wxyz, dtype=np.float32)
    chair_sqs: List[Superquadric] = []
    meshes: List[Mesh] = []

    for idx in range(int(outdict["scale"].shape[1])):
        if float(outdict["exist"][0, idx]) <= 0.5:
            continue
        scale = np.asarray(outdict["scale"][0, idx], dtype=np.float32)
        exps = np.asarray(outdict["shape"][0, idx], dtype=np.float32)
        R = np.asarray(outdict["rotate"][0, idx], dtype=np.float32)
        t = np.asarray(outdict["trans"][0, idx], dtype=np.float32)
        t_w, R_w = _apply_scene_transform(t.tolist(), R, t_np, q_np)
        pose = [float(t_w[0]), float(t_w[1]), float(t_w[2]), *_rotation_matrix_to_wxyz(R_w)]

        if world_representation == "superquadrics":
            chair_sqs.append(Superquadric(name=f"chair_sq_{idx}", pose=pose, radii=scale.tolist(), shape=exps.tolist()))
        else:
            mesh_tm = _primitive_mesh(scale.tolist(), exps.tolist(), R_w, t_w.tolist())
            meshes.append(Mesh(name=f"chair_mesh_{idx}", vertices=mesh_tm.vertices.tolist(),
                               faces=mesh_tm.faces.tolist(), pose=[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]))

    all_sqs = chair_sqs + (extra_sqs or [])
    if world_representation == "superquadrics":
        scene = SceneCfg(cuboid=[TABLE], superquadric=all_sqs)
    else:
        scene = SceneCfg(cuboid=[TABLE], mesh=meshes, superquadric=(extra_sqs or []) or None)

    return scene, chair_sqs


# ---------------------------------------------------------------------------
# Planning helpers
# ---------------------------------------------------------------------------

def _goal_pose(planner: MotionPlanner, xyz: Sequence[float]) -> GoalToolPose:
    n_links = len(planner.tool_frames)
    pos = torch.tensor(xyz, device=planner.device_cfg.device, dtype=torch.float32).reshape(1, 1, 1, 1, 3)
    pos = pos.repeat(1, 1, n_links, 1, 1)
    quat = torch.tensor([1.0, 0.0, 0.0, 0.0], device=planner.device_cfg.device, dtype=torch.float32).reshape(1, 1, 1, 1, 4)
    quat = quat.repeat(1, 1, n_links, 1, 1)
    return GoalToolPose(tool_frames=planner.tool_frames, position=pos, quaternion=quat)


def _last_js(result, planner: MotionPlanner) -> JointState:
    """Extract the last waypoint from a plan_pose TrajOptSolverResult as a 2-D JointState."""
    js = result.js_solution
    pos = js.position.reshape(-1, js.position.shape[-1])[-1:, :]
    state = JointState.from_position(pos, joint_names=js.joint_names)
    return planner.kinematics.get_active_js(state)


def _last_waypoint_js(trajectory: JointState, planner: MotionPlanner) -> JointState:
    """Extract the last waypoint from a JointState trajectory as an active-js state."""
    pos_tensor = torch.as_tensor(trajectory.position, device=planner.device_cfg.device, dtype=torch.float32)
    pos = pos_tensor.reshape(-1, pos_tensor.shape[-1])[-1:, :]
    state = JointState.from_position(pos, joint_names=trajectory.joint_names)
    return planner.kinematics.get_active_js(state)


def _build_grasp_pose_from_bbox(pick_bbox: Cuboid, device_cfg: DeviceCfg, tool_frames: List[str]) -> GoalToolPose:
    """Build a single world-aligned grasp target pose above the top face of an object's OBB."""
    assert pick_bbox.pose is not None
    assert pick_bbox.dims is not None

    bbox_center = np.asarray(pick_bbox.pose[:3], dtype=np.float32)
    half_dims = np.asarray(pick_bbox.dims, dtype=np.float32) / 2.0
    grasp_xyz = [float(bbox_center[0]), float(bbox_center[1]), float(bbox_center[2] + half_dims[2] + 0.15)]

    grasp_pose = Pose.from_list([*grasp_xyz, 1.0, 0.0, 0.0, 0.0], device_cfg)
    assert grasp_pose.position is not None
    assert grasp_pose.quaternion is not None

    position = grasp_pose.position.view(1, 1, 1, 1, 3)
    quaternion = grasp_pose.quaternion.view(1, 1, 1, 1, 4)
    return GoalToolPose(tool_frames=tool_frames, position=position, quaternion=quaternion)


def _pose_from_handle(handle) -> Pose:
    return Pose.from_numpy(np.asarray(handle.position), np.asarray(handle.wxyz))


def _animate(trajectory: JointState, viser_viz) -> None:
    traj = trajectory.squeeze(0)
    pos: torch.Tensor = traj.position  # type: ignore[assignment]
    for i in range(pos.shape[-2]):
        viser_viz.set_joint_state(
            JointState.from_position(pos[0, i, :].unsqueeze(0), joint_names=traj.joint_names).squeeze(0)
        )
        time.sleep(0.02)


# ---------------------------------------------------------------------------
# Main pick-and-place sequence
# ---------------------------------------------------------------------------

def run_pick_and_place(
    planner: MotionPlanner,
    scene_cfg: SceneCfg,
    chair_sqs: List[Superquadric],
    device_cfg: DeviceCfg,
    attached_target_handle=None,
    viser_viz=None,
) -> None:
    home_js = JointState.from_position(
        torch.as_tensor(planner.default_joint_state.position, device=device_cfg.device, dtype=torch.float32).unsqueeze(0),
        joint_names=planner.joint_names,
    )
    home_js = planner.kinematics.get_active_js(home_js)

    assert PICK_CUBE.pose is not None
    pick_xyz = list(PICK_CUBE.pose[:3])
    pick_bbox = PICK_CUBE.get_cuboid()
    assert pick_bbox.pose is not None
    pick_bbox_pose = Pose.from_list(list(pick_bbox.pose), device_cfg)
    grasp_pose = _build_grasp_pose_from_bbox(pick_bbox, device_cfg, planner.tool_frames)
    target_xyz = list(PLACE_POSITION)
    if attached_target_handle is not None:
        print("\n[interactive] Drag attached_target in the viewer, then press Enter to continue ...")
        input()
        target_pose = _pose_from_handle(attached_target_handle)
        assert target_pose.position is not None
        target_xyz = target_pose.position.squeeze().tolist()

    # ------------------------------------------------------------------
    # Phase 1: Native grasp planning to the cube
    #
    # The cube SQ is an active collision obstacle, which would cause the
    # arm's sphere model to fail IK for any configuration near it.
    # Remove it from the world scene first so the planner can reach the
    # grasp poses freely. Then use plan_grasp for the approach/grasp/lift
    # sequence.
    # ------------------------------------------------------------------
    print("\n[1/7] plan_grasp → pick cube (cube removed from world) ...")
    scene_no_cube = SceneCfg(cuboid=[TABLE], superquadric=chair_sqs if chair_sqs else None)
    planner.update_world(scene_no_cube)

    print(f"  Planning grasp to world-aligned pose above {pick_bbox.pose[:3]} ...")

    grasp_result = planner.plan_grasp(
        grasp_poses=grasp_pose,
        current_state=home_js,
        grasp_approach_offset=0.15,
        grasp_lift_offset=0.15,
        grasp_lift_in_tool_frame=True,
        plan_approach_to_grasp=True,
        plan_grasp_to_lift=True,
    )
    if grasp_result is None or grasp_result.success is None or not grasp_result.success.any():
        print(f"  FAIL — {getattr(grasp_result, 'status', 'no result')}")
        planner.update_world(scene_cfg)
        return
    print("  OK")
    for segment in (
        grasp_result.approach_interpolated_trajectory,
        grasp_result.grasp_interpolated_trajectory,
        grasp_result.lift_interpolated_trajectory,
    ):
        if viser_viz is not None and segment is not None:
            _animate(segment, viser_viz)

    lift_traj = grasp_result.lift_interpolated_trajectory
    if lift_traj is None:
        lift_traj = grasp_result.lift_trajectory
    if lift_traj is None:
        print("  FAIL — grasp result did not provide a lift trajectory")
        planner.update_world(scene_cfg)
        return
    lift_js = _last_waypoint_js(lift_traj, planner)

    # ------------------------------------------------------------------
    # Phase 2: Attach the SQ's OBB to the EE
    #
    # Use the cube's oriented bounding box as the attachment proxy and fit
    # spheres from that OBB before moving through the rest of the scene.
    # ------------------------------------------------------------------
    print("\n[2/7] Attaching pick_cube OBB to attached_object link ...")
    planner.attachment_manager.attach(
        joint_states=lift_js,
        obstacles=[pick_bbox],
        link_name="attached_object",
        num_spheres=None,
        sphere_fit_type=SphereFitType.VOXEL,
        world_objects_pose_offset=pick_bbox_pose,
    )
    print("  OK — OBB spheres now active on arm (collision volume includes cube bounds)")

    # ------------------------------------------------------------------
    # Phase 3: Transport above the target position (SQ collision active)
    # ------------------------------------------------------------------
    above_target = [target_xyz[0], target_xyz[1], target_xyz[2] + 0.18]
    print(f"\n[3/7] plan_pose → above target {above_target}  (with attached SQ) ...")
    result = planner.plan_pose(_goal_pose(planner, above_target), lift_js, max_attempts=5)
    if result is None or not result.success.any():
        print("  FAIL — transport failed, detaching and aborting")
        planner.attachment_manager.detach()
        planner.update_world(scene_cfg)
        return
    print("  OK")
    if viser_viz is not None and result.get_interpolated_plan() is not None:
        _animate(result.get_interpolated_plan(), viser_viz)
    transport_js = _last_js(result, planner)

    # ------------------------------------------------------------------
    # Phase 4: Lower to place position (SQ still attached)
    # ------------------------------------------------------------------
    print(f"\n[4/7] plan_pose → place position {target_xyz}  (with attached SQ) ...")
    result = planner.plan_pose(_goal_pose(planner, target_xyz), transport_js, max_attempts=5)
    if result is None or not result.success.any():
        print("  FAIL — place failed, detaching and aborting")
        planner.attachment_manager.detach()
        planner.update_world(scene_cfg)
        return
    print("  OK")
    if viser_viz is not None and result.get_interpolated_plan() is not None:
        _animate(result.get_interpolated_plan(), viser_viz)
    place_js = _last_js(result, planner)

    # ------------------------------------------------------------------
    # Phase 5: Retreat above the place position (SQ still attached)
    #
    # The EE is at PLACE_POSITION. Detaching here would immediately restore
    # the cube as a world obstacle at the same location, putting the arm in
    # collision and making home planning fail. Retreat first.
    # ------------------------------------------------------------------
    print(f"\n[5/7] plan_pose → retreat above target  (with attached SQ) ...")
    result = planner.plan_pose(_goal_pose(planner, above_target), place_js, max_attempts=5)
    if result is None or not result.success.any():
        print("  FAIL — retreat failed, detaching and aborting")
        planner.attachment_manager.detach()
        planner.update_world(scene_cfg)
        return
    print("  OK")
    if viser_viz is not None and result.get_interpolated_plan() is not None:
        _animate(result.get_interpolated_plan(), viser_viz)
    retreat_js = _last_js(result, planner)

    # ------------------------------------------------------------------
    # Phase 6: Detach; update world with cube at its new position
    # ------------------------------------------------------------------
    print("\n[6/7] Detaching SQ; updating world ...")
    planner.attachment_manager.detach()

    placed_cube = Superquadric(
        name="pick_cube",
        pose=[*target_xyz, 1.0, 0.0, 0.0, 0.0],
        radii=list(PICK_CUBE.radii),
        shape=list(PICK_CUBE.shape),
    )
    new_scene = SceneCfg(cuboid=[TABLE], superquadric=(chair_sqs or []) + [placed_cube])
    planner.update_world(new_scene)
    print("  OK — cube now at place position in world")

    # ------------------------------------------------------------------
    # Phase 7: Return home
    # ------------------------------------------------------------------
    print("\n[7/7] plan_pose → home ...")
    result = planner.plan_pose(_goal_pose(planner, [0.0, 0.0, 0.65]), retreat_js, max_attempts=5)
    if result is None or not result.success.any():
        print("  FAIL — home plan failed")
    else:
        print("  OK")
        if viser_viz is not None and result.get_interpolated_plan() is not None:
            _animate(result.get_interpolated_plan(), viser_viz)

    print("\nPick-and-place complete.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    args = _parse_args()
    if _WARP_AVAILABLE and wp is not None:
        wp.init()

    device_cfg = DeviceCfg(device=args.device)

    print("Loading SuperDec outputs ...")
    t0 = time.perf_counter()
    outdict = _load_superdec_outputs(args.ply_path, args.checkpoint_folder)
    print(f"  done in {time.perf_counter() - t0:.2f}s")

    scene_cfg, chair_sqs = _build_scene(
        outdict=outdict,
        world_representation=args.world_representation,
        scene_translation=DEFAULT_SCENE_TRANSLATION.tolist(),
        scene_quat_wxyz=DEFAULT_SCENE_QUAT_WXYZ.tolist(),
        extra_sqs=[PICK_CUBE],
    )
    print(
        f"Scene: {_count_items(scene_cfg.cuboid)} cuboid(s), "
        f"{_count_items(scene_cfg.superquadric)} SQ(s) (incl. pick cube), "
        f"{_count_items(scene_cfg.mesh)} mesh(es)"
    )

    print("Initializing planner ...")
    t0 = time.perf_counter()
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
        cache={
            "cuboid": _count_items(scene_cfg.cuboid),
            "mesh": _count_items(scene_cfg.mesh),
            "superquadric": _count_items(scene_cfg.superquadric),
        },
    )
    planner = MotionPlanner(planner_cfg)
    print(f"  done in {time.perf_counter() - t0:.2f}s")

    print("Warming up ...")
    t0 = time.perf_counter()
    planner.warmup(enable_graph=False, num_warmup_iterations=5)
    print(f"  done in {time.perf_counter() - t0:.2f}s")

    viser_viz = None
    attached_target_handle = None
    if args.visualize or args.attached_target:
        from curobo.viewer import ViserVisualizer
        viser_viz = ViserVisualizer(
            content_path=ContentPath(robot_config_file="franka.yml"),
            connect_ip="0.0.0.0",
            connect_port=8080,
            add_control_frames=True,
            visualize_robot_spheres=False,
        )
        viser_viz.add_scene(scene_cfg, add_control_frames=True)
        target_pose = Pose.from_list([*PLACE_POSITION, 1.0, 0.0, 0.0, 0.0], device_cfg)
        if args.attached_target:
            attached_target_handle = viser_viz.add_control_frame("attached_target", target_pose, scale=0.15)
            print("\nOpen http://localhost:8080, drag attached_target, then use the Move and Grasp buttons.")
        else:
            print("\nOpen http://localhost:8080 to watch.  Waiting 3 s for browser ...")
            time.sleep(3.0)

    if args.attached_target and viser_viz is not None:
        current_state = planner.kinematics.get_active_js(
            planner.default_joint_state.clone().unsqueeze(0)
        )
        scene_no_cube = SceneCfg(cuboid=[TABLE], superquadric=chair_sqs if chair_sqs else None)
        attached_state = {"value": False}

        def _target_xyz() -> List[float]:
            if attached_target_handle is None:
                return list(PLACE_POSITION)
            target_pose = _pose_from_handle(attached_target_handle)
            assert target_pose.position is not None
            return target_pose.position.squeeze().tolist()

        def _plan_to_target() -> None:
            nonlocal current_state
            result = planner.plan_pose(_goal_pose(planner, _target_xyz()), current_state, max_attempts=5)
            if result is None or not result.success.any():
                print("Move failed")
                return
            if result.get_interpolated_plan() is not None:
                _animate(result.get_interpolated_plan(), viser_viz)
            current_state = _last_js(result, planner)

        def _grasp_cube() -> None:
            nonlocal current_state
            if attached_state["value"]:
                print("Cube is already attached")
                return
            planner.update_world(scene_no_cube)
            grasp_result = planner.plan_grasp(
                grasp_poses=grasp_pose,
                current_state=current_state,
                grasp_approach_offset=0.15,
                grasp_lift_offset=0.15,
                grasp_lift_in_tool_frame=True,
                plan_approach_to_grasp=True,
                plan_grasp_to_lift=True,
            )
            if grasp_result is None or grasp_result.success is None or not grasp_result.success.any():
                print(f"Grasp failed: {getattr(grasp_result, 'status', 'no result')}")
                planner.update_world(scene_cfg)
                return
            for segment in (
                grasp_result.approach_interpolated_trajectory,
                grasp_result.grasp_interpolated_trajectory,
                grasp_result.lift_interpolated_trajectory,
            ):
                if segment is not None:
                    _animate(segment, viser_viz)
            lift_traj = grasp_result.lift_interpolated_trajectory or grasp_result.lift_trajectory
            if lift_traj is None:
                print("Grasp failed: missing lift trajectory")
                planner.update_world(scene_cfg)
                return
            lift_js = _last_waypoint_js(lift_traj, planner)
            planner.attachment_manager.attach(
                joint_states=lift_js,
                obstacles=[pick_bbox],
                link_name="attached_object",
                num_spheres=None,
                sphere_fit_type=SphereFitType.VOXEL,
                world_objects_pose_offset=pick_bbox_pose,
            )
            attached_state["value"] = True
            current_state = lift_js
            print("Grasp OK: SQ attached to the arm")

        server = viser_viz._server
        move_btn = server.gui.add_button("Move", color="green")
        move_btn.on_click(lambda _=None: _plan_to_target())
        grasp_btn = server.gui.add_button("Grasp", color="blue")
        grasp_btn.on_click(lambda _=None: _grasp_cube())

        print("Viewer open. Click Grasp to attach the SQ, then Move to carry it to the target.")
        print("Press Ctrl+C to exit.")
        try:
            while True:
                time.sleep(0.1)
        except KeyboardInterrupt:
            pass
        return

    run_pick_and_place(planner, scene_cfg, chair_sqs, device_cfg, attached_target_handle, viser_viz)

    if viser_viz is not None:
        print("\nViewer open — press Ctrl+C to exit.")
        try:
            while True:
                time.sleep(0.1)
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()
