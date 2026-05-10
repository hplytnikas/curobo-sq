"""Minimal superquadric obstacle motion planning demo with Isaac Sim GUI.

One Franka arm reaches a red cube while avoiding a single superquadric obstacle.
Switch --world_representation between 'superquadrics' and 'mesh' to compare
native SQ collision vs. mesh approximation of the same geometry.

Timing is printed to stdout and saved to logs/timing/ as a CSV.

Uses old curobo (0.7.x) via ~/isaacsim/python.sh — NOT curobov2.
For the curobov2 headless benchmark see:
    curobov2/curobo/curobo/examples/getting_started/motion_gen_sq_simple.py

Run (GUI window, SQ collision):
    PATH=/usr/local/cuda-12.8/bin:/usr/bin:$PATH \\
    ~/isaacsim/python.sh \\
        curobo/examples/isaac_sim/motion_gen_sq_simple.py

Run (GUI window, mesh representation for comparison):
    PATH=/usr/local/cuda-12.8/bin:/usr/bin:$PATH \\
    ~/isaacsim/python.sh \\
        curobo/examples/isaac_sim/motion_gen_sq_simple.py --world_representation mesh

Run (headless, stop after 2000 sim steps):
    PATH=/usr/local/cuda-12.8/bin:/usr/bin:$PATH \\
    ~/isaacsim/python.sh \\
        curobo/examples/isaac_sim/motion_gen_sq_simple.py --headless --max_frames 2000
"""

try:
    import isaacsim  # noqa: F401
except ImportError:
    pass

import torch
_cuda_warmup = torch.zeros(4, device="cuda:0")

import argparse
import csv
import json
import queue
import threading
import time
import traceback
from datetime import datetime
from pathlib import Path

import numpy as np

# ── CLI ──────────────────────────────────────────────────────────────────────

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--headless_mode", type=str, default=None,
                    help="Headless mode: 'native' or 'websocket'.")
parser.add_argument("--headless", action="store_true", default=False,
                    help="Shorthand for --headless_mode native.")
parser.add_argument("--world_representation", choices=["superquadrics", "mesh"],
                    default="superquadrics",
                    help="Collision world representation.")
parser.add_argument("--max_frames", type=int, default=0,
                    help="Exit after N simulation steps (0 = run forever).")
parser.add_argument("--auto_cube_targets", type=str, default=None,
                    help="JSON array of [x,y,z] cube positions to cycle (e.g. '[[0.5,0,0.5]]').")
parser.add_argument("--auto_target_interval", type=int, default=300,
                    help="Simulation steps between automatic target changes.")
parser.add_argument("--plan_timeout", type=float, default=5.0,
                    help="Per-plan timeout in seconds.")
args = parser.parse_args()

if args.headless:
    args.headless_mode = "native"

# ── Isaac Sim startup ─────────────────────────────────────────────────────────

from omni.isaac.kit import SimulationApp

simulation_app = SimulationApp({
    "headless": args.headless_mode is not None,
    "width": "1920",
    "height": "1080",
})

import carb
from helper import add_extensions, add_robot_to_scene
from omni.isaac.core import World
from omni.isaac.core.objects import cuboid
from omni.isaac.core.utils.types import ArticulationAction

from curobo.geom.sdf.world import CollisionCheckerType
from curobo.geom.types import Mesh, Superquadric, WorldConfig
from curobo.types.base import TensorDeviceType
from curobo.types.math import Pose
from curobo.types.state import JointState
from curobo.util.logger import setup_curobo_logger
from curobo.util.usd_helper import UsdHelper
from curobo.util_file import get_robot_configs_path, get_world_configs_path, join_path, load_yaml
from curobo.wrap.reacher.motion_gen import (
    MotionGen,
    MotionGenConfig,
    MotionGenPlanConfig,
)

# ── Hardcoded obstacle ────────────────────────────────────────────────────────
# A rounded-box superquadric placed in the robot's reachable workspace.
# eps=[0.4, 0.4] → fairly box-like; eps=[1.0, 1.0] → ellipsoid.

OBSTACLE = Superquadric(
    name="sq_obstacle",
    pose=[0.45, 0.0, 0.35, 1.0, 0.0, 0.0, 0.0],  # [x, y, z, qw, qx, qy, qz]
    radii=[0.08, 0.12, 0.10],                       # semi-axes in metres
    eps=[0.4, 0.4],                                  # shape exponents
    color=[0.2, 0.6, 0.9, 1.0],
)

# ── Mesh approximation (used when --world_representation mesh) ────────────────

def _sq_mesh(sq: Superquadric, resolution: int = 40) -> Mesh:
    """Parametric superquadric surface → triangle mesh."""
    resolution = max(resolution, 8)
    eta = np.linspace(-np.pi / 2.0, np.pi / 2.0, resolution, endpoint=True)
    omega = np.linspace(-np.pi, np.pi, resolution, endpoint=True)
    eg, og = np.meshgrid(eta, omega, indexing="ij")

    def sp(v, p):
        return np.sign(v) * np.abs(v) ** p

    x = sq.radii[0] * sp(np.cos(eg), sq.eps[0]) * sp(np.cos(og), sq.eps[1])
    y = sq.radii[1] * sp(np.cos(eg), sq.eps[0]) * sp(np.sin(og), sq.eps[1])
    z = sq.radii[2] * sp(np.sin(eg), sq.eps[0])
    vertices = np.stack((x, y, z), axis=-1).reshape(-1, 3)

    faces = []
    for row in range(resolution - 1):
        for col in range(resolution - 1):
            i = row * resolution + col
            faces += [[i, i + 1, i + resolution], [i + resolution, i + 1, i + resolution + 1]]
        r, c = row * resolution + (resolution - 1), (row + 1) * resolution + (resolution - 1)
        faces += [[r, row * resolution, c], [c, row * resolution, c - (resolution - 1)]]

    color = list(sq.color) if len(sq.color) == 4 else list(sq.color) + [1.0]
    return Mesh(
        name=f"{sq.name}_mesh",
        vertices=vertices,
        faces=np.asarray(faces, dtype=np.int32),
        pose=sq.pose,
        color=color,
    )


# ── World construction ────────────────────────────────────────────────────────

def build_worlds() -> tuple[WorldConfig, WorldConfig]:
    """Return (collision_world, visual_world)."""
    table = WorldConfig.from_dict(
        load_yaml(join_path(get_world_configs_path(), "collision_table.yml"))
    )
    table.cuboid[0].pose[2] -= 0.02

    obstacle_mesh = _sq_mesh(OBSTACLE)

    if args.world_representation == "superquadrics":
        collision_world = WorldConfig(
            cuboid=table.cuboid,
            superquadric=[OBSTACLE],
        )
    else:
        collision_world = WorldConfig(
            cuboid=table.cuboid,
            mesh=[obstacle_mesh],
        )

    visual_world = WorldConfig(cuboid=[], mesh=[obstacle_mesh])
    return collision_world, visual_world


# ── Motion gen construction ───────────────────────────────────────────────────

def build_motion_gen(robot_cfg, collision_world: WorldConfig,
                     tensor_args: TensorDeviceType) -> MotionGen:
    has_sq = bool(collision_world.superquadric)
    checker_type = (
        CollisionCheckerType.PRIMITIVE if has_sq
        else CollisionCheckerType.MESH
    )
    collision_cache = {"obb": len(collision_world.cuboid)}
    if has_sq:
        collision_cache["superquadric"] = len(collision_world.superquadric)
    else:
        collision_cache["mesh"] = len(collision_world.mesh)

    # SQ kernel uses dynamic-shape ops incompatible with CUDA graph capture.
    use_cuda_graph = not has_sq

    cfg = MotionGenConfig.load_from_robot_config(
        robot_cfg,
        collision_world,
        tensor_args,
        collision_checker_type=checker_type,
        num_ik_seeds=16,
        num_trajopt_seeds=4,
        num_graph_seeds=4,
        collision_cache=collision_cache,
        use_cuda_graph=use_cuda_graph,
    )
    mg = MotionGen(cfg)
    print(f"[motion_gen] world_rep={args.world_representation} "
          f"checker={mg.world_coll_checker.__class__.__name__} "
          f"cuda_graph={use_cuda_graph}", flush=True)
    print("[motion_gen] warming up...", flush=True)
    mg.warmup(enable_graph=True, warmup_js_trajopt=False)
    print("[motion_gen] ready.", flush=True)
    return mg


# ── Timing log ────────────────────────────────────────────────────────────────

def _make_log() -> tuple[Path, str]:
    logs_dir = Path(__file__).resolve().parents[3] / "logs" / "timing"
    logs_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = logs_dir / f"sq_simple_{args.world_representation}_{stamp}.csv"
    with path.open("w", newline="") as f:
        csv.writer(f).writerow([
            "plan_id", "step", "success", "status",
            "wall_s", "total_s", "ik_s", "graph_s", "trajopt_s",
            "world_rep", "target_x", "target_y", "target_z",
        ])
    return path, stamp


def _log_row(path: Path, plan_id: int, step: int, success: bool,
             status: str, wall_s: float, result, target_pos) -> None:
    tx, ty, tz = (float(target_pos[i]) for i in range(3))

    def _f(attr):
        try:
            return float(getattr(result, attr, float("nan")))
        except Exception:
            return float("nan")

    with path.open("a", newline="") as f:
        csv.writer(f).writerow([
            plan_id, step, success, status, f"{wall_s:.4f}",
            f"{_f('total_time'):.4f}", f"{_f('ik_time'):.4f}",
            f"{_f('graph_time'):.4f}", f"{_f('trajopt_time'):.4f}",
            args.world_representation, f"{tx:.4f}", f"{ty:.4f}", f"{tz:.4f}",
        ])


# ── Background plan thread ────────────────────────────────────────────────────

def _plan_worker(mg: MotionGen, cu_js: JointState, ik_goal: Pose,
                 plan_cfg: MotionGenPlanConfig, has_sq: bool,
                 result_q: queue.Queue, default_stream: torch.cuda.Stream,
                 plan_id: int) -> None:
    stream = torch.cuda.Stream(device=mg.tensor_args.device)
    t0 = time.perf_counter()
    with torch.cuda.stream(stream):
        stream.wait_stream(default_stream)
        try:
            result = mg.plan_single(cu_js.unsqueeze(0), ik_goal, plan_cfg)
            # SQ: if start-state collision, retry with validity check off
            if has_sq and result.status.name == "INVALID_START_STATE_WORLD_COLLISION":
                cfg2 = MotionGenPlanConfig(
                    max_attempts=plan_cfg.max_attempts,
                    timeout=plan_cfg.timeout,
                    enable_graph=plan_cfg.enable_graph,
                    enable_finetune_trajopt=True,
                    check_start_validity=False,
                )
                result = mg.plan_single(cu_js.unsqueeze(0), ik_goal, cfg2)
            stream.synchronize()
            result_q.put(("ok", result, time.perf_counter() - t0))
        except Exception:
            stream.synchronize()
            result_q.put(("error", traceback.format_exc(), time.perf_counter() - t0))


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    if args.world_representation == "superquadrics":
        from curobo.curobolib.geom import geom_cu
        if not hasattr(geom_cu, "closest_point_superquadric"):
            raise RuntimeError(
                "CuRobo was not compiled with superquadric support. "
                "Use --world_representation mesh or rebuild."
            )

    my_world = World(stage_units_in_meters=1.0)
    stage = my_world.stage
    xform = stage.DefinePrim("/World", "Xform")
    stage.SetDefaultPrim(xform)
    stage.DefinePrim("/curobo", "Xform")

    # Red cube target
    target = cuboid.VisualCuboid(
        "/World/target",
        position=np.array([0.5, 0.0, 0.5]),
        orientation=np.array([0.0, 1.0, 0.0, 0.0]),
        color=np.array([1.0, 0.0, 0.0]),
        size=0.05,
    )

    setup_curobo_logger("warn")
    usd_helper = UsdHelper()
    tensor_args = TensorDeviceType()

    robot_cfg = load_yaml(join_path(get_robot_configs_path(), "franka.yml"))["robot_cfg"]
    joint_names = robot_cfg["kinematics"]["cspace"]["joint_names"]
    default_config = robot_cfg["kinematics"]["cspace"]["retract_config"]
    robot, _ = add_robot_to_scene(robot_cfg, my_world)
    articulation_controller = None

    collision_world, visual_world = build_worlds()
    motion_gen = build_motion_gen(robot_cfg, collision_world, tensor_args)
    has_sq = bool(collision_world.superquadric)

    log_path, _ = _make_log()
    print(f"[timing] log → {log_path}", flush=True)

    # Plan config: simple, no retry machinery
    plan_cfg = MotionGenPlanConfig(
        enable_graph=True,
        max_attempts=1,
        timeout=args.plan_timeout,
        enable_finetune_trajopt=True,
        time_dilation_factor=0.5,
        check_start_validity=not has_sq,
    )

    # Parse auto targets
    auto_positions = None
    if args.auto_cube_targets is not None:
        try:
            raw = json.loads(args.auto_cube_targets)
            auto_positions = [np.array(t, dtype=np.float64) for t in raw]
        except Exception as exc:
            print(f"[auto-targets] parse failed: {exc}", flush=True)
    else:
        # Default circuit around the obstacle
        auto_positions = [
            np.array([0.50,  0.00, 0.50]),
            np.array([0.40,  0.30, 0.55]),
            np.array([0.40, -0.30, 0.55]),
            np.array([0.55,  0.00, 0.65]),
        ]
    print(f"[auto-targets] {len(auto_positions)} positions, "
          f"interval={args.auto_target_interval} steps", flush=True)
    auto_idx = -1

    add_extensions(simulation_app, args.headless_mode)
    usd_helper.load_stage(my_world.stage)
    usd_helper.add_world_to_stage(visual_world, base_frame="/World")
    my_world.scene.add_default_ground_plane()

    if args.headless_mode is not None:
        my_world.play()

    # ── Sim-loop state ─────────────────────────────────────────────────────
    cmd_plan = None
    cmd_idx = 0
    idx_list = []
    past_pose = None
    past_orientation = None
    target_pose = None
    target_orientation = None
    stationary_steps = 0
    last_plan_step = -100_000
    plan_id = 0
    plan_thread: threading.Thread = None
    result_q: queue.Queue = queue.Queue(maxsize=1)
    active_id = -1
    active_start = 0.0
    active_goal_pos = None
    active_goal_ori = None
    idle_steps = 0

    while simulation_app.is_running():
        my_world.step(render=True)
        if not my_world.is_playing():
            if idle_steps % 100 == 0:
                print("**** Click Play to start simulation ****")
            idle_steps += 1
            continue

        step = my_world.current_time_step_index

        if args.max_frames > 0 and step >= args.max_frames:
            print(f"[headless] max_frames={args.max_frames} reached; exiting.", flush=True)
            break

        if articulation_controller is None:
            articulation_controller = robot.get_articulation_controller()
        if step < 10:
            robot._articulation_view.initialize()
            idx_list = [robot.get_dof_index(n) for n in joint_names]
            robot.set_joint_positions(default_config, idx_list)
            robot._articulation_view.set_max_efforts(
                values=np.array([5000] * len(idx_list)), joint_indices=idx_list
            )
        if step < 20:
            continue

        # ── Auto-target cycling ────────────────────────────────────────────
        interval = max(args.auto_target_interval, 1)
        new_idx = (step // interval) % len(auto_positions)
        if new_idx != auto_idx:
            auto_idx = new_idx
            target.set_world_pose(position=auto_positions[auto_idx])
            print(f"[auto-targets] step={step} → #{auto_idx} "
                  f"{np.array2string(auto_positions[auto_idx], precision=3)}", flush=True)

        # ── Read robot & target state ──────────────────────────────────────
        cube_pos, cube_ori = target.get_world_pose()
        if not (np.all(np.isfinite(cube_pos)) and np.all(np.isfinite(cube_ori))):
            continue
        cube_ori = cube_ori / max(np.linalg.norm(cube_ori), 1e-8)

        if past_pose is None:  past_pose = cube_pos.copy()
        if target_pose is None: target_pose = cube_pos.copy()
        if target_orientation is None: target_orientation = cube_ori.copy()
        if past_orientation is None: past_orientation = cube_ori.copy()

        sim_js = robot.get_joints_state()
        if sim_js is None:
            continue
        if not (np.all(np.isfinite(sim_js.positions)) and np.all(np.isfinite(sim_js.velocities))):
            continue

        cu_js = JointState(
            position=tensor_args.to_device(sim_js.positions),
            velocity=tensor_args.to_device(sim_js.velocities) * 0.0,
            acceleration=tensor_args.to_device(sim_js.velocities) * 0.0,
            jerk=tensor_args.to_device(sim_js.velocities) * 0.0,
            joint_names=robot.dof_names,
        )
        cu_js = cu_js.get_ordered_joint_state(motion_gen.kinematics.joint_names)

        # Track target motion for stationary detection
        target_changed = (
            np.linalg.norm(cube_pos - target_pose) > 1e-3
            or np.linalg.norm(cube_ori - target_orientation) > 1e-3
        )
        target_stationary = (
            np.linalg.norm(past_pose - cube_pos) < 1e-6
            and np.linalg.norm(past_orientation - cube_ori) < 1e-6
        )
        stationary_steps = (stationary_steps + 1) if target_stationary else 0
        past_pose = cube_pos.copy()
        past_orientation = cube_ori.copy()

        # ── Collect plan result ────────────────────────────────────────────
        if plan_thread is not None and not plan_thread.is_alive():
            plan_thread = None
            try:
                status, payload, elapsed = result_q.get_nowait()
            except queue.Empty:
                status, payload, elapsed = None, None, 0.0

            if status == "ok":
                result = payload
                success = bool(result.success.item())
                status_name = result.status.name if hasattr(result.status, "name") else str(result.status)
                wall_s = time.perf_counter() - active_start

                # Timing summary to stdout
                ik_s = float(getattr(result, "ik_time", float("nan")))
                graph_s = float(getattr(result, "graph_time", float("nan")))
                traj_s = float(getattr(result, "trajopt_time", float("nan")))
                total_s = float(getattr(result, "total_time", float("nan")))
                print(
                    f"[plan#{active_id}] {'OK' if success else 'FAIL'} "
                    f"status={status_name} world={args.world_representation} "
                    f"wall={wall_s:.3f}s  total={total_s:.3f}s  "
                    f"ik={ik_s:.3f}s  graph={graph_s:.3f}s  traj={traj_s:.3f}s",
                    flush=True,
                )
                _log_row(log_path, active_id, step, success, status_name,
                         wall_s, result, active_goal_pos)

                if success:
                    cmd_plan = motion_gen.get_full_js(result.get_interpolated_plan())
                    common = [n for n in robot.dof_names if n in cmd_plan.joint_names]
                    idx_list = [robot.get_dof_index(n) for n in common]
                    cmd_plan = cmd_plan.get_ordered_joint_state(common)
                    cmd_idx = 0
                    target_pose = cube_pos.copy()
                    target_orientation = cube_ori.copy()
                else:
                    carb.log_warn(f"plan#{active_id} failed: {status_name}")

            elif status == "error":
                print(f"[plan#{active_id}] EXCEPTION:\n{payload}", flush=True)

        # ── Trigger new plan ───────────────────────────────────────────────
        cooldown = 30
        need_plan = (
            plan_thread is None
            and stationary_steps >= 5
            and (target_changed or step - last_plan_step >= cooldown)
            and step - last_plan_step >= cooldown
        )
        if need_plan:
            ik_goal = Pose(
                position=tensor_args.to_device(cube_pos),
                quaternion=tensor_args.to_device(cube_ori),
            )
            default_stream = torch.cuda.current_stream()
            plan_thread = threading.Thread(
                target=_plan_worker,
                args=(motion_gen, cu_js.clone(), ik_goal, plan_cfg, has_sq,
                      result_q, default_stream, plan_id),
                daemon=True,
            )
            active_id = plan_id
            active_start = time.perf_counter()
            active_goal_pos = cube_pos.copy()
            active_goal_ori = cube_ori.copy()
            plan_id += 1
            last_plan_step = step
            plan_thread.start()

        # ── Execute trajectory ─────────────────────────────────────────────
        if cmd_plan is not None:
            if cmd_idx < len(cmd_plan.position):
                cmd_state = cmd_plan[cmd_idx]
                articulation_controller.apply_action(ArticulationAction(
                    joint_positions=cmd_state.position.cpu().numpy(),
                    joint_velocities=cmd_state.velocity.cpu().numpy(),
                    joint_indices=idx_list,
                ))
                cmd_idx += 1
            else:
                cmd_plan = None

    simulation_app.close()


if __name__ == "__main__":
    main()
