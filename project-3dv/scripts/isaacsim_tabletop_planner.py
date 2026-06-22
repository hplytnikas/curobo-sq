"""
isaacsim_tabletop_planner.py
============================
Full pipeline: tabletop perception → superquadric world model → CuRobo MotionGen in Isaac Sim.

Pipeline
--------
  PLY point cloud
    → table removal + DBSCAN segmentation
    → SuperdecFitter (v3 checkpoint, per-object)
    → CuRobo WorldConfig (SQ obstacles)
    → MotionGen (collision-free trajectory planning)
    → Isaac Sim simulation loop (Franka Panda)

Usage
-----
  # Interactive (drag the red cube to set targets):
  omni_python scripts/isaacsim_tabletop_planner.py \\
      --ply /work/courses/3dv/team15/project-3dv/data/rgbd-scenes-v2/pc/05.ply

  # Headless with auto-cycling targets:
  omni_python scripts/isaacsim_tabletop_planner.py \\
      --ply /work/courses/3dv/team15/project-3dv/data/rgbd-scenes-v2/pc/05.ply \\
      --headless \\
      --auto_targets "[[0.4,0.0,0.4],[0.5,0.2,0.3],[0.3,-0.2,0.5]]" \\
      --max_frames 600

  # Use LM fitter (no GPU needed for perception):
  omni_python scripts/isaacsim_tabletop_planner.py \\
      --ply /work/courses/3dv/team15/project-3dv/data/rgbd-scenes-v2/pc/05.ply \\
      --fitter lm

Run command (IMPORTANT — must use Isaac Sim's Python):
  PATH=/usr/local/cuda-12.8/bin:/usr/bin:$PATH ~/isaacsim/python.sh \\
      scripts/isaacsim_tabletop_planner.py [args]

  Or equivalently:
  omni_python scripts/isaacsim_tabletop_planner.py [args]
"""

# ── Isaac Sim warm-up: must happen before SimulationApp ──────────────────────
try:
    import isaacsim  # noqa: F401
except ImportError:
    pass

import torch
_cuda_warmup = torch.zeros(4, device="cuda:0")  # warm up CUDA context before Isaac Sim

# ── Standard library ─────────────────────────────────────────────────────────
import argparse
import copy
import json
import sys
import time
from pathlib import Path
from typing import List, Optional

import numpy as np

# ── Path injection ────────────────────────────────────────────────────────────
_REPO_ROOT  = Path(__file__).resolve().parent.parent
_PERCEPTION = _REPO_ROOT / "src" / "project_3dv" / "perception"
_SUPERDEC   = Path("/work/courses/3dv/team15/superdec")
_CUROBO_SRC = Path("/work/courses/3dv/team15/curobo-sq/curobo/src")

for _p in [str(_PERCEPTION), str(_SUPERDEC), str(_CUROBO_SRC)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

# ── Argument parsing (before SimulationApp) ───────────────────────────────────
def build_argparser() -> argparse.ArgumentParser:
    ckpt_v3 = Path(
        "/work/courses/3dv/team15/checkpoints/superdec_tabletop"
        "/superdec_tabletop_finetune_v3"
    )
    default_ply = Path(
        "/work/courses/3dv/team15/project-3dv/data/rgbd-scenes-v2/pc/05.ply"
    )
    p = argparse.ArgumentParser(
        description="Tabletop perception → SQ world → CuRobo MotionGen in Isaac Sim",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # ── Perception ────────────────────────────────────────────────────────────
    p.add_argument("--ply", default=str(default_ply),
                   help="Input PLY point cloud (tabletop scene).")
    p.add_argument("--fitter", choices=["superdec", "lm"], default="superdec",
                   help="Superquadric fitter to use.")
    p.add_argument("--checkpoint_folder", default=str(ckpt_v3),
                   help="SuperdecFitter checkpoint folder (contains epoch_*.pt + config.yaml).")
    p.add_argument("--collision_tolerance", type=float, default=0.01,
                   help="Shrink each SQ radius by this margin (m) to reduce over-conservative collision.")
    p.add_argument("--scene_translation", nargs=3, type=float, default=None,
                   metavar=("x", "y", "z"),
                   help="Override XYZ translation applied to the SQ world (world frame). "
                        "Auto-computed from fitted centroids when not set.")
    # ── Simulation ────────────────────────────────────────────────────────────
    p.add_argument("--robot", default="franka.yml",
                   help="CuRobo robot config YAML.")
    p.add_argument("--headless", action="store_true",
                   help="Run headless (no GUI).")
    p.add_argument("--max_frames", type=int, default=0,
                   help="Exit after N simulation steps (0 = run forever).")
    p.add_argument("--auto_targets", default=None,
                   help='JSON array of [[x,y,z], ...] target positions to cycle through.')
    p.add_argument("--auto_target_interval", type=int, default=200,
                   help="Steps between automatic target changes.")
    p.add_argument("--plan_timeout", type=float, default=5.0,
                   help="Per-plan timeout in seconds.")
    p.add_argument("--plan_stable_steps", type=int, default=8,
                   help="Stationary frames required before triggering a plan.")
    p.add_argument("--plan_cooldown_steps", type=int, default=45,
                   help="Minimum steps between planning attempts.")
    # ── Debug ─────────────────────────────────────────────────────────────────
    p.add_argument("--print_sq_stats", action="store_true",
                   help="Print SQ parameter statistics after perception.")
    return p


args = build_argparser().parse_args()

# ── Isaac Sim bootstrap ───────────────────────────────────────────────────────
from omni.isaac.kit import SimulationApp  # noqa: E402

simulation_app = SimulationApp({
    "headless": args.headless,
    "width": 1280,
    "height": 720,
})

# ── Isaac Sim / CuRobo imports (after SimulationApp) ─────────────────────────
import carb  # noqa: E402
from omni.isaac.core import World  # noqa: E402
from omni.isaac.core.objects import cuboid  # noqa: E402
from omni.isaac.core.utils.extensions import enable_extension  # noqa: E402
from omni.isaac.core.utils.nucleus import get_assets_root_path  # noqa: E402

from curobo.geom.sdf.world import CollisionCheckerType  # noqa: E402
from curobo.geom.types import Mesh, Superquadric, WorldConfig  # noqa: E402
from curobo.types.base import TensorDeviceType  # noqa: E402
from curobo.types.math import Pose  # noqa: E402
from curobo.types.robot import JointState  # noqa: E402
from curobo.util.logger import setup_curobo_logger  # noqa: E402
from curobo.util.usd_helper import UsdHelper  # noqa: E402
from curobo.util_file import (  # noqa: E402
    get_robot_configs_path,
    get_world_configs_path,
    join_path,
    load_yaml,
)
from curobo.wrap.reacher.motion_gen import (  # noqa: E402
    MotionGen,
    MotionGenConfig,
    MotionGenPlanConfig,
    MotionGenStatus,
)

# ── Isaac Sim helper (add_robot_to_scene) ─────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parents[1].parent /
                       "curobo-sq" / "curobo" / "examples" / "isaac_sim"))
from helper import add_extensions, add_robot_to_scene  # noqa: E402


# ──────────────────────────────────────────────────────────────────────────────
# Perception
# ──────────────────────────────────────────────────────────────────────────────

def run_perception(ply_path: str, fitter_type: str, checkpoint_folder: str):
    """Load PLY → table removal → segmentation → SQ fitting.

    Returns
    -------
    fits : list[MultiSQFit]
    """
    import open3d as o3d  # noqa: E402
    from pipeline import (  # noqa: E402
        remove_table,
        segment_instances,
        merge_nearby_segments,
    )

    print(f"\n[perception] Loading {ply_path} …")
    pcd = o3d.io.read_point_cloud(ply_path)
    pts = np.asarray(pcd.points, dtype=np.float64)
    if pts.size == 0:
        raise RuntimeError(f"No points in {ply_path}")
    print(f"[perception]   {len(pts):,} raw points")

    # Subsample for speed
    if len(pts) > 80_000:
        idx = np.random.choice(len(pts), 80_000, replace=False)
        pts = pts[idx]

    # Table removal
    try:
        obj_pts, *_ = remove_table(pts, max_height_above_table=0.4,
                                   depth_margin=np.inf, xy_radius=np.inf)
        print(f"[perception]   {len(obj_pts):,} foreground points after table removal")
    except Exception as exc:
        carb.log_warn(f"[perception] Table removal failed ({exc}), using full cloud.")
        obj_pts = pts

    # Segmentation
    segs = segment_instances(
        obj_pts, adaptive_eps=True, eps_multiplier=3.0,
        cluster_min_points=5, eps_max=0.08,
    )
    segs = merge_nearby_segments(segs, merge_dist=0.15)
    segs = [s for s in segs
            if len(s) >= 100 and (s.max(0) - s.min(0)).max() <= 0.70]
    print(f"[perception]   {len(segs)} object segments after filtering")

    if not segs:
        raise RuntimeError(
            "Perception found no valid segments — check PLY scale and table geometry."
        )

    # Fitting
    if fitter_type == "superdec":
        from superdec_fitter import SuperdecFitter  # noqa: E402
        fitter = SuperdecFitter(
            superdec_dir=str(_SUPERDEC),
            checkpoint_path=_find_best_checkpoint(checkpoint_folder),
        )
        fits = fitter.fit_batch([s for s in segs if len(s) >= 30])
    else:
        from superquadric import SuperquadricFitter  # noqa: E402
        lm = SuperquadricFitter(n_restarts=2, n_lm_rounds=10, subsample=256)
        fits = [lm.fit_adaptive(s) for s in segs if len(s) >= 30]

    fits = [f for f in fits if f and f.primitives]
    n_prims = sum(len(f.primitives) for f in fits)
    print(f"[perception]   {len(fits)} objects → {n_prims} SQ primitives")
    return fits


def _find_best_checkpoint(folder: str) -> str:
    """Return the highest-epoch .pt file in folder (or ckpt.pt if present)."""
    folder = Path(folder)
    ckpt = folder / "ckpt.pt"
    if ckpt.exists():
        return str(ckpt)
    pts = sorted(folder.glob("epoch_*.pt"),
                 key=lambda p: int(p.stem.split("_")[1]))
    if not pts:
        raise FileNotFoundError(f"No .pt checkpoint found in {folder}")
    return str(pts[-1])


# ──────────────────────────────────────────────────────────────────────────────
# World building
# ──────────────────────────────────────────────────────────────────────────────

MIN_RADIUS = 0.01  # 1 cm — avoids SDF kernel overflow for near-zero radii


def build_sq_mesh(sq: Superquadric, resolution: int = 32) -> Mesh:
    """Parametric SQ surface → triangle mesh for Isaac Sim visualization."""
    eta   = np.linspace(-np.pi / 2, np.pi / 2, resolution, endpoint=True)
    omega = np.linspace(-np.pi, np.pi, resolution, endpoint=True)
    eg, og = np.meshgrid(eta, omega, indexing="ij")

    def sp(v, p):
        return np.sign(v) * np.abs(v) ** p

    x = sq.radii[0] * sp(np.cos(eg), sq.eps[0]) * sp(np.cos(og), sq.eps[1])
    y = sq.radii[1] * sp(np.cos(eg), sq.eps[0]) * sp(np.sin(og), sq.eps[1])
    z = sq.radii[2] * sp(np.sin(eg), sq.eps[0])

    verts = np.stack([x, y, z], axis=-1).reshape(-1, 3)
    # Flatten poles
    verts[:resolution, 0] = 0.0
    verts[-resolution:, 0] = 0.0

    faces = []
    R = resolution
    for row in range(R - 1):
        for col in range(R - 1):
            i = row * R + col
            faces += [[i, i + 1, i + R], [i + R, i + 1, i + R + 1]]
        i = row * R + (R - 1)
        faces += [[i, row * R, i + R], [i + R, row * R, (row + 1) * R]]
    faces += [[(R-1)*R + (R-1), (R-1)*R, R-1], [R-1, (R-1)*R, 0]]

    color = list(sq.color) if sq.color else [0.3, 0.6, 0.9, 0.7]
    if len(color) == 3:
        color.append(0.7)

    return Mesh(
        name=f"{sq.name}_mesh",
        vertices=verts,
        faces=np.asarray(faces, dtype=np.int32),
        pose=sq.pose,
        color=color,
    )


def build_worlds(fits) -> tuple:
    """Build collision and visual WorldConfigs from perception fits.

    Returns
    -------
    collision_world : WorldConfig  (SQ + table cuboid for planning)
    visual_world    : WorldConfig  (SQ meshes for rendering)
    """
    from superdec_fitter import fits_to_curobo_world  # noqa: E402

    sq_world = fits_to_curobo_world(fits)

    # Apply collision tolerance (shrink radii slightly)
    tol = args.collision_tolerance
    shrunk = []
    for sq in sq_world.superquadric:
        sq2 = copy.deepcopy(sq)
        sq2.radii = [max(r - tol, MIN_RADIUS) for r in sq2.radii]
        shrunk.append(sq2)

    # Table plane
    table_cfg = load_yaml(join_path(get_world_configs_path(), "collision_table.yml"))
    table_world = WorldConfig.from_dict(table_cfg)
    table_world.cuboid[0].pose[2] -= 0.02  # slight downward offset

    collision_world = WorldConfig(
        cuboid=table_world.cuboid,
        superquadric=shrunk,
    )

    mesh_obstacles = [build_sq_mesh(sq) for sq in sq_world.superquadric]
    visual_world = WorldConfig(cuboid=[], mesh=mesh_obstacles)

    if args.print_sq_stats:
        radii_all = np.array([sq.radii for sq in sq_world.superquadric])
        eps_all   = np.array([sq.eps   for sq in sq_world.superquadric])
        print(f"[world] SQ count: {len(sq_world.superquadric)}")
        print(f"[world] radii min/max: {radii_all.min():.4f} / {radii_all.max():.4f} m")
        print(f"[world] eps   min/max: {eps_all.min():.3f}  / {eps_all.max():.3f}")

    return collision_world, visual_world


# ──────────────────────────────────────────────────────────────────────────────
# Motion generation
# ──────────────────────────────────────────────────────────────────────────────

def build_motion_gen(robot_cfg: dict, collision_world: WorldConfig,
                     tensor_args: TensorDeviceType) -> MotionGen:
    """Create and warm up a MotionGen instance for SQ collision."""
    has_sq = bool(collision_world.superquadric)
    n_sq   = len(collision_world.superquadric) if has_sq else 0

    # The SQ radial-distance kernel uses mask.nonzero() → dynamic tensor shapes
    # incompatible with CUDA graph capture. Disable graphs when SQs are present.
    use_cuda_graph = not has_sq

    collision_cache = {"obb": len(collision_world.cuboid)}
    if has_sq:
        collision_cache["superquadric"] = n_sq

    # Balanced profile for interactive SQ mode (reduces VRAM vs "quality")
    num_ik_seeds        = 4
    num_batch_ik_seeds  = 4
    num_trajopt_seeds   = 4
    num_graph_seeds     = 8 if has_sq else 0
    trajopt_tsteps      = 32
    interpolation_steps = 2000

    print(f"[motion_gen] Building MotionGen: {n_sq} SQ obstacles, "
          f"use_cuda_graph={use_cuda_graph}")

    mg_config = MotionGenConfig.load_from_robot_config(
        robot_cfg,
        collision_world,
        tensor_args,
        collision_checker_type=(
            CollisionCheckerType.PRIMITIVE if has_sq else CollisionCheckerType.MESH
        ),
        num_ik_seeds=num_ik_seeds,
        num_batch_ik_seeds=num_batch_ik_seeds,
        num_trajopt_seeds=num_trajopt_seeds,
        num_graph_seeds=num_graph_seeds,
        interpolation_steps=interpolation_steps,
        interpolation_dt=0.05,
        collision_cache=collision_cache,
        optimize_dt=True,
        trajopt_tsteps=trajopt_tsteps,
        use_cuda_graph=use_cuda_graph,
    )

    mg = MotionGen(mg_config)
    print("[motion_gen] Warming up …")
    mg.warmup(enable_graph=(num_graph_seeds > 0), warmup_js_trajopt=False)
    print("[motion_gen] Ready.")
    return mg


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    # ── 1. Perception ────────────────────────────────────────────────────────
    fits = run_perception(args.ply, args.fitter, args.checkpoint_folder)
    collision_world, visual_world = build_worlds(fits)

    # ── 2. Isaac Sim scene setup ─────────────────────────────────────────────
    setup_curobo_logger("warn")

    my_world = World(stage_units_in_meters=1.0)
    stage    = my_world.stage
    xform    = stage.DefinePrim("/World", "Xform")
    stage.SetDefaultPrim(xform)
    stage.DefinePrim("/curobo", "Xform")

    # Target cube (drag in GUI to set goal)
    target = cuboid.VisualCuboid(
        "/World/target",
        position=np.array([0.5, 0.0, 0.45]),
        orientation=np.array([1.0, 0.0, 0.0, 0.0]),
        color=np.array([1.0, 0.0, 0.0]),
        size=0.05,
    )

    tensor_args = TensorDeviceType()

    # Robot
    robot_cfg_path = get_robot_configs_path()
    robot_cfg      = load_yaml(join_path(robot_cfg_path, args.robot))["robot_cfg"]
    joint_names    = robot_cfg["kinematics"]["cspace"]["joint_names"]
    default_config = robot_cfg["kinematics"]["cspace"]["retract_config"]

    robot, _ = add_robot_to_scene(robot_cfg, my_world)
    articulation_controller = None

    # ── 3. CuRobo MotionGen ──────────────────────────────────────────────────
    motion_gen = build_motion_gen(robot_cfg, collision_world, tensor_args)

    # Check retract pose validity
    retract_js = JointState.from_position(
        motion_gen.get_retract_config().view(1, -1),
        joint_names=motion_gen.kinematics.joint_names,
    )
    retract_ok, retract_status = motion_gen.check_start_state(retract_js)
    if not retract_ok:
        carb.log_warn(
            f"[startup] Retract pose is in world collision (status={retract_status}). "
            "Planning will attempt with check_start_validity=False. "
            "Consider increasing --collision_tolerance if this persists."
        )
    else:
        print(f"[startup] Retract pose collision check: VALID")

    # ── 4. Visualize world ───────────────────────────────────────────────────
    add_extensions(simulation_app, "native" if args.headless else None)
    usd_helper = UsdHelper()
    usd_helper.load_stage(my_world.stage)
    usd_helper.add_world_to_stage(visual_world, base_frame="/World")

    my_world.scene.add_default_ground_plane()

    # Auto-targets for headless/debug
    auto_cube_positions = None
    if args.auto_targets is not None:
        try:
            raw = json.loads(args.auto_targets)
            auto_cube_positions = [np.array(t, dtype=np.float64) for t in raw]
            print(f"[auto-targets] {len(auto_cube_positions)} positions loaded, "
                  f"cycling every {args.auto_target_interval} steps.")
        except Exception as exc:
            carb.log_warn(f"[auto-targets] Failed to parse --auto_targets: {exc}")

    # ── 5. Simulation loop ───────────────────────────────────────────────────
    if args.headless:
        my_world.play()

    plan_config = MotionGenPlanConfig(
        enable_graph=bool(collision_world.superquadric),
        max_attempts=1,
        timeout=args.plan_timeout,
        enable_finetune_trajopt=True,
        time_dilation_factor=0.5,
        check_start_validity=(not bool(collision_world.superquadric)),
    )

    cmd_plan         = None
    cmd_idx          = 0
    idx_list         = []
    past_pose        = None
    past_orientation = None
    idle_steps       = 0
    last_plan_step   = -1_000_000
    stationary_steps = 0
    auto_target_idx  = 0

    while simulation_app.is_running():
        my_world.step(render=True)

        if not my_world.is_playing():
            if idle_steps % 200 == 0:
                print("*** Click Play to start simulation ***")
            idle_steps += 1
            continue

        step_index = my_world.current_time_step_index

        if args.max_frames > 0 and step_index >= args.max_frames:
            print(f"[headless] max_frames={args.max_frames} reached; exiting.")
            break

        # Init robot
        if articulation_controller is None:
            articulation_controller = robot.get_articulation_controller()
        if step_index < 10:
            robot._articulation_view.initialize()
            idx_list = [robot.get_dof_index(name) for name in joint_names]
            robot.set_joint_positions(default_config, idx_list)
            robot._articulation_view.set_max_efforts(
                values=np.array([5000] * len(idx_list)), joint_indices=idx_list
            )
        if step_index < 20:
            continue

        # Auto-target cycling
        if auto_cube_positions is not None:
            interval = max(args.auto_target_interval, 1)
            new_idx = (step_index // interval) % len(auto_cube_positions)
            if new_idx != auto_target_idx:
                auto_target_idx = new_idx
                target.set_world_pose(position=auto_cube_positions[auto_target_idx])
                print(f"[auto-targets] step={step_index} → target #{auto_target_idx} "
                      f"pos={auto_cube_positions[auto_target_idx]}")

        # Robot state
        sim_js = robot.get_joints_state()
        cu_js  = JointState(
            position=tensor_args.to_device(sim_js.positions),
            velocity=tensor_args.to_device(sim_js.velocities * 0.0),
            acceleration=tensor_args.to_device(sim_js.velocities * 0.0),
            jerk=tensor_args.to_device(sim_js.velocities * 0.0),
            joint_names=joint_names,
        )
        cu_js = cu_js.get_ordered_joint_state(motion_gen.kinematics.joint_names)

        # Check if target has moved
        cube_pos, cube_ori = target.get_world_pose()
        target_changed = False
        if past_pose is None or np.linalg.norm(cube_pos - past_pose) > 0.005:
            target_changed = True
            stationary_steps = 0
        else:
            stationary_steps += 1
        past_pose        = cube_pos.copy()
        past_orientation = cube_ori.copy()

        # Trigger planning when target is stable and cooldown has elapsed
        cooldown_ok   = (step_index - last_plan_step) >= max(args.plan_cooldown_steps, 0)
        stable_enough = stationary_steps >= max(args.plan_stable_steps, 1)
        robot_moving  = cmd_plan is not None

        if stable_enough and cooldown_ok and not robot_moving:
            ik_goal = Pose(
                position=tensor_args.to_device(cube_pos[np.newaxis]),
                quaternion=tensor_args.to_device(cube_ori[np.newaxis]),
            )
            t0 = time.perf_counter()
            result = motion_gen.plan_single(cu_js.unsqueeze(0), ik_goal, plan_config)
            dt_ms  = (time.perf_counter() - t0) * 1000

            if result.success.item():
                cmd_plan = result.get_interpolated_plan()
                cmd_plan = motion_gen.get_full_js(cmd_plan)
                cmd_idx  = 0
                last_plan_step   = step_index
                stationary_steps = 0
                print(f"[plan] SUCCESS  {dt_ms:.0f} ms  "
                      f"motion={result.motion_time:.2f}s  "
                      f"waypoints={cmd_plan.position.shape[0]}")
            else:
                last_plan_step   = step_index  # back off even on failure
                stationary_steps = 0
                carb.log_warn(
                    f"[plan] FAILED  status={result.status}  {dt_ms:.0f} ms  "
                    f"ik={result.ik_time*1000:.0f}ms  "
                    f"trajopt={result.trajopt_time*1000:.0f}ms"
                )

        # Execute trajectory
        if cmd_plan is not None:
            cmd_state  = cmd_plan[cmd_idx]
            art_action_pos = cmd_state.position.cpu().numpy()
            art_action_vel = cmd_state.velocity.cpu().numpy()

            from omni.isaac.core.utils.types import ArticulationAction
            articulation_controller.apply_action(
                ArticulationAction(
                    joint_positions=art_action_pos,
                    joint_velocities=art_action_vel,
                    joint_indices=idx_list,
                )
            )

            # Extra physics sub-steps for smoother execution
            for _ in range(2):
                my_world.step(render=False)

            cmd_idx += 1
            if cmd_idx >= cmd_plan.position.shape[0]:
                cmd_plan = None
                cmd_idx  = 0

    simulation_app.close()


if __name__ == "__main__":
    main()
