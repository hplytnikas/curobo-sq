"""Minimal superquadric obstacle motion planning benchmark for cuRobo v2.

One Franka arm plans to a series of target poses while avoiding a single
superquadric obstacle.  Switch ``--world_representation`` between
``superquadrics`` and ``mesh`` to compare native SQ collision against a
mesh approximation of the same geometry.

Timing is printed to stdout after every plan and saved to a CSV log.

Run (SQ collision, default 6 targets):
    PATH=/usr/local/cuda-12.8/bin:/usr/bin:$PATH ~/isaacsim/python_v2.sh curobov2/curobo/curobo/examples/getting_started/motion_gen_sq_simple.py

Run (mesh representation for comparison):
    PATH=/usr/local/cuda-12.8/bin:/usr/bin:$PATH \\
    ~/isaacsim/python_v2.sh \\
        curobov2/curobo/curobo/examples/getting_started/motion_gen_sq_simple.py \\
        --world_representation mesh

Run (warm-cache timing, repeat each target 3x):
    PATH=/usr/local/cuda-12.8/bin:/usr/bin:$PATH \\
    ~/isaacsim/python_v2.sh \\
        curobov2/curobo/curobo/examples/getting_started/motion_gen_sq_simple.py \\
        --num_repeats 3
"""

import argparse
import csv
import json
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import warp as wp

wp.init()

from curobo._src.geom.collision.collision_scene import SceneCollisionCfg
from curobo._src.geom.types import Cuboid, Mesh, SceneCfg, Superquadric
from curobo._src.types.device_cfg import DeviceCfg
from curobo.motion_planner import MotionPlanner, MotionPlannerCfg
from curobo.types import GoalToolPose, JointState, Pose

# ── CLI ───────────────────────────────────────────────────────────────────────

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument(
    "--world_representation",
    choices=["superquadrics", "mesh"],
    default="superquadrics",
    help="Collision world representation.",
)
parser.add_argument(
    "--auto_cube_targets",
    type=str,
    default=None,
    help="JSON array of [x,y,z] target positions (e.g. '[[0.5,0,0.5]]'). "
         "Defaults to a built-in circuit around the obstacle.",
)
parser.add_argument(
    "--num_repeats",
    type=int,
    default=1,
    help="Repeat each target this many times (useful for warm-cache timing).",
)
parser.add_argument(
    "--log_dir",
    type=str,
    default=None,
    help="Directory for timing CSV. Defaults to <repo>/logs/timing/.",
)
args = parser.parse_args()

# ── Obstacle ──────────────────────────────────────────────────────────────────
# A rounded-box superquadric placed in the Franka workspace.
# eps=[0.4, 0.4] gives a fairly boxy shape; [1.0, 1.0] gives an ellipsoid.

OBSTACLE = Superquadric(
    name="sq_obstacle",
    pose=[0.45, 0.0, 0.35, 1.0, 0.0, 0.0, 0.0],  # [x, y, z, qw, qx, qy, qz]
    radii=[0.08, 0.12, 0.10],
    shape=[0.4, 0.4],
    color=[0.2, 0.6, 0.9, 1.0],
)

# Also include the table so the planner doesn't drive through it.
TABLE = Cuboid(
    name="table",
    pose=[0.0, 0.0, -0.05, 1.0, 0.0, 0.0, 0.0],
    dims=[2.0, 2.0, 0.05],
)

# ── Auto-targets ─────────────────────────────────────────────────────────────
# Default circuit of end-effector goal positions around the obstacle.

DEFAULT_TARGETS = [
    [0.50,  0.00, 0.50],
    [0.40,  0.30, 0.55],
    [0.40, -0.30, 0.55],
    [0.55,  0.00, 0.65],
    [0.35,  0.15, 0.45],
    [0.35, -0.15, 0.45],
]

# Default end-effector orientation: pointing downward (wrist facing floor).
GOAL_QUAT = [0.0, 1.0, 0.0, 0.0]  # [qw, qx, qy, qz]

# ── Mesh approximation of the SQ ─────────────────────────────────────────────

def _sq_to_mesh(sq: Superquadric, resolution: int = 40) -> Mesh:
    """Parametric superquadric surface → triangle Mesh obstacle."""
    resolution = max(resolution, 8)
    eta = np.linspace(-np.pi / 2.0, np.pi / 2.0, resolution, endpoint=True)
    omega = np.linspace(-np.pi, np.pi, resolution, endpoint=True)
    eg, og = np.meshgrid(eta, omega, indexing="ij")

    def sp(v, p):
        return np.sign(v) * np.abs(v) ** p

    eps1, eps2 = sq.shape[0], sq.shape[1]
    x = sq.radii[0] * sp(np.cos(eg), eps1) * sp(np.cos(og), eps2)
    y = sq.radii[1] * sp(np.cos(eg), eps1) * sp(np.sin(og), eps2)
    z = sq.radii[2] * sp(np.sin(eg), eps1)
    verts = np.stack((x, y, z), axis=-1).reshape(-1, 3)

    faces = []
    for row in range(resolution - 1):
        for col in range(resolution - 1):
            i = row * resolution + col
            faces += [[i, i + 1, i + resolution], [i + resolution, i + 1, i + resolution + 1]]
        r = row * resolution + (resolution - 1)
        r2 = (row + 1) * resolution + (resolution - 1)
        faces += [[r, row * resolution, r2], [r2, row * resolution, r2 - (resolution - 1)]]

    color = list(sq.color) if len(sq.color) == 4 else list(sq.color) + [1.0]
    return Mesh(
        name=f"{sq.name}_mesh",
        vertices=verts.tolist(),
        faces=np.asarray(faces, dtype=np.int32).tolist(),
        pose=list(sq.pose),
        color=color,
    )


# ── Scene construction ────────────────────────────────────────────────────────

def build_scene_cfg(world_representation: str) -> SceneCfg:
    if world_representation == "superquadrics":
        return SceneCfg(
            cuboid=[TABLE],
            superquadric=[OBSTACLE],
        )
    else:
        return SceneCfg(
            cuboid=[TABLE],
            mesh=[_sq_to_mesh(OBSTACLE)],
        )


# ── Planner construction ──────────────────────────────────────────────────────

def build_planner(world_representation: str) -> MotionPlanner:
    has_sq = world_representation == "superquadrics"
    device_cfg = DeviceCfg()

    # SQ Warp kernel uses dynamic-shape Warp arrays; disable CUDA graph capture.
    use_cuda_graph = not has_sq

    # Build base planner config (no scene_model yet — we inject it below).
    cfg = MotionPlannerCfg.create(
        robot="franka.yml",
        num_ik_seeds=16,
        num_trajopt_seeds=4,
        use_cuda_graph=use_cuda_graph,
    )

    # Build scene collision config and inject it into the planner cfg.
    scene_cfg = build_scene_cfg(world_representation)
    cache = (
        {"cuboid": 2, "superquadric": 4} if has_sq
        else {"cuboid": 2, "mesh": 4}
    )
    cfg.scene_collision_cfg = SceneCollisionCfg(
        device_cfg=device_cfg,
        scene_model=scene_cfg,
        cache=cache,
    )

    planner = MotionPlanner(cfg)
    print(f"[planner] world_rep={world_representation}  cuda_graph={use_cuda_graph}", flush=True)

    print("[planner] warming up...", flush=True)
    planner.warmup(enable_graph=True, num_warmup_iterations=3)
    print("[planner] ready.", flush=True)

    return planner


# ── Timing log ────────────────────────────────────────────────────────────────

def _make_log(world_rep: str, log_dir: str | None) -> tuple[Path, str]:
    if log_dir is not None:
        logs_dir = Path(log_dir)
    else:
        logs_dir = Path(__file__).resolve().parents[5] / "logs" / "timing"
    logs_dir.mkdir(parents=True, exist_ok=True)

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = logs_dir / f"sq_simple_{world_rep}_{stamp}.csv"
    with path.open("w", newline="") as f:
        csv.writer(f).writerow([
            "plan_id", "target_id", "world_rep",
            "success", "wall_s", "solve_s", "total_s",
            "target_x", "target_y", "target_z",
        ])
    return path, stamp


def _log_row(path: Path, plan_id: int, target_id: int, world_rep: str,
             success: bool, wall_s: float, result, target_pos) -> None:
    tx, ty, tz = float(target_pos[0]), float(target_pos[1]), float(target_pos[2])
    solve_s = float(getattr(result, "solve_time", float("nan"))) if result is not None else float("nan")
    total_s = float(getattr(result, "total_time", float("nan"))) if result is not None else float("nan")
    with path.open("a", newline="") as f:
        csv.writer(f).writerow([
            plan_id, target_id, world_rep,
            success, f"{wall_s:.4f}", f"{solve_s:.4f}", f"{total_s:.4f}",
            f"{tx:.4f}", f"{ty:.4f}", f"{tz:.4f}",
        ])


# ── Summary stats ─────────────────────────────────────────────────────────────

def _print_summary(rows: list) -> None:
    if not rows:
        return
    successes = [r for r in rows if r["success"]]
    n = len(rows)
    n_ok = len(successes)
    if n_ok > 0:
        wall_times = [r["wall_s"] for r in successes]
        print(
            f"\n── Summary ({n} plans, {n_ok} successful) ──────────────────────\n"
            f"  wall time  min={min(wall_times):.3f}s  "
            f"mean={sum(wall_times)/len(wall_times):.3f}s  "
            f"max={max(wall_times):.3f}s",
            flush=True,
        )
    else:
        print(f"\n── Summary ({n} plans, 0 successful) ──────────────────────", flush=True)


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    # Load auto-target positions
    if args.auto_cube_targets is not None:
        try:
            targets = [np.array(t, dtype=np.float64) for t in json.loads(args.auto_cube_targets)]
        except Exception as exc:
            print(f"[auto-targets] parse failed: {exc}", flush=True)
            targets = [np.array(t) for t in DEFAULT_TARGETS]
    else:
        targets = [np.array(t) for t in DEFAULT_TARGETS]

    if args.num_repeats > 1:
        targets = targets * args.num_repeats
    print(f"[auto-targets] {len(targets)} positions, world_rep={args.world_representation}", flush=True)

    # Build planner
    planner = build_planner(args.world_representation)

    # Timing log
    log_path, _ = _make_log(args.world_representation, args.log_dir)
    print(f"[timing] log → {log_path}\n", flush=True)

    # Start state: robot retract/default configuration [1, n_joints]
    def _default_start():
        return JointState.from_position(
            planner.default_joint_state.position.unsqueeze(0),
            joint_names=planner.joint_names,
        )

    start_state = _default_start()

    # GoalToolPose shape: [batch, horizon, num_links, num_goalset, 3/4]
    # With batch=1, horizon=1, num_links=1 (end-effector), num_goalset=1.
    goal_quat_t = torch.tensor(
        [[[[GOAL_QUAT]]]],   # [1, 1, 1, 1, 4]
        dtype=torch.float32,
        device=planner.device_cfg.device,
    )

    rows = []
    plan_id = 0

    for target_id, target_pos in enumerate(targets):
        goal_pos_t = torch.tensor(
            [[[[ target_pos.tolist() ]]]],  # [1, 1, 1, 1, 3]
            dtype=torch.float32,
            device=planner.device_cfg.device,
        )
        goal = GoalToolPose(
            tool_frames=planner.tool_frames,
            position=goal_pos_t,
            quaternion=goal_quat_t,
        )

        t0 = time.perf_counter()
        result = planner.plan_pose(goal, start_state)
        wall_s = time.perf_counter() - t0

        success = (result is not None) and bool(result.success.any())
        solve_s = float(getattr(result, "solve_time", float("nan"))) if result is not None else float("nan")
        total_s = float(getattr(result, "total_time", float("nan"))) if result is not None else float("nan")

        status_tag = "OK" if success else "FAIL"
        print(
            f"[plan#{plan_id:03d}] target#{target_id:02d} "
            f"pos=({target_pos[0]:.3f},{target_pos[1]:.3f},{target_pos[2]:.3f})  "
            f"{status_tag}  wall={wall_s:.3f}s  solve={solve_s:.3f}s  total={total_s:.3f}s",
            flush=True,
        )

        _log_row(log_path, plan_id, target_id, args.world_representation,
                 success, wall_s, result, target_pos)
        rows.append({"success": success, "wall_s": wall_s})

        if success:
            # Advance start state to end of planned trajectory for chained planning.
            # position may be 3D [B, T, DOF] or 4D [B, seeds, T, DOF]; flatten
            # batch/seed dims so [0, -1, :] always picks the last timestep.
            interpolated = result.get_interpolated_plan()
            pos = interpolated.position
            n_dof = len(planner.joint_names)
            last_pos = pos.reshape(-1, pos.shape[-2], pos.shape[-1])[0, -1, :n_dof].unsqueeze(0)
            start_state = JointState.from_position(
                last_pos,
                joint_names=planner.joint_names,
            )
        else:
            # Reset to retract on failure
            start_state = _default_start()

        plan_id += 1

    _print_summary(rows)
    print(f"\n[timing] log saved → {log_path}", flush=True)


if __name__ == "__main__":
    main()
