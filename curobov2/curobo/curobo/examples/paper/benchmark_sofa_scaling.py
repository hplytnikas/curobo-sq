"""Object-size scaling benchmark (a.k.a. the "sofa" experiment).

Loads a single large object (default: a chair point cloud), then progressively
increases the number of instances placed around the robot and measures planning
time for each collision representation.  This isolates the regime where the
analytic superquadric representation yields its largest speed-ups (large objects;
see the paper's size/speed-up section).

It is the recreatable, self-contained ``paper/`` counterpart of the ``--scaling_test``
mode of ``getting_started/motion_planning_sq.py``.  It reuses the main benchmark
infrastructure (``benchmark_sq_vs_mesh.py``) so the CSV schema and plots are
identical to the tabletop experiment.

Run under the ``3dv`` conda environment, e.g.::

    conda run -n 3dv python benchmark_sofa_scaling.py                 # default chair.ply
    conda run -n 3dv python benchmark_sofa_scaling.py --ply_path sofa.ply --counts 1,3,6,21,51
    conda run -n 3dv python plot_benchmark_paper.py --csv eval_out/results_sofa.csv --out-prefix sofa_
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
import trimesh
import warp as wp
from scipy.spatial import cKDTree

# Import the demo + main benchmark as libraries (same directory).
sys.path.insert(0, str(Path(__file__).resolve().parent))
import motion_planning_sq_demo as demo  # noqa: E402
import benchmark_sq_vs_mesh as bench  # noqa: E402
import sofa_viz  # noqa: E402

from curobo._src.geom.types import SceneCfg, Mesh  # noqa: E402
from curobo._src.types.device_cfg import DeviceCfg  # noqa: E402
from curobo.types import GoalToolPose, JointState, Pose  # noqa: E402


# ── configuration ───────────────────────────────────────────────────────────────

OUT_DIR = bench.OUT_DIR
RESULTS_SOFA_CSV = OUT_DIR / "results_sofa.csv"

# instance counts: one object, then progressively more (= 1 + the original
# getting_started SCALING_TEST_SIZES [0, 2, 5, 20, 50]).
SOFA_COUNTS = [1, 3, 6, 21, 51]

# Default targets (from getting_started DEFAULT_SCALING_TARGETS), as positions;
# extended below with a fixed downward end-effector orientation -> 7D wxyz poses.
_DEFAULT_TARGET_POSITIONS = [
    [0.7, 0.17, 0.5],
    [-0.1, -0.368, 0.5],
    [0.4, 0.4, 0.6],
    [0.5, -0.3, 0.4],
]
_DOWN_QUAT_WXYZ = [0.0, 1.0, 0.0, 0.0]   # EE pointing down
DEFAULT_TARGETS = [p + _DOWN_QUAT_WXYZ for p in _DEFAULT_TARGET_POSITIONS]

DEFAULT_PLY = str(demo.SUPERDEC_ROOT / "examples" / "chair.ply")


# ── object loading ───────────────────────────────────────────────────────────────

def _load_object_points(ply_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """Load a single object's point cloud as (points [N,3], colors [N,3] in [0,1]).

    Supports a real ``.ply`` (point cloud or mesh, via trimesh) and an ``.npz``
    with ``xyz``/``color`` (and optional ``instance_label``).
    """
    path = Path(ply_path)
    if path.suffix.lower() == ".npz":
        data = np.load(ply_path, allow_pickle=True)
        xyz = data["xyz"].astype(np.float32)
        rgb = data["color"].astype(np.float32) if "color" in data else None
        if "instance_label" in data:
            inst = data["instance_label"].astype(np.int64)
            ids = [i for i in np.unique(inst) if i not in (0,)]  # skip 0 (table)
            if ids:
                # largest non-table instance
                iid = max(ids, key=lambda i: int((inst == i).sum()))
                mask = inst == iid
                xyz = xyz[mask]
                rgb = rgb[mask] if rgb is not None else None
    else:
        m = trimesh.load(ply_path, process=False)
        if isinstance(m, trimesh.Scene):
            m = m.dump(concatenate=True)
        xyz = np.asarray(m.vertices, dtype=np.float32)
        rgb = None
        vis = getattr(m, "visual", None)
        vc = getattr(vis, "vertex_colors", None)
        if vc is not None and len(vc) == len(xyz):
            rgb = np.asarray(vc, dtype=np.float32)[:, :3]

    if rgb is None:
        rgb = np.tile(np.array([[0.6, 0.6, 0.65]], dtype=np.float32), (len(xyz), 1))
    if rgb.max() > 1.0:
        rgb = rgb / 255.0
    return xyz.astype(np.float32), rgb.astype(np.float32)


def _prepare_base_object(
    ply_path: str, scale_factor: float, checkpoint_folder: str, mesh_resolution: int,
) -> Tuple["bench._CachedPred", float]:
    """Load the object, fit SuperDec once, return (origin-frame cached object, xy extent)."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pts, colors = _load_object_points(ply_path)

    pts = pts * float(scale_factor)
    # centre in xy and floor at z = 0 so placement offsets position the object
    pts[:, 0] -= float(pts[:, 0].mean())
    pts[:, 1] -= float(pts[:, 1].mean())
    pts[:, 2] -= float(pts[:, 2].min())

    extent = float(max(np.ptp(pts[:, 0]), np.ptp(pts[:, 1])))
    print(f"Loaded object: {len(pts)} points, xy extent {extent:.2f} m")

    model = demo._load_model(checkpoint_folder, device)
    pred = demo._infer_single_object(pts, model, device, mesh_resolution, iid=0)
    if pred is None or pred.mesh is None:
        raise SystemExit("SuperDec produced no valid (non-needle) decomposition for this object.")

    cached = bench._CachedPred(
        outdict=pred.outdict,
        sq_meshes=bench._per_sq_meshes(pred.outdict, mesh_resolution),
        orig_mesh=None,
        pts=pts.astype(np.float32),
        colors=colors.astype(np.float32),
    )
    return cached, extent


# ── scene building ───────────────────────────────────────────────────────────────

def _sofa_geometry(n: int, extent: float) -> Tuple[float, float, float, float]:
    """Return (inner_r, outer_r, table_half, min_sep) for n large objects."""
    inner_r = max(1.0, 0.6 * extent + 0.4)         # clear robot base + object radius
    spacing = max(0.6, extent * 1.1)
    outer_r = inner_r + spacing * (0.5 + np.sqrt(max(n, 1)))
    min_sep = extent * 0.9                          # allow objects to nearly touch
    table_half = outer_r + extent
    return inner_r, outer_r, table_half, min_sep


def _build_scene(cached: "bench._CachedPred", n: int, extent: float, seed: int) -> Tuple[dict, float]:
    """Place n instances via Poisson sampling; return (scene dict, table_half)."""
    inner_r, outer_r, table_half, min_sep = _sofa_geometry(n, extent)
    rng = np.random.RandomState(seed)
    centers = demo._place_on_table(n, rng, inner_r=inner_r, outer_r=outer_r, min_sep=min_sep)

    object_pts, object_colors, predictions = [], [], []
    for slot, c in enumerate(centers):
        placed = bench._place_cached(cached, np.asarray(c, dtype=np.float32), iid=slot)
        predictions.append({k: placed[k] for k in ("iid", "outdict", "sq_meshes", "orig_mesh")})
        object_pts.append(placed["pts"])
        object_colors.append(placed["colors"])

    scene = {
        "name": f"sofa_{n:03d}",
        "n_objects": n,
        "object_pts": object_pts,
        "object_colors": object_colors,
        "predictions": predictions,
        "has_orig_mesh": False,
    }
    return scene, table_half


def _build_scene_cfg(scene: dict, representation: str, table) -> Tuple[SceneCfg, int]:
    """Like bench._build_scene_cfg but with an explicit (sofa-sized) table."""
    demo.TABLE = table  # demo._prediction_to_scene_cfg reads this global
    if representation == "superquadrics":
        cfg = demo._prediction_to_scene_cfg(
            bench._scene_predictions(scene), "superquadrics", 1.0,
            demo.SCENE_TRANSLATION, demo.SCENE_QUAT_WXYZ, torch.zeros(1),
        )
        return cfg, demo._count_items(cfg.superquadric)

    meshes: List[Mesh] = []
    if representation == "mesh":
        for p in scene["predictions"]:
            for k, (v, f) in enumerate(p["sq_meshes"]):
                meshes.append(Mesh(name=f"obj_{p['iid']}_sq_{k}",
                                   vertices=v.tolist(), faces=f.tolist(),
                                   pose=[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]))
    else:
        raise ValueError(f"Unknown representation: {representation!r}")

    cfg = SceneCfg(cuboid=[table], mesh=meshes)
    print(f"Built scene: {len(meshes)} {representation} mesh primitive(s)")
    return cfg, len(meshes)


# ── timing ───────────────────────────────────────────────────────────────────────

def _run_scene(
    scene: dict, table, representation: str, targets: List[List[float]],
    device_cfg: DeviceCfg, tree: cKDTree, visualize,
) -> List[dict]:
    """Sequential tour for one (scene, representation). Mirrors bench.benchmark_one."""
    scene_cfg, n_primitives = _build_scene_cfg(scene, representation, table)
    planner = demo._rebuild_planner(scene_cfg, bench._PLANNER_REP[representation], device_cfg)
    tool_frame = planner.kinematics.tool_frames[0]
    current_state = planner.default_joint_state.clone().unsqueeze(0)

    # throwaway plan to remove first-call overhead
    if targets:
        try:
            planner.plan_pose(
                GoalToolPose.from_poses({tool_frame: Pose.from_list(targets[0])}, num_goalset=1),
                planner.kinematics.get_active_js(current_state.clone()),
                use_implicit_goal=True, max_attempts=3,
            )
        except Exception as exc:
            print(f"  [warn] warmup plan failed: {exc}")

    rows: List[dict] = []
    for leg, target in enumerate(targets):
        goal = GoalToolPose.from_poses({tool_frame: Pose.from_list(target)}, num_goalset=1)
        active_js = planner.kinematics.get_active_js(current_state.clone())

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        try:
            result = planner.plan_pose(goal, active_js, use_implicit_goal=True, max_attempts=3)
        except Exception as exc:
            print(f"  [{representation}] {scene['name']} leg {leg + 1}: plan_pose raised: {exc}")
            result = None
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        plan_wall = time.perf_counter() - t0

        success = bool(result is not None and result.success.any())
        row = {
            "scene": scene["name"], "n_objects": scene["n_objects"],
            "n_primitives": n_primitives, "representation": representation,
            "leg": leg + 1, "plan_success": int(success), "plan_wall_s": plan_wall,
            "plan_solver_total_s": float(getattr(result, "total_time", float("nan")))
            if result is not None else float("nan"),
            "motion_time_s": float("nan"), "playback_s": float("nan"),
            "n_steps_total": 0, "n_steps_collide": 0, "n_spheres_collide": 0,
            "frac_in_collision": float("nan"), "max_penetration_m": float("nan"),
        }

        if success:
            interp = planner.kinematics.get_active_js(result.get_interpolated_plan())
            row["motion_time_s"] = float(np.asarray(result.motion_time().cpu()).reshape(-1)[0])
            pos_per_step = bench._interp_positions_per_step(interp).contiguous()
            n_steps = int(pos_per_step.shape[0])
            row["playback_s"] = n_steps / bench.PLAYBACK_HZ
            spheres = bench._trajectory_spheres(planner, pos_per_step)
            row.update(bench._collision_metrics(spheres, tree))
            last = pos_per_step[-1:].contiguous()
            current_state = planner.kinematics.get_full_js(
                JointState.from_position(last, joint_names=interp.joint_names)
            )
            if visualize is not None:
                sofa_viz.animate(visualize, pos_per_step, interp.joint_names, bench.PLAYBACK_HZ)
        else:
            print(f"  [{representation}] {scene['name']} leg {leg + 1}: FAILED to plan")

        rows.append(row)
        print(f"  [{representation}] {scene['name']} leg {leg + 1}: "
              f"success={success} plan={plan_wall:.3f}s "
              f"motion={row['motion_time_s']:.3f}s coll_frac={row['frac_in_collision']}")

    del planner
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return rows


# ── main ─────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ply_path", type=str, default=DEFAULT_PLY,
                        help="Input object point cloud (.ply or .npz). Default: chair.ply")
    parser.add_argument("--checkpoint_folder", type=str, default=demo.CHECKPOINT_FOLDER)
    parser.add_argument("--mesh_resolution", type=int, default=48)
    parser.add_argument("--scale_factor", type=float, default=1.0)
    parser.add_argument("--counts", type=bench._parse_counts, default=None,
                        help=f"Comma-separated instance counts (default {SOFA_COUNTS})")
    parser.add_argument("--targets", type=str, default=None,
                        help="JSON file with a list of [x,y,z,qw,qx,qy,qz] targets")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--visualize", action="store_true")
    parser.add_argument("--port", type=int, default=8083)
    args = parser.parse_args()

    wp.init()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    counts = args.counts if args.counts else SOFA_COUNTS

    targets = DEFAULT_TARGETS
    if args.targets:
        with open(args.targets) as f:
            targets = json.load(f)
    print(f"Using {len(targets)} targets, counts={counts}")

    device_cfg = DeviceCfg(device=args.device)
    cached, extent = _prepare_base_object(
        args.ply_path, args.scale_factor, args.checkpoint_folder, args.mesh_resolution
    )

    visualize = None
    if args.visualize:
        visualize = sofa_viz.make_visualizer(args.port)

    all_rows: List[dict] = []
    for n in counts:
        scene, table_half = _build_scene(cached, n, extent, seed=args.seed + n)
        table = bench._make_table(table_half)
        all_pts = np.concatenate(scene["object_pts"]).astype(np.float32)
        tree = cKDTree(all_pts)
        print(f"\n=== {scene['name']} ({n} instances, {len(all_pts)} points) ===")
        for rep in bench.REPRESENTATIONS:
            if rep in ("shapenet_mesh", "pointcloud"):
                continue  # no original meshes; pointcloud primitives removed
            all_rows.extend(_run_scene(scene, table, rep, targets, device_cfg, tree, visualize))

    with open(RESULTS_SOFA_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=bench.BENCH_FIELDNAMES)
        writer.writeheader()
        for r in all_rows:
            writer.writerow(r)
    print(f"\nWrote {len(all_rows)} rows to {RESULTS_SOFA_CSV}")


if __name__ == "__main__":
    main()
