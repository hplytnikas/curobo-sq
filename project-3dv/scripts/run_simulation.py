#!/usr/bin/env python3
"""
run_simulation.py
=================
End-to-end tabletop simulation: perception → SQ world model → path planning.

Usage examples
--------------
CPU-only (no GPU/cuRobo required):
    python3 scripts/run_simulation.py --scene 1 --no-planning --no-superdec

SuperDec + RRT* planner (GPU node):
    python3 scripts/run_simulation.py --scene 1 \\
        --checkpoint /work/courses/3dv/team15/checkpoints/superdec_tabletop/\\
superdec_tabletop_finetune_v2/epoch_300.pt

SuperDec + cuRobo MotionGen:
    python3 scripts/run_simulation.py --scene 1 --planner curobo \\
        --checkpoint /work/courses/3dv/team15/checkpoints/superdec_tabletop/\\
superdec_tabletop_finetune_v2/epoch_300.pt
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import open3d as o3d

# ── PYTHONPATH ────────────────────────────────────────────────────────────────
_REPO_ROOT  = Path(__file__).resolve().parent.parent
_SRC        = _REPO_ROOT / "src" / "project_3dv" / "perception"
_SUPERDEC   = Path("/work/courses/3dv/team15/superdec")
_CUROBO_SRC = Path("/work/courses/3dv/team15/curobo-sq/curobo/src")

for _p in [str(_SRC), str(_SUPERDEC), str(_CUROBO_SRC)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

# ── RGB-D Scenes v2 loader ────────────────────────────────────────────────────
_RGBD_DATA = _REPO_ROOT / "data" / "rgbd-scenes-v2" / "pc"


def _load_scene(scene_id: int) -> np.ndarray:
    sys.path.insert(0, str(_REPO_ROOT / "scripts"))
    from test_on_superdec_split import RGBDScenesV2
    scene = RGBDScenesV2(data_root=str(_RGBD_DATA), scene_id=scene_id)
    pts, _, _labels = scene.load()
    return pts


def _run_lm_pipeline(pts: np.ndarray) -> list:
    """LM fitter via Stage 2 of two-stage pipeline — CPU, no checkpoint needed."""
    from pipeline import fit_superquadrics_world
    from superquadric import SuperquadricFitter

    N = len(pts)
    if N > 50_000:
        idx = np.random.choice(N, 50_000, replace=False)
        pts = pts[idx]

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts.astype(np.float64))

    fitter = SuperquadricFitter(n_restarts=2, n_lm_rounds=10, subsample=256)
    sqs = fit_superquadrics_world(
        pcd,
        fitter              = fitter,
        min_object_points   = 100,
        max_object_extent   = 0.60,
        cluster_min_points  = 5,
        max_height          = 0.4,
    )
    return sqs


def _run_superdec_pipeline(pts: np.ndarray, checkpoint_path: str) -> list:
    """SuperDec fitter via Stage 2 of two-stage pipeline — GPU required."""
    from pipeline import fit_superquadrics_world
    from superdec_fitter import SuperdecFitter

    N = len(pts)
    if N > 50_000:
        idx = np.random.choice(N, 50_000, replace=False)
        pts = pts[idx]

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts.astype(np.float64))

    fitter = SuperdecFitter(
        superdec_dir=str(_SUPERDEC),
        checkpoint_path=checkpoint_path,
    )
    sqs = fit_superquadrics_world(
        pcd,
        fitter              = fitter,
        min_object_points   = 100,
        max_object_extent   = 0.60,
        cluster_min_points  = 5,
        max_height          = 0.4,
    )
    return sqs


# ── Planning ──────────────────────────────────────────────────────────────────

def _plan_rrtstar(fits: list, n_queries: int = 5, out_dir: Path = None) -> dict:
    """RRT* planning via superdec_planner (CPU/GPU)."""
    from superdec_fitter import fits_to_superdec_npz
    from superdec_planner.superdec import Scene
    from superdec_planner.rrt_superdec import PathPlanner
    import tempfile, os

    # Save npz, load Scene
    with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as f:
        tmp_npz = f.name
    try:
        fits_to_superdec_npz(fits, save_path=tmp_npz)
        scene = Scene(tmp_npz)
    finally:
        os.unlink(tmp_npz)

    # Table bounding box from primitive translations
    # fits is now List[SuperquadricWorld]; each has a .position attribute
    all_trans = np.array([sq.position for sq in fits])
    if len(all_trans) == 0:
        return {"success_rate": 0.0, "mean_latency_ms": 0.0, "paths": []}

    lo = all_trans.min(axis=0) - 0.3
    hi = all_trans.max(axis=0) + 0.3
    bound = {
        "low_x": float(lo[0]), "high_x": float(hi[0]),
        "low_y": float(lo[1]), "high_y": float(hi[1]),
        "low_z": float(lo[2]), "high_z": float(hi[2]),
    }

    planner = PathPlanner()
    planner.update_collision_radius(0.05)
    planner.update_sp(bound, scene)

    rng = np.random.default_rng(42)
    latencies = []
    successes = 0
    paths = []

    for q in range(n_queries):
        # Random free start/goal (sample inside bbox, reject if in collision)
        for _ in range(50):
            start_pos = rng.uniform(lo, hi)
            goal_pos  = rng.uniform(lo, hi)
            if np.linalg.norm(start_pos - goal_pos) > 0.2:
                break

        start = {"pos": start_pos, "quat": np.array([0., 0., 0., 1.])}
        goal  = {"pos": goal_pos,  "quat": np.array([0., 0., 0., 1.])}
        planner.update_start_goal(start, goal)

        t0 = time.perf_counter()
        try:
            planner.solve(time_limit=1.0, method="rrtstar")
            solution = planner.get_solution()
            latency_ms = (time.perf_counter() - t0) * 1000
            successes += 1
            paths.append(solution)
        except Exception:
            latency_ms = (time.perf_counter() - t0) * 1000
            paths.append([])
        latencies.append(latency_ms)
        print(f"  Query {q+1}/{n_queries}: {latency_ms:.1f} ms  "
              f"({'found' if paths[-1] else 'failed'})")

    return {
        "success_rate":    successes / n_queries,
        "mean_latency_ms": float(np.mean(latencies)),
        "paths":           paths,
        "latencies_ms":    latencies,
    }


def _plan_curobo(sqs: list, n_queries: int = 5) -> dict:
    """cuRobo MotionGen planning (GPU required)."""
    from pipeline import superquadrics_to_curobo_world

    try:
        from curobo.wrap.reacher.motion_gen import (
            MotionGen, MotionGenConfig, MotionGenPlanConfig)
        from curobo.types.math import Pose
        from curobo.types.robot import JointState
        import torch
    except ImportError as e:
        print(f"  cuRobo MotionGen not available: {e}")
        return {"success_rate": 0.0, "mean_latency_ms": 0.0, "paths": []}

    world = superquadrics_to_curobo_world(sqs)

    # Standard Franka Panda config
    robot_cfg = "franka.yml"
    try:
        mg_cfg = MotionGenConfig.load_from_robot_config(
            robot_cfg,
            world,
            tensor_args={"device": "cuda", "dtype": torch.float32},
        )
        mg = MotionGen(mg_cfg)
        mg.warmup(n_goalset=1)
    except Exception as e:
        print(f"  MotionGen init failed: {e}")
        return {"success_rate": 0.0, "mean_latency_ms": 0.0, "paths": []}

    plan_cfg   = MotionGenPlanConfig(max_attempts=3)
    rng        = np.random.default_rng(0)
    latencies  = []
    successes  = 0
    paths      = []
    n_dof      = 7  # Franka has 7 DOF

    for q in range(n_queries):
        q_start = torch.from_numpy(
            rng.uniform(-2.5, 2.5, n_dof).astype(np.float32)).cuda()
        q_goal  = torch.from_numpy(
            rng.uniform(-2.5, 2.5, n_dof).astype(np.float32)).cuda()

        start_state = JointState.from_position(q_start.unsqueeze(0))
        goal_state  = JointState.from_position(q_goal.unsqueeze(0))

        t0 = time.perf_counter()
        try:
            result     = mg.plan_single_js(start_state, goal_state, plan_cfg)
            latency_ms = (time.perf_counter() - t0) * 1000
            if result.success[0]:
                successes += 1
                paths.append(result.get_interpolated_plan())
            else:
                paths.append(None)
        except Exception:
            latency_ms = (time.perf_counter() - t0) * 1000
            paths.append(None)
        latencies.append(latency_ms)
        print(f"  Query {q+1}/{n_queries}: {latency_ms:.1f} ms  "
              f"({'ok' if paths[-1] is not None else 'failed'})")

    return {
        "success_rate":    successes / n_queries,
        "mean_latency_ms": float(np.mean(latencies)),
        "paths":           paths,
        "latencies_ms":    latencies,
    }


# ── Visualization ─────────────────────────────────────────────────────────────

def _visualize(fits: list, plan_result: dict, out_path: Path, pts_raw: np.ndarray = None):
    """Save a top-down matplotlib figure of SQ centroids + planned paths."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    fig, ax = plt.subplots(figsize=(8, 7))

    # Raw point cloud (subsample for speed)
    if pts_raw is not None and len(pts_raw) > 0:
        pts_sub = pts_raw[::max(1, len(pts_raw)//3000)]
        ax.scatter(pts_sub[:, 0], pts_sub[:, 2], s=0.3, c="lightgray",
                   alpha=0.4, rasterized=True)

    # SQ primitives — ellipses proportional to sx/sz
    for m in fits:
        for p in m.primitives:
            e = mpatches.Ellipse(
                (p.tx, p.tz),
                width=2 * p.sx, height=2 * p.sz,
                angle=np.degrees(p.ry),
                alpha=0.55,
                facecolor="steelblue", edgecolor="navy", linewidth=0.8,
            )
            ax.add_patch(e)

    # Planned paths
    colors = plt.cm.tab10.colors
    for i, path in enumerate(plan_result.get("paths", [])):
        if not path:
            continue
        if hasattr(path, "__iter__") and not isinstance(path, np.ndarray):
            try:
                waypoints = np.array([w["pos"] for w in path])
                ax.plot(waypoints[:, 0], waypoints[:, 2],
                        "-o", markersize=3, linewidth=1.2,
                        color=colors[i % len(colors)],
                        label=f"path {i+1}")
            except (KeyError, TypeError, IndexError):
                pass

    # Summary text
    sr   = plan_result.get("success_rate", float("nan"))
    ml   = plan_result.get("mean_latency_ms", float("nan"))
    n_prims = sum(len(m.primitives) for m in fits)
    title = (f"Scene overview  |  {len(fits)} objects, {n_prims} primitives\n"
             f"Planning: success {sr*100:.0f}%  |  mean latency {ml:.0f} ms")
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Z (m)")
    ax.set_aspect("equal", "datalim")
    ax.grid(True, linewidth=0.4, alpha=0.5)
    if any(p for p in plan_result.get("paths", []) if p):
        ax.legend(fontsize=7)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Saved figure → {out_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Perception → SQ world model → planning simulation"
    )
    parser.add_argument("--scene",       type=int, default=1,
                        help="RGB-D Scenes v2 scene ID (1–14)")
    parser.add_argument("--checkpoint",  default=None,
                        help="SuperDec .pt checkpoint file (enables SuperDec fitter)")
    parser.add_argument("--no-superdec", action="store_true",
                        help="Use LM fitter instead of SuperDec")
    parser.add_argument("--planner",     choices=["rrtstar", "curobo", "none"],
                        default="rrtstar",
                        help="Path planner to use (default: rrtstar)")
    parser.add_argument("--no-planning", action="store_true",
                        help="Skip planning entirely")
    parser.add_argument("--n-queries",   type=int, default=5,
                        help="Number of planning queries (default: 5)")
    parser.add_argument("--out-dir",     default=None,
                        help="Output directory for PNG and npz (default: outputs/)")
    args = parser.parse_args()

    out_dir = Path(args.out_dir) if args.out_dir else _REPO_ROOT / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n=== run_simulation.py  (scene {args.scene:02d}) ===\n")

    # ── 1. Load scene ─────────────────────────────────────────────────────────
    print("Step 1: loading scene …")
    t0 = time.perf_counter()
    try:
        pts_raw = _load_scene(args.scene)
    except Exception as e:
        print(f"  ERROR: {e}")
        sys.exit(1)
    print(f"  {len(pts_raw):,} points  ({time.perf_counter()-t0:.1f}s)")

    # ── 2. Perception ─────────────────────────────────────────────────────────
    print("\nStep 2: running perception pipeline …")
    t0 = time.perf_counter()
    use_superdec = (not args.no_superdec and args.checkpoint is not None)
    if use_superdec:
        print(f"  Fitter: SuperdecFitter  checkpoint={args.checkpoint}")
        fits = _run_superdec_pipeline(pts_raw, args.checkpoint)
    else:
        if not args.no_superdec and args.checkpoint is None:
            print("  WARNING: --checkpoint not provided; falling back to LM fitter")
        print("  Fitter: SuperquadricFitter (LM)")
        fits = _run_lm_pipeline(pts_raw)
    t1 = time.perf_counter()
    n_prims = sum(len(f.primitives) for f in fits)
    print(f"  {len(fits)} segments → {n_prims} primitives  ({t1-t0:.1f}s)")

    if not fits:
        print("  ERROR: no fits produced")
        sys.exit(1)

    # ── 3. Export npz ─────────────────────────────────────────────────────────
    print("\nStep 3: exporting superdec .npz …")
    from superdec_fitter import fits_to_superdec_npz, fits_to_curobo_world
    npz_path = out_dir / f"scene_{args.scene:02d}.npz"
    arrays = fits_to_superdec_npz(fits, save_path=str(npz_path))
    print(f"  exist       : {arrays['exist'].shape}")
    print(f"  scale       : {arrays['scale'].shape}")
    print(f"  exponents   : {arrays['exponents'].shape}")
    print(f"  rotation    : {arrays['rotation'].shape}")
    print(f"  translation : {arrays['translation'].shape}")
    print(f"  saved → {npz_path}")

    # ── 4. cuRobo WorldConfig ─────────────────────────────────────────────────
    print("\nStep 4: building cuRobo WorldConfig …")
    try:
        world = fits_to_curobo_world(fits)
        n_sq = len(world.superquadric)
        print(f"  {n_sq} Superquadric obstacles  ✓")
        # Verify quaternion norms
        bad = sum(1 for sq in world.superquadric
                  if abs((sq.pose[3]**2 + sq.pose[4]**2 +
                           sq.pose[5]**2 + sq.pose[6]**2)**0.5 - 1.0) > 1e-5)
        if bad:
            print(f"  WARNING: {bad} quaternions with |q| ≠ 1")
        else:
            print(f"  All quaternions unit-norm  ✓")
    except ImportError as e:
        print(f"  cuRobo not available (expected on login node): {e}")

    # ── 5. Planning ───────────────────────────────────────────────────────────
    plan_result = {"success_rate": float("nan"), "mean_latency_ms": float("nan"),
                   "paths": []}
    if not args.no_planning and args.planner != "none":
        print(f"\nStep 5: planning ({args.planner}, {args.n_queries} queries) …")
        budget_ms = 200.0
        if args.planner == "rrtstar":
            plan_result = _plan_rrtstar(fits, n_queries=args.n_queries,
                                        out_dir=out_dir)
        elif args.planner == "curobo":
            plan_result = _plan_curobo(fits, n_queries=args.n_queries)

        sr  = plan_result["success_rate"]
        ml  = plan_result["mean_latency_ms"]
        budget_ok = ml <= budget_ms or np.isnan(ml)
        print(f"\n  Success rate    : {sr*100:.0f}% / 100%")
        print(f"  Mean latency    : {ml:.1f} ms  (budget {budget_ms:.0f} ms "
              f"{'✓' if budget_ok else '✗ — exceeds budget'})")
    else:
        print("\nStep 5: planning skipped (--no-planning)")

    # ── 6. Visualize ──────────────────────────────────────────────────────────
    print("\nStep 6: saving visualization …")
    png_path = out_dir / f"scene_{args.scene:02d}_simulation.png"
    _visualize(fits, plan_result, png_path, pts_raw=pts_raw)

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n=== Summary ===")
    print(f"  Scene           : {args.scene:02d}")
    print(f"  Fitter          : {'SuperDec' if use_superdec else 'LM'}")
    print(f"  Segments        : {len(fits)}")
    print(f"  Primitives      : {n_prims}")
    print(f"  npz shape       : N={arrays['exist'].shape[0]}  "
          f"K={arrays['exist'].shape[1]}")
    if not np.isnan(plan_result["success_rate"]):
        print(f"  Planning SR     : {plan_result['success_rate']*100:.0f}%")
        print(f"  Mean latency    : {plan_result['mean_latency_ms']:.1f} ms")
    print(f"  Output PNG      : {png_path}")
    print()


if __name__ == "__main__":
    main()
