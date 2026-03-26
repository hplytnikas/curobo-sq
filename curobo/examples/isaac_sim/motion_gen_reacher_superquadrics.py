"""Isaac Sim motion generation example using SuperDec-derived superquadrics.

Run this with ``omni_python`` and switch ``--world_representation`` between
``superquadrics`` and ``mesh`` to compare the native CuRobo superquadric
collision path against a mesh approximation built from the same superquadric
parameters.
"""

try:
    import isaacsim  # noqa: F401
except ImportError:
    pass

# Warm up the CUDA context before SimulationApp initializes; omitting this
# causes Isaac Sim to crash during startup.
import torch

_cuda_warmup = torch.zeros(4, device="cuda:0")

import argparse
import copy
import queue
import threading
import sys
import time
import traceback
from pathlib import Path
from typing import List
import numpy as np


def build_argparser() -> argparse.ArgumentParser:
    workspace_root = Path(__file__).resolve().parents[3]
    superdec_root = workspace_root / "superdec"

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--headless_mode",
        type=str,
        default=None,
        help="To run headless, use one of [native, websocket], webrtc might not work.",
    )
    parser.add_argument(
        "--robot",
        type=str,
        default="franka.yml",
        help="Robot configuration to load.",
    )
    parser.add_argument(
        "--external_asset_path",
        type=str,
        default=None,
        help="Path to external assets when loading an externally located robot.",
    )
    parser.add_argument(
        "--external_robot_configs_path",
        type=str,
        default=None,
        help="Path to external robot config when loading an external robot.",
    )
    parser.add_argument(
        "--visualize_spheres",
        action="store_true",
        default=False,
        help="When True, visualizes robot spheres.",
    )
    parser.add_argument(
        "--reactive",
        action="store_true",
        default=False,
        help="When True, runs in reactive mode.",
    )
    parser.add_argument(
        "--constrain_grasp_approach",
        action="store_true",
        default=False,
        help="When True, approaches grasp with fixed orientation and motion only along z axis.",
    )
    parser.add_argument(
        "--reach_partial_pose",
        nargs=6,
        metavar=("qx", "qy", "qz", "x", "y", "z"),
        type=float,
        default=None,
        help="Reach partial pose.",
    )
    parser.add_argument(
        "--hold_partial_pose",
        nargs=6,
        metavar=("qx", "qy", "qz", "x", "y", "z"),
        type=float,
        default=None,
        help="Hold partial pose while moving to goal.",
    )
    parser.add_argument(
        "--world_representation",
        choices=["superquadrics", "mesh"],
        default="superquadrics",
        help="Collision world representation to test.",
    )
    parser.add_argument(
        "--auto_fallback_to_mesh",
        action="store_true",
        default=False,
        help=(
            "When superquadric start-state collision persists after tolerance auto-relax, "
            "rebuild planner with mesh collision from the same inferred primitives."
        ),
    )
    parser.add_argument(
        "--disable_superquadric_trajopt_retry",
        action="store_true",
        default=False,
        help=(
            "Disable one automatic retry with relaxed planning attempts when "
            "superquadric planning fails with TRAJOPT_FAIL."
        ),
    )
    parser.add_argument(
        "--disable_superquadric_graph_only_fallback",
        action="store_true",
        default=False,
        help=(
            "Disable graph-only fallback in native superquadric mode. When enabled "
            "(default), a final retry can return a geometric path if trajopt keeps failing."
        ),
    )
    parser.add_argument(
        "--disable_superquadric_retract_escape",
        action="store_true",
        default=False,
        help=(
            "Disable automatic escape-to-retract fallback in native superquadric mode. "
            "When enabled (default), planner failures can trigger a temporary retreat "
            "to retract configuration before re-attempting the target."
        ),
    )
    parser.add_argument(
        "--disable_superquadric_ik_retry",
        action="store_true",
        default=False,
        help=(
            "Disable automatic IK recalculation retry in native superquadric mode. "
            "When enabled (default), IK_FAIL triggers one retry with increased IK seeds."
        ),
    )
    parser.add_argument(
        "--trajopt_only",
        action="store_true",
        default=False,
        help=(
            "Use IK + trajectory optimization only. Disables graph planner usage and "
            "all graph-based fallbacks (graph-only and retract-escape)."
        ),
    )
    parser.add_argument(
        "--ply_path",
        type=str,
        default=str(superdec_root / "examples" / "chair.ply"),
        help="Input point cloud used by SuperDec.",
    )
    parser.add_argument(
        "--checkpoint_folder",
        type=str,
        default=str(superdec_root / "checkpoints" / "normalized"),
        help="SuperDec checkpoint folder containing ckpt.pt and config.yaml.",
    )
    parser.add_argument(
        "--superdec_root",
        type=str,
        default=str(superdec_root),
        help="Path to the SuperDec repository root.",
    )
    parser.add_argument(
        "--superquadric_scale",
        type=float,
        default=1.0,
        help="Uniform scale applied to the inferred superquadrics before scene placement.",
    )
    parser.add_argument(
        "--superquadric_collision_tolerance",
        type=float,
        default=0.01,
        help=(
            "Shrink each collision superquadric radius by this margin (meters) to counter "
            "small signed-distance overestimation in the native superquadric kernel. "
            "Set to 0.0 to disable."
        ),
    )
    parser.add_argument(
        "--superquadric_auto_relax_max_tolerance",
        type=float,
        default=0.05,
        help=(
            "Maximum radius-shrink tolerance (meters) explored by start-state auto-relax. "
            "Lower this to keep obstacle inflation conservative."
        ),
    )
    parser.add_argument(
        "--superquadric_auto_relax_max_eps_blend",
        type=float,
        default=0.35,
        help=(
            "Maximum exponent blend toward ellipsoids explored by start-state auto-relax. "
            "0.0 disables eps blending; smaller values are more collision-conservative."
        ),
    )
    parser.add_argument(
        "--persist_superquadric_auto_relax_world",
        action="store_true",
        default=False,
        help=(
            "When enabled, a collision-relaxed SQ world found during auto-relax becomes the "
            "active planning world. Disabled by default so auto-relax is used only for "
            "start-state validation and the planner remains on the baseline collision world."
        ),
    )
    parser.add_argument(
        "--superquadric_min_eps",
        type=float,
        default=0.05,
        help=(
            "Lower clamp for SuperDec exponents before collision world creation. "
            "Increase to reduce very boxy primitives that can destabilize distance queries."
        ),
    )
    parser.add_argument(
        "--superquadric_max_eps",
        type=float,
        default=2.0,
        help="Upper clamp for SuperDec exponents before collision world creation.",
    )
    parser.add_argument(
        "--print_superquadric_stats",
        action="store_true",
        default=False,
        help="Print inferred superquadric parameter ranges and clamp counts.",
    )
    parser.add_argument(
        "--superquadric_max_radius",
        type=float,
        default=1.5,
        help=(
            "Upper clamp for each SuperDec principal radius (meters) after scale application. "
            "Prevents a single over-sized primitive from falsely covering the entire robot workspace. "
            "Set <= 0 to disable clamping."
        ),
    )
    parser.add_argument(
        "--superquadric_translation",
        nargs=3,
        type=float,
        default=[-0.29955, -0.68389, 0.13559],
        metavar=("x", "y", "z"),
        help="Scene translation applied to the inferred superquadrics.",
    )
    parser.add_argument(
        "--superquadric_orientation",
        nargs=4,
        type=float,
        default=[0.7071, 0.7071, 0.0, 0.0],
        metavar=("qw", "qx", "qy", "qz"),
        help="Scene orientation applied to the inferred superquadrics.",
    )
    parser.add_argument(
        "--surface_resolution",
        type=int,
        default=40,
        help="Resolution of the generated comparison meshes.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed used when subsampling the point cloud for SuperDec inference.",
    )
    parser.add_argument(
        "--plan_timeout",
        type=float,
        default=4.0,
        help="Per-plan timeout in seconds. Reduce to keep Isaac Sim responsive.",
    )
    parser.add_argument(
        "--plan_stable_steps",
        type=int,
        default=5,
        help="Require this many consecutive stationary frames before triggering a plan.",
    )
    parser.add_argument(
        "--plan_cooldown_steps",
        type=int,
        default=20,
        help="Minimum simulation steps between planning attempts.",
    )
    parser.add_argument(
        "--planner_memory_profile",
        choices=["auto", "quality", "balanced", "low"],
        default="auto",
        help=(
            "Planner memory profile. auto picks a safer profile for interactive "
            "superquadric scenes."
        ),
    )
    parser.add_argument(
        "--disable_superquadric_graph",
        action="store_true",
        default=False,
        help=(
            "Disable graph planner in native superquadric mode. By default graph planning "
            "is enabled in superquadric mode to help with far goals where trajopt-only fails."
        ),
    )
    parser.add_argument(
        "--superquadric_graph_seeds",
        type=int,
        default=8,
        help=(
            "Number of graph seeds to allocate for native superquadric mode when graph "
            "planning is enabled."
        ),
    )
    parser.add_argument(
        "--max_superquadrics",
        type=int,
        default=48,
        help=(
            "Maximum number of inferred superquadrics used for collision. "
            "Set <=0 to disable capping."
        ),
    )
    parser.add_argument(
        "--trajopt_tsteps_override",
        type=int,
        default=None,
        help="Optional override for trajopt_tsteps.",
    )
    parser.add_argument(
        "--trajopt_seeds_override",
        type=int,
        default=None,
        help="Optional override for num_trajopt_seeds.",
    )
    parser.add_argument(
        "--ik_seeds_override",
        type=int,
        default=None,
        help="Optional override for num_ik_seeds.",
    )
    parser.add_argument(
        "--interpolation_steps_override",
        type=int,
        default=None,
        help="Optional override for interpolation_steps buffer size.",
    )
    parser.add_argument(
        "--disable_interpolated_collision_check",
        action="store_true",
        default=False,
        help="Disable interpolated trajectory collision evaluation to reduce memory.",
    )
    parser.add_argument(
        "--disable_swept_collision",
        action="store_true",
        default=False,
        help=(
            "Disable swept collision kernels during trajopt/finetune and fall back to "
            "discrete per-timestep collision checks."
        ),
    )
    parser.add_argument(
        "--disable_finetune_trajopt",
        action="store_true",
        default=False,
        help="Disable finetune trajopt pass to reduce peak memory.",
    )
    parser.add_argument(
        "--debug_runtime",
        action="store_true",
        default=False,
        help="Print loop and planner timing diagnostics for interactive debugging.",
    )
    parser.add_argument(
        "--debug_runtime_interval",
        type=float,
        default=0.5,
        help="Seconds between loop heartbeat debug prints when --debug_runtime is enabled.",
    )
    parser.add_argument(
        "--debug_plan_watchdog_secs",
        type=float,
        default=1.5,
        help=(
            "Emit a warning when an async planning request runs longer than this many seconds "
            "while --debug_runtime is enabled."
        ),
    )
    parser.add_argument(
        "--debug_ik_fail_details",
        action="store_true",
        default=False,
        help=(
            "Store IK debug tensors and print extra IK diagnostics when a plan fails."
        ),
    )
    parser.add_argument(
        "--visualize_sdf",
        action="store_true",
        default=False,
        help="Visualize ESDF samples in Isaac Sim for collision debugging.",
    )
    parser.add_argument(
        "--sdf_bbox_center",
        nargs=3,
        type=float,
        default=[0.4, 0.0, 0.45],
        metavar=("x", "y", "z"),
        help="Center of ESDF sampling bounding box in world frame.",
    )
    parser.add_argument(
        "--sdf_bbox_dims",
        nargs=3,
        type=float,
        default=[1.6, 1.6, 1.2],
        metavar=("dx", "dy", "dz"),
        help="Dimensions of ESDF sampling bounding box in meters.",
    )
    parser.add_argument(
        "--sdf_voxel_size",
        type=float,
        default=0.035,
        help="Voxel size used for ESDF sampling.",
    )
    parser.add_argument(
        "--sdf_band",
        type=float,
        default=0.05,
        help="Visualize points with sdf >= -band (meters).",
    )
    parser.add_argument(
        "--sdf_max_points",
        type=int,
        default=3500,
        help="Maximum number of ESDF points to draw per update.",
    )
    parser.add_argument(
        "--sdf_point_size",
        type=float,
        default=0.008,
        help="Rendered size of ESDF debug points.",
    )
    parser.add_argument(
        "--sdf_update_steps",
        type=int,
        default=30,
        help="Update ESDF visualization every N simulation steps.",
    )
    parser.add_argument(
        "--visualize_sdf_gradients",
        action="store_true",
        default=False,
        help="Visualize ESDF gradient vectors (normals) in Isaac Sim.",
    )
    parser.add_argument(
        "--sdf_gradient_max_vectors",
        type=int,
        default=400,
        help="Maximum number of SDF gradient vectors drawn per update.",
    )
    parser.add_argument(
        "--sdf_gradient_scale",
        type=float,
        default=0.08,
        help="Scale factor for rendered SDF gradient vector lengths.",
    )
    parser.add_argument(
        "--sdf_gradient_line_width",
        type=float,
        default=0.0025,
        help="Rendered line width for SDF gradient vectors.",
    )
    parser.add_argument(
        "--sdf_gradient_tip_size",
        type=float,
        default=0.02,
        help="Rendered size for gradient tip point markers.",
    )
    parser.add_argument(
        "--sdf_gradient_shaft_point_size",
        type=float,
        default=0.01,
        help="Rendered size for gradient shaft point markers.",
    )
    parser.add_argument(
        "--sdf_gradient_shaft_points",
        type=int,
        default=6,
        help="Number of point samples used to draw each gradient arrow shaft.",
    )
    # ── Headless / automated debugging ──────────────────────────────────────
    parser.add_argument(
        "--headless",
        action="store_true",
        default=False,
        help="Shorthand for --headless_mode native. Enables headless Isaac Sim with auto-play.",
    )
    parser.add_argument(
        "--max_frames",
        type=int,
        default=0,
        help="Exit after this many simulation steps (0 = run forever).",
    )
    parser.add_argument(
        "--auto_cube_targets",
        type=str,
        default=None,
        help=(
            "JSON array of [x,y,z] cube target positions to cycle through automatically "
            "(e.g. '[[0.5,0,0.5],[0.4,0.3,0.5],[0.4,-0.3,0.5]]'). "
            "Enables automatic planning without manual interaction."
        ),
    )
    parser.add_argument(
        "--auto_target_interval",
        type=int,
        default=120,
        help="Simulation steps between automatic cube target changes (used with --auto_cube_targets).",
    )
    parser.add_argument(
        "--verbose_plan_diag",
        action="store_true",
        default=False,
        help="Print detailed planning diagnostics to stdout on every plan attempt and result.",
    )
    return parser


args = build_argparser().parse_args()
if args.headless and args.headless_mode is None:
    args.headless_mode = "native"


def runtime_debug(message: str) -> None:
    if not args.debug_runtime:
        return
    stamp = time.strftime("%H:%M:%S")
    print(f"[runtime-debug {stamp}] {message}", flush=True)


def vprint(message: str) -> None:
    """Print to stdout only when --verbose_plan_diag is active."""
    if not args.verbose_plan_diag:
        return
    stamp = time.strftime("%H:%M:%S")
    print(f"[diag {stamp}] {message}", flush=True)


def _sdf_debug_color(distance: float, band: float):
    """Map signed distance to RGB color (red=inside, blue=outside)."""
    band = max(float(band), 1.0e-6)
    if distance >= 0.0:
        t = min(distance / band, 1.0)
        return (1.0, 0.8 * (1.0 - t), 0.0)
    t = min(abs(distance) / band, 1.0)
    return (0.1 * (1.0 - t), 0.5 + 0.5 * (1.0 - t), 1.0)


def create_sdf_points_prim(stage):
    """Create a USD points prim used for ESDF visualization."""
    from pxr import UsdGeom

    points = UsdGeom.Points.Define(stage, "/curobo/sdf_debug_points")
    points.CreatePointsAttr().Set([])
    points.CreateWidthsAttr().Set([])
    color_primvar = points.CreateDisplayColorPrimvar(UsdGeom.Tokens.vertex)
    color_primvar.Set([])
    return points


def update_sdf_points_prim(points_prim, motion_gen) -> None:
    """Update ESDF debug points from the current world collision field."""
    from pxr import Gf, Vt

    bbox_dims = [max(float(x), 1.0e-3) for x in args.sdf_bbox_dims]
    sdf_band = max(float(args.sdf_band), 1.0e-6)
    voxel_size = max(float(args.sdf_voxel_size), 1.0e-3)

    query_box = Cuboid(
        name="sdf_debug_box",
        pose=list(args.sdf_bbox_center) + [1.0, 0.0, 0.0, 0.0],
        dims=bbox_dims,
    )
    voxel_grid = motion_gen.world_coll_checker.get_esdf_in_bounding_box(
        cuboid=query_box,
        voxel_size=voxel_size,
        dtype=torch.float32,
    )

    xyz = voxel_grid.xyzr_tensor[:, :3].detach().cpu().numpy()
    sdf = voxel_grid.feature_tensor.detach().cpu().numpy()
    valid_mask = np.isfinite(sdf)
    sdf_mask = sdf >= -sdf_band
    keep = np.logical_and(valid_mask, sdf_mask)

    if not np.any(keep):
        points_prim.GetPointsAttr().Set(Vt.Vec3fArray())
        points_prim.GetWidthsAttr().Set(Vt.FloatArray())
        points_prim.GetDisplayColorPrimvar().Set(Vt.Vec3fArray())
        return

    xyz = xyz[keep]
    sdf = sdf[keep]

    max_points = max(int(args.sdf_max_points), 1)
    if xyz.shape[0] > max_points:
        sample_idx = np.linspace(0, xyz.shape[0] - 1, max_points, dtype=np.int32)
        xyz = xyz[sample_idx]
        sdf = sdf[sample_idx]

    pts = Vt.Vec3fArray([Gf.Vec3f(float(p[0]), float(p[1]), float(p[2])) for p in xyz])
    cols = Vt.Vec3fArray([Gf.Vec3f(*_sdf_debug_color(float(d), sdf_band)) for d in sdf])
    widths = Vt.FloatArray([float(args.sdf_point_size)] * len(xyz))

    points_prim.GetPointsAttr().Set(pts)
    points_prim.GetWidthsAttr().Set(widths)
    points_prim.GetDisplayColorPrimvar().Set(cols)


def _sdf_gradient_color(magnitude: float, max_magnitude: float):
    """Map gradient magnitude to RGB color (cyan=small, yellow=large)."""
    mmax = max(float(max_magnitude), 1.0e-6)
    t = float(np.clip(magnitude / mmax, 0.0, 1.0))
    return (0.0 + 0.9 * t, 0.8 + 0.2 * t, 1.0 - 0.8 * t)


def create_sdf_gradient_prim(stage):
    """Create a USD BasisCurves prim used for SDF gradient visualization."""
    from pxr import UsdGeom

    curves = UsdGeom.BasisCurves.Define(stage, "/curobo/sdf_gradient_vectors")
    curves.CreateTypeAttr().Set(UsdGeom.Tokens.linear)
    curves.CreateCurveVertexCountsAttr().Set([])
    curves.CreatePointsAttr().Set([])
    curves.CreateWidthsAttr().Set([])
    color_primvar = curves.CreateDisplayColorPrimvar(UsdGeom.Tokens.vertex)
    color_primvar.SetInterpolation(UsdGeom.Tokens.vertex)
    color_primvar.Set([])
    return curves


def create_sdf_gradient_tip_prim(stage):
    """Create a USD points prim used as a fallback for gradient tip visualization."""
    from pxr import UsdGeom

    points = UsdGeom.Points.Define(stage, "/curobo/sdf_gradient_tips")
    points.CreatePointsAttr().Set([])
    points.CreateWidthsAttr().Set([])
    color_primvar = points.CreateDisplayColorPrimvar(UsdGeom.Tokens.vertex)
    color_primvar.SetInterpolation(UsdGeom.Tokens.vertex)
    color_primvar.Set([])
    return points


def create_sdf_gradient_shaft_prim(stage):
    """Create a USD points prim used to render gradient arrow shafts as point chains."""
    from pxr import UsdGeom

    points = UsdGeom.Points.Define(stage, "/curobo/sdf_gradient_shafts")
    points.CreatePointsAttr().Set([])
    points.CreateWidthsAttr().Set([])
    color_primvar = points.CreateDisplayColorPrimvar(UsdGeom.Tokens.vertex)
    color_primvar.SetInterpolation(UsdGeom.Tokens.vertex)
    color_primvar.Set([])
    return points


def update_sdf_gradient_prim(
    gradient_prim,
    motion_gen,
    gradient_tip_prim=None,
    gradient_shaft_prim=None,
) -> None:
    """Update ESDF gradient vectors in a bounding box around the robot workspace."""
    from pxr import Gf, Vt

    bbox_dims = [max(float(x), 1.0e-3) for x in args.sdf_bbox_dims]
    voxel_size = max(float(args.sdf_voxel_size), 1.0e-3)
    sdf_band = max(float(args.sdf_band), 1.0e-6)
    max_vectors = max(int(args.sdf_gradient_max_vectors), 1)
    vec_scale = max(float(args.sdf_gradient_scale), 1.0e-6)
    line_width = max(float(args.sdf_gradient_line_width), 1.0e-5)

    query_box = Cuboid(
        name="sdf_gradient_box",
        pose=list(args.sdf_bbox_center) + [1.0, 0.0, 0.0, 0.0],
        dims=bbox_dims,
    )
    voxel_grid = motion_gen.world_coll_checker.get_esdf_in_bounding_box(
        cuboid=query_box,
        voxel_size=voxel_size,
        dtype=torch.float32,
    )

    xyz = voxel_grid.xyzr_tensor[:, :3]
    sdf = voxel_grid.feature_tensor
    valid = torch.isfinite(sdf)
    near_surface = sdf >= -sdf_band
    keep = torch.logical_and(valid, near_surface)
    total_voxels = int(xyz.shape[0])
    kept_voxels = int(torch.count_nonzero(keep).item())

    if kept_voxels > 0:
        sdf_kept = sdf[keep]
        sdf_min = float(torch.min(sdf_kept).item())
        sdf_max = float(torch.max(sdf_kept).item())
        sdf_mean = float(torch.mean(sdf_kept).item())
    else:
        sdf_min = float("nan")
        sdf_max = float("nan")
        sdf_mean = float("nan")

    # print(
    #     "[sdf-grad] "
    #     f"voxels_total={total_voxels} near_surface={kept_voxels} "
    #     f"sdf[min/mean/max]=({sdf_min:.4f}/{sdf_mean:.4f}/{sdf_max:.4f}) "
    #     f"band={sdf_band:.4f}",
    #     flush=True,
    # )

    if int(torch.count_nonzero(keep).item()) == 0:
        gradient_prim.GetCurveVertexCountsAttr().Set(Vt.IntArray())
        gradient_prim.GetPointsAttr().Set(Vt.Vec3fArray())
        gradient_prim.GetWidthsAttr().Set(Vt.FloatArray())
        gradient_prim.GetDisplayColorPrimvar().Set(Vt.Vec3fArray())
        if gradient_tip_prim is not None:
            gradient_tip_prim.GetPointsAttr().Set(Vt.Vec3fArray())
            gradient_tip_prim.GetWidthsAttr().Set(Vt.FloatArray())
            gradient_tip_prim.GetDisplayColorPrimvar().Set(Vt.Vec3fArray())
        if gradient_shaft_prim is not None:
            gradient_shaft_prim.GetPointsAttr().Set(Vt.Vec3fArray())
            gradient_shaft_prim.GetWidthsAttr().Set(Vt.FloatArray())
            gradient_shaft_prim.GetDisplayColorPrimvar().Set(Vt.Vec3fArray())
        return

    xyz = xyz[keep]
    sdf = sdf[keep]

    if xyz.shape[0] > max_vectors:
        sample_idx = torch.linspace(
            0,
            xyz.shape[0] - 1,
            steps=max_vectors,
            device=xyz.device,
            dtype=torch.float32,
        ).round().to(dtype=torch.long)
        xyz = xyz.index_select(0, sample_idx)
        sdf = sdf.index_select(0, sample_idx)

    device = xyz.device
    query = torch.zeros((xyz.shape[0], 1, 1, 4), dtype=torch.float32, device=device)
    query[:, 0, 0, :3] = xyz
    query.requires_grad_(True)

    coll_buffer = CollisionQueryBuffer()
    coll_buffer.update_buffer_shape(
        query.shape,
        motion_gen.world_coll_checker.tensor_args,
        motion_gen.world_coll_checker.collision_types,
    )
    weight = motion_gen.world_coll_checker.tensor_args.to_device([1.0])

    dist = motion_gen.world_coll_checker.get_sphere_distance(
        query,
        coll_buffer,
        weight,
        motion_gen.world_coll_checker.max_distance,
        sum_collisions=False,
        compute_esdf=True,
    )
    grad_vec = None

    # Preferred path: use collision buffer gradients produced by collision kernels.
    try:
        grad_buffer = coll_buffer.get_gradient_buffer()
        if grad_buffer is not None:
            candidate = grad_buffer[:, 0, 0, :3]
            if torch.count_nonzero(torch.isfinite(candidate)).item() > 0:
                grad_vec = candidate.detach()
    except Exception:
        grad_vec = None

    # Fallback path: autograd from ESDF scalar field.
    if grad_vec is None:
        grad = torch.autograd.grad(
            outputs=dist.sum(),
            inputs=query,
            retain_graph=False,
            create_graph=False,
            allow_unused=True,
        )[0]
        if grad is not None:
            grad_vec = grad[:, 0, 0, :3].detach()

    if grad_vec is None:
        gradient_prim.GetCurveVertexCountsAttr().Set(Vt.IntArray())
        gradient_prim.GetPointsAttr().Set(Vt.Vec3fArray())
        gradient_prim.GetWidthsAttr().Set(Vt.FloatArray())
        gradient_prim.GetDisplayColorPrimvar().Set(Vt.Vec3fArray())
        if gradient_tip_prim is not None:
            gradient_tip_prim.GetPointsAttr().Set(Vt.Vec3fArray())
            gradient_tip_prim.GetWidthsAttr().Set(Vt.FloatArray())
            gradient_tip_prim.GetDisplayColorPrimvar().Set(Vt.Vec3fArray())
        if gradient_shaft_prim is not None:
            gradient_shaft_prim.GetPointsAttr().Set(Vt.Vec3fArray())
            gradient_shaft_prim.GetWidthsAttr().Set(Vt.FloatArray())
            gradient_shaft_prim.GetDisplayColorPrimvar().Set(Vt.Vec3fArray())
        carb.log_warn("SDF gradient visualization: no gradients available for sampled ESDF points.")
        return

    grad_mag = torch.linalg.vector_norm(grad_vec, ord=2, dim=1)
    finite_grad = torch.isfinite(grad_mag)
    non_zero = grad_mag > 1.0e-8
    keep_grad = torch.logical_and(finite_grad, non_zero)
    finite_count = int(torch.count_nonzero(finite_grad).item())
    non_zero_count = int(torch.count_nonzero(non_zero).item())
    keep_grad_count = int(torch.count_nonzero(keep_grad).item())

    if finite_count > 0:
        finite_vals = grad_mag[finite_grad]
        gmin = float(torch.min(finite_vals).item())
        gmax = float(torch.max(finite_vals).item())
        gmean = float(torch.mean(finite_vals).item())
    else:
        gmin = float("nan")
        gmax = float("nan")
        gmean = float("nan")

    # print(
    #     "[sdf-grad] "
    #     f"gradients finite={finite_count} non_zero={non_zero_count} rendered={keep_grad_count} "
    #     f"|g|[min/mean/max]=({gmin:.4e}/{gmean:.4e}/{gmax:.4e})",
    #     flush=True,
    # )

    if int(torch.count_nonzero(keep_grad).item()) == 0:
        gradient_prim.GetCurveVertexCountsAttr().Set(Vt.IntArray())
        gradient_prim.GetPointsAttr().Set(Vt.Vec3fArray())
        gradient_prim.GetWidthsAttr().Set(Vt.FloatArray())
        gradient_prim.GetDisplayColorPrimvar().Set(Vt.Vec3fArray())
        if gradient_tip_prim is not None:
            gradient_tip_prim.GetPointsAttr().Set(Vt.Vec3fArray())
            gradient_tip_prim.GetWidthsAttr().Set(Vt.FloatArray())
            gradient_tip_prim.GetDisplayColorPrimvar().Set(Vt.Vec3fArray())
        if gradient_shaft_prim is not None:
            gradient_shaft_prim.GetPointsAttr().Set(Vt.Vec3fArray())
            gradient_shaft_prim.GetWidthsAttr().Set(Vt.FloatArray())
            gradient_shaft_prim.GetDisplayColorPrimvar().Set(Vt.Vec3fArray())
        # print("[sdf-grad] no renderable gradient vectors after filtering", flush=True)
        return

    xyz = xyz[keep_grad].detach().cpu().numpy()
    grad_vec = grad_vec[keep_grad].cpu().numpy()
    grad_mag = grad_mag[keep_grad].cpu().numpy()
    max_mag = float(np.max(grad_mag)) if grad_mag.size > 0 else 1.0

    grad_dir = grad_vec / np.maximum(grad_mag[:, None], 1.0e-8)
    # ESDF returned here is positive inside obstacles and negative outside;
    # use -grad to point toward increasing free space.
    grad_dir = -grad_dir
    grad_norm = grad_mag / max(max_mag, 1.0e-8)
    # Keep a visible minimum vector length so near-flat regions are still rendered.
    vec_len = vec_scale * (0.25 + 0.75 * grad_norm)
    endpoints = xyz + grad_dir * vec_len[:, None]

    preview_n = min(3, xyz.shape[0])
    # for i in range(preview_n):
    #     print(
    #         "[sdf-grad] "
    #         f"sample[{i}] p=({xyz[i][0]:.3f},{xyz[i][1]:.3f},{xyz[i][2]:.3f}) "
    #         f"g=({grad_vec[i][0]:.3e},{grad_vec[i][1]:.3e},{grad_vec[i][2]:.3e}) "
    #         f"|g|={grad_mag[i]:.3e} len={vec_len[i]:.3e}",
    #         flush=True,
    #     )

    points = []
    colors = []
    curve_counts = []
    widths = []
    for i in range(xyz.shape[0]):
        p0 = xyz[i]
        p1 = endpoints[i]
        color = _sdf_gradient_color(float(grad_mag[i]), max_mag)

        points.append(Gf.Vec3f(float(p0[0]), float(p0[1]), float(p0[2])))
        points.append(Gf.Vec3f(float(p1[0]), float(p1[1]), float(p1[2])))
        colors.append(Gf.Vec3f(*color))
        colors.append(Gf.Vec3f(*color))
        curve_counts.append(2)
        widths.append(line_width)

    gradient_prim.GetCurveVertexCountsAttr().Set(Vt.IntArray(curve_counts))
    gradient_prim.GetPointsAttr().Set(Vt.Vec3fArray(points))
    gradient_prim.GetWidthsAttr().Set(Vt.FloatArray(widths))
    gradient_prim.GetDisplayColorPrimvar().Set(Vt.Vec3fArray(colors))

    if gradient_tip_prim is not None:
        tip_points = [Gf.Vec3f(float(p[0]), float(p[1]), float(p[2])) for p in endpoints]
        tip_colors = [Gf.Vec3f(0.0, 1.0, 0.2) for _ in endpoints]
        tip_max_size = float(max(args.sdf_gradient_tip_size, 1.0e-4))
        tip_min_size = 0.2 * tip_max_size
        tip_widths = [
            float(tip_min_size + (tip_max_size - tip_min_size) * float(gn))
            for gn in grad_norm
        ]
        gradient_tip_prim.GetPointsAttr().Set(Vt.Vec3fArray(tip_points))
        gradient_tip_prim.GetWidthsAttr().Set(Vt.FloatArray(tip_widths))
        gradient_tip_prim.GetDisplayColorPrimvar().Set(Vt.Vec3fArray(tip_colors))

    if gradient_shaft_prim is not None:
        shaft_pts = []
        shaft_cols = []
        shaft_w = []
        shaft_point_count = max(int(args.sdf_gradient_shaft_points), 2)
        shaft_size = float(max(args.sdf_gradient_shaft_point_size, 1.0e-4))
        for i in range(xyz.shape[0]):
            color = _sdf_gradient_color(float(grad_mag[i]), max_mag)
            p0 = xyz[i]
            p1 = endpoints[i]
            for k in range(shaft_point_count):
                t = k / float(shaft_point_count - 1)
                pp = p0 * (1.0 - t) + p1 * t
                shaft_pts.append(Gf.Vec3f(float(pp[0]), float(pp[1]), float(pp[2])))
                shaft_cols.append(Gf.Vec3f(*color))
                shaft_w.append(shaft_size)
        gradient_shaft_prim.GetPointsAttr().Set(Vt.Vec3fArray(shaft_pts))
        gradient_shaft_prim.GetWidthsAttr().Set(Vt.FloatArray(shaft_w))
        gradient_shaft_prim.GetDisplayColorPrimvar().Set(Vt.Vec3fArray(shaft_cols))


def quat_geodesic_distance_degrees(q_a, q_b) -> float:
    """Return shortest-angle quaternion distance in degrees."""
    q_a = np.asarray(q_a, dtype=np.float64)
    q_b = np.asarray(q_b, dtype=np.float64)
    q_a_norm = np.linalg.norm(q_a)
    q_b_norm = np.linalg.norm(q_b)
    if q_a_norm < 1e-8 or q_b_norm < 1e-8:
        return float("nan")
    q_a = q_a / q_a_norm
    q_b = q_b / q_b_norm
    dot = float(np.clip(np.abs(np.dot(q_a, q_b)), 0.0, 1.0))
    return float(np.degrees(2.0 * np.arccos(dot)))


def _safe_float(value, default: float = float("nan")) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _format_grad_tensor_stats(name: str, tensor: torch.Tensor) -> str:
    """Return concise gradient magnitude statistics for a tensor."""
    t = tensor.detach()
    if t.numel() == 0:
        return f"{name}: empty"

    t_f = t.to(dtype=torch.float32)
    flat = t_f.reshape(-1)
    abs_max = float(torch.max(torch.abs(flat)).item())

    if t_f.ndim == 0:
        l2_mean = float(torch.abs(t_f).item())
        l2_max = l2_mean
    elif t_f.shape[-1] > 1:
        vec = t_f.reshape(-1, t_f.shape[-1])
        l2 = torch.linalg.vector_norm(vec, ord=2, dim=1)
        l2_mean = float(torch.mean(l2).item())
        l2_max = float(torch.max(l2).item())
    else:
        l2 = torch.abs(flat)
        l2_mean = float(torch.mean(l2).item())
        l2_max = float(torch.max(l2).item())

    return (
        f"{name}: shape={tuple(t.shape)} "
        f"l2_mean={l2_mean:.3e} l2_max={l2_max:.3e} abs_max={abs_max:.3e}"
    )


def _log_gradient_magnitudes(plan_id: int, label: str, payload) -> None:
    """Log gradient tensor magnitudes found in planner debug payloads."""
    grad_summaries = []

    def add_summary(name: str, value) -> None:
        if isinstance(value, torch.Tensor):
            grad_summaries.append(_format_grad_tensor_stats(name, value))
        elif isinstance(value, (list, tuple)):
            for idx, item in enumerate(value):
                if isinstance(item, torch.Tensor):
                    grad_summaries.append(_format_grad_tensor_stats(f"{name}[{idx}]", item))

    if isinstance(payload, dict):
        for key, value in payload.items():
            if "grad" in str(key).lower():
                add_summary(str(key), value)

    for attr in dir(payload):
        if attr.startswith("_") or "grad" not in attr.lower():
            continue
        try:
            value = getattr(payload, attr)
        except Exception:
            continue
        if callable(value):
            continue
        add_summary(attr, value)

    if not grad_summaries:
        carb.log_warn(f"plan#{plan_id} {label} gradient magnitudes: unavailable")
        return

    for summary in grad_summaries:
        carb.log_warn(f"plan#{plan_id} {label} gradient magnitudes {summary}")


def _format_plan_status(status, success: bool | None = None) -> str:
    """Render planner status with a stable fallback when status is missing."""
    if isinstance(status, MotionGenStatus):
        return f"{status.name} ({status.value})"
    if status is None:
        if success is False:
            return "UNKNOWN_FAILURE"
        if success is True:
            return "SUCCESS"
        return "NOT_ATTEMPTED"
    return str(status)


def _extract_ee_pose_tensors(kin_state):
    """Return end-effector position and quaternion tensors across CuRobo versions."""
    ee_pos = getattr(kin_state, "ee_pos_seq", None)
    ee_quat = getattr(kin_state, "ee_quat_seq", None)

    if ee_pos is None:
        ee_pos = getattr(kin_state, "ee_position", None)
    if ee_quat is None:
        ee_quat = getattr(kin_state, "ee_quaternion", None)

    if (ee_pos is None or ee_quat is None) and hasattr(kin_state, "ee_pose"):
        ee_pose = getattr(kin_state, "ee_pose")
        if ee_pos is None:
            ee_pos = getattr(ee_pose, "position", None)
        if ee_quat is None:
            ee_quat = getattr(ee_pose, "quaternion", None)

    if ee_pos is None or ee_quat is None:
        raise AttributeError(
            "KinematicModelState did not expose ee pose tensors; "
            "expected one of (ee_pos_seq, ee_quat_seq), "
            "(ee_position, ee_quaternion), or ee_pose.{position,quaternion}."
        )
    return ee_pos, ee_quat


def log_plan_failure_diagnostics(
    motion_gen,
    plan_id: int,
    start_js,
    goal_position,
    goal_orientation,
    plan_result,
) -> None:
    """Print lightweight diagnostics for failed plan requests."""
    try:
        fk_state = start_js if len(start_js.position.shape) > 1 else start_js.unsqueeze(0)
        kin_state = motion_gen.compute_kinematics(fk_state)
        ee_pos_tensor, ee_quat_tensor = _extract_ee_pose_tensors(kin_state)
        ee_position = np.asarray(
            ee_pos_tensor.detach().view(-1)[0:3].cpu().numpy(),
            dtype=np.float64,
        )
        ee_orientation = np.asarray(
            ee_quat_tensor.detach().view(-1)[0:4].cpu().numpy(),
            dtype=np.float64,
        )
        goal_position = np.asarray(goal_position, dtype=np.float64)
        goal_orientation = np.asarray(goal_orientation, dtype=np.float64)
        pos_err = float(np.linalg.norm(goal_position - ee_position))
        rot_err_deg = quat_geodesic_distance_degrees(goal_orientation, ee_orientation)
    except Exception as exc:
        carb.log_warn(f"plan#{plan_id} diagnostics skipped; FK failed: {exc}")
        return

    try:
        start_valid, start_status = motion_gen.check_start_state(start_js)
    except Exception as exc:
        start_valid, start_status = False, f"check_start_state_error: {exc}"

    status = getattr(plan_result, "status", None)
    status_label = _format_plan_status(status, success=False)
    attempts = _safe_float(getattr(plan_result, "attempts", float("nan")))
    total_time = _safe_float(getattr(plan_result, "total_time", float("nan")))
    ik_time = _safe_float(getattr(plan_result, "ik_time", float("nan")))
    trajopt_time = _safe_float(getattr(plan_result, "trajopt_time", float("nan")))
    graph_time = _safe_float(getattr(plan_result, "graph_time", float("nan")))
    carb.log_warn(
        f"plan#{plan_id} failure status={status_label} attempts={attempts:.0f} "
        f"time(total/ik/graph/trajopt)=({total_time:.3f}/{ik_time:.3f}/{graph_time:.3f}/{trajopt_time:.3f})s "
        f"start_valid={start_valid} start_status={start_status} "
        f"ee_pos_err={pos_err:.4f}m ee_rot_err={rot_err_deg:.2f}deg"
    )

    debug_info = getattr(plan_result, "debug_info", None)
    ik_result = debug_info.get("ik_result") if isinstance(debug_info, dict) else None
    if ik_result is not None:
        try:
            _log_gradient_magnitudes(plan_id, "ik", ik_result)
            success_count = int(torch.count_nonzero(ik_result.success).item())
            total_count = int(ik_result.success.numel())
            min_pos_err = float(torch.min(ik_result.position_error).item())
            min_rot_err = float(torch.min(ik_result.rotation_error).item())
            carb.log_warn(
                f"plan#{plan_id} IK seed stats success={success_count}/{total_count} "
                f"best_pos_err={min_pos_err:.5f}m best_rot_err={min_rot_err:.5f}rad"
            )

            # Diagnose whether IK failed due feasibility (collision/limits) despite low pose error.
            if hasattr(ik_result, "solution") and ik_result.solution is not None:
                flat_solution = ik_result.solution.detach().view(-1, ik_result.solution.shape[-1])
                if flat_solution.numel() > 0:
                    flat_error = (ik_result.position_error + ik_result.rotation_error).detach().view(-1)
                    best_seed_idx = int(torch.argmin(flat_error).item())
                    best_q = flat_solution[best_seed_idx].view(1, -1)
                    best_js = JointState.from_position(
                        best_q,
                        joint_names=motion_gen.kinematics.joint_names,
                    )
                    best_valid, best_status = motion_gen.check_start_state(best_js)
                    carb.log_warn(
                        f"plan#{plan_id} best IK candidate validity={best_valid} status={best_status}"
                    )

                    if (not best_valid) and best_status == MotionGenStatus.INVALID_START_STATE_WORLD_COLLISION:
                        carb.log_warn(
                            "IK converged in pose space but candidate violates world-collision constraints. "
                            "This usually means the collision world is too conservative near the arm/goal "
                            "or the scene transform places obstacles into the robot workspace."
                        )
        except Exception as exc:
            carb.log_warn(f"plan#{plan_id} IK debug stats unavailable: {exc}")

    trajopt_result = debug_info.get("trajopt_result") if isinstance(debug_info, dict) else None
    if trajopt_result is not None:
        try:
            _log_gradient_magnitudes(plan_id, "trajopt", trajopt_result)
            traj_success_count = int(torch.count_nonzero(trajopt_result.success).item())
            traj_total_count = int(trajopt_result.success.numel())
            traj_msg = (
                f"plan#{plan_id} trajopt seed stats success={traj_success_count}/{traj_total_count}"
            )
            if getattr(trajopt_result, "position_error", None) is not None:
                traj_best_pos = float(torch.min(trajopt_result.position_error).item())
                traj_msg += f" best_pos_err={traj_best_pos:.5f}m"
            if getattr(trajopt_result, "rotation_error", None) is not None:
                traj_best_rot = float(torch.min(trajopt_result.rotation_error).item())
                traj_msg += f" best_rot_err={traj_best_rot:.5f}rad"
            carb.log_warn(traj_msg)
        except Exception as exc:
            carb.log_warn(f"plan#{plan_id} trajopt debug stats unavailable: {exc}")

    if status == MotionGenStatus.IK_FAIL and pos_err < 0.08 and rot_err_deg > 30.0:
        carb.log_warn(
            "IK_FAIL hint: target position is near current EE pose but orientation error is large. "
            "Try rotating the cube orientation or using --reach_partial_pose for position-priority tasks."
        )

    if (
        status == MotionGenStatus.IK_FAIL
        and not start_valid
        and start_status == MotionGenStatus.INVALID_START_STATE_WORLD_COLLISION
    ):
        carb.log_warn(
            "IK_FAIL hint: start state is already in world collision. "
            "For superquadrics, verify --superquadric_translation/--superquadric_scale and try "
            "--world_representation mesh to compare collision conservativeness."
        )

    if status in [MotionGenStatus.TRAJOPT_FAIL, MotionGenStatus.FINETUNE_TRAJOPT_FAIL]:
        if pos_err < 0.08 and rot_err_deg > 30.0:
            carb.log_warn(
                "TRAJOPT_FAIL hint: position can be reached but orientation is difficult under "
                "current constraints. Try rotating the target cube or use --reach_partial_pose "
                "for position-priority tasks."
            )
        if pos_err > 0.30 and start_valid and not bool(getattr(plan_result, "used_graph", False)):
            carb.log_warn(
                "TRAJOPT_FAIL hint: goal is far from current EE pose and this solve did not "
                "use graph planning. Enable graph seeds (superquadrics default) or move the "
                "target incrementally; --world_representation mesh can be used as a baseline."
            )


from omni.isaac.kit import SimulationApp


simulation_app = SimulationApp(
    {
        "headless": args.headless_mode is not None,
        "width": "1920",
        "height": "1080",
    }
)

import carb
from helper import add_extensions, add_robot_to_scene
from omni.isaac.core import World
from omni.isaac.core.objects import cuboid, sphere
from omni.isaac.core.utils.types import ArticulationAction

from curobo.curobolib.geom import geom_cu
from curobo.geom.sdf.world import CollisionCheckerType, CollisionQueryBuffer
from curobo.geom.types import Cuboid, Mesh, Superquadric, WorldConfig
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
    MotionGenStatus,
    PoseCostMetric,
)


def maybe_limit_superquadrics(superquadrics: List[Superquadric]) -> List[Superquadric]:
    max_superquadrics = args.max_superquadrics
    if max_superquadrics is None or max_superquadrics <= 0:
        return superquadrics
    if len(superquadrics) <= max_superquadrics:
        return superquadrics

    # Keep the largest primitives by volume because they usually capture the
    # dominant collision geometry while bounding planner memory usage.
    volumes = np.asarray(
        [
            float(np.prod(np.maximum(np.asarray(obstacle.radii, dtype=np.float32), 1.0e-6)))
            for obstacle in superquadrics
        ],
        dtype=np.float32,
    )
    keep_indices = np.argsort(volumes)[-max_superquadrics:]
    keep_indices = np.sort(keep_indices)
    limited = [superquadrics[index] for index in keep_indices]
    carb.log_warn(
        "Capping inferred superquadrics from "
        f"{len(superquadrics)} to {len(limited)} based on --max_superquadrics={max_superquadrics}."
    )
    return limited


def ensure_superquadric_support() -> None:
    if not hasattr(geom_cu, "closest_point_superquadric"):
        raise RuntimeError(
            "CuRobo was built without native superquadric support. Rebuild with OpenGJK "
            "available at openGJK/gpu, or run this example with --world_representation mesh."
        )


def import_superdec_modules(superdec_root: Path):
    if str(superdec_root) not in sys.path:
        sys.path.append(str(superdec_root))

    import open3d as o3d
    from omegaconf import OmegaConf
    from superdec.data.dataloader import denormalize_outdict, denormalize_points, normalize_points
    from superdec.superdec import SuperDec
    from superdec.utils.predictions_handler import PredictionHandler
    from superdec.utils.transforms import mat2quat

    return {
        "OmegaConf": OmegaConf,
        "PredictionHandler": PredictionHandler,
        "SuperDec": SuperDec,
        "denormalize_outdict": denormalize_outdict,
        "denormalize_points": denormalize_points,
        "mat2quat": mat2quat,
        "normalize_points": normalize_points,
        "o3d": o3d,
    }


def infer_superquadric_world() -> WorldConfig:
    superdec_root = Path(args.superdec_root).expanduser().resolve()
    checkpoint_folder = Path(args.checkpoint_folder).expanduser().resolve()
    ply_path = Path(args.ply_path).expanduser().resolve()

    if not superdec_root.exists():
        raise FileNotFoundError(f"SuperDec root was not found: {superdec_root}")
    if not checkpoint_folder.exists():
        raise FileNotFoundError(f"Checkpoint folder was not found: {checkpoint_folder}")
    if not ply_path.exists():
        raise FileNotFoundError(f"PLY file was not found: {ply_path}")

    modules = import_superdec_modules(superdec_root)
    OmegaConf = modules["OmegaConf"]
    PredictionHandler = modules["PredictionHandler"]
    SuperDec = modules["SuperDec"]
    denormalize_outdict = modules["denormalize_outdict"]
    denormalize_points = modules["denormalize_points"]
    mat2quat = modules["mat2quat"]
    normalize_points = modules["normalize_points"]
    o3d = modules["o3d"]

    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    checkpoint = torch.load(
        checkpoint_folder / "ckpt.pt",
        map_location=device,
        weights_only=False,
    )
    configs = OmegaConf.load(checkpoint_folder / "config.yaml")
    model = SuperDec(configs.superdec).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    point_cloud = o3d.io.read_point_cloud(str(ply_path))
    points_raw = np.asarray(point_cloud.points)
    if points_raw.size == 0:
        raise RuntimeError(f"No points were loaded from {ply_path}")

    sample_count = min(4096, len(points_raw))
    sample_indices = rng.choice(len(points_raw), 4096, replace=len(points_raw) < 4096)
    points = points_raw[sample_indices]
    points, translation, scale = normalize_points(points)
    points_tensor = torch.from_numpy(points).unsqueeze(0).to(device).float()

    with torch.no_grad():
        outdict = model(points_tensor)
        for key, value in outdict.items():
            if isinstance(value, torch.Tensor):
                outdict[key] = value.cpu()
        outdict = denormalize_outdict(
            outdict,
            np.array([translation]),
            np.array([scale]),
            False,
        )
        points_tensor = denormalize_points(
            points_tensor.cpu(),
            np.array([translation]),
            np.array([scale]),
            False,
        )

    predictions = PredictionHandler.from_outdict(outdict, points_tensor, [ply_path.stem])
    quaternions = mat2quat(torch.from_numpy(predictions.rotation).float()).cpu().numpy()

    eps_raw_all = []
    eps_clamped_all = []
    radii_all = []
    clamp_count = 0
    radius_clamp_count = 0
    raw_radius_max_seen = 0.0
    superquadrics = []
    for index in range(predictions.scale.shape[1]):
        if predictions.exist[0, index] <= 0.5:
            continue

        translation_vec = np.asarray(predictions.translation[0, index], dtype=np.float32)
        quaternion = np.asarray(quaternions[0, index], dtype=np.float32)
        color = (np.asarray(predictions.colors[index], dtype=np.float32) / 255.0).tolist()
        radii = np.asarray(predictions.scale[0, index], dtype=np.float32)
        raw_eps = np.asarray(predictions.exponents[0, index], dtype=np.float32)
        eps = np.clip(raw_eps, args.superquadric_min_eps, args.superquadric_max_eps)
        if np.any(np.abs(eps - raw_eps) > 1e-6):
            clamp_count += 1

        max_r = float(args.superquadric_max_radius)
        raw_max_radius = float(np.max(radii))
        if max_r > 0.0 and raw_max_radius > max_r:
            raw_radius_max_seen = max(raw_radius_max_seen, raw_max_radius)
            radii = np.minimum(radii, max_r)
            radius_clamp_count += 1

        eps_raw_all.append(raw_eps)
        eps_clamped_all.append(eps)
        radii_all.append(radii)

        superquadrics.append(
            Superquadric(
                name=f"sq_{index}",
                pose=translation_vec.tolist() + quaternion.tolist(),
                radii=radii.tolist(),
                eps=eps.tolist(),
                color=color,
            )
        )

    total_superquadrics = len(superquadrics)
    if not superquadrics:
        raise RuntimeError(
            f"SuperDec did not return any active primitives from {ply_path}. "
            f"The model sampled {sample_count} input points."
        )

    superquadrics = maybe_limit_superquadrics(superquadrics)

    radii_np = np.asarray(radii_all, dtype=np.float32)
    if radius_clamp_count > 0:
        print(
            f"Clamped {radius_clamp_count} superquadric(s) with radii exceeding "
            f"--superquadric_max_radius={args.superquadric_max_radius:.3f}m "
            f"(max raw radius was {raw_radius_max_seen:.4f}m). "
            "If this is unexpected, check your PLY scale or adjust --superquadric_scale."
        )

    if args.print_superquadric_stats:
        eps_raw_np = np.asarray(eps_raw_all, dtype=np.float32)
        eps_clamped_np = np.asarray(eps_clamped_all, dtype=np.float32)
        print(
            "Superquadric stats:",
            "count_inferred=", total_superquadrics,
            "count_used=", len(superquadrics),
            "raw_eps[min,max]=",
            (float(np.min(eps_raw_np)), float(np.max(eps_raw_np))),
            "clamped_eps[min,max]=",
            (float(np.min(eps_clamped_np)), float(np.max(eps_clamped_np))),
            "radii[min,max]=",
            (float(np.min(radii_np)), float(np.max(radii_np))),
            "eps_clamped_primitives=", clamp_count,
            "radius_clamped_primitives=", radius_clamp_count,
        )

    return WorldConfig(superquadric=superquadrics)


MIN_RADIUS = 0.01  # 1cm minimum — below this the SDF kernel overflows

def apply_scene_transform(superquadric_world: WorldConfig) -> WorldConfig:
    placement_pose = Pose.from_list(
        list(args.superquadric_translation) + list(args.superquadric_orientation)
    )
    transformed_superquadrics = []

    for obstacle in superquadric_world.superquadric:
        obstacle_copy = copy.deepcopy(obstacle)
        obstacle_copy.radii = [args.superquadric_scale * r for r in obstacle_copy.radii]
        obstacle_copy.pose[:3] = [args.superquadric_scale * v for v in obstacle_copy.pose[:3]]

        # Clamp degenerate radii before they reach the SDF kernel.
        # Values near zero cause powf(ax, p2) overflow → NaN → false collision.
        obstacle_copy.radii = [max(r, MIN_RADIUS) for r in obstacle_copy.radii]

        obstacle_pose = Pose.from_list(obstacle_copy.pose)
        obstacle_copy.pose = placement_pose.multiply(obstacle_pose).tolist()
        transformed_superquadrics.append(obstacle_copy)

    return WorldConfig(superquadric=transformed_superquadrics)


def apply_superquadric_collision_tolerance(
    superquadric_world: WorldConfig,
    tolerance: float | None = None,
) -> WorldConfig:
    """Shrink superquadric radii used for collision to reduce over-conservative contacts."""
    tol_value = args.superquadric_collision_tolerance if tolerance is None else tolerance
    tol = max(float(tol_value), 0.0)
    if tol <= 0.0:
        return superquadric_world

    adjusted = []
    adjusted_count = 0
    for obstacle in superquadric_world.superquadric:
        obstacle_copy = copy.deepcopy(obstacle)
        new_radii = []
        for radius in obstacle_copy.radii:
            radius_f = float(radius)
            shrunk = max(radius_f - tol, MIN_RADIUS)
            if shrunk < radius_f:
                adjusted_count += 1
            new_radii.append(shrunk)
        obstacle_copy.radii = new_radii
        adjusted.append(obstacle_copy)

    carb.log_warn(
        "Applying superquadric collision tolerance "
        f"{tol:.4f}m (radius shrink) to reduce false-positive collisions; "
        f"adjusted {adjusted_count} radii."
    )
    return WorldConfig(superquadric=adjusted)


def apply_superquadric_collision_eps_blend(
    superquadric_world: WorldConfig,
    eps_blend: float,
) -> WorldConfig:
    """Blend SQ exponents toward ellipsoids (eps=1) for collision robustness."""
    blend = float(np.clip(eps_blend, 0.0, 1.0))
    if blend <= 0.0:
        return superquadric_world

    adjusted = []
    changed_count = 0
    for obstacle in superquadric_world.superquadric:
        obstacle_copy = copy.deepcopy(obstacle)
        old_eps = [float(obstacle_copy.eps[0]), float(obstacle_copy.eps[1])]
        new_eps = [
            float(np.clip((1.0 - blend) * old_eps[0] + blend * 1.0, 0.2, 1.0)),
            float(np.clip((1.0 - blend) * old_eps[1] + blend * 1.0, 0.2, 1.0)),
        ]
        if abs(new_eps[0] - old_eps[0]) > 1.0e-6 or abs(new_eps[1] - old_eps[1]) > 1.0e-6:
            changed_count += 1
        obstacle_copy.eps = new_eps
        adjusted.append(obstacle_copy)

    if changed_count > 0:
        carb.log_warn(
            "Applying collision-only superquadric eps blend toward ellipsoids "
            f"(blend={blend:.2f}); adjusted {changed_count} primitives."
        )
    return WorldConfig(superquadric=adjusted)


def build_superquadric_mesh(superquadric: Superquadric, resolution: int) -> Mesh:
    resolution = max(resolution, 8)
    eta = np.linspace(-np.pi / 2.0, np.pi / 2.0, resolution, endpoint=True)
    omega = np.linspace(-np.pi, np.pi, resolution, endpoint=True)
    eta_grid, omega_grid = np.meshgrid(eta, omega, indexing="ij")

    def signed_power(values: np.ndarray, power: float) -> np.ndarray:
        return np.sign(values) * np.power(np.abs(values), power)

    x = (
        superquadric.radii[0]
        * signed_power(np.cos(eta_grid), superquadric.eps[0])
        * signed_power(np.cos(omega_grid), superquadric.eps[1])
    )
    y = (
        superquadric.radii[1]
        * signed_power(np.cos(eta_grid), superquadric.eps[0])
        * signed_power(np.sin(omega_grid), superquadric.eps[1])
    )
    z = superquadric.radii[2] * signed_power(np.sin(eta_grid), superquadric.eps[0])

    vertices = np.stack((x, y, z), axis=-1).reshape(-1, 3)
    vertices[:resolution, 0] = 0.0
    vertices[-resolution:, 0] = 0.0

    faces = []
    for row in range(resolution - 1):
        for col in range(resolution - 1):
            current_idx = row * resolution + col
            faces.append([current_idx, current_idx + 1, current_idx + resolution])
            faces.append([current_idx + resolution, current_idx + 1, current_idx + resolution + 1])
        faces.append([
            row * resolution + (resolution - 1),
            row * resolution,
            (row + 1) * resolution + (resolution - 1),
        ])
        faces.append([
            (row + 1) * resolution + (resolution - 1),
            row * resolution,
            (row + 1) * resolution,
        ])

    faces.append([(resolution - 1) * resolution + (resolution - 1), (resolution - 1) * resolution, resolution - 1])
    faces.append([resolution - 1, (resolution - 1) * resolution, 0])

    color = superquadric.color
    if len(color) == 3:
        color = color + [1.0]

    return Mesh(
        name=f"{superquadric.name}_mesh",
        vertices=vertices,
        faces=np.asarray(faces, dtype=np.int32),
        pose=superquadric.pose,
        color=color,
    )


def build_collision_and_visual_worlds() -> tuple[WorldConfig, WorldConfig, WorldConfig]:
    table_world = WorldConfig.from_dict(
        load_yaml(join_path(get_world_configs_path(), "collision_table.yml"))
    )
    table_world.cuboid[0].pose[2] -= 0.02

    superquadric_world = apply_scene_transform(infer_superquadric_world())
    mesh_obstacles = [
        build_superquadric_mesh(obstacle, args.surface_resolution)
        for obstacle in superquadric_world.superquadric
    ]
    visual_world = WorldConfig(cuboid=[], mesh=mesh_obstacles)

    collision_superquadric_world = apply_superquadric_collision_tolerance(superquadric_world)

    if args.world_representation == "superquadrics":
        collision_world = WorldConfig(
            cuboid=[],
            superquadric=collision_superquadric_world.superquadric,
        )
    else:
        collision_world = WorldConfig(cuboid=[], mesh=mesh_obstacles)

    print(
        "Loaded",
        len(superquadric_world.superquadric),
        "SuperDec primitives with collision mode",
        args.world_representation,
    )
    return collision_world, visual_world, superquadric_world


def build_motion_gen(robot_cfg, collision_world: WorldConfig, tensor_args: TensorDeviceType) -> MotionGen:
    trajopt_dt = None
    optimize_dt = True
    trajopt_tsteps = 32
    trim_steps = None
    interpolation_dt = 0.05
    if args.reactive:
        trajopt_tsteps = 40
        trajopt_dt = 0.04
        optimize_dt = False
        trim_steps = [1, None]
        interpolation_dt = trajopt_dt

    # The OpenGJK superquadric kernel does not support CUDA stream capture.
    # We still allow graph planning in native SQ mode by disabling CUDA graph
    # capture globally for IK/trajopt/graph rollouts (use_cuda_graph=False).
    has_superquadrics = bool(collision_world.superquadric)
    superquadric_count = len(collision_world.superquadric) if has_superquadrics else 0
    interactive_mode = args.headless_mode is None
    checker_type = (
        CollisionCheckerType.PRIMITIVE
        if has_superquadrics and len(collision_world.mesh) == 0
        else CollisionCheckerType.MESH
    )
    enable_superquadric_graph = (
        has_superquadrics and (not args.disable_superquadric_graph) and (not args.trajopt_only)
    )
    if has_superquadrics:
        num_graph_seeds = max(int(args.superquadric_graph_seeds), 1) if enable_superquadric_graph else 0
    else:
        num_graph_seeds = 0 if args.trajopt_only else 12

    memory_profile = args.planner_memory_profile
    if memory_profile == "auto":
        if has_superquadrics and interactive_mode:
            memory_profile = "balanced"   # was "low"; balanced gives 2x seeds vs low
        else:
            memory_profile = "quality"

    collision_cache = {
        "obb": len(collision_world.cuboid),
    }
    if checker_type == CollisionCheckerType.MESH:
        collision_cache["mesh"] = len(collision_world.mesh)
    if has_superquadrics:
        collision_cache["superquadric"] = len(collision_world.superquadric)

    # Superquadric trajopt benefits from more seeds; balanced profile caps to 2/4 of these.
    num_trajopt_seeds = 8 if has_superquadrics else 12
    num_ik_seeds = 16 if has_superquadrics else 32
    num_batch_ik_seeds = 16 if has_superquadrics else 32
    interpolation_steps = 5000
    evaluate_interpolated_trajectory = True

    if memory_profile == "balanced":
        num_ik_seeds = min(num_ik_seeds, 4)
        num_batch_ik_seeds = min(num_batch_ik_seeds, 4)
        num_trajopt_seeds = min(num_trajopt_seeds, 2)
        trajopt_tsteps = min(trajopt_tsteps, 24)
        interpolation_steps = min(interpolation_steps, 1200)
    elif memory_profile == "low":
        num_ik_seeds = min(num_ik_seeds, 2)
        num_batch_ik_seeds = min(num_batch_ik_seeds, 2)
        num_trajopt_seeds = min(num_trajopt_seeds, 1)
        trajopt_tsteps = min(trajopt_tsteps, 18)
        optimize_dt = False
        interpolation_steps = min(interpolation_steps, 400)
        evaluate_interpolated_trajectory = False
        if trajopt_dt is None:
            trajopt_dt = 0.04

    if args.trajopt_tsteps_override is not None:
        trajopt_tsteps = max(args.trajopt_tsteps_override, 8)
    if args.trajopt_seeds_override is not None:
        num_trajopt_seeds = max(args.trajopt_seeds_override, 1)
    if args.ik_seeds_override is not None:
        num_ik_seeds = max(args.ik_seeds_override, 1)
        num_batch_ik_seeds = max(args.ik_seeds_override, 1)
    if args.interpolation_steps_override is not None:
        interpolation_steps = max(args.interpolation_steps_override, 64)
    if args.disable_interpolated_collision_check:
        evaluate_interpolated_trajectory = False

    gradient_trajopt_file = "gradient_trajopt.yml"
    finetune_trajopt_file = None
    if args.disable_swept_collision:
        gradient_trajopt_file = "gradient_trajopt_no_sweep.yml"
        finetune_trajopt_file = "finetune_trajopt_no_sweep.yml"
        carb.log_warn(
            "Swept collision disabled: using discrete collision configs "
            f"({gradient_trajopt_file}, {finetune_trajopt_file})."
        )

    print(
        "Planner config:",
        "profile=",
        memory_profile,
        "trajopt_tsteps=",
        trajopt_tsteps,
        "num_trajopt_seeds=",
        num_trajopt_seeds,
        "num_graph_seeds=",
        num_graph_seeds,
        "graph_enabled=",
        (num_graph_seeds > 0),
        "num_ik_seeds=",
        num_ik_seeds,
        "checker_type=",
        checker_type,
        "interpolation_steps=",
        interpolation_steps,
        "eval_interp=",
        evaluate_interpolated_trajectory,
        "swept_collision=",
        (not args.disable_swept_collision),
        "superquadrics=",
        superquadric_count,
        "trajopt_only=",
        args.trajopt_only,
    )

    # The OpenGJK superquadric kernel does not support CUDA stream capture.
    # CUDA graphs are used not only by the graph planner but also internally by
    # the IK and trajopt particle optimisers (particle_opt_base._initialize_cuda_graph).
    # Setting use_cuda_graph=False disables graph capture across all solvers so
    # the superquadric kernel is called eagerly rather than inside a CUDA graph.
    
    use_cuda_graph = not has_superquadrics
    # use_cuda_graph = True

    motion_gen_config = MotionGenConfig.load_from_robot_config(
        robot_cfg,
        collision_world,
        tensor_args,
        collision_checker_type=checker_type,
        num_ik_seeds=num_ik_seeds,
        num_trajopt_seeds=num_trajopt_seeds,
        num_batch_ik_seeds=num_batch_ik_seeds,
        num_graph_seeds=num_graph_seeds,
        interpolation_steps=interpolation_steps,
        interpolation_dt=interpolation_dt,
        collision_cache=collision_cache,
        optimize_dt=optimize_dt,
        trajopt_dt=trajopt_dt,
        trajopt_tsteps=trajopt_tsteps,
        evaluate_interpolated_trajectory=evaluate_interpolated_trajectory,
        gradient_trajopt_file=gradient_trajopt_file,
        finetune_trajopt_file=finetune_trajopt_file,
        trim_steps=trim_steps,
        use_cuda_graph=use_cuda_graph,
        store_debug_in_result=(args.debug_runtime or args.debug_ik_fail_details),
    )
    motion_gen = MotionGen(motion_gen_config)
    checker_name = motion_gen.world_coll_checker.__class__.__name__
    print(
        "World checker:",
        checker_name,
        "collision_types=",
        motion_gen.world_coll_checker.collision_types,
    )
    if has_superquadrics and motion_gen.world_coll_checker.collision_types.get("mesh", False):
        # Some checker paths can still expose mesh-combine reshape branches even
        # for native superquadric worlds; disable mesh collision explicitly.
        carb.log_warn("Disabling mesh collision branch for native superquadric planning.")
        motion_gen.world_coll_checker.collision_types["mesh"] = False
    if not args.reactive:
        print("warming up...")
        motion_gen.warmup(enable_graph=(num_graph_seeds > 0), warmup_js_trajopt=False)
    return motion_gen


def _plan_worker(
    motion_gen: MotionGen,
    cu_js: JointState,
    ik_goal: Pose,
    plan_config: MotionGenPlanConfig,
    has_superquadrics: bool,
    result_queue: queue.Queue,
    default_stream: torch.cuda.Stream,
    plan_request_id: int,
    debug_runtime: bool,
) -> None:
    """Run motion planning in a background thread on a dedicated CUDA stream.

    Using a stream separate from Isaac Sim's default stream means planning CUDA
    kernels do not serialise with the renderer, so the viewport stays live.
    """
    device = motion_gen.tensor_args.device
    planning_stream = torch.cuda.Stream(device=device)
    start_time = time.perf_counter()
    if debug_runtime:
        runtime_debug(
            f"plan#{plan_request_id} worker started on device={device}; waiting for default stream"
        )
    with torch.cuda.stream(planning_stream):
        # Wait for any ops on the default stream (e.g. cu_js.clone()) to finish
        # before we start issuing work on this stream.
        planning_stream.wait_stream(default_stream)
        try:
            result = motion_gen.plan_single(cu_js.unsqueeze(0), ik_goal, plan_config)
            if (
                has_superquadrics
                and result.status == MotionGenStatus.INVALID_START_STATE_WORLD_COLLISION
            ):
                carb.log_warn(
                    "Retrying plan with check_start_validity=False after "
                    "INVALID_START_STATE_WORLD_COLLISION in superquadric mode."
                )
                retry_config = copy.deepcopy(plan_config)
                retry_config.check_start_validity = False
                result = motion_gen.plan_single(cu_js.unsqueeze(0), ik_goal, retry_config)
            if (
                has_superquadrics
                and (not args.disable_superquadric_ik_retry)
                and (not bool(result.success.item()))
                and result.status == MotionGenStatus.IK_FAIL
            ):
                retry_config = copy.deepcopy(plan_config)
                retry_config.max_attempts = max(int(plan_config.max_attempts), 2)
                retry_config.partial_ik_opt = False
                retry_config.check_start_validity = False

                current_ik_seeds = int(getattr(motion_gen, "ik_seeds", 8))
                requested_ik_seeds = int(plan_config.num_ik_seeds or 0)
                retry_config.num_ik_seeds = max(current_ik_seeds * 2, requested_ik_seeds * 2, 16)

                if retry_config.timeout is not None:
                    retry_timeout_floor = 2.0 if args.headless_mode is None else 6.0
                    retry_config.timeout = max(float(retry_config.timeout), retry_timeout_floor)

                motion_gen.reset_seed()
                carb.log_warn(
                    "Retrying superquadric IK after IK_FAIL with "
                    f"num_ik_seeds={retry_config.num_ik_seeds} "
                    f"max_attempts={retry_config.max_attempts} timeout={retry_config.timeout}."
                )
                retry_result = motion_gen.plan_single(cu_js.unsqueeze(0), ik_goal, retry_config)
                if bool(retry_result.success.item()):
                    result = retry_result
            if (
                has_superquadrics
                and (not args.disable_superquadric_trajopt_retry)
                and (not bool(result.success.item()))
                and result.status in [MotionGenStatus.TRAJOPT_FAIL, MotionGenStatus.FINETUNE_TRAJOPT_FAIL]
                and (not bool(getattr(result, "used_graph", False)))
            ):
                retry_config = copy.deepcopy(plan_config)
                retry_config.max_attempts = max(int(plan_config.max_attempts), 2)
                if retry_config.timeout is not None:
                    retry_timeout_floor = 2.0 if args.headless_mode is None else 6.0
                    retry_config.timeout = max(float(retry_config.timeout), retry_timeout_floor)
                # Keep this retry lightweight: skip finetune and rely on core trajopt feasibility.
                retry_config.enable_finetune_trajopt = False
                retry_config.parallel_finetune = False
                carb.log_warn(
                    "Retrying superquadric plan after trajopt failure with "
                    f"max_attempts={retry_config.max_attempts} timeout={retry_config.timeout}."
                )
                retry_result = motion_gen.plan_single(cu_js.unsqueeze(0), ik_goal, retry_config)
                if bool(retry_result.success.item()):
                    result = retry_result
            if (
                has_superquadrics
                and (not args.disable_superquadric_graph_only_fallback)
                and (not args.trajopt_only)
                and (not bool(result.success.item()))
                and result.status in [MotionGenStatus.TRAJOPT_FAIL, MotionGenStatus.FINETUNE_TRAJOPT_FAIL]
            ):
                graph_only_config = copy.deepcopy(plan_config)
                graph_only_config.enable_graph = True
                graph_only_config.need_graph_success = True
                graph_only_config.enable_opt = False
                graph_only_config.enable_finetune_trajopt = False
                graph_only_config.parallel_finetune = False
                graph_only_config.max_attempts = max(int(plan_config.max_attempts), 2)
                graph_only_config.time_dilation_factor = 1.0
                if graph_only_config.timeout is not None:
                    graph_timeout_floor = 2.0 if args.headless_mode is None else 6.0
                    graph_only_config.timeout = max(
                        float(graph_only_config.timeout),
                        graph_timeout_floor,
                    )
                carb.log_warn(
                    "Retrying superquadric plan with graph-only fallback "
                    f"(max_attempts={graph_only_config.max_attempts} timeout={graph_only_config.timeout})."
                )
                graph_only_result = motion_gen.plan_single(
                    cu_js.unsqueeze(0),
                    ik_goal,
                    graph_only_config,
                )
                if bool(graph_only_result.success.item()):
                    carb.log_warn(
                        "Superquadric graph-only fallback succeeded; returning collision-validated "
                        "geometric path without trajopt refinement."
                    )
                    result = graph_only_result
            if (
                has_superquadrics
                and (not args.disable_superquadric_retract_escape)
                and (not args.trajopt_only)
                and (not bool(result.success.item()))
                and result.status
                in [
                    MotionGenStatus.TRAJOPT_FAIL,
                    MotionGenStatus.FINETUNE_TRAJOPT_FAIL,
                    MotionGenStatus.GRAPH_FAIL,
                ]
            ):
                escape_config = copy.deepcopy(plan_config)
                escape_config.enable_graph = True
                escape_config.enable_graph_attempt = None
                escape_config.need_graph_success = False
                escape_config.enable_opt = True
                escape_config.enable_finetune_trajopt = False
                escape_config.parallel_finetune = False
                escape_config.check_start_validity = False
                escape_config.max_attempts = max(int(plan_config.max_attempts), 2)
                escape_config.time_dilation_factor = 1.0
                if escape_config.timeout is not None:
                    escape_timeout_floor = 2.0 if args.headless_mode is None else 6.0
                    escape_config.timeout = max(float(escape_config.timeout), escape_timeout_floor)

                retract_goal = JointState.from_position(
                    motion_gen.get_retract_config().view(1, -1),
                    joint_names=motion_gen.kinematics.joint_names,
                )
                carb.log_warn(
                    "Attempting superquadric escape trajectory to retract configuration "
                    f"(max_attempts={escape_config.max_attempts} timeout={escape_config.timeout})."
                )
                escape_result = motion_gen.plan_single_js(
                    cu_js.unsqueeze(0),
                    retract_goal,
                    escape_config,
                )
                if bool(escape_result.success.item()):
                    setattr(escape_result, "escape_retract", True)
                    carb.log_warn(
                        "Superquadric retract-escape fallback succeeded; executing retreat "
                        "trajectory and will re-attempt target planning."
                    )
                    result = escape_result
            # CPU-block this worker thread until all planning CUDA work is done so
            # that result tensors are safe to consume on the main thread.
            planning_stream.synchronize()
            elapsed = time.perf_counter() - start_time
            if debug_runtime:
                runtime_debug(
                    f"plan#{plan_request_id} worker finished in {elapsed:.3f}s with status={result.status}"
                )
            result_queue.put(("ok", result))
        except Exception as exc:
            elapsed = time.perf_counter() - start_time
            if debug_runtime:
                runtime_debug(
                    f"plan#{plan_request_id} worker failed after {elapsed:.3f}s: {exc}"
                )
            result_queue.put(("error", traceback.format_exc()))


def main() -> None:
    if args.world_representation == "superquadrics":
        ensure_superquadric_support()

    my_world = World(stage_units_in_meters=1.0)
    stage = my_world.stage
    xform = stage.DefinePrim("/World", "Xform")
    stage.SetDefaultPrim(xform)
    stage.DefinePrim("/curobo", "Xform")

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

    robot_cfg_path = get_robot_configs_path()
    if args.external_robot_configs_path is not None:
        robot_cfg_path = args.external_robot_configs_path
    robot_cfg = load_yaml(join_path(robot_cfg_path, args.robot))["robot_cfg"]
    if args.external_asset_path is not None:
        robot_cfg["kinematics"]["external_asset_path"] = args.external_asset_path
    if args.external_robot_configs_path is not None:
        robot_cfg["kinematics"]["external_robot_configs_path"] = args.external_robot_configs_path

    joint_names = robot_cfg["kinematics"]["cspace"]["joint_names"]
    default_config = robot_cfg["kinematics"]["cspace"]["retract_config"]
    robot, _ = add_robot_to_scene(robot_cfg, my_world)
    articulation_controller = None

    collision_world, visual_world, superquadric_world = build_collision_and_visual_worlds()
    motion_gen = build_motion_gen(robot_cfg, collision_world, tensor_args)
    
    # ── Diagnostic: dump sq_params to verify radii reach the kernel ──────────
    try:
        wc = motion_gen.world_coll_checker
        sq_list = wc._sq_tensor_list
        print(f"=== _sq_tensor_list: {len(sq_list)} env(s)")
        for env_idx, sq_tensor in enumerate(sq_list):
            sq_np = sq_tensor.detach().cpu().numpy()
            print(f"  env={env_idx} raw shape={sq_np.shape}")
            # Squeeze out any leading dimensions to get (n_obs, n_params)
            sq_flat = sq_np.reshape(-1, sq_np.shape[-1])
            names = wc._env_superquadrics_names[env_idx] if env_idx < len(wc._env_superquadrics_names) else []
            for obs_idx in range(sq_flat.shape[0]):
                row = sq_flat[obs_idx]
                name = names[obs_idx] if obs_idx < len(names) else "?"
                # Print all columns so we can identify layout regardless of version
                cols = "  ".join(f"{v:.4f}" for v in row)
                print(f"    [{obs_idx:2d}] {name:10s}  [{cols}]")

        # Also print n_superquadrics per env
        print(f"=== _env_n_superquadrics: {wc._env_n_superquadrics}")
        
        print("=== motion_gen collision-related attrs:")
        for attr in sorted(dir(motion_gen)):
            if any(k in attr.lower() for k in ['sq', 'super', 'coll', 'world', 'obs']):
                try:
                    val = getattr(motion_gen, attr)
                    if hasattr(val, 'shape'):
                        print(f"  TENSOR {attr}: shape={val.shape}")
                    elif not callable(val):
                        print(f"  {attr}: {repr(val)[:80]}")
                    else:
                        print(f"  method {attr}")
                except Exception as e:
                    print(f"  {attr}: error={e}")
        
        retract_js = JointState.from_position(
            motion_gen.get_retract_config().view(1, -1),
            joint_names=motion_gen.kinematics.joint_names,
        )
        kin_state = motion_gen.compute_kinematics(retract_js.unsqueeze(0))
        # robot spheres at retract pose
        spheres = motion_gen.kinematics.get_robot_as_spheres(retract_js.position)
        if spheres is not None:
            print("=== Robot spheres at retract (x, y, z, r):")
            if hasattr(spheres, 'detach'):
                sph_np = spheres.detach().cpu().numpy()
            elif isinstance(spheres, (list, tuple)):
                # get_robot_as_spheres returns [[sphere0, sphere1, ...]] (nested: batch × spheres)
                flat_spheres = []
                for batch_item in spheres:
                    if isinstance(batch_item, (list, tuple)):
                        flat_spheres.extend(batch_item)
                    elif hasattr(batch_item, 'position'):
                        flat_spheres.append(batch_item)
                sph_np = np.array(
                    [[float(s.position[0]), float(s.position[1]),
                      float(s.position[2]), float(s.radius)]
                     for s in flat_spheres if hasattr(s, 'position')],
                    dtype=np.float32,
                ) if flat_spheres else np.empty((0, 4), dtype=np.float32)
            else:
                sph_np = np.array(spheres, dtype=np.float32)
            sph_flat = sph_np.reshape(-1, 4) if sph_np.size > 0 else np.empty((0, 4), dtype=np.float32)
            active_spheres = [(i, s) for i, s in enumerate(sph_flat) if s[3] > 0]
            for i, s in active_spheres:
                print(f"  sphere[{i:2d}]  pos=({s[0]:7.4f}, {s[1]:7.4f}, {s[2]:7.4f})  r={s[3]:.4f}")

            # Per-SQ SDF for each active sphere — identifies which obstacles cause
            # the retract-state collision and whether values are plausible
            if wc._sq_tensor_list is not None and len(active_spheres) > 0:
                from curobo.geom.sdf.world import WorldCollisionConfig, CollisionQueryBuffer
                sq_params = wc._sq_tensor_list[0]   # (1, n_obs, 12)
                sq_enable = wc._sq_tensor_list[1]    # (1, n_obs)
                n_obs = sq_params.shape[1]
                print(f"=== Per-SQ SDF at retract (positive = collision in CuRobo convention):")
                for obs_idx in range(n_obs):
                    if not sq_enable[0, obs_idx]:
                        continue
                    obs_name = wc._env_superquadrics_names[0][obs_idx]
                    obs_cfg_sq = wc._sq_tensor_list[0][0:1, obs_idx:obs_idx+1, :]  # (1,1,12)
                    obs_en = torch.ones(1, 1, dtype=torch.uint8, device=sq_params.device)
                    from curobo.geom.types import Superquadric, WorldConfig as WC2
                    from curobo.geom.sdf.world import WorldPrimitiveCollision, WorldCollisionConfig as WCC
                    row = sq_params[0, obs_idx].cpu().tolist()
                    sq_single = Superquadric(
                        name=obs_name or f"sq_{obs_idx}",
                        pose=[row[5], row[6], row[7],
                              row[11], row[8], row[9], row[10]],
                        radii=[row[0], row[1], row[2]],
                        eps=[row[3], row[4]],
                    )
                    wcfg = WCC(tensor_args=wc.tensor_args, world_model=WC2(superquadric=[sq_single]),
                               cache={"obb": 0, "superquadric": 1})
                    w_single = WorldPrimitiveCollision(wcfg)
                    min_d = float('inf')
                    worst_sph = -1
                    for si, s in active_spheres:
                        q = CollisionQueryBuffer.initialize_from_shape(
                            (1, 1, 1, 4), wc.tensor_args, w_single.collision_types)
                        sph_t = wc.tensor_args.to_device(
                            [[float(s[0]), float(s[1]), float(s[2]), float(s[3])]]
                        ).view(1, 1, 1, 4)
                        w_single.get_sphere_distance(
                            sph_t, q,
                            wc.tensor_args.to_device([1.0]),
                            wc.tensor_args.to_device([0.0]),
                            env_query_idx=wc.tensor_args.to_device([0]).to(torch.int32),
                            compute_esdf=True,
                        )
                        d = q.superquadric_collision_buffer.distance_buffer.item()
                        if d < min_d:
                            min_d = d
                            worst_sph = si
                    flag = " *** COLLISION ***" if min_d > 0 else ""
                    print(f"  {obs_name or obs_idx:10s}  min_sdf={min_d:+.4f}  "
                          f"(worst sphere[{worst_sph}]){flag}")
    except Exception as e:
        print(f"=== sq_params diagnostic failed: {e}")

    print("CuRobo is ready")
    persistent_start_collision = False
    request_mesh_fallback = False
    active_sq_collision_eps_blend = 0.0

    def _ensure_superquadric_start_state_valid(
        start_js: JointState,
        context: str,
        persist_relaxed_world: bool = False,
    ) -> tuple[bool, object, WorldConfig | None, float | None, float | None]:
        """Try to auto-relax SQ collision tolerance when start state is in world collision.

        Returns:
            is_valid: Whether a collision-free start state was found.
            status: Resulting start-state status.
            relaxed_world_for_plan: Temporary collision world that validated start state
                when persist_relaxed_world=False.
            relaxed_tolerance: Radius-shrink tolerance used for relaxed_world_for_plan.
            relaxed_eps_blend: Exponent blend used for relaxed_world_for_plan.
        """
        nonlocal collision_world, active_sq_collision_eps_blend

        baseline_collision_world = collision_world

        start_valid, start_status = motion_gen.check_start_state(start_js)
        if start_valid:
            return True, start_status, None, None, None

        if start_status != MotionGenStatus.INVALID_START_STATE_WORLD_COLLISION:
            return False, start_status, None, None, None

        base_tol = max(float(args.superquadric_collision_tolerance), 0.0)
        max_eps_blend = float(np.clip(args.superquadric_auto_relax_max_eps_blend, 0.0, 1.0))
        eps_candidates = [float(np.clip(active_sq_collision_eps_blend, 0.0, max_eps_blend))]
        for candidate in [0.35, 0.65, 1.0]:
            candidate = float(np.clip(candidate, 0.0, max_eps_blend))
            if all(abs(candidate - x) > 1.0e-6 for x in eps_candidates):
                eps_candidates.append(candidate)

        max_tol = max(float(args.superquadric_auto_relax_max_tolerance), 0.0)
        tol_candidates = [min(base_tol, max_tol)]
        for candidate in [0.02, 0.03, 0.05, 0.08, 0.12, 0.16, 0.20, 0.25]:
            candidate = min(float(candidate), max_tol)
            if all(abs(candidate - x) > 1.0e-6 for x in tol_candidates):
                tol_candidates.append(candidate)

        carb.log_warn(
            "SQ auto-relax search bounds: "
            f"max_tolerance={max_tol:.3f}m max_eps_blend={max_eps_blend:.2f} "
            f"persist_relaxed_world={persist_relaxed_world}."
        )

        for eps_blend in eps_candidates:
            sq_world_blended = apply_superquadric_collision_eps_blend(superquadric_world, eps_blend)
            for candidate_tol in tol_candidates:
                # Skip exact current configuration already tested above.
                if (
                    abs(eps_blend - active_sq_collision_eps_blend) <= 1.0e-6
                    and abs(candidate_tol - base_tol) <= 1.0e-6
                ):
                    continue

                relaxed_sq_world = apply_superquadric_collision_tolerance(
                    sq_world_blended,
                    tolerance=candidate_tol,
                )
                relaxed_collision_world = WorldConfig(
                    cuboid=collision_world.cuboid,
                    superquadric=relaxed_sq_world.superquadric,
                )
                motion_gen.update_world(relaxed_collision_world)
                trial_valid, trial_status = motion_gen.check_start_state(start_js)
                if trial_valid:
                    if persist_relaxed_world:
                        args.superquadric_collision_tolerance = candidate_tol
                        active_sq_collision_eps_blend = eps_blend
                        collision_world = relaxed_collision_world
                        carb.log_warn(
                            "Auto-relaxed SQ collision model for "
                            f"{context}: tolerance={candidate_tol:.3f}m eps_blend={eps_blend:.2f}; "
                            "start state is now collision-free and relaxed world is active."
                        )
                    else:
                        motion_gen.update_world(baseline_collision_world)
                        carb.log_warn(
                            "Auto-relax found a start-state-valid SQ configuration for "
                            f"{context}: tolerance={candidate_tol:.3f}m eps_blend={eps_blend:.2f}; "
                            "keeping baseline collision world globally and using relaxed "
                            "world only for this plan request."
                        )
                        return (
                            True,
                            trial_status,
                            relaxed_collision_world,
                            candidate_tol,
                            eps_blend,
                        )
                    return True, trial_status, None, None, None
                carb.log_warn(
                    f"Auto-relax trial for {context} with "
                    f"tolerance={candidate_tol:.3f}m eps_blend={eps_blend:.2f} "
                    f"still invalid (status={trial_status})."
                )

        # If no candidate validated, restore baseline world so trial updates do not leak.
        motion_gen.update_world(baseline_collision_world)
        return False, start_status, None, None, None

    # Verify that the robot's default/retract configuration is not inside the collision world.
    # A spurious collision here (especially in superquadric mode) means the SDF kernel is over-
    # estimating distances -- typically because superquadric radii are too large for the scene
    # scale.  Warn immediately so the user doesn't have to wait for a 40-second planning failure
    # to discover the problem.
    if collision_world.superquadric:
        try:
            _default_js_check = JointState.from_position(
                motion_gen.get_retract_config().view(1, -1),
                joint_names=motion_gen.kinematics.joint_names,
            )
            (
                _retract_valid,
                _retract_status,
                _retract_relaxed_world,
                _retract_relaxed_tolerance,
                _retract_relaxed_eps_blend,
            ) = _ensure_superquadric_start_state_valid(
                _default_js_check,
                "startup retract pose",
                persist_relaxed_world=args.persist_superquadric_auto_relax_world,
            )
            if (
                _retract_valid
                and _retract_relaxed_world is not None
                and (not args.persist_superquadric_auto_relax_world)
            ):
                carb.log_warn(
                    "Startup auto-relax found a one-shot-valid SQ world "
                    f"(tolerance={_retract_relaxed_tolerance:.3f}m "
                    f"eps_blend={_retract_relaxed_eps_blend:.2f}); runtime planning "
                    "will apply this as a temporary per-plan world when needed."
                )
            if _retract_valid:
                print(
                    f"[startup] Retract pose collision check: VALID "
                    f"(relaxed_tol={_retract_relaxed_tolerance} "
                    f"eps_blend={_retract_relaxed_eps_blend}). Planning can proceed.",
                    flush=True,
                )
            if not _retract_valid:
                print(
                    f"[startup] *** RETRACT POSE IN WORLD COLLISION (status={_retract_status}) ***\n"
                    f"  Superquadric SDF is over-estimating distances for the retract config.\n"
                    f"  Auto-relax tried tolerances up to {args.superquadric_auto_relax_max_tolerance:.3f}m "
                    f"and eps_blend up to {args.superquadric_auto_relax_max_eps_blend:.2f} — all failed.\n"
                    f"  → ALL PLAN REQUESTS WILL BE SKIPPED until this is resolved.\n"
                    f"  Options: (1) increase --superquadric_auto_relax_max_tolerance (current "
                    f"{args.superquadric_auto_relax_max_tolerance:.3f}m), "
                    f"(2) increase --superquadric_collision_tolerance "
                    f"(current {args.superquadric_collision_tolerance:.4f}m), "
                    f"(3) --world_representation mesh for comparison.",
                    flush=True,
                )
                carb.log_warn(
                    f"Robot retract/default configuration is already in world collision "
                    f"(status={_retract_status}). "
                    "This almost certainly means the superquadric SDF is over-estimating distances. "
                    "Common causes: (1) PLY coordinates are not in meters -- check "
                    "--superquadric_scale; (2) SuperDec output radii are large -- try "
                    "--print_superquadric_stats to see the actual values; (3) the kernel "
                    "is too conservative for the current exponents -- increase "
                    "--superquadric_collision_tolerance (current: "
                    f"{args.superquadric_collision_tolerance:.4f}m) or lower "
                    "--superquadric_max_radius "
                    f"(current: {args.superquadric_max_radius:.3f}m)."
                )
                persistent_start_collision = True
                request_mesh_fallback = True
                carb.log_warn(
                    "Persistent start-state world collision remains after auto-relax trials. "
                    "To avoid long futile IK runs, this script will skip planning until the "
                    "robot or world moves out of collision. Consider using --world_representation mesh "
                    "to compare behavior."
                )
        except Exception as _exc:
            carb.log_warn(f"Startup collision check skipped: {_exc}")
            print(f"[startup] Startup collision check exception: {_exc}", flush=True)

    interactive_mode = args.headless_mode is None

    add_extensions(simulation_app, args.headless_mode)
    usd_helper.load_stage(my_world.stage)
    usd_helper.add_world_to_stage(visual_world, base_frame="/World")
    has_superquadrics = bool(collision_world.superquadric)
    max_attempts = 1 if (args.reactive or has_superquadrics) else 4
    enable_finetune_trajopt = (not args.reactive) and (not args.disable_finetune_trajopt)
    effective_plan_stable_steps = max(args.plan_stable_steps, 1)
    effective_plan_cooldown_steps = max(args.plan_cooldown_steps, 0)
    effective_plan_timeout = args.plan_timeout if args.plan_timeout > 0.0 else None
    cmd_substeps = 2

    if interactive_mode:
        if effective_plan_stable_steps < 8:
            carb.log_warn(
                f"Increasing --plan_stable_steps from {effective_plan_stable_steps} to 8 "
                "in interactive mode for stability."
            )
            effective_plan_stable_steps = 8
        if effective_plan_cooldown_steps < 45:
            carb.log_warn(
                f"Increasing --plan_cooldown_steps from {effective_plan_cooldown_steps} to 45 "
                "in interactive mode for stability."
            )
            effective_plan_cooldown_steps = 45
        # No timeout cap — allow long runs for convergence investigation.
        # (Previously capped at 5s for SQs / 2s for mesh in interactive mode.)
        cmd_substeps = 0

    enable_superquadric_graph = (
        has_superquadrics and (not args.disable_superquadric_graph) and (not args.trajopt_only)
    )
    plan_config = MotionGenPlanConfig(
        enable_graph=enable_superquadric_graph,
        enable_graph_attempt=(None if (enable_superquadric_graph or args.trajopt_only) else (0 if has_superquadrics else 2)),
        max_attempts=max_attempts,
        timeout=effective_plan_timeout,
        enable_finetune_trajopt=enable_finetune_trajopt,
        time_dilation_factor=0.5,
        check_start_validity=not has_superquadrics,
    )

    def _switch_to_mesh_collision_mode(reason: str) -> bool:
        """Rebuild planner with mesh collision using the same inferred geometry."""
        nonlocal collision_world, motion_gen, has_superquadrics, persistent_start_collision

        if not has_superquadrics:
            return False

        mesh_collision_world = WorldConfig(
            cuboid=collision_world.cuboid,
            mesh=visual_world.mesh,
        )
        try:
            motion_gen = build_motion_gen(robot_cfg, mesh_collision_world, tensor_args)
            collision_world = mesh_collision_world
            has_superquadrics = False
            persistent_start_collision = False

            # Update plan config to match mesh mode behavior after runtime fallback.
            plan_config.enable_graph = False
            plan_config.enable_graph_attempt = 2
            plan_config.max_attempts = 1 if args.reactive else 4
            plan_config.check_start_validity = True

            carb.log_warn(
                "Switched collision mode to mesh fallback after persistent superquadric start-collision "
                f"({reason})."
            )
            return True
        except Exception as exc:
            carb.log_warn(f"Failed to switch to mesh collision fallback: {exc}")
            return False

    if request_mesh_fallback and has_superquadrics:
        if args.auto_fallback_to_mesh:
            _switch_to_mesh_collision_mode("startup")
        else:
            carb.log_warn(
                "Persistent SQ start-collision detected; rerun with --auto_fallback_to_mesh "
                "to switch automatically to mesh collision mode."
            )

    cmd_plan = None
    cmd_idx = 0
    idx_list = []
    num_targets = 0
    past_cmd = None
    past_pose = None
    past_orientation = None
    pose_metric = None
    spheres = None
    target_pose = None
    target_orientation = None
    idle_steps = 0
    last_plan_step = -10_000_000
    stationary_steps = 0
    oom_backoff_until_step = -1
    plan_queue: queue.Queue = queue.Queue(maxsize=1)
    plan_thread: threading.Thread = None
    plan_request_id = 0
    active_plan_request_id = -1
    active_plan_start_time = 0.0
    active_plan_start_js = None
    active_goal_position = None
    active_goal_orientation = None
    active_plan_uses_relaxed_sq_world = False
    active_plan_relaxed_tolerance = None
    active_plan_relaxed_eps_blend = None
    last_watchdog_time = 0.0
    last_heartbeat_time = 0.0
    force_plan_request = False
    sdf_points_prim = None
    sdf_gradient_prim = None
    sdf_gradient_tip_prim = None
    sdf_gradient_shaft_prim = None
    last_sdf_update_step = -10_000_000

    my_world.scene.add_default_ground_plane()

    if args.visualize_sdf:
        sdf_points_prim = create_sdf_points_prim(stage)
        carb.log_warn(
            "SDF visualization enabled. Red points are inside obstacles, blue points are outside "
            "(within configured band)."
        )
    if args.visualize_sdf_gradients:
        sdf_gradient_prim = create_sdf_gradient_prim(stage)
        sdf_gradient_tip_prim = create_sdf_gradient_tip_prim(stage)
        sdf_gradient_shaft_prim = create_sdf_gradient_shaft_prim(stage)
        carb.log_warn(
            "SDF gradient visualization enabled. Arrows are rendered as shaft points + green tips."
        )

    # Parse automatic cube target positions (headless testing / debugging).
    auto_cube_positions = None
    if args.auto_cube_targets is not None:
        import json as _json
        try:
            raw_targets = _json.loads(args.auto_cube_targets)
            auto_cube_positions = [np.array(t, dtype=np.float64) for t in raw_targets]
            print(f"[auto-targets] {len(auto_cube_positions)} positions loaded, "
                  f"cycling every {args.auto_target_interval} steps.", flush=True)
        except Exception as _exc:
            print(f"[auto-targets] Failed to parse --auto_cube_targets: {_exc}", flush=True)
    auto_target_idx = 0

    # In headless mode the world is never played interactively — start it automatically.
    if not interactive_mode:
        my_world.play()
        print("[headless] Simulation started (auto-play).", flush=True)

    while simulation_app.is_running():
        my_world.step(render=not (not interactive_mode))
        if not my_world.is_playing():
            if idle_steps % 100 == 0:
                print("**** Click Play to start simulation *****")
            idle_steps += 1
            continue

        step_index = my_world.current_time_step_index

        # Auto-exit after --max_frames steps.
        if args.max_frames > 0 and step_index >= args.max_frames:
            print(f"[headless] Reached max_frames={args.max_frames}; exiting.", flush=True)
            break

        if articulation_controller is None:
            articulation_controller = robot.get_articulation_controller()
        if step_index < 10:
            robot._articulation_view.initialize()
            idx_list = [robot.get_dof_index(name) for name in joint_names]
            robot.set_joint_positions(default_config, idx_list)
            robot._articulation_view.set_max_efforts(
                values=np.array([5000 for _ in range(len(idx_list))]),
                joint_indices=idx_list,
            )
        if step_index < 20:
            continue

        # Automatic cube target cycling (headless / debugging mode).
        if auto_cube_positions is not None and step_index > 20:
            interval = max(args.auto_target_interval, 1)
            new_auto_idx = (step_index // interval) % len(auto_cube_positions)
            if new_auto_idx != auto_target_idx:
                auto_target_idx = new_auto_idx
                new_pos = auto_cube_positions[auto_target_idx]
                target.set_world_pose(position=new_pos)
                print(
                    f"[auto-targets] step={step_index} → target #{auto_target_idx} "
                    f"pos={np.array2string(new_pos, precision=3)}",
                    flush=True,
                )

        if (step_index - last_sdf_update_step) >= max(args.sdf_update_steps, 1):
            if args.visualize_sdf and sdf_points_prim is not None:
                try:
                    update_sdf_points_prim(sdf_points_prim, motion_gen)
                except Exception as exc:
                    carb.log_warn(f"Failed to update SDF visualization: {exc}")
            if args.visualize_sdf_gradients and sdf_gradient_prim is not None:
                try:
                    update_sdf_gradient_prim(
                        sdf_gradient_prim,
                        motion_gen,
                        sdf_gradient_tip_prim,
                        sdf_gradient_shaft_prim,
                    )
                except Exception as exc:
                    carb.log_warn(f"Failed to update SDF gradient visualization: {exc}")
            last_sdf_update_step = step_index

        cube_position, cube_orientation = target.get_world_pose()
        if not np.all(np.isfinite(cube_position)) or not np.all(np.isfinite(cube_orientation)):
            carb.log_warn("Target pose contained non-finite values; skipping this frame.")
            stationary_steps = 0
            continue
        cube_orientation_norm = np.linalg.norm(cube_orientation)
        if cube_orientation_norm < 1e-8:
            carb.log_warn("Target orientation norm is near zero; skipping this frame.")
            stationary_steps = 0
            continue
        cube_orientation = cube_orientation / cube_orientation_norm
        if past_pose is None:
            past_pose = cube_position
        if target_pose is None:
            target_pose = cube_position
        if target_orientation is None:
            target_orientation = cube_orientation
        if past_orientation is None:
            past_orientation = cube_orientation

        sim_js = robot.get_joints_state()
        if sim_js is None:
            print("sim_js is None")
            continue
        if (not np.all(np.isfinite(sim_js.positions))) or (not np.all(np.isfinite(sim_js.velocities))):
            carb.log_warn("Isaac Sim returned non-finite joint state values; skipping this frame.")
            stationary_steps = 0
            cmd_plan = None
            past_cmd = None
            continue

        cu_js = JointState(
            position=tensor_args.to_device(sim_js.positions),
            velocity=tensor_args.to_device(sim_js.velocities),
            acceleration=tensor_args.to_device(sim_js.velocities) * 0.0,
            jerk=tensor_args.to_device(sim_js.velocities) * 0.0,
            joint_names=robot.dof_names,
        )
        if not args.reactive:
            cu_js.velocity *= 0.0
            cu_js.acceleration *= 0.0
        if args.reactive and past_cmd is not None:
            cu_js.position[:] = past_cmd.position
            cu_js.velocity[:] = past_cmd.velocity
            cu_js.acceleration[:] = past_cmd.acceleration
        cu_js = cu_js.get_ordered_joint_state(motion_gen.kinematics.joint_names)

        if args.visualize_spheres and step_index % 2 == 0:
            sphere_list = motion_gen.kinematics.get_robot_as_spheres(cu_js.position)
            if spheres is None:
                spheres = []
                for sphere_index, robot_sphere in enumerate(sphere_list[0]):
                    sphere_prim = sphere.VisualSphere(
                        prim_path=f"/curobo/robot_sphere_{sphere_index}",
                        position=np.ravel(robot_sphere.position),
                        radius=float(robot_sphere.radius),
                        color=np.array([0.0, 0.8, 0.2]),
                    )
                    spheres.append(sphere_prim)
            else:
                for sphere_index, robot_sphere in enumerate(sphere_list[0]):
                    if not np.isnan(robot_sphere.position[0]):
                        spheres[sphere_index].set_world_pose(position=np.ravel(robot_sphere.position))
                        spheres[sphere_index].set_radius(float(robot_sphere.radius))

        robot_static = (np.max(np.abs(sim_js.velocities)) < 0.5) or args.reactive
        target_changed = (
            np.linalg.norm(cube_position - target_pose) > 1e-3
            or np.linalg.norm(cube_orientation - target_orientation) > 1e-3
        )
        target_stationary = (
            np.linalg.norm(past_pose - cube_position) < 1e-6
            and np.linalg.norm(past_orientation - cube_orientation) < 1e-6
        )
        if target_stationary:
            stationary_steps += 1
        else:
            stationary_steps = 0

        # ---- collect result from background plan thread (non-blocking) ----
        if plan_thread is not None and not plan_thread.is_alive():
            plan_thread = None
            if args.debug_runtime and active_plan_start_time > 0.0:
                runtime_debug(
                    f"plan#{active_plan_request_id} thread joined after "
                    f"{time.perf_counter() - active_plan_start_time:.3f}s"
                )
            try:
                plan_status, plan_result = plan_queue.get_nowait()
            except queue.Empty:
                plan_status, plan_result = None, None

            if plan_status == "ok":
                success = plan_result.success.item()
                is_escape_retract = bool(getattr(plan_result, "escape_retract", False))
                _result_status = _format_plan_status(
                    getattr(plan_result, "status", None), success=bool(success)
                )
                print(
                    f"[diag step={step_index}] plan#{active_plan_request_id} "
                    f"{'SUCCESS' if success else 'FAILED'} status={_result_status}",
                    flush=True,
                )
                if args.debug_runtime:
                    runtime_debug(
                        f"plan#{active_plan_request_id} result success={bool(success)} "
                        f"status={plan_result.status}"
                    )
                if num_targets == 1:
                    if args.constrain_grasp_approach:
                        pose_metric = PoseCostMetric.create_grasp_approach_metric()
                    if args.reach_partial_pose is not None:
                        reach_vec = motion_gen.tensor_args.to_device(args.reach_partial_pose)
                        pose_metric = PoseCostMetric(
                            reach_partial_pose=True,
                            reach_vec_weight=reach_vec,
                        )
                    if args.hold_partial_pose is not None:
                        hold_vec = motion_gen.tensor_args.to_device(args.hold_partial_pose)
                        pose_metric = PoseCostMetric(
                            hold_partial_pose=True,
                            hold_vec_weight=hold_vec,
                        )
                if success:
                    if is_escape_retract:
                        force_plan_request = True
                    else:
                        num_targets += 1
                        force_plan_request = False
                    cmd_plan = motion_gen.get_full_js(plan_result.get_interpolated_plan())
                    common_joint_names = []
                    idx_list = []
                    for name in robot.dof_names:
                        if name in cmd_plan.joint_names:
                            idx_list.append(robot.get_dof_index(name))
                            common_joint_names.append(name)
                    cmd_plan = cmd_plan.get_ordered_joint_state(common_joint_names)
                    cmd_idx = 0
                    if args.debug_runtime:
                        runtime_debug(
                            f"plan#{active_plan_request_id} accepted with "
                            f"{len(cmd_plan.position)} waypoints"
                        )
                    if is_escape_retract:
                        carb.log_warn(
                            "Executed retract-escape trajectory; re-attempting original target."
                        )
                else:
                    status_label = _format_plan_status(
                        getattr(plan_result, "status", None),
                        success=False,
                    )
                    carb.log_warn("Plan did not converge to a solution: " + status_label)
                    print(
                        f"[diag step={step_index}] plan#{active_plan_request_id} FAILED: {status_label} "
                        f"target={np.array2string(active_goal_position, precision=4) if active_goal_position is not None else 'N/A'}",
                        flush=True,
                    )
                    if active_plan_start_js is not None and active_goal_position is not None:
                        log_plan_failure_diagnostics(
                            motion_gen,
                            active_plan_request_id,
                            active_plan_start_js,
                            active_goal_position,
                            active_goal_orientation,
                            plan_result,
                        )
            elif plan_status == "error":
                carb.log_warn(f"Planner threw an exception: {plan_result}")
                print(f"[diag step={step_index}] plan#{active_plan_request_id} EXCEPTION: {plan_result}", flush=True)
                if "out of memory" in str(plan_result).lower():
                    torch.cuda.empty_cache()
                    backoff_steps = max(effective_plan_cooldown_steps, 120)
                    oom_backoff_until_step = max(oom_backoff_until_step, step_index + backoff_steps)
                    carb.log_warn(
                        "CUDA OOM detected; cleared PyTorch cache and pausing new plan attempts "
                        f"for {backoff_steps} simulation steps. Consider "
                        "--planner_memory_profile low or lowering --max_superquadrics."
                    )
            if active_plan_uses_relaxed_sq_world:
                motion_gen.update_world(collision_world)
                carb.log_warn(
                    "Restored baseline SQ collision world after "
                    f"plan#{active_plan_request_id} "
                    f"(temporary tolerance={active_plan_relaxed_tolerance:.3f}m "
                    f"eps_blend={active_plan_relaxed_eps_blend:.2f})."
                )
                active_plan_uses_relaxed_sq_world = False
                active_plan_relaxed_tolerance = None
                active_plan_relaxed_eps_blend = None
            active_plan_request_id = -1
            active_plan_start_time = 0.0
            active_plan_start_js = None
            active_goal_position = None
            active_goal_orientation = None

        # ---- trigger a new plan asynchronously when target is stable -------
        planning_active = plan_thread is not None
        now = time.perf_counter()
        debug_interval = max(args.debug_runtime_interval, 0.1)
        if args.debug_runtime and (now - last_heartbeat_time) >= debug_interval:
            cmd_progress = "none"
            if cmd_plan is not None:
                cmd_progress = f"{cmd_idx}/{len(cmd_plan.position)}"
            runtime_debug(
                "loop "
                f"step={step_index} planning_active={planning_active} stationary_steps={stationary_steps} "
                f"target_changed={target_changed} robot_static={robot_static} cmd_progress={cmd_progress}"
            )
            last_heartbeat_time = now
        if (
            args.debug_runtime
            and planning_active
            and active_plan_start_time > 0.0
            and (now - active_plan_start_time) >= max(args.debug_plan_watchdog_secs, 0.1)
            and (now - last_watchdog_time) >= debug_interval
        ):
            runtime_debug(
                f"plan#{active_plan_request_id} still running after {now - active_plan_start_time:.3f}s"
            )
            last_watchdog_time = now
        can_attempt_plan = (
            (step_index - last_plan_step) >= effective_plan_cooldown_steps
            and not planning_active
            and step_index >= oom_backoff_until_step
        )
        target_stable = stationary_steps >= effective_plan_stable_steps

        if ((target_changed and target_stable) or (force_plan_request and target_stable)) and robot_static and can_attempt_plan:
            vprint(
                f"step={step_index} PLAN TRIGGERED  target={np.array2string(cube_position, precision=4)} "
                f"target_changed={target_changed} force={force_plan_request} "
                f"persistent_collision={persistent_start_collision}"
            )
            plan_relaxed_world = None
            plan_relaxed_tolerance = None
            plan_relaxed_eps_blend = None
            if has_superquadrics:
                (
                    current_valid,
                    current_status,
                    plan_relaxed_world,
                    plan_relaxed_tolerance,
                    plan_relaxed_eps_blend,
                ) = _ensure_superquadric_start_state_valid(
                    cu_js,
                    f"runtime step {step_index}",
                    persist_relaxed_world=args.persist_superquadric_auto_relax_world,
                )
                if current_valid:
                    if persistent_start_collision:
                        persistent_start_collision = False
                        carb.log_warn(
                            "Start-state collision cleared; re-enabling superquadric planning."
                        )
                        vprint("Start-state collision CLEARED. Planning will proceed.")
                else:
                    persistent_start_collision = True
                    if args.auto_fallback_to_mesh and _switch_to_mesh_collision_mode(
                        f"runtime step {step_index}"
                    ):
                        target_pose = cube_position
                        target_orientation = cube_orientation
                        continue
                    print(
                        f"[diag step={step_index}] PLAN SKIPPED: start state in world collision "
                        f"(status={current_status}). "
                        "Use --world_representation mesh to compare, or check SQ scene transform.",
                        flush=True,
                    )
                    carb.log_warn(
                        "Skipping plan request: start state remains in world collision "
                        f"(status={current_status}). Avoiding expensive IK run. "
                        "Move robot/scene or use --world_representation mesh for comparison."
                    )
                    target_pose = cube_position
                    target_orientation = cube_orientation
                    continue

            # Only pre-check start validity when the planner is also configured to check it.
            # With superquadrics check_start_validity=False because the superquadric SDF kernel
            # can report false-positive collisions (known distance overestimation), so we let
            # plan_single proceed without a start-state gate.
            if plan_config.check_start_validity:
                start_valid, start_status = motion_gen.check_start_state(cu_js)
                if not start_valid and start_status == MotionGenStatus.INVALID_START_STATE_WORLD_COLLISION:
                    carb.log_warn(
                        "Skipping plan request because current robot start state is in world collision. "
                        "Move robot away from obstacles or adjust superquadric scene placement "
                        "(--superquadric_translation/--superquadric_scale)."
                    )
                    # Mark this target as processed to avoid repeated expensive checks for
                    # the same cube pose until the user moves it again.
                    target_pose = cube_position
                    target_orientation = cube_orientation
                    continue

            last_plan_step = step_index
            plan_request_id += 1
            active_plan_request_id = plan_request_id
            active_plan_start_time = time.perf_counter()
            last_watchdog_time = active_plan_start_time
            if args.debug_runtime:
                runtime_debug(
                    f"plan#{active_plan_request_id} queued at step={step_index} "
                    f"target_pos={np.array2string(cube_position, precision=4)}"
                )

            if has_superquadrics and plan_relaxed_world is not None:
                # Use the relaxed world that validated start state for this request only.
                motion_gen.update_world(plan_relaxed_world)
                active_plan_uses_relaxed_sq_world = True
                active_plan_relaxed_tolerance = plan_relaxed_tolerance
                active_plan_relaxed_eps_blend = plan_relaxed_eps_blend
                carb.log_warn(
                    "Using temporary auto-relaxed SQ collision world for "
                    f"plan#{active_plan_request_id} "
                    f"(tolerance={plan_relaxed_tolerance:.3f}m "
                    f"eps_blend={plan_relaxed_eps_blend:.2f})."
                )
            else:
                active_plan_uses_relaxed_sq_world = False
                active_plan_relaxed_tolerance = None
                active_plan_relaxed_eps_blend = None

            ik_goal = Pose(
                position=tensor_args.to_device(cube_position),
                quaternion=tensor_args.to_device(cube_orientation),
            )
            plan_config.pose_cost_metric = pose_metric
            cu_js_snapshot = cu_js.clone()
            active_plan_start_js = cu_js_snapshot.clone()
            active_goal_position = np.asarray(cube_position, dtype=np.float64).copy()
            active_goal_orientation = np.asarray(cube_orientation, dtype=np.float64).copy()
            # Capture the current default stream *after* the clone so the worker
            # can wait for the clone to complete before issuing planning ops.
            current_default_stream = torch.cuda.current_stream(tensor_args.device)
            plan_thread = threading.Thread(
                target=_plan_worker,
                args=(
                    motion_gen,
                    cu_js_snapshot,
                    ik_goal,
                    plan_config,
                    has_superquadrics,
                    plan_queue,
                    current_default_stream,
                    active_plan_request_id,
                    args.debug_runtime,
                ),
                daemon=True,
            )
            plan_thread.start()
            force_plan_request = False
            target_pose = cube_position
            target_orientation = cube_orientation

        past_pose = cube_position
        past_orientation = cube_orientation

        if cmd_plan is not None:
            cmd_state = cmd_plan[cmd_idx]
            past_cmd = cmd_state.clone()
            try:
                articulation_controller.apply_action(
                    ArticulationAction(
                        cmd_state.position.cpu().numpy(),
                        cmd_state.velocity.cpu().numpy(),
                        joint_indices=idx_list,
                    )
                )
            except Exception as exc:
                carb.log_warn(f"Failed to apply articulation command; clearing plan: {exc}")
                cmd_idx = 0
                cmd_plan = None
                past_cmd = None
                continue
            cmd_idx += 1
            for _ in range(cmd_substeps):
                my_world.step(render=False)
            if cmd_idx >= len(cmd_plan.position):
                cmd_idx = 0
                cmd_plan = None
                past_cmd = None

    simulation_app.close()


if __name__ == "__main__":
    main()
