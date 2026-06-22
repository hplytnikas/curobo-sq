"""Reproducible benchmark: superquadric vs. mesh collision representation.

This is an evaluation harness for the paper.  It reuses the scene-generation,
SuperDec inference, planner construction and SQ/mesh switching helpers from
``motion_planning_sq_demo.py`` (imported as ``demo``) and adds measurement on top.

Workflow (three subcommands):

  1. build        Build a fixed family of tabletop scenes with increasing object
                  counts (1, 5, 10, 15, 25, 50, 100, 200), run SuperDec once, and
                  pickle everything to ``eval_out/scenes_cache.pkl`` so the scenes
                  are computed exactly once and are fully reproducible.

  2. set-targets  Open a viser UI to set & save 4 end-effector targets per scene
                  (so the planned trajectories are non-trivial).  Saved to
                  ``eval_out/targets.json``.

  3. benchmark    For each scene and each representation {superquadrics, mesh},
                  plan a sequential tour home->T1->T2->T3->T4, recording planning
                  time, time-optimal motion time, wall-clock playback time, and a
                  ground-truth collision metric (does the planned arm path actually
                  pierce any object point-cloud point).  Written to
                  ``eval_out/results.csv``.

Run everything under the ``3dv`` conda environment, e.g.::

    conda run -n 3dv python benchmark_sq_vs_mesh.py build
    conda run -n 3dv python benchmark_sq_vs_mesh.py set-targets --port 8081
    conda run -n 3dv python benchmark_sq_vs_mesh.py benchmark
    conda run -n 3dv python plot_benchmark.py
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import pickle
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import trimesh
import warp as wp
from scipy.spatial import cKDTree

# Import the demo module as a library (same directory).
sys.path.insert(0, str(Path(__file__).resolve().parent))
import motion_planning_sq_demo as demo  # noqa: E402

from curobo._src.geom.types import Cuboid, Mesh, SceneCfg  # noqa: E402
from curobo._src.types.device_cfg import DeviceCfg  # noqa: E402
from curobo.types import GoalToolPose, JointState, Pose  # noqa: E402

from superdec.utils.predictions_handler import PredictionHandler  # noqa: E402


# ── configuration ───────────────────────────────────────────────────────────────

OUT_DIR = Path(__file__).resolve().parent / "eval_out"
SCENES_CACHE = OUT_DIR / "scenes_cache.pkl"
TARGETS_JSON = OUT_DIR / "targets.json"
RESULTS_CSV = OUT_DIR / "results.csv"

OBJECT_COUNTS = [1, 5, 10, 15, 25, 50, 100, 200]
N_TARGETS = 4                       # targets per scene (sequential tour waypoints)
BASE_SEED = 1000                    # per-scene seed = BASE_SEED + index
PLAYBACK_HZ = 50.0                  # wall-clock playback rate (matches demo: 0.02 s/step)

# Representations to benchmark:
#   superquadrics  - analytic SQ SDF kernel, one primitive per superquadric
#   mesh           - SQ surfaces tessellated to triangles, ONE MESH PER SUPERQUADRIC
#                    (same primitive granularity as the SQ mode -> fair per-primitive plot)
#   pointcloud     - the REAL object point cloud turned into a watertight voxel-surface
#                    mesh via curobo's native Mesh.from_pointcloud (one mesh per object);
#                    faithful "ground-truth shape" baseline.
#   shapenet_mesh  - the ORIGINAL ShapeNet object mesh (one per object); only runs if
#                    build found mesh files (the ONet data ships only point clouds, so
#                    this is normally skipped).
REPRESENTATIONS = ["superquadrics", "mesh", "pointcloud"]

# For planner construction, the mesh/pointcloud/shapenet_mesh modes all use the mesh backend.
_PLANNER_REP = {
    "superquadrics": "superquadrics", "mesh": "mesh",
    "pointcloud": "mesh", "shapenet_mesh": "mesh",
}

# Voxel pitch (m) for Mesh.from_pointcloud in the 'pointcloud' representation.
# Smaller = more faithful to the real surface but more triangles.
PC_PITCH = 0.01

# Candidate file names for an original ShapeNet mesh inside a model directory,
# tried in order.  Extended with a generic glob for *.obj / *.off / *.ply.
ORIG_MESH_NAMES = ["model_normalized.obj", "model.obj", "mesh.obj", "mesh.off",
                   "mesh.ply", "model.off", "model.ply"]

# Collision skin: a robot sphere counts as a true collision with the object
# point cloud when the nearest point is closer than (radius - skin).  Keep at 0
# for a strict "any penetration" criterion.
COLLISION_SKIN_M = 0.0

# CSV schema shared by the tabletop and sofa-scaling benchmarks.
BENCH_FIELDNAMES = [
    "scene", "n_objects", "n_primitives", "representation", "leg", "plan_success",
    "plan_wall_s", "plan_solver_total_s", "motion_time_s", "playback_s",
    "n_steps_total", "n_steps_collide", "n_spheres_collide",
    "frac_in_collision", "max_penetration_m",
]


def scene_geometry(n: int) -> Tuple[float, float, float, float]:
    """Return (inner_r, outer_r, table_half, object_size_m) for ``n`` objects.

    The placement annulus and the table grow with object count (outer radius
    ~ sqrt(n)) so that objects spread out instead of piling up; object size
    shrinks slightly for dense scenes.  Minor overlap is acceptable.
    """
    inner_r = 0.28
    outer_r = max(0.65, 0.28 + 0.6 * math.sqrt(n / 10.0))
    if n == 50:
        outer_r = 1.2
    elif n == 200:
        outer_r = 2.0
    object_size_m = 0.54 if n <= 25 else (0.45 if n <= 100 else 0.36)
    table_half = outer_r + 0.20
    return inner_r, outer_r, table_half, object_size_m


def _make_table(table_half: float) -> Cuboid:
    return Cuboid(
        name="table",
        pose=[0.0, 0.0, -0.05, 1.0, 0.0, 0.0, 0.0],
        dims=[2.0 * table_half, 2.0 * table_half, 0.05],
    )


# ── scene building (with-replacement sampling + per-model inference cache) ────────

@dataclass
class _CachedPred:
    """A SuperDec prediction at the origin, reusable across placements."""
    outdict: dict
    sq_meshes: List[Tuple[np.ndarray, np.ndarray]]   # one (verts, faces) per superquadric
    orig_mesh: Optional[Tuple[np.ndarray, np.ndarray]]  # original ShapeNet mesh, or None
    pts: np.ndarray       # origin-centred object points [N, 3]
    colors: np.ndarray    # [N, 3]


def _per_sq_meshes(outdict: dict, resolution: int) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Tessellate each existing superquadric of an object into its own mesh."""
    handler = PredictionHandler.from_outdict(outdict, torch.zeros(1, 1, 3), ["0"])
    meshes: List[Tuple[np.ndarray, np.ndarray]] = []
    P = handler.scale.shape[1]
    for p in range(P):
        if handler.exist[0, p] > 0.5:
            v, f = handler._superquadric_mesh(
                handler.scale[0, p], handler.exponents[0, p],
                handler.rotation[0, p], handler.translation[0, p], resolution,
            )
            meshes.append((np.asarray(v, dtype=np.float32), np.asarray(f, dtype=np.int64)))
    return meshes


def _find_orig_mesh_path(dataset, idx: int) -> Optional[str]:
    """Locate an original mesh file in the model directory, if the dataset exposes paths."""
    items = getattr(dataset, "_items", None)
    if items is None:
        return None
    model_dir = os.path.dirname(items[idx])
    for name in ORIG_MESH_NAMES:
        p = os.path.join(model_dir, name)
        if os.path.isfile(p):
            return p
    # generic fallback: first mesh-like file in the directory
    for ext in (".obj", ".off", ".ply", ".glb"):
        for f in sorted(os.listdir(model_dir)):
            if f.lower().endswith(ext):
                return os.path.join(model_dir, f)
    return None


def _load_orig_mesh(
    dataset, idx: int, object_size_m: float, pts0: np.ndarray,
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """Load the original ShapeNet mesh and align it to the object's point cloud.

    The mesh is oriented y-up -> z-up (matching ``_shapenet_item_to_zup``) and then
    bbox-aligned (per-axis scale + translate) to ``pts0`` so it coincides with the
    point cloud / SQ fit regardless of how the source mesh was normalised.
    Returns origin-frame (verts, faces) or None if no mesh file is found.
    """
    path = _find_orig_mesh_path(dataset, idx)
    if path is None:
        return None
    try:
        m = trimesh.load(path, process=False, force="mesh")
        if isinstance(m, trimesh.Scene):
            m = m.dump(concatenate=True)
        v = np.asarray(m.vertices, dtype=np.float32)
        f = np.asarray(m.faces, dtype=np.int64)
        if len(v) == 0 or len(f) == 0:
            return None
        # y-up -> z-up (same rotation as the point cloud)
        v = demo.rotate_around_axis(v, axis=(1, 0, 0), angle=np.pi / 2, center_point=np.zeros(3))
        # per-axis bbox alignment to the (already scaled/rotated/floored) point cloud
        pmin, pmax = pts0.min(0), pts0.max(0)
        vmin, vmax = v.min(0), v.max(0)
        scale = (pmax - pmin) / np.maximum(vmax - vmin, 1e-6)
        v = (v - vmin) * scale + pmin
        return v.astype(np.float32), f
    except Exception as exc:
        print(f"  [orig-mesh] failed to load {path}: {exc}")
        return None


def _infer_origin_object(
    dataset, idx: int, object_size_m: float, color_idx: int,
    dataset_type: str, model, device: str, mesh_resolution: int,
) -> Optional[_CachedPred]:
    """Run SuperDec on object ``idx`` placed at the origin. None if needle-thin."""
    pts, colors = demo._shapenet_item_to_zup(dataset[idx], object_size_m, color_idx, dataset_type)
    pred = demo._infer_single_object(pts, model, device, mesh_resolution, iid=0)
    if pred is None or pred.mesh is None:
        return None
    return _CachedPred(
        outdict=pred.outdict,
        sq_meshes=_per_sq_meshes(pred.outdict, mesh_resolution),
        orig_mesh=_load_orig_mesh(dataset, idx, object_size_m, pts.astype(np.float32)),
        pts=pts.astype(np.float32),
        colors=colors.astype(np.float32),
    )


def _place_cached(cached: _CachedPred, center: np.ndarray, iid: int) -> dict:
    """Offset a cached origin prediction to a table ``center`` (x, y)."""
    offset = np.array([center[0], center[1], 0.0], dtype=np.float32)
    outdict = {
        k: (v.copy() if isinstance(v, np.ndarray) else v)
        for k, v in cached.outdict.items()
    }
    trans = np.asarray(outdict["trans"], dtype=np.float32).copy()
    trans[..., 0] += center[0]
    trans[..., 1] += center[1]
    outdict["trans"] = trans
    orig = None
    if cached.orig_mesh is not None:
        orig = (cached.orig_mesh[0] + offset, cached.orig_mesh[1])
    return {
        "iid": iid,
        "outdict": outdict,
        "sq_meshes": [(v + offset, f) for (v, f) in cached.sq_meshes],
        "orig_mesh": orig,
        "pts": cached.pts + offset,
        "colors": cached.colors,
    }


def build_benchmark_scene(
    dataset, n: int, seed: int, model, device: str, mesh_resolution: int,
) -> dict:
    """Build one scene of ``n`` objects (sampling with replacement; inference cached).

    Returns a plain-dict scene (picklable).  Distinct placements of the same model
    are separate obstacles, so scenes with more objects than the dataset are fine.
    """
    inner_r, outer_r, _table_half, object_size_m = scene_geometry(n)
    rng = np.random.RandomState(seed)

    centers = demo._place_on_table(
        n, rng,
        inner_r=inner_r, outer_r=outer_r,
        min_sep=object_size_m * 2.0 * 0.12,   # 5x tighter than original
    )

    pred_cache: Dict[int, Optional[_CachedPred]] = {}   # model idx -> cached pred or None (bad)
    object_pts: List[np.ndarray] = []
    object_colors: List[np.ndarray] = []
    predictions: List[dict] = []

    n_models = len(dataset)
    for slot, center in enumerate(centers):
        cached: Optional[_CachedPred] = None
        attempt = 0
        while cached is None:
            attempt += 1
            idx = int(rng.randint(n_models))
            if idx in pred_cache:
                cached = pred_cache[idx]            # may be None (known needle-thin)
                if cached is None:
                    continue
                break
            cached = _infer_origin_object(
                dataset, idx, object_size_m, slot, "shapenet", model, device, mesh_resolution
            )
            pred_cache[idx] = cached
            if cached is None and attempt > 200:
                raise RuntimeError(f"scene n={n}: could not find a valid object after 200 tries")

        placed = _place_cached(cached, center, iid=slot)
        predictions.append({k: placed[k] for k in ("iid", "outdict", "sq_meshes", "orig_mesh")})
        object_pts.append(placed["pts"])
        object_colors.append(placed["colors"])

    n_cached = sum(1 for v in pred_cache.values() if v is not None)
    has_orig_mesh = bool(predictions) and all(p["orig_mesh"] is not None for p in predictions)
    print(f"  scene n={n}: {len(predictions)} objects placed "
          f"({n_cached} unique models inferred, {len(pred_cache) - n_cached} rejected); "
          f"original meshes available: {has_orig_mesh}")
    return {
        "name": f"scene_{n:03d}",
        "n_objects": n,
        "object_pts": object_pts,
        "object_colors": object_colors,
        "predictions": predictions,
        "has_orig_mesh": has_orig_mesh,
    }


def _fuse_meshes(meshes: List[Tuple[np.ndarray, np.ndarray]]) -> trimesh.Trimesh:
    """Concatenate per-superquadric meshes into one trimesh (for display only)."""
    verts, faces, offset = [], [], 0
    for v, f in meshes:
        verts.append(v)
        faces.append(f + offset)
        offset += len(v)
    if not verts:
        return trimesh.Trimesh()
    return trimesh.Trimesh(
        vertices=np.concatenate(verts), faces=np.concatenate(faces), process=False
    )


def _scene_predictions(scene: dict) -> List[demo.SuperDecPrediction]:
    """Reconstruct demo.SuperDecPrediction objects (fused mesh) for display."""
    preds = []
    for p in scene["predictions"]:
        mesh = _fuse_meshes(p["sq_meshes"])
        preds.append(demo.SuperDecPrediction(iid=p["iid"], outdict=p["outdict"], mesh=mesh))
    return preds


def cmd_build(args: argparse.Namespace) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    wp.init()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Loading ShapeNet dataset…")
    dataset = demo._load_shapenet(args.shapenet_root, demo.TABLETOP_CATEGORIES)
    print(f"ShapeNet loaded: {len(dataset)} models")
    model = demo._load_model(args.checkpoint_folder, device)

    counts = args.counts if args.counts else OBJECT_COUNTS
    scenes = []
    for i, n in enumerate(counts):
        print(f"\nBuilding scene with {n} objects (seed {BASE_SEED + i})…")
        scenes.append(
            build_benchmark_scene(dataset, n, BASE_SEED + i, model, device, args.mesh_resolution)
        )

    with open(SCENES_CACHE, "wb") as f:
        pickle.dump(scenes, f)
    print(f"\nWrote {len(scenes)} scenes to {SCENES_CACHE}")


def _load_scenes() -> List[dict]:
    if not SCENES_CACHE.exists():
        raise SystemExit(f"No scene cache at {SCENES_CACHE}. Run the 'build' subcommand first.")
    with open(SCENES_CACHE, "rb") as f:
        return pickle.load(f)


# ── target setting UI ─────────────────────────────────────────────────────────────

def _load_targets() -> Dict[str, List[List[float]]]:
    if TARGETS_JSON.exists():
        with open(TARGETS_JSON) as f:
            return json.load(f)
    return {}


def cmd_set_targets(args: argparse.Namespace) -> None:
    wp.init()
    scenes = _load_scenes()
    targets = _load_targets()

    from curobo.types import ContentPath
    from curobo.viewer import ViserVisualizer

    viser_viz = ViserVisualizer(
        content_path=ContentPath(robot_config_file="franka.yml"),
        connect_ip="0.0.0.0",
        connect_port=args.port,
        add_control_frames=True,
        visualize_robot_spheres=False,
    )
    server = viser_viz._server
    state = {"scene_idx": 0, "overlays": [], "planner": None, "planner_scene_idx": -1}

    def _scene_name(i: int) -> str:
        return scenes[i]["name"]

    def _redraw(scene_idx: int) -> None:
        # clear previous obstacles / overlays
        for h in state["overlays"]:
            try:
                h.remove()
            except Exception:
                pass
        state["overlays"] = []
        for nm in ("/obstacles", "/scene"):
            try:
                server.scene.remove_by_name(nm)
            except Exception:
                pass

        scene = scenes[scene_idx]
        preds = _scene_predictions(scene)
        table = _make_table(scene_geometry(scene["n_objects"])[2])
        disp, centroids = demo._superdec_display_scene_cfg(
            preds, [table], demo.SCENE_TRANSLATION, demo.SCENE_QUAT_WXYZ, 1.0
        )
        viser_viz.add_scene(disp, add_control_frames=False)
        state["overlays"] = demo._add_superdec_overlays(
            viser_viz, scene["object_pts"], scene["object_colors"], preds,
            demo.SCENE_TRANSLATION, demo.SCENE_QUAT_WXYZ, 1.0,
            centroids=centroids, add_meshes=True,
        )

    def _frame_handle():
        # single tool frame (panda_hand); return its TransformControls handle
        return next(iter(viser_viz._control_frames.values()))

    def _set_frame_pose(pose7: List[float]) -> None:
        h = _frame_handle()
        h.position = np.array(pose7[:3], dtype=np.float32)
        h.wxyz = np.array(pose7[3:7], dtype=np.float32)

    def _status_text() -> str:
        lines = ["**Saved targets per scene:**"]
        for s in scenes:
            k = len(targets.get(s["name"], []))
            lines.append(f"- {s['name']}: {k}/{N_TARGETS}")
        return "\n".join(lines)

    scene_dd = server.gui.add_dropdown(
        "Scene", options=[s["name"] for s in scenes], initial_value=_scene_name(0)
    )
    status_md = server.gui.add_markdown(_status_text())
    save_btns = [server.gui.add_button(f"Save as Target {i + 1}") for i in range(N_TARGETS)]
    recall_dd = server.gui.add_dropdown(
        "Recall target", options=["-"] + [f"T{i + 1}" for i in range(N_TARGETS)], initial_value="-"
    )
    move_btn = server.gui.add_button("Move to current pose", color="blue")
    move_status_md = server.gui.add_markdown("")
    write_btn = server.gui.add_button("Write targets.json", color="green")

    _redraw(0)

    @scene_dd.on_update
    def _(_e) -> None:
        state["scene_idx"] = [s["name"] for s in scenes].index(scene_dd.value)
        state["planner"] = None   # invalidate cached planner for new scene
        _redraw(state["scene_idx"])
        status_md.content = _status_text()

    def _save(slot: int) -> None:
        name = _scene_name(state["scene_idx"])
        h = _frame_handle()
        pose7 = [*[float(x) for x in h.position], *[float(x) for x in h.wxyz]]
        cur = targets.get(name, [[0, 0, 0, 1, 0, 0, 0]] * N_TARGETS)
        cur = (cur + [[0, 0, 0, 1, 0, 0, 0]] * N_TARGETS)[:N_TARGETS]
        cur[slot] = pose7
        targets[name] = cur
        # persist immediately so progress is never lost
        with open(TARGETS_JSON, "w") as f:
            json.dump(targets, f, indent=2)
        status_md.content = _status_text()
        print(f"saved {name} target {slot + 1}: {pose7}")

    for i, btn in enumerate(save_btns):
        btn.on_click(lambda _e, i=i: _save(i))

    @recall_dd.on_update
    def _(_e) -> None:
        if recall_dd.value == "-":
            return
        slot = int(recall_dd.value[1:]) - 1
        name = _scene_name(state["scene_idx"])
        if name in targets and slot < len(targets[name]):
            _set_frame_pose(targets[name][slot])

    @move_btn.on_click
    def _(_e) -> None:
        import threading

        def _plan_and_animate() -> None:
            scene_idx = state["scene_idx"]
            scene = scenes[scene_idx]

            if state["planner"] is None or state["planner_scene_idx"] != scene_idx:
                move_status_md.content = "_Building planner for this scene…_"
                preds = _scene_predictions(scene)
                demo.TABLE = _make_table(scene_geometry(scene["n_objects"])[2])
                scene_cfg = demo._prediction_to_scene_cfg(
                    preds, "superquadrics", 1.0,
                    demo.SCENE_TRANSLATION, demo.SCENE_QUAT_WXYZ, torch.zeros(1),
                )
                device_cfg = DeviceCfg(device="cuda" if torch.cuda.is_available() else "cpu")
                state["planner"] = demo._rebuild_planner(scene_cfg, "superquadrics", device_cfg)
                state["planner_scene_idx"] = scene_idx

            planner = state["planner"]
            h = _frame_handle()
            pose7 = [*[float(x) for x in h.position], *[float(x) for x in h.wxyz]]

            move_status_md.content = "_Planning…_"
            tool_frame = planner.kinematics.tool_frames[0]
            goal = GoalToolPose.from_poses(
                {tool_frame: Pose.from_list(pose7)}, num_goalset=1
            )
            current_js = planner.default_joint_state.clone().unsqueeze(0)
            active_js = planner.kinematics.get_active_js(current_js)

            try:
                result = planner.plan_pose(
                    goal, active_js, use_implicit_goal=True, max_attempts=3
                )
            except Exception as exc:
                move_status_md.content = f"**Plan error:** {exc}"
                return

            if result is None or not result.success.any():
                move_status_md.content = "**Unreachable — no plan found.**"
                return

            move_status_md.content = "_Animating…_"
            interp = result.get_interpolated_plan()
            interp = planner.kinematics.get_active_js(interp)
            _animate(viser_viz, interp)
            move_status_md.content = "Reached."

        threading.Thread(target=_plan_and_animate, daemon=True).start()

    @write_btn.on_click
    def _(_e) -> None:
        with open(TARGETS_JSON, "w") as f:
            json.dump(targets, f, indent=2)
        status_md.content = _status_text() + f"\n\n**Wrote {TARGETS_JSON}**"
        print(f"Wrote {TARGETS_JSON}")

    print(f"Viser running at http://localhost:{args.port} — set targets, then 'Write targets.json'.")
    while True:
        time.sleep(1.0)


# ── benchmark ─────────────────────────────────────────────────────────────────────

def _trajectory_spheres(planner, pos_per_step: torch.Tensor) -> np.ndarray:
    """Robot collision spheres along a trajectory.

    Args:
        pos_per_step: arm positions in canonical active order, shape [T, dof].

    Returns:
        np.ndarray [T, S, 4] of (x, y, z, r) spheres per timestep.
    """
    n_steps = int(pos_per_step.shape[0])
    # joint_names left as None: positions are already in the canonical active order
    kin = planner.kinematics.compute_kinematics(JointState.from_position(pos_per_step))
    sph = kin.robot_spheres.detach().cpu().numpy().reshape(n_steps, -1, 4)
    return sph


def _collision_metrics(spheres: np.ndarray, tree: cKDTree) -> dict:
    """Ground-truth collision of robot spheres vs. object point cloud.

    A sphere collides if the nearest object point is within (radius - skin).

    Args:
        spheres: [T, S, 4] robot spheres per timestep.
    """
    n_steps = spheres.shape[0]
    n_steps_collide = 0
    n_spheres_collide = 0
    max_pen = 0.0
    for t in range(n_steps):
        sph = spheres[t]
        sph = sph[sph[:, 3] > 0.0]
        if sph.shape[0] == 0:
            continue
        dist, _ = tree.query(sph[:, :3], k=1)
        pen = sph[:, 3] - COLLISION_SKIN_M - dist
        hit = pen > 0.0
        if hit.any():
            n_steps_collide += 1
            n_spheres_collide += int(hit.sum())
            max_pen = max(max_pen, float(pen.max()))
    return {
        "n_steps_total": n_steps,
        "n_steps_collide": n_steps_collide,
        "n_spheres_collide": n_spheres_collide,
        "frac_in_collision": (n_steps_collide / n_steps) if n_steps else 0.0,
        "max_penetration_m": max_pen,
    }


def _interp_positions_per_step(interp: JointState) -> torch.Tensor:
    """Return interpolated arm positions as [T, dof]."""
    pos = interp.position
    while pos.ndim > 2 and pos.shape[0] == 1:
        pos = pos[0]
    return pos


def _build_scene_cfg(scene: dict, representation: str) -> Tuple[SceneCfg, int]:
    """Build the collision SceneCfg for one representation; return (cfg, n_primitives).

    superquadrics  one Superquadric per primitive (via demo._prediction_to_scene_cfg)
    mesh           one fused Mesh per object (same as demo._prediction_to_scene_cfg)
    shapenet_mesh  one Mesh per object, from the original ShapeNet mesh
    """
    table = _make_table(scene_geometry(scene["n_objects"])[2])
    demo.TABLE = table  # demo._prediction_to_scene_cfg reads this global

    if representation in ("superquadrics", "mesh"):
        cfg = demo._prediction_to_scene_cfg(
            _scene_predictions(scene), representation, 1.0,
            demo.SCENE_TRANSLATION, demo.SCENE_QUAT_WXYZ, torch.zeros(1),
        )
        n = (demo._count_items(cfg.superquadric) if representation == "superquadrics"
             else demo._count_items(cfg.mesh))
        return cfg, n

    meshes: List[Mesh] = []
    if representation == "pointcloud":
        # Native curobo: voxel-surface mesh of the real per-object point cloud.
        for p, pts in zip(scene["predictions"], scene["object_pts"]):
            meshes.append(Mesh.from_pointcloud(
                np.asarray(pts, dtype=np.float64), pitch=PC_PITCH,
                name=f"obj_{p['iid']}_pc",
            ))
    elif representation == "shapenet_mesh":
        for p in scene["predictions"]:
            if p["orig_mesh"] is None:
                continue
            v, f = p["orig_mesh"]
            meshes.append(Mesh(
                name=f"obj_{p['iid']}",
                vertices=v.tolist(), faces=f.tolist(),
                pose=[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
            ))
    else:
        raise ValueError(f"Unknown representation: {representation!r}")

    cfg = SceneCfg(cuboid=[table], mesh=meshes)
    print(f"Built scene: {len(meshes)} {representation} mesh primitive(s)")
    return cfg, len(meshes)


def benchmark_one(
    scene: dict, representation: str, targets: List[List[float]],
    device_cfg: DeviceCfg, tree: cKDTree, visualize, args,
) -> List[dict]:
    """Run the sequential tour for one (scene, representation). Returns per-leg rows."""
    scene_cfg, n_primitives = _build_scene_cfg(scene, representation)
    planner = demo._rebuild_planner(scene_cfg, _PLANNER_REP[representation], device_cfg)
    tool_frame = planner.kinematics.tool_frames[0]

    current_state = planner.default_joint_state.clone().unsqueeze(0)

    # Throwaway plan to first target to remove first-call / lazy-init overhead.
    if targets:
        try:
            warm_goal = GoalToolPose.from_poses(
                {tool_frame: Pose.from_list(targets[0])}, num_goalset=1
            )
            planner.plan_pose(
                warm_goal, planner.kinematics.get_active_js(current_state.clone()),
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
            "scene": scene["name"],
            "n_objects": scene["n_objects"],
            "n_primitives": n_primitives,
            "representation": representation,
            "leg": leg + 1,
            "plan_success": int(success),
            "plan_wall_s": plan_wall,
            "plan_solver_total_s": float(getattr(result, "total_time", float("nan")))
            if result is not None else float("nan"),
            "motion_time_s": float("nan"),
            "playback_s": float("nan"),
            "n_steps_total": 0,
            "n_steps_collide": 0,
            "n_spheres_collide": 0,
            "frac_in_collision": float("nan"),
            "max_penetration_m": float("nan"),
        }

        if success:
            interp = result.get_interpolated_plan()
            # Reorder once to canonical active order (interp carries joint_names).
            interp = planner.kinematics.get_active_js(interp)
            row["motion_time_s"] = float(np.asarray(result.motion_time().cpu()).reshape(-1)[0])
            pos_per_step = _interp_positions_per_step(interp).contiguous()
            n_steps = int(pos_per_step.shape[0])
            row["playback_s"] = n_steps / PLAYBACK_HZ

            # Ground-truth accuracy: robot spheres per step vs. object point cloud.
            spheres = _trajectory_spheres(planner, pos_per_step)
            row.update(_collision_metrics(spheres, tree))

            # Advance state for the next leg (continue from end of this trajectory).
            last = pos_per_step[-1:].contiguous()
            current_state = JointState.from_position(last, joint_names=interp.joint_names)
            current_state = planner.kinematics.get_full_js(current_state)

            if visualize is not None:
                _animate(visualize, interp)
        else:
            print(f"  [{representation}] {scene['name']} leg {leg + 1}: FAILED to plan")

        rows.append(row)
        c = row["frac_in_collision"]
        print(f"  [{representation}] {scene['name']} leg {leg + 1}: "
              f"success={success} plan={plan_wall:.3f}s "
              f"motion={row['motion_time_s']:.3f}s coll_frac={c}")

    del planner
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return rows


def _animate(viser_viz, interp: JointState) -> None:
    pos = _interp_positions_per_step(interp)
    jn = interp.joint_names
    for t in range(pos.shape[0]):
        viser_viz.set_joint_state(
            JointState.from_position(pos[t:t + 1], joint_names=jn).squeeze(0)
        )
        time.sleep(1.0 / PLAYBACK_HZ)


def cmd_benchmark(args: argparse.Namespace) -> None:
    wp.init()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    scenes = _load_scenes()
    targets = _load_targets()
    if not targets:
        raise SystemExit(f"No targets at {TARGETS_JSON}. Run the 'set-targets' subcommand first.")

    if args.counts:
        scenes = [s for s in scenes if s["n_objects"] in args.counts]

    device_cfg = DeviceCfg(device=args.device)

    visualize = None
    if args.visualize:
        from curobo.types import ContentPath
        from curobo.viewer import ViserVisualizer
        visualize = ViserVisualizer(
            content_path=ContentPath(robot_config_file="franka.yml"),
            connect_ip="0.0.0.0", connect_port=args.port,
            add_control_frames=False, visualize_robot_spheres=False,
        )
        print(f"Viser running at http://localhost:{args.port}")

    fieldnames = BENCH_FIELDNAMES
    all_rows: List[dict] = []

    for scene in scenes:
        scene_targets = targets.get(scene["name"])
        if not scene_targets:
            print(f"[skip] {scene['name']}: no saved targets")
            continue
        # KD-tree over the real object surface points (table excluded).
        all_pts = np.concatenate(scene["object_pts"]).astype(np.float32)
        tree = cKDTree(all_pts)
        print(f"\n=== {scene['name']} ({scene['n_objects']} objects, "
              f"{len(all_pts)} points) ===")
        for rep in REPRESENTATIONS:
            if rep == "shapenet_mesh" and not scene.get("has_orig_mesh", False):
                print(f"[skip] {scene['name']} shapenet_mesh: no original meshes in dataset")
                continue
            rows = benchmark_one(scene, rep, scene_targets, device_cfg, tree, visualize, args)
            all_rows.extend(rows)

    with open(RESULTS_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in all_rows:
            writer.writerow(r)
    print(f"\nWrote {len(all_rows)} rows to {RESULTS_CSV}")


# ── CLI ────────────────────────────────────────────────────────────────────────────

def _parse_counts(s: Optional[str]) -> Optional[List[int]]:
    if not s:
        return None
    return [int(x) for x in s.split(",") if x.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    def _common(p):
        p.add_argument("--shapenet_root", type=str,
                       default=str(demo.WORKSPACE_ROOT / "data" / "ShapeNet"))
        p.add_argument("--checkpoint_folder", type=str, default=demo.CHECKPOINT_FOLDER)
        p.add_argument("--mesh_resolution", type=int, default=48)
        p.add_argument("--counts", type=_parse_counts, default=None,
                       help="Comma-separated object counts to restrict to, e.g. 1,5,10")

    p_build = sub.add_parser("build", help="Build and cache the scene family")
    _common(p_build)
    p_build.add_argument("--force", action="store_true", help="Rebuild even if cache exists")

    p_targets = sub.add_parser("set-targets", help="Viser UI to set & save 4 targets per scene")
    _common(p_targets)
    p_targets.add_argument("--port", type=int, default=8081)

    p_bench = sub.add_parser("benchmark", help="Run the SQ-vs-mesh benchmark")
    _common(p_bench)
    p_bench.add_argument("--device", type=str, default="cuda")
    p_bench.add_argument("--visualize", action="store_true", help="Animate runs in viser")
    p_bench.add_argument("--port", type=int, default=8082)

    args = parser.parse_args()
    if args.command == "build":
        cmd_build(args)
    elif args.command == "set-targets":
        cmd_set_targets(args)
    elif args.command == "benchmark":
        cmd_benchmark(args)


if __name__ == "__main__":
    main()
