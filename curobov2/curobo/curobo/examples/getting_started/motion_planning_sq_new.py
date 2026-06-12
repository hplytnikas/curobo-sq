"""Browse SuperDec SQ predictions on val-set npz scenes with interactive motion planning.

Loads the first N .npz files from the val directory, runs SuperDec inference per scene
on demand, and displays:
  - Franka robot at origin (scene objects placed via a moveable scene-root frame)
  - SQ primitive meshes (per-instance toggle)
  - Raw point cloud (per-instance toggle)
  - Interactive target frame + Move / Grasp / Reset buttons
  - TX/TY/TZ/RX/RY/RZ sliders to reposition the whole scene; per-scene transform is
    saved to <scenes_dir>/.scene_transforms.json and restored on next load.

Usage:
    conda run -n 3dv python motion_planning_sq_new \\
        [--scenes_dir /path/to/val] \\
        [--checkpoint_folder /path/to/ckpt] \\
        [--n_scenes 10] \\
        [--port 8080] \\
        [--scene_translation -0.1 -0.5 -0.77]
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
import warp as wp
from omegaconf import OmegaConf
import trimesh
from trimesh.visual.color import ColorVisuals as _ColorVisuals
from scipy.spatial.transform import Rotation as SciRotation

from curobo._src.geom.types import Cuboid, Mesh, SceneCfg, Superquadric
from curobo._src.geom.collision.collision_scene import SceneCollisionCfg
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
from superdec.utils.predictions_handler import PredictionHandler  # type: ignore[import]
from superdec.data.dataloader import normalize_points, denormalize_outdict
from superdec.data.transform import rotate_around_axis

DEFAULT_SCENES_DIR = Path(
    "/mnt/seagate/code/3dv/npz-TO-vanilla/Volumes/ubuntu18/output/npz/TO-vanilla/val"
)
DEFAULT_CHECKPOINT = Path("/home/haroldas/3DV/superdec/checkpoints/finetuned")
DEFAULT_N_SCENES = 10
SAMPLE_POINTS = 8192
CKPT_FILE = "ckpt.pt"
SKIP_INSTANCES = {0}
MIN_RADIUS_M = 0.005
MESH_RESOLUTION = 30

DEFAULT_SCENE_TRANSLATION = np.array([-0.1, -0.5, -0.77], dtype=np.float32)
DEFAULT_SCENE_QUAT_WXYZ = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)

TABLE = Cuboid(name="table", pose=[0.0, 0.0, -0.05, 1.0, 0.0, 0.0, 0.0], dims=[1.4, 1.4, 0.05])

_COLORS: List[Tuple[int, int, int, int]] = [
    (220, 80,  80,  200), (80,  160, 220, 200), (80,  200, 100, 200),
    (220, 180, 60,  200), (180, 80,  220, 200), (80,  220, 200, 200),
    (220, 120, 60,  200), (150, 150, 220, 200), (220, 60,  150, 200),
    (100, 220, 150, 200), (180, 220, 80,  200), (220, 160, 160, 200),
    (60,  120, 180, 200), (180, 120, 60,  200), (120, 60,  180, 200),
    (60,  180, 120, 200),
]


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--scenes_dir", type=str, default=str(DEFAULT_SCENES_DIR))
    p.add_argument("--checkpoint_folder", type=str, default=str(DEFAULT_CHECKPOINT))
    p.add_argument("--n_scenes", type=int, default=DEFAULT_N_SCENES)
    p.add_argument("--port", type=int, default=8080)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument(
        "--world_representation",
        type=str,
        choices=("superquadrics", "mesh"),
        default="superquadrics",
        help="Which scene representation to plan against",
    )
    p.add_argument(
        "--scene_translation", type=float, nargs=3,
        default=DEFAULT_SCENE_TRANSLATION.tolist(),
        metavar=("TX", "TY", "TZ"),
        help="Default translation: scene-frame → robot frame",
    )
    p.add_argument(
        "--scene_quat_wxyz", type=float, nargs=4,
        default=DEFAULT_SCENE_QUAT_WXYZ.tolist(),
        metavar=("QW", "QX", "QY", "QZ"),
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def _rotation_matrix_to_wxyz(R: np.ndarray) -> List[float]:
    q = SciRotation.from_matrix(R).as_quat()
    return [float(q[3]), float(q[0]), float(q[1]), float(q[2])]


def _get_outdict_scalar(outdict: dict, key: str, idx: int) -> float:
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
    return float(val.item()) if val.size == 1 else float(val.ravel()[0])


def _apply_scene_transform(
    translation: np.ndarray,
    rotation_matrix: np.ndarray,
    scene_translation: np.ndarray,
    scene_quat_wxyz: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    scene_quat_xyzw = np.array(
        [scene_quat_wxyz[1], scene_quat_wxyz[2], scene_quat_wxyz[3], scene_quat_wxyz[0]],
        dtype=np.float32,
    )
    scene_rot = SciRotation.from_quat(scene_quat_xyzw)
    new_translation = scene_rot.apply(np.asarray(translation, dtype=np.float32)) + scene_translation
    new_rotation = (scene_rot * SciRotation.from_matrix(rotation_matrix)).as_matrix()
    return new_translation, new_rotation


def _euler_to_wxyz(rx_deg: float, ry_deg: float, rz_deg: float) -> Tuple[float, float, float, float]:
    q = SciRotation.from_euler("xyz", [rx_deg, ry_deg, rz_deg], degrees=True).as_quat()
    return (float(q[3]), float(q[0]), float(q[1]), float(q[2]))


def _transform_trimesh_mesh(
    mesh_tm: trimesh.Trimesh,
    rotation_matrix: np.ndarray,
    translation: np.ndarray,
) -> trimesh.Trimesh:
    vertices = (rotation_matrix @ np.asarray(mesh_tm.vertices, dtype=np.float32).T).T + translation
    return trimesh.Trimesh(vertices=vertices, faces=np.asarray(mesh_tm.faces), process=False)


def _colorize(mesh: trimesh.Trimesh, rgba: Tuple[int, int, int, int]) -> trimesh.Trimesh:
    colored = trimesh.Trimesh(vertices=mesh.vertices, faces=mesh.faces, process=False)
    colored.visual = _ColorVisuals(
        mesh=colored,
        vertex_colors=np.tile(np.array(rgba, dtype=np.uint8), (len(mesh.vertices), 1)),
    )
    return colored


# ---------------------------------------------------------------------------
# CuRobo scene + planner builders
# ---------------------------------------------------------------------------

def _build_scene_cfg(
    outdicts: dict,
    scene_translation: np.ndarray,
    scene_quat_wxyz: np.ndarray,
    world_representation: str = "superquadrics",
    results: List[Tuple[int, "trimesh.Trimesh"]] | None = None,
) -> SceneCfg:
    superquadrics: List[Superquadric] = []
    for iid, outdict in outdicts.items():
        n_prims = int(outdict["scale"].shape[1])
        for idx in range(n_prims):
            if _get_outdict_scalar(outdict, "exist", idx) <= 0.5:
                continue
            scale = np.asarray(outdict["scale"][0, idx], dtype=np.float32)
            exponents = np.asarray(outdict["shape"][0, idx], dtype=np.float32)
            rotation = np.asarray(outdict["rotate"][0, idx], dtype=np.float32)
            translation = np.asarray(outdict["trans"][0, idx], dtype=np.float32)
            t_trans, t_rot = _apply_scene_transform(
                translation, rotation, scene_translation, scene_quat_wxyz
            )
            pose = [float(t_trans[0]), float(t_trans[1]), float(t_trans[2]),
                    *_rotation_matrix_to_wxyz(t_rot)]
            superquadrics.append(Superquadric(
                name=f"inst_{iid}_sq_{idx}",
                pose=pose,
                radii=scale.tolist(),
                shape=exponents.tolist(),
            ))

    if world_representation == "superquadrics":
        return SceneCfg(cuboid=[TABLE], superquadric=superquadrics)

    # mesh representation: convert each instance's trimesh to a Mesh obstacle
    scene_quat_xyzw = np.array(
        [scene_quat_wxyz[1], scene_quat_wxyz[2], scene_quat_wxyz[3], scene_quat_wxyz[0]],
        dtype=np.float32,
    )
    rot = SciRotation.from_quat(scene_quat_xyzw).as_matrix()
    meshes: List[Mesh] = []
    for iid, mesh_tm in (results or []):
        if mesh_tm is None:
            continue
        tm = _transform_trimesh_mesh(mesh_tm, rot, scene_translation)
        meshes.append(Mesh(
            name=f"inst_{iid}",
            vertices=tm.vertices.tolist(),
            faces=tm.faces.tolist(),
            pose=[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
        ))
    return SceneCfg(cuboid=[TABLE], mesh=meshes)


def _build_planner(
    scene_cfg: SceneCfg,
    device_cfg: DeviceCfg,
    world_representation: str = "superquadrics",
) -> MotionPlanner:
    n_sq = len(scene_cfg.superquadric or [])
    n_cub = len(scene_cfg.cuboid or [])
    n_mesh = len(scene_cfg.mesh or [])
    use_cuda_graph = world_representation != "superquadrics"
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
        cache={"cuboid": n_cub, "mesh": n_mesh, "superquadric": n_sq},
    )
    planner = MotionPlanner(planner_cfg)
    planner.warmup(enable_graph=use_cuda_graph, num_warmup_iterations=5)
    return planner


# ---------------------------------------------------------------------------
# SuperDec inference
# ---------------------------------------------------------------------------

def _load_model(ckpt_dir: str, device: str) -> SuperDec:
    cfg = OmegaConf.load(Path(ckpt_dir) / "config.yaml")
    model = SuperDec(cfg.superdec).to(device)
    model.lm_optimization = False
    ckpt = torch.load(Path(ckpt_dir) / CKPT_FILE, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"Loaded SuperDec from {ckpt_dir}")
    return model


def _subsample(pts: np.ndarray, n: int) -> np.ndarray:
    if pts.shape[0] == n:
        return pts
    idx = np.random.choice(pts.shape[0], n, replace=pts.shape[0] < n)
    return pts[idx]


def run_inference(
    model: SuperDec,
    npz_path: Path,
    device: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[Tuple[int, trimesh.Trimesh]], dict]:
    """Return (xyz, rgb_float01, inst, [(iid, mesh), ...], {iid: outdict}) for one scene."""
    data = np.load(npz_path, allow_pickle=True)
    xyz = data["xyz"].astype(np.float32)
    rgb = data["color"].astype(np.float32)
    if rgb.max() > 1.0:
        rgb = rgb / 255.0
    inst = data["instance_label"].astype(np.float64)

    print(f"Instance IDs in {npz_path.name}: {np.unique(inst)}")
    instance_ids = [i for i in np.unique(inst) if i not in SKIP_INSTANCES]

    outdicts: dict = {}
    results: List[Tuple[int, trimesh.Trimesh]] = []

    for iid in instance_ids:
        obj_pts_scene = xyz[inst == iid]
        obj_pts = _subsample(obj_pts_scene, SAMPLE_POINTS)

        obj_yup = rotate_around_axis(obj_pts, axis=(1, 0, 0), angle=-np.pi / 2, center_point=np.zeros(3))
        pts_norm, translation, scale = normalize_points(obj_yup)
        pts_t = torch.from_numpy(pts_norm).unsqueeze(0).to(device).float()

        with torch.no_grad():
            out = model(pts_t)
        out = {k: (v.cpu() if isinstance(v, torch.Tensor) else v) for k, v in out.items()}
        out = denormalize_outdict(out, np.array([translation]), np.array([scale]), z_up=True)

        s_np = np.asarray(out["scale"]).astype(np.float32)
        drop_mask = (
            (np.asarray(out["exist"]).reshape(s_np.shape[1]) > 0.5)
            & (s_np[0].min(axis=1) < MIN_RADIUS_M)
        )
        out["exist"][0, np.where(drop_mask)[0]] = 0.0

        pts_for_handler = torch.from_numpy(obj_pts_scene[None].astype(np.float32))
        handler = PredictionHandler.from_outdict(out, pts_for_handler[:, :SAMPLE_POINTS], [str(int(iid))])
        mesh = handler.get_meshes(resolution=MESH_RESOLUTION)[0]
        if mesh is None:
            print(f"    instance {int(iid)}: no mesh, skipping")
            continue
        outdicts[int(iid)] = out
        results.append((int(iid), mesh))
        print(f"    instance {int(iid)}: {len(mesh.vertices)} verts")

    return xyz, rgb, inst, results, outdicts


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = _parse_args()
    wp.init()

    scenes_dir = Path(args.scenes_dir)
    npz_files = sorted(scenes_dir.glob("*.npz"))[: args.n_scenes]
    if not npz_files:
        print(f"No .npz files found in {scenes_dir}")
        return
    print(f"Found {len(npz_files)} scenes")

    device = args.device if torch.cuda.is_available() else "cpu"
    device_cfg = DeviceCfg(device=torch.device(device))
    model = _load_model(args.checkpoint_folder, device)

    # Default transform from args (used when no saved transform exists for a scene)
    default_translation = np.array(args.scene_translation, dtype=np.float32)
    default_quat_wxyz = np.array(args.scene_quat_wxyz, dtype=np.float32)
    default_quat_xyzw = np.array([
        default_quat_wxyz[1], default_quat_wxyz[2],
        default_quat_wxyz[3], default_quat_wxyz[0],
    ])
    default_euler = SciRotation.from_quat(default_quat_xyzw).as_euler("xyz", degrees=True)

    # Per-scene saved transforms
    transforms_file = scenes_dir / ".scene_transforms.json"
    if transforms_file.exists():
        with open(transforms_file) as f:
            saved_transforms: dict = json.load(f)
    else:
        saved_transforms = {}

    cache: dict[str, tuple] = {}
    _lock = threading.Lock()
    _busy = [False]

    viser_viz = ViserVisualizer(
        content_path=ContentPath(robot_config_file="franka.yml"),
        connect_ip="0.0.0.0",
        connect_port=args.port,
        add_control_frames=True,
        visualize_robot_spheres=False,
    )
    server = viser_viz._server
    server.scene.set_up_direction("+z")

    _initial_control_frames = {
        name: (tuple(frame.position), tuple(frame.wxyz))
        for name, frame in viser_viz._control_frames.items()
    }

    # Persistent scene-root frame — all scene objects live under /scene_root/
    # so updating this frame's position/wxyz moves everything at once.
    scene_root_frame = server.scene.add_frame(
        "/scene_root",
        wxyz=tuple(default_quat_wxyz.tolist()),
        position=tuple(default_translation.tolist()),
        show_axes=False,
    )

    # ------------------------------------------------------------------
    # GUI layout
    # ------------------------------------------------------------------

    scene_names = [f.stem for f in npz_files]
    scene_dropdown = server.gui.add_dropdown("Scene", options=scene_names, initial_value=scene_names[0])
    status_text = server.gui.add_text("Status", initial_value="Loading…")
    repr_dropdown = server.gui.add_dropdown(
        "Representation",
        options=["superquadrics", "mesh"],
        initial_value=args.world_representation,
    )
    current_representation = [args.world_representation]
    all_sq_toggle = server.gui.add_checkbox("All Superquadrics", initial_value=True)
    all_pc_toggle = server.gui.add_checkbox("All Point Clouds", initial_value=True)

    with server.gui.add_folder("Scene Transform"):
        tx_slider = server.gui.add_slider("TX", min=-3.0, max=3.0, step=0.005,
                                          initial_value=float(default_translation[0]))
        ty_slider = server.gui.add_slider("TY", min=-3.0, max=3.0, step=0.005,
                                          initial_value=float(default_translation[1]))
        tz_slider = server.gui.add_slider("TZ", min=-3.0, max=3.0, step=0.005,
                                          initial_value=float(default_translation[2]))
        rx_slider = server.gui.add_slider("RX°", min=-180.0, max=180.0, step=1.0,
                                          initial_value=float(default_euler[0]))
        ry_slider = server.gui.add_slider("RY°", min=-180.0, max=180.0, step=1.0,
                                          initial_value=float(default_euler[1]))
        rz_slider = server.gui.add_slider("RZ°", min=-180.0, max=180.0, step=1.0,
                                          initial_value=float(default_euler[2]))
        save_tf_btn = server.gui.add_button("Save Transform", color="green")
        rebuild_btn = server.gui.add_button("Rebuild Planner", color="blue")

    # Per-instance visual handles: (folder_h, sq_cb, mesh_h, pc_cb, pc_h)
    inst_toggle_handles: list = []
    bg_pc_handle: list = []
    bg_pc_toggle: list = []

    current_planner: list = [None]
    current_state: list = [None]
    is_moving = [False]
    transform_dirty = [False]

    # ------------------------------------------------------------------
    # Transform helpers
    # ------------------------------------------------------------------

    def _get_current_transform() -> Tuple[np.ndarray, np.ndarray]:
        t = np.array([tx_slider.value, ty_slider.value, tz_slider.value], dtype=np.float32)
        wxyz = np.array(
            _euler_to_wxyz(rx_slider.value, ry_slider.value, rz_slider.value),
            dtype=np.float32,
        )
        return t, wxyz

    def _sync_frame() -> None:
        t, wxyz = _get_current_transform()
        scene_root_frame.position = tuple(t.tolist())
        scene_root_frame.wxyz = tuple(wxyz.tolist())
        transform_dirty[0] = True

    for _sl in (tx_slider, ty_slider, tz_slider, rx_slider, ry_slider, rz_slider):
        @_sl.on_update
        def _(_event) -> None:
            _sync_frame()

    # ------------------------------------------------------------------
    # Visual helpers
    # ------------------------------------------------------------------

    def _wire_inst_toggles(sq_cb, mesh_h, pc_cb, pc_h) -> None:
        @sq_cb.on_update
        def _(_event) -> None:
            mesh_h.visible = all_sq_toggle.value and sq_cb.value

        @pc_cb.on_update
        def _(_event) -> None:
            pc_h.visible = all_pc_toggle.value and pc_cb.value

    @all_sq_toggle.on_update
    def _(_) -> None:
        for _, sq_cb, mesh_h, _, _ in inst_toggle_handles:
            mesh_h.visible = all_sq_toggle.value and sq_cb.value

    @all_pc_toggle.on_update
    def _(_) -> None:
        for _, _, _, pc_cb, pc_h in inst_toggle_handles:
            pc_h.visible = all_pc_toggle.value and pc_cb.value
        if bg_pc_handle:
            bg_val = bg_pc_toggle[0].value if bg_pc_toggle else True
            bg_pc_handle[0].visible = all_pc_toggle.value and bg_val

    def _clear() -> None:
        for folder_h, sq_cb, mesh_h, pc_cb, pc_h in inst_toggle_handles:
            for obj in (folder_h, sq_cb, mesh_h, pc_cb, pc_h):
                try:
                    obj.remove()
                except Exception:
                    pass
        inst_toggle_handles.clear()
        if bg_pc_handle:
            try:
                bg_pc_handle[0].remove()
            except Exception:
                pass
            bg_pc_handle.clear()
        if bg_pc_toggle:
            try:
                bg_pc_toggle[0].remove()
            except Exception:
                pass
            bg_pc_toggle.clear()

    # ------------------------------------------------------------------
    # Motion planning callbacks
    # ------------------------------------------------------------------

    def execute_trajectory(trajectory) -> None:
        traj = trajectory.squeeze(0)
        for i in range(traj.position.shape[-2]):
            if not is_moving[0]:
                return
            viser_viz.set_joint_state(JointState.from_position(
                traj.position[0, i, :].unsqueeze(0), joint_names=traj.joint_names,
            ).squeeze(0))
            time.sleep(0.02)
        current_state[0] = JointState.from_position(
            traj.position[0, -1, :].unsqueeze(0), joint_names=traj.joint_names,
        )

    def on_move(_) -> None:
        if is_moving[0] or current_planner[0] is None:
            return

        def _plan() -> None:
            is_moving[0] = True
            planner = current_planner[0]
            target_poses = viser_viz.get_control_frame_pose()
            active_js = planner.kinematics.get_active_js(current_state[0].clone())
            result = planner.plan_pose(
                GoalToolPose.from_poses(target_poses, num_goalset=1),
                active_js, use_implicit_goal=True, max_attempts=3,
            )
            if result is not None and result.success.any():
                execute_trajectory(result.get_interpolated_plan())
            else:
                print("Motion planning failed; status:", getattr(result, "status", None))
            is_moving[0] = False

        threading.Thread(target=_plan, daemon=True).start()

    def on_grasp(_) -> None:
        if is_moving[0] or current_planner[0] is None:
            return

        def _plan() -> None:
            is_moving[0] = True
            planner = current_planner[0]
            target_poses = viser_viz.get_control_frame_pose()
            active_js = planner.kinematics.get_active_js(current_state[0].clone())
            offset = Pose.from_list([0.0, 0.0, -0.15, 1.0, 0.0, 0.0, 0.0])
            approach_poses = {f: p.multiply(offset) for f, p in target_poses.items()}

            approach_result = planner.plan_pose(
                GoalToolPose.from_poses(approach_poses, num_goalset=1),
                active_js, max_attempts=5,
            )
            if approach_result is None or not approach_result.success.any():
                print("Grasp failed: approach unreachable")
                is_moving[0] = False
                return

            approach_end = planner.kinematics.get_active_js(JointState.from_position(
                approach_result.js_solution.position[0, 0, -1, :].unsqueeze(0),
                joint_names=approach_result.js_solution.joint_names,
            ))
            grasp_result = planner.plan_pose(
                GoalToolPose.from_poses(target_poses, num_goalset=1),
                approach_end, max_attempts=5,
            )
            if grasp_result is None or not grasp_result.success.any():
                print("Grasp failed: grasp unreachable from approach")
                is_moving[0] = False
                return

            grasp_end = planner.kinematics.get_active_js(JointState.from_position(
                grasp_result.js_solution.position[0, 0, -1, :].unsqueeze(0),
                joint_names=grasp_result.js_solution.joint_names,
            ))
            lift_result = planner.plan_pose(
                GoalToolPose.from_poses(approach_poses, num_goalset=1),
                grasp_end, max_attempts=5,
            )
            execute_trajectory(approach_result.get_interpolated_plan())
            execute_trajectory(grasp_result.get_interpolated_plan())
            if lift_result is not None and lift_result.success.any():
                execute_trajectory(lift_result.get_interpolated_plan())
            else:
                print("Lift planning failed, skipping")
            is_moving[0] = False

        threading.Thread(target=_plan, daemon=True).start()

    def _reset(_) -> None:
        is_moving[0] = False
        if current_planner[0] is not None:
            init_js = current_planner[0].default_joint_state.clone()
            current_state[0] = init_js.unsqueeze(0)
            viser_viz.set_joint_state(init_js)
        for name, (pos, wxyz) in _initial_control_frames.items():
            if name in viser_viz._control_frames:
                viser_viz._control_frames[name].position = pos
                viser_viz._control_frames[name].wxyz = wxyz

    move_btn = server.gui.add_button("Move", color="green")
    move_btn.on_click(on_move)
    grasp_btn = server.gui.add_button("Grasp", color="blue")
    grasp_btn.on_click(on_grasp)
    reset_btn = server.gui.add_button("Reset", color="red")
    reset_btn.on_click(_reset)

    # ------------------------------------------------------------------
    # Save transform / rebuild planner
    # ------------------------------------------------------------------

    @save_tf_btn.on_click
    def _(_event) -> None:
        stem = scene_dropdown.value
        saved_transforms[stem] = {
            "tx": float(tx_slider.value), "ty": float(ty_slider.value),
            "tz": float(tz_slider.value), "rx": float(rx_slider.value),
            "ry": float(ry_slider.value), "rz": float(rz_slider.value),
        }
        with open(transforms_file, "w") as fh:
            json.dump(saved_transforms, fh, indent=2)
        status_text.value = f"Saved transform for '{stem}'"
        print(f"[transform] saved for '{stem}' → {transforms_file}")

    @repr_dropdown.on_update
    def _(_event) -> None:
        stem = scene_dropdown.value
        if stem not in cache or is_moving[0]:
            return

        def _switch() -> None:
            is_moving[0] = True
            new_rep = repr_dropdown.value
            current_representation[0] = new_rep
            status_text.value = f"Switching to {new_rep}…"
            try:
                t, q_wxyz = _get_current_transform()
                _, _, _, results_local, outdicts = cache[stem]
                scene_cfg = _build_scene_cfg(outdicts, t, q_wxyz, new_rep, results_local)
                planner = _build_planner(scene_cfg, device_cfg, new_rep)
                current_planner[0] = planner
                init_js = planner.default_joint_state.clone()
                current_state[0] = init_js.unsqueeze(0)
                viser_viz.set_joint_state(init_js)
                transform_dirty[0] = False
                n = len(scene_cfg.superquadric or []) + len(scene_cfg.mesh or [])
                status_text.value = f"{stem} — {new_rep}, {n} primitives"
            except Exception as exc:
                status_text.value = f"Switch failed: {exc}"
                import traceback
                traceback.print_exc()
            finally:
                is_moving[0] = False

        threading.Thread(target=_switch, daemon=True).start()

    @rebuild_btn.on_click
    def _(_event) -> None:
        stem = scene_dropdown.value
        if stem not in cache or is_moving[0]:
            return

        def _rebuild() -> None:
            is_moving[0] = False
            status_text.value = f"Rebuilding planner for {stem}…"
            try:
                t, q_wxyz = _get_current_transform()
                _, _, _, results_local, outdicts = cache[stem]
                rep = current_representation[0]
                scene_cfg = _build_scene_cfg(outdicts, t, q_wxyz, rep, results_local)
                planner = _build_planner(scene_cfg, device_cfg, rep)
                current_planner[0] = planner
                init_js = planner.default_joint_state.clone()
                current_state[0] = init_js.unsqueeze(0)
                viser_viz.set_joint_state(init_js)
                transform_dirty[0] = False
                n = len(scene_cfg.superquadric or []) + len(scene_cfg.mesh or [])
                status_text.value = f"{stem} — planner rebuilt ({rep}), {n} primitives"
            except Exception as exc:
                status_text.value = f"Rebuild failed: {exc}"
                import traceback
                traceback.print_exc()

        threading.Thread(target=_rebuild, daemon=True).start()

    # ------------------------------------------------------------------
    # Scene loader
    # ------------------------------------------------------------------

    def _show_scene(stem: str) -> None:
        with _lock:
            if _busy[0]:
                return
            _busy[0] = True
        is_moving[0] = False
        try:
            status_text.value = f"Running inference on {stem}…"
            if stem not in cache:
                npz_path = next(f for f in npz_files if f.stem == stem)
                cache[stem] = run_inference(model, npz_path, device)

            xyz, rgb, inst, results, outdicts = cache[stem]
            _clear()

            # Restore saved (or default) transform for this scene
            if stem in saved_transforms:
                t = saved_transforms[stem]
                tx_slider.value = t["tx"]
                ty_slider.value = t["ty"]
                tz_slider.value = t["tz"]
                rx_slider.value = t["rx"]
                ry_slider.value = t["ry"]
                rz_slider.value = t["rz"]
            else:
                tx_slider.value = float(default_translation[0])
                ty_slider.value = float(default_translation[1])
                tz_slider.value = float(default_translation[2])
                rx_slider.value = float(default_euler[0])
                ry_slider.value = float(default_euler[1])
                rz_slider.value = float(default_euler[2])
            _sync_frame()

            rgb_uint8 = (rgb * 255).clip(0, 255).astype(np.uint8)

            # Add all scene objects in raw scene-frame coords under /scene_root/
            # The scene_root frame applies the transform → no pre-transformation needed.
            for inst_i, (iid, mesh) in enumerate(results):
                mesh_h = server.scene.add_mesh_trimesh(
                    name=f"/scene_root/sq/inst_{iid}",
                    mesh=_colorize(mesh, _COLORS[inst_i % len(_COLORS)]),
                    visible=True,
                )
                mask = inst == iid
                pc_h = server.scene.add_point_cloud(
                    name=f"/scene_root/pointcloud/inst_{iid}",
                    points=xyz[mask], colors=rgb_uint8[mask],
                    point_size=0.003, visible=True,
                )
                with server.gui.add_folder(f"Inst {iid}") as folder_h:
                    sq_cb = server.gui.add_checkbox("Superquadric", initial_value=True)
                    pc_cb = server.gui.add_checkbox("Point cloud", initial_value=True)
                inst_toggle_handles.append((folder_h, sq_cb, mesh_h, pc_cb, pc_h))
                _wire_inst_toggles(sq_cb, mesh_h, pc_cb, pc_h)

            bg_h = server.scene.add_point_cloud(
                name="/scene_root/pointcloud/bg",
                points=xyz[inst == 0], colors=rgb_uint8[inst == 0],
                point_size=0.003, visible=True,
            )
            bg_pc_handle.append(bg_h)
            cb_bg = server.gui.add_checkbox("Background PC", initial_value=True)
            bg_pc_toggle.append(cb_bg)

            @cb_bg.on_update
            def _(_) -> None:
                bg_h.visible = cb_bg.value

            status_text.value = f"Building planner for {stem}…"
            cur_t, cur_q = _get_current_transform()
            rep = current_representation[0]
            scene_cfg = _build_scene_cfg(outdicts, cur_t, cur_q, rep, results)
            planner = _build_planner(scene_cfg, device_cfg, rep)
            current_planner[0] = planner
            init_js = planner.default_joint_state.clone()
            current_state[0] = init_js.unsqueeze(0)
            viser_viz.set_joint_state(init_js)
            transform_dirty[0] = False

            n = len(scene_cfg.superquadric or []) + len(scene_cfg.mesh or [])
            status_text.value = f"{stem} — {len(results)} instances, {n} primitives ({rep})"
        except Exception as exc:
            status_text.value = f"Error: {exc}"
            import traceback
            traceback.print_exc()
        finally:
            with _lock:
                _busy[0] = False

    @scene_dropdown.on_update
    def _(_event) -> None:
        threading.Thread(target=_show_scene, args=(scene_dropdown.value,), daemon=True).start()

    threading.Thread(target=_show_scene, args=(scene_names[0],), daemon=True).start()

    print(f"\nInteractive viewer at http://localhost:{args.port}")
    print("  - Drag the target frame to set goal pose")
    print("  - Use Scene Transform sliders to align the scene, then Save Transform")
    print("  - After moving the scene, click 'Rebuild Planner' before planning")
    print("Press Ctrl+C to exit.")
    try:
        while True:
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("\nShutting down.")


if __name__ == "__main__":
    main()
