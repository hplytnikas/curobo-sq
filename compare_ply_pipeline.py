"""Compare chair.ply (working) vs obj_00.ply (broken) through the full pipeline.

Prints side-by-side diagnostics at every stage:
  1. Raw PLY properties
  2. Pre-processing / normalisation
  3. Model inference outputs (per-primitive)
  4. Denormalised SQ parameters in world frame
  5. Collision-mesh properties (watertight, normals, volume, bbox)

Run with:
    conda run -n 3dv python compare_ply_pipeline.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import torch
import trimesh
import trimesh.repair
from omegaconf import OmegaConf
from scipy.spatial.transform import Rotation as SciRotation

WORKSPACE_ROOT = Path(__file__).resolve().parent
SUPERDEC_ROOT = WORKSPACE_ROOT / "superdec"
if str(SUPERDEC_ROOT) not in sys.path:
    sys.path.append(str(SUPERDEC_ROOT))

from superdec.superdec import SuperDec
from superdec.utils.predictions_handler import PredictionHandler

# ── config ────────────────────────────────────────────────────────────────────
CHAIR_PLY      = "/home/haroldas/3DV/superdec/examples/chair.ply"
OBJ00_PLY      = "/home/haroldas/3DV/superdec/examples/Archive/objects_pc_normalized/obj_00.ply"
NORM_NPZ       = "/home/haroldas/3DV/superdec/examples/Archive/objects_pc_normalized/normalization.npz"
CHECKPOINT_DIR = "/home/haroldas/3DV/superdec/checkpoints/normalized"

# chair.ply pre-processing (from _load_superdec_outputs)
CHAIR_ROTATE_DEG_X   = 90.0
CHAIR_TRANSLATION    = np.array([0.3, 0.8, -0.2], dtype=np.float32)
CHAIR_SAMPLE_POINTS  = 8192
OBJ_SAMPLE_POINTS    = 4096

SEP = "─" * 70


# ── helpers ───────────────────────────────────────────────────────────────────

def load_ply_points(path: str) -> np.ndarray:
    pc = trimesh.load(path, process=False)
    if isinstance(pc, trimesh.Scene):
        pts = np.concatenate([
            np.asarray(g.vertices) for g in pc.geometry.values()
            if hasattr(g, "vertices")
        ], axis=0)
    else:
        pts = np.asarray(pc.vertices)
    return pts.astype(np.float32)


def subsample(pts: np.ndarray, n: int) -> np.ndarray:
    idx = np.random.default_rng(42).choice(len(pts), n, replace=len(pts) < n)
    return pts[idx]


def normalize(pts: np.ndarray):
    center = pts.mean(axis=0)
    centered = pts - center
    scale = float(2.0 * np.abs(centered).max())
    return (centered / scale).astype(np.float32), center, scale


def denorm_outdict(out: dict, center: np.ndarray, scale: float) -> dict:
    s = np.asarray([[scale]], dtype=np.float32)
    t = center.reshape(1, 1, 3)
    out = dict(out)
    out["scale"] = out["scale"] * s[:, :, None]
    out["trans"] = out["trans"] * s[:, :, None] + t
    return out


def print_raw(label: str, pts: np.ndarray) -> None:
    print(f"\n{'━'*70}")
    print(f"  {label}")
    print(f"{'━'*70}")
    print(f"  points          : {pts.shape[0]:,}")
    print(f"  dtype           : {pts.dtype}")
    bb_min = pts.min(axis=0)
    bb_max = pts.max(axis=0)
    print(f"  bbox min (xyz)  : {bb_min}")
    print(f"  bbox max (xyz)  : {bb_max}")
    print(f"  bbox extent     : {bb_max - bb_min}")
    print(f"  centroid        : {pts.mean(axis=0)}")


def print_norm(label: str, pts_norm: np.ndarray, center: np.ndarray, scale: float) -> None:
    print(f"\n[norm] {label}")
    print(f"  center          : {center}")
    print(f"  scale           : {scale:.6f}")
    print(f"  norm range      : [{pts_norm.min():.4f}, {pts_norm.max():.4f}]")
    print(f"  norm centroid   : {pts_norm.mean(axis=0)}")


def _active(out: dict):
    """Return (scales, shapes, trans) for active primitives, handling any batch shape."""
    exist = np.asarray(out["exist"]).flatten()   # (n_primitives,)
    scales = np.asarray(out["scale"]).reshape(-1, 3)
    shapes = np.asarray(out["shape"]).reshape(-1, 2)
    trans  = np.asarray(out["trans"]).reshape(-1, 3)
    active = exist > 0.5
    return exist, active, scales[active], shapes[active], trans[active]


def print_outdict(label: str, out: dict) -> None:
    print(f"\n[model output] {label}")
    # print raw shapes for diagnostics
    for k in ("exist", "scale", "shape", "trans", "rotate"):
        if k in out:
            v = out[k]
            shp = tuple(v.shape) if hasattr(v, "shape") else "?"
            print(f"  tensor '{k}' shape: {shp}")

    exist, active, scales, shapes, trans = _active(out)
    n_active = active.sum()
    print(f"  primitives total / active : {len(exist)} / {n_active}")

    if n_active == 0:
        print("  (no active primitives)")
        return

    print(f"  radii  min/max  : {scales.min():.4f} / {scales.max():.4f}")
    print(f"  radii  mean     : {scales.mean(axis=0)}")
    print(f"  shape  min/max  : {shapes.min():.4f} / {shapes.max():.4f}  (eps1, eps2)")
    print(f"  shape  mean     : {shapes.mean(axis=0)}")
    print(f"  trans  min/max  : {trans.min():.4f} / {trans.max():.4f}")
    print(f"  trans  mean     : {trans.mean(axis=0)}")

    # flag suspicious values
    if shapes.min() < 0.05:
        print(f"  ⚠ very small exponent(s): {shapes[shapes.min(axis=1) < 0.05]}")
    if scales.max() > 0.6:
        print(f"  ⚠ large normalised radius: {scales.max():.4f}")


def print_world(label: str, out_world: dict) -> None:
    print(f"\n[world-frame SQs] {label}")
    _, active, scales, shapes, trans = _active(out_world)

    print(f"  radii  min/max  : {scales.min():.4f} / {scales.max():.4f} m")
    print(f"  radii  mean     : {scales.mean(axis=0)} m")
    print(f"  trans  min/max  : {trans.min():.4f} / {trans.max():.4f} m")
    print(f"  trans  mean     : {trans.mean(axis=0)} m")
    print(f"  z range         : [{trans[:, 2].min():.3f}, {trans[:, 2].max():.3f}] m")

    large = scales.max(axis=1) > 0.4
    if large.any():
        print(f"  ⚠ {large.sum()} SQ(s) with max-radius > 0.4 m (world frame)")

    # Print per-primitive summary
    print(f"\n  {'idx':>4}  {'eps1':>6}  {'eps2':>6}  {'rx':>7}  {'ry':>7}  {'rz':>7}  {'tx':>7}  {'ty':>7}  {'tz':>7}")
    for i, (sc, sh, tr) in enumerate(zip(scales, shapes, trans)):
        print(f"  {i:>4}  {sh[0]:>6.3f}  {sh[1]:>6.3f}  {sc[0]:>7.4f}  {sc[1]:>7.4f}  {sc[2]:>7.4f}  {tr[0]:>7.3f}  {tr[1]:>7.3f}  {tr[2]:>7.3f}")


def print_mesh_props(label: str, mesh_tm: trimesh.Trimesh) -> None:
    print(f"\n[collision mesh] {label}")
    print(f"  vertices        : {len(mesh_tm.vertices):,}")
    print(f"  faces           : {len(mesh_tm.faces):,}")
    print(f"  is_watertight   : {mesh_tm.is_watertight}")
    print(f"  is_winding_cons.: {mesh_tm.is_winding_consistent}")
    print(f"  volume          : {mesh_tm.volume:.6f} m³")
    bb = mesh_tm.bounds
    print(f"  bbox min        : {bb[0]}")
    print(f"  bbox max        : {bb[1]}")
    print(f"  bbox extent     : {bb[1] - bb[0]}")
    # check for degenerate / NaN
    nan_v = np.isnan(mesh_tm.vertices).any()
    nan_f = np.isnan(mesh_tm.faces.astype(float)).any()
    print(f"  NaN in vertices : {nan_v}")
    print(f"  NaN in faces    : {nan_f}")


# ── load model once ───────────────────────────────────────────────────────────

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")
checkpoint = torch.load(os.path.join(CHECKPOINT_DIR, "ckpt.pt"), map_location=device, weights_only=False)
configs    = OmegaConf.load(os.path.join(CHECKPOINT_DIR, "config.yaml"))
model      = SuperDec(configs.superdec).to(device)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()
print("Model loaded.")


# ════════════════════════════════════════════════════════════════════════════
# CHAIR.PLY  (working path)
# ════════════════════════════════════════════════════════════════════════════
print(f"\n{'═'*70}")
print("  CHAIR.PLY  (working)")
print(f"{'═'*70}")

chair_raw = load_ply_points(CHAIR_PLY)
print_raw("chair.ply — raw", chair_raw)

# pre-process: rotate +90° X, translate
rot_x = SciRotation.from_euler("x", CHAIR_ROTATE_DEG_X, degrees=True)
chair_processed = rot_x.apply(chair_raw) + CHAIR_TRANSLATION
print_raw("chair.ply — after rotate+translate", chair_processed)

chair_sampled = subsample(chair_processed, CHAIR_SAMPLE_POINTS)
chair_norm, chair_center, chair_scale = normalize(chair_sampled)
print_norm("chair.ply", chair_norm, chair_center, chair_scale)

pts_t = torch.from_numpy(chair_norm).unsqueeze(0).to(device)
with torch.no_grad():
    chair_out = model(pts_t)
chair_out = {k: (v.cpu() if isinstance(v, torch.Tensor) else v) for k, v in chair_out.items()}

print_outdict("chair.ply", chair_out)

chair_out_world = denorm_outdict(chair_out, np.asarray(chair_center, np.float32), chair_scale)
print_world("chair.ply", chair_out_world)

# native mesh
pts_tensor_chair = torch.from_numpy(chair_norm[None]).float()
pred_handler_chair = PredictionHandler.from_outdict(chair_out, pts_tensor_chair, ["chair"])
meshes_chair = pred_handler_chair.get_meshes(resolution=48, colors=False)
if meshes_chair and meshes_chair[0] is not None:
    parts_chair = meshes_chair[0].split() or [meshes_chair[0]]
    print(f"\n  Native mesh parts: {len(parts_chair)}")
    for i, m in enumerate(parts_chair[:3]):
        tm = trimesh.Trimesh(vertices=m.vertices, faces=m.faces, process=False)
        print_mesh_props(f"chair part {i}", tm)
        repaired = tm.copy()
        trimesh.repair.fill_holes(repaired)
        trimesh.repair.fix_normals(repaired, multibody=True)
        print(f"    → after repair: watertight={repaired.is_watertight}")
else:
    print("  No native mesh produced.")


# ════════════════════════════════════════════════════════════════════════════
# OBJ_00.PLY  (broken path)
# ════════════════════════════════════════════════════════════════════════════
print(f"\n{'═'*70}")
print("  OBJ_00.PLY  (broken)")
print(f"{'═'*70}")

obj_raw = load_ply_points(OBJ00_PLY)
print_raw("obj_00.ply — as loaded (already normalised)", obj_raw)

# load stored normalisation params for obj_id=0
norm_data = np.load(NORM_NPZ, allow_pickle=True)
idx0 = list(norm_data["obj_id"]).index(0)
obj_center = np.asarray(norm_data["center"][idx0], dtype=np.float32)
obj_scale  = float(norm_data["scale"][idx0])
print(f"\n[stored norm params]  center={obj_center}  scale={obj_scale:.6f}")

obj_sampled = subsample(obj_raw, OBJ_SAMPLE_POINTS)
# no further normalisation — points are already in [-0.5, 0.5]
print_norm("obj_00.ply (from stored params)", obj_sampled, obj_center, obj_scale)

pts_t2 = torch.from_numpy(obj_sampled).unsqueeze(0).to(device)
with torch.no_grad():
    obj_out = model(pts_t2)
obj_out = {k: (v.cpu() if isinstance(v, torch.Tensor) else v) for k, v in obj_out.items()}

print_outdict("obj_00.ply", obj_out)

obj_out_world = denorm_outdict(obj_out, obj_center, obj_scale)
print_world("obj_00.ply", obj_out_world)

# native mesh — use normalised points (what model saw) for PredictionHandler
pts_tensor_obj = torch.from_numpy(obj_sampled[None]).float()
pred_handler_obj = PredictionHandler.from_outdict(obj_out, pts_tensor_obj, ["obj00"])
meshes_obj = pred_handler_obj.get_meshes(resolution=48, colors=False)
if meshes_obj and meshes_obj[0] is not None:
    parts_obj = meshes_obj[0].split() or [meshes_obj[0]]
    print(f"\n  Native mesh parts: {len(parts_obj)}")
    for i, m in enumerate(parts_obj[:3]):
        tm = trimesh.Trimesh(vertices=m.vertices, faces=m.faces, process=False)
        print_mesh_props(f"obj_00 part {i}", tm)
        repaired = tm.copy()
        trimesh.repair.fill_holes(repaired)
        trimesh.repair.fix_normals(repaired, multibody=True)
        print(f"    → after repair: watertight={repaired.is_watertight}")
else:
    print("  No native mesh produced.")


# ════════════════════════════════════════════════════════════════════════════
# POST-SCENE-TRANSFORM positions  (where do SQs land in ROBOT frame?)
# ════════════════════════════════════════════════════════════════════════════
print(f"\n{'═'*70}")
print("  POST-SCENE-TRANSFORM POSITIONS IN ROBOT FRAME")
print(f"{'═'*70}")

def apply_scene(trans_world: np.ndarray, quat_wxyz: np.ndarray, translation: np.ndarray) -> np.ndarray:
    """Apply rigid scene transform [quat + translation] to an array of 3D points."""
    xyzw = np.array([quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]], dtype=np.float32)
    rot = SciRotation.from_quat(xyzw)
    return rot.apply(trans_world) + translation

CHAIR_SCENE_QUAT  = np.array([0.70711, 0.70711, 0.0, 0.0], dtype=np.float32)  # +90° X
CHAIR_SCENE_T     = np.array([-0.29955, -0.68389, 0.13559], dtype=np.float32)
NPZ_SCENE_QUAT    = np.array([0.70711, -0.70711, 0.0, 0.0], dtype=np.float32)  # -90° X
NPZ_SCENE_T       = np.array([-0.3, -0.2, 0.03], dtype=np.float32)

_, _, sc_cw, _, tr_cw_world = _active(chair_out_world)
_, _, sc_ow, _, tr_ow_world = _active(obj_out_world)

tr_cw_robot = apply_scene(tr_cw_world, CHAIR_SCENE_QUAT, CHAIR_SCENE_T)
tr_ow_robot = apply_scene(tr_ow_world, NPZ_SCENE_QUAT,   NPZ_SCENE_T)

print(f"\n  chair.ply  (using DEFAULT_SCENE transform, +90° X)")
print(f"  {'idx':>4}  {'tx':>7}  {'ty':>7}  {'tz':>7}  {'r_max':>7}")
for i, (t, s) in enumerate(zip(tr_cw_robot, sc_cw)):
    print(f"  {i:>4}  {t[0]:>7.3f}  {t[1]:>7.3f}  {t[2]:>7.3f}  {s.max():>7.4f}")
print(f"  robot-frame z: [{tr_cw_robot[:,2].min():.3f}, {tr_cw_robot[:,2].max():.3f}] m")
print(f"  (table surface at z=0.0, arm default config ~ z=0.3–0.8 m)")

print(f"\n  obj_00.ply  (using DEFAULT_NPZ_SCENE transform, -90° X)")
print(f"  {'idx':>4}  {'tx':>7}  {'ty':>7}  {'tz':>7}  {'r_max':>7}")
for i, (t, s) in enumerate(zip(tr_ow_robot, sc_ow)):
    print(f"  {i:>4}  {t[0]:>7.3f}  {t[1]:>7.3f}  {t[2]:>7.3f}  {s.max():>7.4f}")
print(f"  robot-frame z: [{tr_ow_robot[:,2].min():.3f}, {tr_ow_robot[:,2].max():.3f}] m")

# also try IDENTITY rotation
print(f"\n  obj_00.ply  (IDENTITY rotation — stored coords as robot frame)")
tr_ow_identity = tr_ow_world + NPZ_SCENE_T   # just translation, no rotation
print(f"  {'idx':>4}  {'tx':>7}  {'ty':>7}  {'tz':>7}  {'r_max':>7}")
for i, (t, s) in enumerate(zip(tr_ow_identity, sc_ow)):
    print(f"  {i:>4}  {t[0]:>7.3f}  {t[1]:>7.3f}  {t[2]:>7.3f}  {s.max():>7.4f}")
print(f"  robot-frame z: [{tr_ow_identity[:,2].min():.3f}, {tr_ow_identity[:,2].max():.3f}] m")

# ════════════════════════════════════════════════════════════════════════════
# KEY DIFFERENCES SUMMARY
# ════════════════════════════════════════════════════════════════════════════
print(f"\n{'═'*70}")
print("  KEY DIFFERENCES SUMMARY")
print(f"{'═'*70}")

_, _, sc_c,  sh_c, tr_c  = _active(chair_out)
_, _, sc_o,  sh_o, tr_o  = _active(obj_out)
_, _, sc_cw, _,   tr_cw  = _active(chair_out_world)
_, _, sc_ow, _,   tr_ow  = _active(obj_out_world)

rows = [
    ("active primitives",     f"{len(sc_c)}",                    f"{len(sc_o)}"),
    ("norm radii max",        f"{sc_c.max():.4f}",               f"{sc_o.max():.4f}"),
    ("norm radii mean",       f"{sc_c.mean():.4f}",              f"{sc_o.mean():.4f}"),
    ("norm trans max|abs|",   f"{np.abs(tr_c).max():.4f}",       f"{np.abs(tr_o).max():.4f}"),
    ("shape eps min",         f"{sh_c.min():.4f}",               f"{sh_o.min():.4f}"),
    ("shape eps max",         f"{sh_c.max():.4f}",               f"{sh_o.max():.4f}"),
    ("world radii max (m)",   f"{sc_cw.max():.4f}",              f"{sc_ow.max():.4f}"),
    ("world z min (m)",       f"{tr_cw[:, 2].min():.3f}",        f"{tr_ow[:, 2].min():.3f}"),
    ("world z max (m)",       f"{tr_cw[:, 2].max():.3f}",        f"{tr_ow[:, 2].max():.3f}"),
    ("norm input range",      f"[{chair_norm.min():.3f}, {chair_norm.max():.3f}]",
                              f"[{obj_sampled.min():.3f}, {obj_sampled.max():.3f}]"),
]

print(f"\n  {'metric':<28}  {'chair.ply':>18}  {'obj_00.ply':>18}")
print(f"  {'-'*28}  {'-'*18}  {'-'*18}")
for name, vc, vo in rows:
    flag = "  ← DIFF" if vc != vo else ""
    print(f"  {name:<28}  {vc:>18}  {vo:>18}{flag}")
