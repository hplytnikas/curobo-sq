"""Scan every obj_XX.ply through the pipeline and print per-file diagnostics."""

import os, sys
import numpy as np
import torch
import trimesh
import trimesh.repair
from pathlib import Path
from omegaconf import OmegaConf

SUPERDEC_ROOT = Path("/home/haroldas/3DV/superdec")
sys.path.insert(0, str(SUPERDEC_ROOT))
from superdec.superdec import SuperDec
from superdec.utils.predictions_handler import PredictionHandler

FOLDER   = "/home/haroldas/3DV/superdec/examples/Archive/objects_pc_normalized"
NORM_NPZ = os.path.join(FOLDER, "normalization.npz")
CKPT_DIR = "/home/haroldas/3DV/superdec/checkpoints/normalized"
N_POINTS = 4096
NPZ_T    = np.array([-0.3, -0.2, 0.03], dtype=np.float32)   # identity rotation

device = "cuda" if torch.cuda.is_available() else "cpu"
checkpoint = torch.load(os.path.join(CKPT_DIR, "ckpt.pt"), map_location=device, weights_only=False)
configs    = OmegaConf.load(os.path.join(CKPT_DIR, "config.yaml"))
model      = SuperDec(configs.superdec).to(device)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()
print("Model loaded.\n")

norm_data   = np.load(NORM_NPZ, allow_pickle=True)
norm_lookup = {
    int(norm_data["obj_id"][i]): (
        np.asarray(norm_data["center"][i], dtype=np.float32),
        float(norm_data["scale"][i]),
    )
    for i in range(len(norm_data["obj_id"]))
}


def subsample(pts, n):
    idx = np.random.default_rng(42).choice(len(pts), n, replace=len(pts) < n)
    return pts[idx]


def load_pts(path):
    pc = trimesh.load(path, process=False)
    if isinstance(pc, trimesh.Scene):
        pts = np.concatenate(
            [np.asarray(g.vertices) for g in pc.geometry.values() if hasattr(g, "vertices")], 0
        )
    else:
        pts = np.asarray(pc.vertices)
    return pts.astype(np.float32)


hdr = (
    f"{'file':<12} {'n_act':>5} {'r_max_norm':>10} {'r_max_world_m':>13} "
    f"{'eps_min':>7} {'z_r_min':>7} {'z_r_max':>7} {'mesh_vol':>9} "
    f"{'watertight':>10} {'after_rep':>9}"
)
print(hdr)
print("-" * len(hdr))

for filename in sorted(f for f in os.listdir(FOLDER) if f.endswith(".ply")):
    obj_id = int(filename.split("_")[1].split(".")[0])
    if obj_id not in norm_lookup:
        print(f"{filename:<12}  no norm data")
        continue
    center, scale = norm_lookup[obj_id]

    pts = subsample(load_pts(os.path.join(FOLDER, filename)), N_POINTS)
    pts_t = torch.from_numpy(pts).unsqueeze(0).to(device)
    with torch.no_grad():
        out = model(pts_t)
    out = {k: v.cpu() if isinstance(v, torch.Tensor) else v for k, v in out.items()}

    exist  = np.asarray(out["exist"]).flatten()
    active = exist > 0.5
    scales = np.asarray(out["scale"]).reshape(-1, 3)[active]
    shapes = np.asarray(out["shape"]).reshape(-1, 2)[active]
    trans  = np.asarray(out["trans"]).reshape(-1, 3)[active]

    # denorm → world frame
    scales_w = scales * scale
    trans_w  = trans  * scale + center

    # robot frame: identity rotation + translation NPZ_T
    trans_r = trans_w + NPZ_T

    # native mesh
    wt_before = wt_after = vol = "?"
    try:
        world_pts = (pts * scale + center).astype(np.float32)
        pts_tensor = torch.from_numpy(world_pts[None])
        out_w = dict(out)
        out_w["scale"] = torch.tensor(
            np.asarray(out["scale"]).reshape(1, -1, 3) * scale, dtype=torch.float32
        )
        out_w["trans"] = torch.tensor(
            np.asarray(out["trans"]).reshape(1, -1, 3) * scale + center.reshape(1, 1, 3),
            dtype=torch.float32,
        )
        ph      = PredictionHandler.from_outdict(out_w, pts_tensor, [filename])
        meshes  = ph.get_meshes(resolution=48, colors=False)
        if meshes and meshes[0] is not None:
            parts = meshes[0].split() or [meshes[0]]
            m = trimesh.util.concatenate(
                [trimesh.Trimesh(p.vertices, p.faces, process=False) for p in parts]
            )
            m.vertices = m.vertices + NPZ_T   # apply scene translation
            wt_before  = m.is_watertight
            mr = m.copy()
            trimesh.repair.fill_holes(mr)
            trimesh.repair.fix_normals(mr, multibody=True)
            wt_after = mr.is_watertight
            vol      = f"{mr.volume:.4f}"
        else:
            wt_before = wt_after = "no_mesh"
            vol = "nan"
    except Exception as e:
        wt_before = wt_after = "err"
        vol = str(e)[:12]

    r_max_n = float(scales.max())   if len(scales) else 0.0
    r_max_w = float(scales_w.max()) if len(scales_w) else 0.0
    eps_min = float(shapes.min())   if len(shapes) else 0.0
    z_min   = float(trans_r[:, 2].min()) if len(trans_r) else 0.0
    z_max   = float(trans_r[:, 2].max()) if len(trans_r) else 0.0

    print(
        f"{filename:<12} {active.sum():>5} {r_max_n:>10.4f} {r_max_w:>13.4f} "
        f"{eps_min:>7.4f} {z_min:>7.3f} {z_max:>7.3f} {vol:>9} "
        f"{str(wt_before):>10} {str(wt_after):>9}"
    )

    # detailed per-primitive dump for obj_03 (the broken one)
    if obj_id == 3:
        print(f"\n  ── obj_03 per-primitive detail ──")
        print(f"  center={center}  scale={scale:.5f}")
        print(f"  {'prim':>4} {'eps1':>6} {'eps2':>6} {'rx':>7} {'ry':>7} {'rz':>7} {'tx_r':>7} {'ty_r':>7} {'tz_r':>7}")
        for i, (sc, sh, tr_r) in enumerate(zip(scales_w, shapes, trans_r)):
            print(f"  {i:>4} {sh[0]:>6.3f} {sh[1]:>6.3f} {sc[0]:>7.4f} {sc[1]:>7.4f} {sc[2]:>7.4f} {tr_r[0]:>7.3f} {tr_r[1]:>7.3f} {tr_r[2]:>7.3f}")
        print()
