"""Single-checkpoint viser visualization across all tabletop test objects.

Auto-detects extended vs convex from the checkpoint's config.yaml and picks the
matching PredictionHandler.
"""

import os
import time
from typing import Dict, Tuple

import numpy as np
import torch
import viser
from omegaconf import OmegaConf

from superdec.superdec import SuperDec
from superdec.data.dataloader import normalize_points
from superdec.utils.predictions_handler import PredictionHandler as PredictionHandlerConvex
from superdec.utils.predictions_handler_extended import PredictionHandler as PredictionHandlerExtended
from superdec.utils.visualizations import generate_ncolors


def segmented_points_np(out: dict, pts_np: np.ndarray):
    """Color each input point by its argmax assignment — bypasses open3d."""
    assign = out["assign_matrix"].cpu().numpy()[0]  # (P, N) or (N, P) depending on impl
    if assign.shape[0] != pts_np.shape[0]:
        assign = assign.T
    P = assign.shape[1]
    seg = np.argmax(assign, axis=1)
    colors = generate_ncolors(P) / 255.0
    return pts_np, colors[seg]


CATEGORIES = ["02876657", "02880940", "03624134", "03642806", "03797390", "gso"]
CATEGORY_NAMES = {
    "02876657": "bottle",
    "02880940": "bowl",
    "03624134": "knife",
    "03642806": "laptop",
    "03797390": "mug",
    "gso": "gso",
}
SHAPENET_ROOT = "data/ShapeNet"


def load_model(ckpt_dir: str, ckpt_file: str, device: str):
    cfg = OmegaConf.load(os.path.join(ckpt_dir, "config.yaml"))
    model = SuperDec(cfg.superdec).to(device)
    model.lm_optimization = False
    ckpt = torch.load(os.path.join(ckpt_dir, ckpt_file), map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    extended = bool(getattr(cfg.superdec, "extended", False))
    print(f"loaded {ckpt_dir}/{ckpt_file} (extended={extended})")
    return model, extended


def gather_test_models(shapenet_root: str = SHAPENET_ROOT, extra_ply: list = None) -> list:
    items = []
    for cat in CATEGORIES:
        lst = os.path.join(shapenet_root, cat, "test.lst")
        if not os.path.isfile(lst):
            continue
        with open(lst) as f:
            for line in f:
                mid = line.strip()
                if not mid:
                    continue
                pc = os.path.join(shapenet_root, cat, mid, "pointcloud.npz")
                if os.path.isfile(pc):
                    items.append((CATEGORY_NAMES.get(cat, cat), mid, pc))
    # Optional extra .ply files/dirs to inspect alongside the test set.
    for path in (extra_ply or []):
        if os.path.isdir(path):
            for fn in sorted(os.listdir(path)):
                if fn.endswith(".ply"):
                    items.append(("extra", os.path.splitext(fn)[0], os.path.join(path, fn)))
        elif os.path.isfile(path) and path.endswith(".ply"):
            items.append(("extra", os.path.splitext(os.path.basename(path))[0], path))
    return items


def load_points(path: str, n: int = 4096) -> np.ndarray:
    if path.endswith(".npz"):
        pts = np.load(path)["points"].astype(np.float32)
    elif path.endswith(".ply"):
        import trimesh
        m = trimesh.load(path, process=False)
        pts = np.asarray(m.vertices, dtype=np.float32)
    else:
        raise ValueError(f"unsupported format: {path}")
    if pts.shape[0] != n:
        idx = np.random.choice(pts.shape[0], n, replace=pts.shape[0] < n)
        pts = pts[idx]
    return pts


def random_rotation(seed: int) -> np.ndarray:
    """Uniform random rotation matrix (Shoemake / quaternion)."""
    rng = np.random.default_rng(seed)
    u1, u2, u3 = rng.random(3)
    q = np.array([
        np.sqrt(1 - u1) * np.sin(2 * np.pi * u2),
        np.sqrt(1 - u1) * np.cos(2 * np.pi * u2),
        np.sqrt(u1) * np.sin(2 * np.pi * u3),
        np.sqrt(u1) * np.cos(2 * np.pi * u3),
    ])
    x, y, z, w = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w),     2 * (x * z + y * w)],
        [2 * (x * y + z * w),     1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w),     2 * (y * z + x * w),     1 - 2 * (x * x + y * y)],
    ], dtype=np.float32)


def occlude_points(points: np.ndarray, frac: float, seed: int) -> np.ndarray:
    """Drop the `frac` portion of points farthest from a random viewpoint
    (simulates a hemispherical view-occlusion)."""
    if frac <= 0.0:
        return points
    rng = np.random.default_rng(seed)
    direction = rng.standard_normal(3).astype(np.float32)
    direction /= np.linalg.norm(direction) + 1e-8
    proj = points @ direction
    keep_n = max(64, int(round((1.0 - frac) * points.shape[0])))
    keep_idx = np.argsort(-proj)[:keep_n]  # keep points facing the camera (largest proj)
    return points[keep_idx]


def run(model: SuperDec, points_t: torch.Tensor) -> dict:
    with torch.no_grad():
        out = model(points_t)
    return {k: (v.cpu() if isinstance(v, torch.Tensor) else v) for k, v in out.items()}


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt_dir", default="checkpoints/expocc_tt_bent",
                    help="folder with epoch_*.pt + config.yaml")
    ap.add_argument("--ckpt_file", default="epoch_100.pt")
    ap.add_argument("--shapenet_root", default=SHAPENET_ROOT,
                    help="Dataset root with <cat>/test.lst + <cat>/<inst>/pointcloud.npz")
    ap.add_argument("--extra_ply", nargs="*", default=None,
                    help="Optional extra .ply files or dirs to view alongside the test set")
    ap.add_argument("--port", type=int, default=8081)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt_dir, ckpt_file = args.ckpt_dir, args.ckpt_file
    resolution = 30

    model, extended = load_model(ckpt_dir, ckpt_file, device)
    Handler = PredictionHandlerExtended if extended else PredictionHandlerConvex

    items = gather_test_models(args.shapenet_root, args.extra_ply)
    if not items:
        raise RuntimeError(f"no test models found under {args.shapenet_root}/<cat>/")
    print(f"{len(items)} test models")

    cache: Dict[Tuple, Tuple[object, np.ndarray, np.ndarray]] = {}

    def render(idx: int, rot_seed: int, occ_frac: float, occ_seed: int):
        key = (idx, rot_seed, round(occ_frac, 3), occ_seed)
        if key in cache:
            return cache[key]
        _, mid, pc_path = items[idx]
        pts_np = load_points(pc_path, n=8192)
        pts_np, _, _ = normalize_points(pts_np)

        if occ_frac > 0:
            pts_np = occlude_points(pts_np, occ_frac, occ_seed)
        if rot_seed >= 0:
            R = random_rotation(rot_seed)
            pts_np = pts_np @ R.T

        # resample/pad to exactly 4096 for the model
        if pts_np.shape[0] != 4096:
            sel = np.random.choice(pts_np.shape[0], 4096, replace=pts_np.shape[0] < 4096)
            pts_in = pts_np[sel]
        else:
            pts_in = pts_np

        pts_t = torch.from_numpy(pts_in).unsqueeze(0).to(device).float()
        out = run(model, pts_t)
        h = Handler.from_outdict(out, pts_t.cpu(), [mid])
        mesh = h.get_meshes(resolution=resolution)[0]
        seg_points, seg_colors = segmented_points_np(out, pts_in)
        cache[key] = (mesh, seg_points, seg_colors)
        return cache[key]

    server = viser.ViserServer(port=args.port)
    server.scene.set_up_direction([0.0, 1.0, 0.0])

    labels = [f"{i:03d}  {tag}/{m}" for i, (tag, m, _) in enumerate(items)]
    dropdown = server.gui.add_dropdown("Test model", labels, initial_value=labels[0])
    rot_toggle = server.gui.add_checkbox("Random rotation", False)
    occ_slider = server.gui.add_slider("Occlusion fraction", 0.0, 0.9, 0.05, 0.0)
    reroll = server.gui.add_button("Reroll rotation/occlusion")
    status = server.gui.add_text("Status", initial_value="ready", disabled=True)

    state = {"rot_seed": 0, "occ_seed": 0}

    def update():
        idx = labels.index(dropdown.value)
        rot_seed = state["rot_seed"] if rot_toggle.value else -1
        occ_frac = float(occ_slider.value)
        status.value = f"computing {idx + 1}/{len(items)}..."
        mesh, seg_points, seg_colors = render(idx, rot_seed, occ_frac, state["occ_seed"])
        server.scene.add_mesh_trimesh("/superquadrics", mesh=mesh, visible=True)
        server.scene.add_point_cloud(
            name="/input_points",
            points=seg_points,
            colors=seg_colors,
            point_size=0.005,
        )
        kept = seg_points.shape[0]
        flags = []
        if rot_seed >= 0:
            flags.append(f"rot#{rot_seed}")
        if occ_frac > 0:
            flags.append(f"occ={occ_frac:.2f}({kept}pts)")
        tag = ", ".join(flags) if flags else "raw"
        status.value = f"shown {idx + 1}/{len(items)}  [{tag}]  ({'extended' if extended else 'convex'})"

    def on_reroll(_):
        state["rot_seed"] += 1
        state["occ_seed"] += 1
        update()

    dropdown.on_update(lambda _: update())
    rot_toggle.on_update(lambda _: update())
    occ_slider.on_update(lambda _: update())
    reroll.on_click(on_reroll)
    update()

    @server.on_client_connect
    def _(client: viser.ClientHandle) -> None:
        client.camera.position = (0.0, 0.6, 2.0)
        client.camera.look_at = (0.0, 0.0, 0.0)

    print(f"viser running on http://localhost:{args.port}")
    while True:
        time.sleep(10.0)


if __name__ == "__main__":
    main()
