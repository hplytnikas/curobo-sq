"""Fit SuperDec to a full scene by running the object-level model per instance.

- Loads a scene npz with keys: xyz, color, semantic_label, instance_label, bbox.
- Splits points by instance_label, runs the model on each object,
  denormalizes the prediction back into scene coordinates, and renders
  all superquadric meshes together with the scene point cloud in viser.

The scene is assumed z-up; each object is converted to y-up for the model
(matching training) and the prediction is converted back.
"""

import argparse
import os
import sys
import time

from pathlib import Path
import numpy as np
import torch
import viser
from omegaconf import OmegaConf

WORKSPACE_ROOT = Path(__file__).resolve().parents[5]
SUPERDEC_ROOT = WORKSPACE_ROOT / "superdec"
if str(SUPERDEC_ROOT) not in sys.path:
    sys.path.append(str(SUPERDEC_ROOT))

from superdec.superdec import SuperDec
from superdec.data.dataloader import normalize_points, denormalize_outdict
from superdec.data.transform import rotate_around_axis
from superdec.utils.predictions_handler import PredictionHandler as PredictionHandlerConvex
# from superdec.utils.predictions_handler_extended import PredictionHandler as PredictionHandlerExtended
from superdec.utils.visualizations import generate_ncolors


SCENE_NPZ = "data/scenes/scene_custom.npz"
CKPT_DIR = "checkpoints/expocc_tt_bent"
CKPT_FILE = "epoch_100.pt"
SKIP_INSTANCES = {0}      # instance 0 is the table — skip (degenerate flat plane)
RESOLUTION = 30
N_POINTS = 4096


def load_model(ckpt_dir, ckpt_file, device):
    cfg = OmegaConf.load(os.path.join(ckpt_dir, "config.yaml"))
    model = SuperDec(cfg.superdec).to(device)
    model.lm_optimization = False
    ckpt = torch.load(os.path.join(ckpt_dir, ckpt_file), map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    extended = bool(getattr(cfg.superdec, "extended", False))
    print(f"loaded {ckpt_dir}/{ckpt_file} (extended={extended})")
    return model, extended


def subsample(points, n):
    if points.shape[0] == n:
        return points
    idx = np.random.choice(points.shape[0], n, replace=points.shape[0] < n)
    return points[idx]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt_dir", default=CKPT_DIR, help="folder with epoch_*.pt + config.yaml")
    ap.add_argument("--ckpt_file", default=CKPT_FILE)
    ap.add_argument("--scene", default=SCENE_NPZ, help="scene npz path")
    ap.add_argument("--port", type=int, default=8081)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, extended = load_model(args.ckpt_dir, args.ckpt_file, device)
    #Handler = PredictionHandlerExtended if extended else PredictionHandlerConvex
    Handler = PredictionHandlerConvex

    data = np.load(args.scene, allow_pickle=True)
    xyz = data["xyz"].astype(np.float32)
    rgb = data["color"].astype(np.float32)
    if rgb.max() > 1.0:
        rgb = rgb / 255.0
    inst = data["instance_label"].astype(np.int64)

    instance_ids = [i for i in np.unique(inst) if i not in SKIP_INSTANCES]
    print(f"{len(instance_ids)} objects to fit")

    server = viser.ViserServer(port=args.port)
    server.scene.set_up_direction([0.0, 0.0, 1.0])  # scene is z-up

    # full scene point cloud (original colors)
    # server.scene.add_point_cloud(
    #     name="/scene",
    #     points=xyz,
    #     colors=rgb,
    #     point_size=0.003,
    #     visible=True,
    # )

    obj_colors = generate_ncolors(max(len(instance_ids), 1)) / 255.0


    for k, iid in enumerate(instance_ids):
        obj_pts_scene = xyz[inst == iid]               # z-up scene coords
        obj_pts = subsample(obj_pts_scene, N_POINTS)

        
        obj_pts_yup = rotate_around_axis(obj_pts, axis=(1, 0, 0),
                                         angle=-np.pi / 2, center_point=np.zeros(3))
        pts_norm, translation, scale = normalize_points(obj_pts_yup)
        pts_t = torch.from_numpy(pts_norm).unsqueeze(0).to(device).float()

        with torch.no_grad():
            out = model(pts_t)
        out = {key: (v.cpu() if isinstance(v, torch.Tensor) else v) for key, v in out.items()}

        # denormalize back into scene (z-up) coordinates
        out = denormalize_outdict(out, np.array([translation]), np.array([scale]), z_up=True)

        print(f"instance {iid}: output: {out}")

        pts_back = torch.from_numpy(obj_pts_scene[None].astype(np.float32))
        handler = Handler.from_outdict(out, pts_back[:, :N_POINTS], [str(iid)])
        mesh = handler.get_meshes(resolution=RESOLUTION)[0]
        if mesh is None:
            print(f"  instance {iid}: no mesh")
            continue

        server.scene.add_mesh_trimesh(f"/fit/obj_{iid}", mesh=mesh, visible=True)
        print(f"  instance {iid}: fitted ({k + 1}/{len(instance_ids)})")

    @server.on_client_connect
    def _(client: viser.ClientHandle) -> None:
        client.camera.position = (1.5, 1.5, 2.0)
        client.camera.look_at = (0.0, 0.0, 0.85)

    print(f"viser running on http://localhost:{args.port}")
    while True:
        time.sleep(10.0)


if __name__ == "__main__":
    main()
