"""Run student inference on test pointclouds and dump predictions for visual inspection.

Outputs (one set per processed object):
    <output>/<label>/pred.npz   raw SQ params (PredictionHandler format)
    <output>/<label>/pred.glb   predicted SQ primitives as a colored mesh
    <output>/<label>/input.glb  input pointcloud rendered as small spheres
    <output>/<label>/combined.glb  pred mesh + input cloud overlay
    <output>/predictions.npz    aggregate of all objects (PredictionHandler format)

Usage:
    # all test shapes, no augmentation
    python scripts/run_inference.py --checkpoint checkpoints/expocc_tt --output predictions/expocc_tt_clean

    # partial views (HPR) and full SO(3) rotation, 50 random shapes
    python scripts/run_inference.py --checkpoint checkpoints/expocc_tt \
        --partial --rotate --max_objects 50 --output predictions/expocc_tt_partial_rotated

    # one custom .npz
    python scripts/run_inference.py --checkpoint checkpoints/expocc_tt \
        --pointcloud my_object.npz --output predictions/custom

The output dir is small and self-contained (just .glb + .npz), so you can
scp the whole tree to your laptop and inspect.
"""
import os
import argparse
import glob

import numpy as np
import torch
from omegaconf import OmegaConf

from superdec.superdec import SuperDec
from superdec.utils.predictions_handler_extended import PredictionHandler
from superdec.data.dataloader import normalize_points, denormalize_outdict, denormalize_points
from superdec.data.transform_occlusions import HRPOcclusion, RandomOcclusion
from superdec.data.transform import rotate_around_axis

# Force on the trainer-side import side-effect of keeping reproducible point sampling
import trimesh

CATEGORIES = {
    "02876657": "bottle", "02880940": "bowl", "03797390": "mug",
    "03642806": "laptop", "03624134": "knife", "gso": "gso",
}


def random_so3_rotation(rng):
    """Uniformly random SO(3) rotation by 3 axis-aligned rotations."""
    R = np.eye(3)
    for axis_idx, axis in enumerate([(1, 0, 0), (0, 1, 0), (0, 0, 1)]):
        angle = rng.uniform(-np.pi, np.pi)
        c, s = np.cos(angle), np.sin(angle)
        if axis_idx == 0:
            R_a = np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
        elif axis_idx == 1:
            R_a = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
        else:
            R_a = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
        R = R_a @ R
    return R


def find_checkpoint(ckpt_dir, epoch):
    if epoch:
        return os.path.join(ckpt_dir, epoch)
    files = sorted(f for f in os.listdir(ckpt_dir) if f.startswith("epoch_") and f.endswith(".pt"))
    if not files:
        raise FileNotFoundError(f"No epoch_*.pt in {ckpt_dir}")
    # sort by epoch number not alphabetic
    files.sort(key=lambda f: int(f.split("_")[1].split(".")[0]))
    return os.path.join(ckpt_dir, files[-1])


def list_test_objects(data_root, max_objects=None, rng=None):
    objs = []
    for cat in sorted(os.listdir(data_root)):
        cat_path = os.path.join(data_root, cat)
        test_lst = os.path.join(cat_path, "test.lst")
        if not os.path.isdir(cat_path) or not os.path.isfile(test_lst):
            continue
        cat_name = CATEGORIES.get(cat, cat)
        for mid in [l.strip() for l in open(test_lst) if l.strip()]:
            pc = os.path.join(cat_path, mid, "pointcloud.npz")
            if os.path.isfile(pc):
                objs.append({"category": cat_name, "model_id": mid,
                             "npz_path": pc, "label": f"{cat_name}__{mid}"})
    if max_objects:
        if rng is not None:
            idx = rng.choice(len(objs), size=min(max_objects, len(objs)), replace=False)
            objs = [objs[i] for i in sorted(idx)]
        else:
            objs = objs[:max_objects]
    return objs


def apply_augmentations(points, normals, partial, rotate, rng):
    """Apply optional eval-time aug. Operates on raw (un-normalized) points."""
    if rotate:
        R = random_so3_rotation(rng)
        points = points @ R.T
        normals = normals @ R.T
    if partial:
        # HPRoclusion expects an axis-symmetric phi range; use the same defaults
        # as the dataloader (equator band) so the partial views look plausible.
        hpr = HRPOcclusion(p=1.0, phi_range=(np.pi / 4, 3 * np.pi / 4))
        out = hpr(points=points, normals=normals)
        if out["points"].shape[0] >= 1024:  # don't accept degenerate
            points, normals = out["points"], out["normals"]
        else:
            # fall back to random spherical mask
            ro = RandomOcclusion(p=1.0, phi_range=(np.pi / 4, 3 * np.pi / 4))
            out = ro(points=points, normals=normals)
            if out["points"].shape[0] >= 1024:
                points, normals = out["points"], out["normals"]
    return points, normals


def points_to_sphere_mesh(points, color=(180, 180, 180, 255), radius=0.005, max_points=2048):
    """Render a point cloud as a small mesh of spheres (handy for .glb overlay)."""
    if points.shape[0] > max_points:
        idx = np.random.choice(points.shape[0], max_points, replace=False)
        points = points[idx]
    sphere = trimesh.creation.icosphere(subdivisions=1, radius=radius)
    V, F = len(sphere.vertices), len(sphere.faces)
    n = points.shape[0]
    verts = np.tile(sphere.vertices, (n, 1, 1)) + points[:, None, :]
    verts = verts.reshape(-1, 3)
    faces = (np.tile(sphere.faces, (n, 1, 1)) + np.arange(n)[:, None, None] * V).reshape(-1, 3)
    fc = np.tile(np.array(color, dtype=np.uint8), (n * F, 1))
    return trimesh.Trimesh(vertices=verts, faces=faces, face_colors=fc)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True, help="Checkpoint dir, e.g. checkpoints/expocc_tt")
    ap.add_argument("--epoch", default=None, help="epoch_NNN.pt; auto-picks latest if omitted")
    ap.add_argument("--data_root", default="data/ShapeNet")
    ap.add_argument("--output", required=True, help="Output dir")
    ap.add_argument("--max_objects", type=int, default=None)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--partial", action="store_true", help="apply HPR partial-view occlusion")
    ap.add_argument("--rotate", action="store_true", help="apply uniform SO(3) rotation")
    ap.add_argument("--pointcloud", default=None,
                    help="Path or glob to a custom .npz with key 'points' (and optionally 'normals'). "
                         "When set, overrides the test-set iteration.")
    ap.add_argument("--mesh_resolution", type=int, default=80,
                    help="Marching-cubes resolution for mesh export. Higher = smoother + slower.")
    ap.add_argument("--no_meshes", action="store_true",
                    help="Skip per-object .glb export (only save aggregate .npz). Faster.")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    rng = np.random.default_rng(args.seed)
    np.random.seed(args.seed)

    ckpt_path = find_checkpoint(args.checkpoint, args.epoch)
    cfg = OmegaConf.load(os.path.join(args.checkpoint, "config.yaml"))
    model = SuperDec(cfg.superdec).to(device)
    sd = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(sd["model_state_dict"])
    model.eval()
    print(f"Checkpoint: {ckpt_path}")
    print(f"  extended={cfg.superdec.extended}  rotation6d={cfg.superdec.rotation6d}")
    print(f"Aug: partial={args.partial}  rotate={args.rotate}")

    # Figure out the input list
    if args.pointcloud is not None:
        paths = sorted(glob.glob(args.pointcloud))
        if not paths:
            raise FileNotFoundError(args.pointcloud)
        objs = [{"category": "custom", "model_id": os.path.splitext(os.path.basename(p))[0],
                 "npz_path": p, "label": f"custom__{os.path.splitext(os.path.basename(p))[0]}"}
                for p in paths]
    else:
        objs = list_test_objects(args.data_root, max_objects=args.max_objects, rng=rng)
    print(f"Processing {len(objs)} objects -> {args.output}")
    os.makedirs(args.output, exist_ok=True)

    pred_handler = None

    for i, obj in enumerate(objs):
        out_dir = os.path.join(args.output, obj["label"])
        os.makedirs(out_dir, exist_ok=True)
        print(f"  [{i+1}/{len(objs)}] {obj['label']}")

        d = np.load(obj["npz_path"])
        points_np = d["points"].astype(np.float32)
        if "normals" in d.files:
            normals_np = d["normals"].astype(np.float32)
        else:
            # placeholder — augmentations need normals
            normals_np = np.zeros_like(points_np)
            normals_np[:, 2] = 1.0  # dummy z-up

        # Optional eval augmentations (in raw coordinates)
        points_np, normals_np = apply_augmentations(points_np, normals_np,
                                                    partial=args.partial,
                                                    rotate=args.rotate,
                                                    rng=rng)

        # Subsample to the model's expected 4096 points
        n = points_np.shape[0]
        idxs = np.random.choice(n, 4096, replace=n < 4096)
        points_np = points_np[idxs]

        # Normalize (the network was trained with normalize=True)
        points_norm, translation, scale = normalize_points(points_np)
        points_t = torch.from_numpy(points_norm).unsqueeze(0).to(device).float()

        with torch.no_grad():
            outdict = model(points_t)
            for k in outdict:
                if isinstance(outdict[k], torch.Tensor):
                    outdict[k] = outdict[k].cpu()
            outdict = denormalize_outdict(outdict, np.array([translation]), np.array([scale]))
            pc_world = denormalize_points(points_t.cpu(),
                                          np.array([translation]), np.array([scale]))

        # Aggregate into a single npz across all objects (paper-format pseudo-GT)
        if pred_handler is None:
            pred_handler = PredictionHandler.from_outdict(outdict, pc_world, [obj["label"]])
        else:
            pred_handler.append_outdict(outdict, pc_world, [obj["label"]])

        # Per-object .glb dumps for visual inspection
        if not args.no_meshes:
            try:
                # build a temp 1-object handler so we can mesh just this prediction
                tmp_h = PredictionHandler.from_outdict(outdict, pc_world, [obj["label"]])
                meshes = tmp_h.get_meshes(resolution=args.mesh_resolution, colors=True)
                pred_mesh = meshes[0]
                if pred_mesh is not None:
                    pred_mesh.export(os.path.join(out_dir, "pred.glb"), file_type="glb")

                pc_mesh = points_to_sphere_mesh(pc_world[0].cpu().numpy(),
                                                color=(80, 80, 80, 255))
                pc_mesh.export(os.path.join(out_dir, "input.glb"), file_type="glb")

                if pred_mesh is not None:
                    combined = trimesh.util.concatenate([pred_mesh, pc_mesh])
                    combined.export(os.path.join(out_dir, "combined.glb"), file_type="glb")
            except Exception as e:
                print(f"    [warn] mesh export failed: {e}")

    # Aggregate npz
    out_npz = os.path.join(args.output, "predictions.npz")
    pred_handler.save_npz(out_npz)
    print(f"\nSaved {len(objs)} predictions -> {out_npz}")
    print(f"Per-object .glb dumps under {args.output}/<label>/")


if __name__ == "__main__":
    main()
