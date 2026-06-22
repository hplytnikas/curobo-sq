"""Generate per-model `points.npz` (occupancy) for the tabletop categories.

Uses the precomputed `model_normalized.solid.binvox` to label uniform samples
as inside/outside — no remeshing or watertight repair needed.

Reads:
    data/ShapeNet/shapenet_tabletop/<cat>/<model_id>/models/model_normalized.solid.binvox
Writes:
    data/ShapeNet/<cat>/<model_id>/points.npz
        points       (N, 3)  float16   uniform samples in [-range, range]^3
        occupancies  (N/8,)  uint8     packed inside/outside bits

CPU only. Parallelized with multiprocessing.
"""

import argparse
import os
import sys
import traceback
from multiprocessing import Pool

import numpy as np
import trimesh


DEFAULT_CATEGORIES = ["02876657", "02880940", "03624134", "03642806", "03797390"]


def occupancy_from_binvox(binvox_path: str, points: np.ndarray) -> np.ndarray:
    """Look up occupancy for each point in the solid binvox grid.

    Points outside the binvox AABB are labelled as outside (False).
    """
    with open(binvox_path, "rb") as fh:
        vg = trimesh.exchange.binvox.load_binvox(fh)

    dense = vg.encoding.dense  # (D, D, D) bool array
    D = dense.shape[0]

    # transform: voxel_idx_homog -> world_homog. Invert to map points to indices.
    T = np.asarray(vg.transform, dtype=np.float64)
    T_inv = np.linalg.inv(T)
    pts_h = np.concatenate([points, np.ones((points.shape[0], 1))], axis=1)
    idx = (pts_h @ T_inv.T)[:, :3]
    # voxel center = idx + 0.5 in voxel space; round-down gives correct bin.
    ijk = np.floor(idx).astype(np.int64)

    inside_aabb = np.all((ijk >= 0) & (ijk < D), axis=1)
    occ = np.zeros(points.shape[0], dtype=bool)
    if inside_aabb.any():
        i, j, k = ijk[inside_aabb].T
        occ[inside_aabb] = dense[i, j, k]
    return occ


def process_one(args):
    binvox_path, dst_npz, n_points, half_range, seed, overwrite = args
    if os.path.exists(dst_npz) and not overwrite:
        return dst_npz, "skip"
    try:
        rng = np.random.default_rng(seed)
        points = rng.uniform(-half_range, half_range, size=(n_points, 3)).astype(np.float64)
        occ = occupancy_from_binvox(binvox_path, points)
        os.makedirs(os.path.dirname(dst_npz), exist_ok=True)
        np.savez_compressed(
            dst_npz,
            points=points.astype(np.float16),
            occupancies=np.packbits(occ.astype(np.uint8)),
        )
        return dst_npz, f"ok inside={occ.mean():.3f}"
    except Exception as e:
        return dst_npz, f"err: {e}\n{traceback.format_exc()}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="data/ShapeNet/shapenet_tabletop")
    ap.add_argument("--dst", default="data/ShapeNet")
    ap.add_argument("--categories", nargs="+", default=DEFAULT_CATEGORIES)
    ap.add_argument("--n_points", type=int, default=100000)
    ap.add_argument("--range", dest="half_range", type=float, default=0.55,
                    help="samples drawn from [-range, range]^3")
    ap.add_argument("--workers", type=int, default=max(1, os.cpu_count() // 2))
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    jobs = []
    for cat in args.categories:
        cat_src = os.path.join(args.src, cat)
        if not os.path.isdir(cat_src):
            print(f"[skip] {cat_src} not found")
            continue
        for model_id in sorted(os.listdir(cat_src)):
            bv = os.path.join(cat_src, model_id, "models", "model_normalized.solid.binvox")
            if not os.path.isfile(bv):
                continue
            dst = os.path.join(args.dst, cat, model_id, "points.npz")
            seed = hash((cat, model_id)) & 0xFFFFFFFF
            jobs.append((bv, dst, args.n_points, args.half_range, seed, args.overwrite))

    print(f"{len(jobs)} models to process, workers={args.workers}")
    ok = err = skip = 0
    with Pool(args.workers) as pool:
        for i, (path, status) in enumerate(pool.imap_unordered(process_one, jobs), 1):
            if status == "skip":
                skip += 1
            elif status.startswith("ok"):
                ok += 1
            else:
                err += 1
                print(f"[err] {path}: {status.splitlines()[0]}", file=sys.stderr)
            if i % 50 == 0 or i == len(jobs):
                print(f"  {i}/{len(jobs)} ok={ok} skip={skip} err={err}")

    print(f"done: ok={ok} skip={skip} err={err}")


if __name__ == "__main__":
    main()
