"""
Generate points.npz (uniformly sampled query points + occupancies) for GSO
objects from their existing pointcloud.npz (surface points + normals).

Format matches ShapeNet/OccNet:
    points:       float16 (N=100000, 3), in raw mesh coords, span ~[-0.55, 0.55]
                  of the object's normalized frame after the dataloader rescales
    occupancies:  uint8, np.packbits of bool array length N

Occupancy is estimated via weighted sign-vote of nearest surface normals:
    sd_i = ((q - p_i) . n_i),  occupied iff weighted sum of sign(sd_i) < 0
where weights are 1 / (||q - p_i|| + eps) over the k-NN.

Run from the repo root:
    python scripts/preprocess_gso_occupancy.py --data_root data/ShapeNet
"""
import argparse
import numpy as np
from pathlib import Path
from scipy.spatial import cKDTree
from tqdm import tqdm


def compute_occupancy(pc, normals, n_query=100000, k=5, query_box=0.55, seed=42):
    rng = np.random.default_rng(seed)

    pc = pc.astype(np.float64)
    nm = normals.astype(np.float64)
    nm = nm / np.maximum(np.linalg.norm(nm, axis=1, keepdims=True), 1e-9)

    # Match dataloader's normalize_points: center + scale to fit max abs to 0.5
    translation = pc.mean(0)
    pc_c = pc - translation
    scale = 2.0 * float(np.abs(pc_c).max())
    if scale < 1e-6:
        scale = 1e-4
    pc_norm = pc_c / scale  # surface points now in [-0.5, 0.5]^3 (max axis)

    # Sample query points in the normalized cube wider than [-0.5, 0.5]
    q_norm = rng.uniform(-query_box, query_box, size=(n_query, 3))

    tree = cKDTree(pc_norm)
    dists, idxs = tree.query(q_norm, k=k)
    if k == 1:
        dists = dists[:, None]
        idxs = idxs[:, None]

    p_neighbors = pc_norm[idxs]                 # (n_query, k, 3)
    n_neighbors = nm[idxs]                      # (n_query, k, 3)
    diff = q_norm[:, None, :] - p_neighbors     # (n_query, k, 3)
    signed = (diff * n_neighbors).sum(axis=-1)  # (n_query, k)
    weights = 1.0 / (dists + 1e-6)              # (n_query, k)
    weighted_sign = (np.sign(signed) * weights).sum(axis=1)
    occupied = weighted_sign < 0

    # Save query points back in raw mesh coords so the dataloader's
    # (points_iou - translation) / scale step puts them into [-query_box, query_box]
    q_raw = q_norm * scale + translation
    return q_raw.astype(np.float16), np.packbits(occupied), float(occupied.mean())


def process_object(obj_dir: Path, k: int, n_query: int, force: bool):
    pc_path = obj_dir / "pointcloud.npz"
    out_path = obj_dir / "points.npz"
    if not pc_path.exists():
        return None
    if out_path.exists() and not force:
        return "skip"
    d = np.load(pc_path)
    pts, occ_packed, occ_rate = compute_occupancy(
        d["points"], d["normals"], n_query=n_query, k=k
    )
    tmp_path = out_path.with_suffix(".npz.tmp")
    np.savez_compressed(tmp_path, points=pts, occupancies=occ_packed)
    tmp_path.rename(out_path)
    return occ_rate


def process_category(data_root: Path, category: str, k: int, n_query: int, force: bool):
    cat_dir = data_root / category
    if not cat_dir.is_dir():
        print(f"[skip] {cat_dir} not found")
        return

    splits = ["train", "val", "test"]
    obj_ids = []
    for s in splits:
        sf = cat_dir / f"{s}.lst"
        if sf.exists():
            obj_ids += [l.strip() for l in sf.read_text().splitlines() if l.strip()]
    obj_ids = sorted(set(obj_ids))
    if not obj_ids:
        print(f"[skip] {category}: no split files")
        return

    print(f"[{category}] {len(obj_ids)} objects, k={k}, n_query={n_query}")
    n_done, n_skip, n_fail = 0, 0, 0
    rates = []
    for obj_id in tqdm(obj_ids, desc=category):
        try:
            r = process_object(cat_dir / obj_id, k=k, n_query=n_query, force=force)
            if r is None:
                n_fail += 1
            elif r == "skip":
                n_skip += 1
            else:
                n_done += 1
                rates.append(r)
        except Exception as e:
            n_fail += 1
            print(f"  fail {obj_id}: {e}")
    msg = f"[{category}] done={n_done} skip={n_skip} fail={n_fail}"
    if rates:
        msg += f" mean_occ_rate={np.mean(rates):.3f} (median={np.median(rates):.3f})"
    print(msg)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", type=Path, default=Path("data/ShapeNet"))
    ap.add_argument("--categories", nargs="+", default=["gso"])
    ap.add_argument("-k", type=int, default=11,
                    help="k-NN for sign vote. Higher = more robust to noisy normals.")
    ap.add_argument("--n_query", type=int, default=100000)
    ap.add_argument("--force", action="store_true", help="overwrite existing points.npz")
    args = ap.parse_args()

    for c in args.categories:
        process_category(args.data_root, c, args.k, args.n_query, args.force)


if __name__ == "__main__":
    main()
