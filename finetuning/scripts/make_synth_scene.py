"""
make_synth_scene.py — synthesise a TO-Scene-compatible tabletop scene
from ShapeNet + GSO pre-sampled point clouds.

Output: data/synth/scene_<seed>.npz with keys
    xyz             (N, 3) float64
    color           (N, 3) float64        — 0..255
    semantic_label  (N,)   float64        — 2 = table, 41..N = objects
    instance_label  (N,)   int64          — 0 = table, 1..N = objects
    bbox            (K, 7) float64        — [cx,cy,cz,dx,dy,dz,sem] per object

Drops straight into run_on_frame.py via --to_scene_npz, with --semantic_min 41
to keep only the placed objects (table sits at sem=2 < 41).
"""
import argparse
import random
from pathlib import Path

import numpy as np

# Default dataset root; override with --shapenet_root. Layout expected:
#   <root>/<category>/<instance>/pointcloud.npz   (+ a "gso" category)
DEFAULT_SHAPENET_ROOT = Path("data/ShapeNet")
SHAPENET_CATEGORIES = [
    "02876657",   # bottle
    "02880940",   # bowl
    "03624134",   # knife
    "03642806",   # laptop
    "03797390",   # mug
]
SEM_TABLE     = 2     # matches ScanNet "floor" — sem < 41, excluded by filter
SEM_OBJ_START = 41    # matches TO-Scene tabletop-added classes

# Human-readable names per category key (ShapeNet synset id, or "gso").
CAT_NAMES = {
    "02876657": "bottle", "02880940": "bowl",   "03624134": "knife",
    "03642806": "laptop", "03797390": "mug",    "gso":      "gso",
}

# Real-world resting orientation per category key. Objects default to "upright"
# (their canonical pose, kept Z-up). "lay_flat" reorients the object to rest on
# its largest face — e.g. a knife lies on its side on the table, it does not
# stand balanced on its tip like a mug. Extend this dict for any category whose
# canonical pose is not how it would naturally sit on a table.
ORIENT_POLICY = {
    "03624134": "lay_flat",   # knife — rests flat on its blade/side
}


def list_objects(shapenet_root: Path):
    """Return list of (source, pc_path, sem_id, cat_key) tuples.

    cat_key is the ShapeNet synset id for ShapeNet objects, or "gso" for GSO —
    used both for the real-world ORIENT_POLICY and for round-robin grouping.
    """
    out = []
    for cat in SHAPENET_CATEGORIES:
        cat_dir = shapenet_root / cat
        if not cat_dir.is_dir():
            continue
        sem = SEM_OBJ_START + SHAPENET_CATEGORIES.index(cat)
        for inst in sorted(cat_dir.iterdir()):
            pc = inst / "pointcloud.npz"
            if pc.is_file():
                out.append(("shapenet", pc, sem, cat))
    gso_root = shapenet_root / "gso"
    if gso_root.is_dir():
        sem = SEM_OBJ_START + len(SHAPENET_CATEGORIES)
        for inst in sorted(gso_root.iterdir()):
            pc = inst / "pointcloud.npz"
            if pc.is_file():
                out.append(("gso", pc, sem, "gso"))
    return out


def pick_round_robin(pool, n_objects, pyrng):
    """Pick n_objects spread evenly across categories.

    Each category's instances are shuffled, then objects are drawn one category
    at a time, cycling through categories until n_objects are collected. This
    guarantees a diverse scene — no category can dominate (unlike a flat
    random.sample, which once put 3 mugs in an 8-object scene).
    """
    groups = {}
    for item in pool:
        groups.setdefault(item[3], []).append(item)
    cat_keys = sorted(groups)
    for k in cat_keys:
        pyrng.shuffle(groups[k])
    picks, cursors, ci = [], {k: 0 for k in cat_keys}, 0
    while len(picks) < n_objects:
        if all(cursors[k] >= len(groups[k]) for k in cat_keys):
            break   # whole pool exhausted
        k = cat_keys[ci % len(cat_keys)]
        ci += 1
        if cursors[k] < len(groups[k]):
            picks.append(groups[k][cursors[k]])
            cursors[k] += 1
    return picks


def _yaw_rotate(pts: np.ndarray, theta: float) -> np.ndarray:
    c, s = np.cos(theta), np.sin(theta)
    R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float64)
    return pts @ R.T


def _lay_flat(pts: np.ndarray) -> np.ndarray:
    """Reorient an object to rest on its largest face.

    Permutes the coordinate axes so the smallest bounding-box extent points up
    (Z) and the largest points along X — the natural way a flat/elongated object
    (e.g. a knife) lies on a table. The permutation is corrected to a proper
    rotation (det = +1) so the object is not mirrored.
    """
    ext = pts.max(0) - pts.min(0)
    order = np.argsort(ext)          # [smallest, middle, largest] axis indices
    perm = np.zeros((3, 3), dtype=np.float64)
    perm[0, order[2]] = 1.0          # X ← largest extent
    perm[1, order[1]] = 1.0          # Y ← middle extent
    perm[2, order[0]] = 1.0          # Z ← smallest extent  → lies flat
    if np.linalg.det(perm) < 0:      # turn a reflection into a proper rotation
        perm[1] *= -1.0
    return pts @ perm.T


def _bbox(pts: np.ndarray) -> np.ndarray:
    """Axis-aligned bbox → [cx, cy, cz, dx, dy, dz]."""
    lo, hi = pts.min(0), pts.max(0)
    c = (lo + hi) / 2
    d = hi - lo
    return np.concatenate([c, d])


def _palette(n: int) -> np.ndarray:
    base = np.array([
        [228,26,28],[55,126,184],[77,175,74],[152,78,163],
        [255,127,0],[166,86,40],[247,129,191],[102,194,165],
        [252,141,98],[141,160,203],[231,138,195],[166,216,84],
    ], dtype=np.float64)
    return base[np.arange(n) % len(base)]


def make_scene(
    seed: int,
    n_objects: int,
    shapenet_root: Path = DEFAULT_SHAPENET_ROOT,
    table_size: tuple = (1.0, 0.7),
    table_height: float = 0.75,
    table_points: int = 8000,
    obj_min_diameter: float = 0.06,
    obj_max_diameter: float = 0.22,
    max_place_tries: int = 60,
):
    rng = np.random.default_rng(seed)
    random.seed(seed)

    pool = list_objects(shapenet_root)
    if not pool:
        raise SystemExit("No ShapeNet/GSO pointcloud.npz found under " + str(shapenet_root))
    # Diverse selection: round-robin across categories instead of a flat sample.
    picks = pick_round_robin(pool, n_objects, random)
    if len(picks) < n_objects:
        print(f"  [make_synth] only {len(picks)} objects available (asked {n_objects})")

    Tx, Ty = table_size
    tx_min, tx_max = -Tx / 2, Tx / 2
    ty_min, ty_max = -Ty / 2, Ty / 2

    # ── 1. table cloud ────────────────────────────────────────────────────────
    table_xy = rng.uniform([tx_min, ty_min], [tx_max, ty_max], size=(table_points, 2))
    table_z  = np.full((table_points, 1), table_height)
    table_pts = np.concatenate([table_xy, table_z], axis=1)
    # tiny noise so the plane isn't degenerate
    table_pts += rng.normal(0.0, 0.001, table_pts.shape)
    table_color = np.tile([200., 200., 200.], (table_points, 1))

    # ── 2. each object ────────────────────────────────────────────────────────
    obj_clouds, obj_colors, obj_sems, obj_iids, obj_bboxes = [], [], [], [], []
    placed_xy_radii: list = []
    palette = _palette(n_objects)

    for k, (src, pc_path, sem_id, cat_key) in enumerate(picks):
        d = np.load(pc_path)
        pts = np.asarray(d["points"], dtype=np.float64)        # unit-sphere normalized
        # ShapeNet canonical is Y-up; rotate 90° around X so Y→Z (now Z-up).
        # GSO already ships Z-up.
        if src == "shapenet":
            R_yup_to_zup = np.array([[1, 0, 0],
                                     [0, 0, -1],
                                     [0, 1, 0]], dtype=np.float64)
            pts = pts @ R_yup_to_zup.T
        # Real-world resting orientation: lay flat categories (e.g. knife) down
        # on their largest face instead of leaving them standing upright.
        if ORIENT_POLICY.get(cat_key) == "lay_flat":
            pts = _lay_flat(pts)
        # Scale: pick a target diameter; unit-sphere → diameter ~ 2
        cur_diam = float((pts.max(0) - pts.min(0)).max())
        target_diam = float(rng.uniform(obj_min_diameter, obj_max_diameter))
        s = target_diam / max(cur_diam, 1e-6)
        pts = pts * s
        # Random yaw around Z (keeps objects upright on the table)
        pts = _yaw_rotate(pts, float(rng.uniform(0, 2 * np.pi)))
        # Lift so min_z sits on the table
        pts[:, 2] += (table_height - pts[:, 2].min())
        # Place: rejection sample so XY footprint clears prior objects
        obj_xy_extent = np.linalg.norm((pts[:, :2].max(0) - pts[:, :2].min(0)) / 2)
        placed = False
        for _ in range(max_place_tries):
            cx = float(rng.uniform(tx_min + obj_xy_extent, tx_max - obj_xy_extent))
            cy = float(rng.uniform(ty_min + obj_xy_extent, ty_max - obj_xy_extent))
            ok = True
            for px, py, pr in placed_xy_radii:
                if (cx - px) ** 2 + (cy - py) ** 2 < (pr + obj_xy_extent) ** 2:
                    ok = False
                    break
            if ok:
                placed = True
                break
        if not placed:
            print(f"  [{k}] {pc_path.parent.name}: placement skipped (table full)")
            continue
        pts[:, 0] += cx
        pts[:, 1] += cy
        placed_xy_radii.append((cx, cy, obj_xy_extent))

        obj_clouds.append(pts)
        obj_colors.append(np.tile(palette[k], (len(pts), 1)))
        obj_sems.append(np.full(len(pts), sem_id, dtype=np.int64))
        obj_iids.append(np.full(len(pts), k + 1, dtype=np.int64))   # 0 reserved for table
        obj_bboxes.append(np.concatenate([_bbox(pts), [sem_id]]))
        orient = ORIENT_POLICY.get(cat_key, "upright")
        print(f"  [{k}] inst={k+1:2d} sem={sem_id} src={src} "
              f"cat={CAT_NAMES.get(cat_key, cat_key)} orient={orient} "
              f"name={pc_path.parent.name} pts={len(pts)} "
              f"diam={target_diam:.3f} at xy=({cx:+.2f},{cy:+.2f})")

    # ── 3. combine + label ────────────────────────────────────────────────────
    n_tbl = len(table_pts)
    xyz = np.concatenate([table_pts] + obj_clouds, axis=0)
    color = np.concatenate([table_color] + obj_colors, axis=0)
    sem = np.concatenate([
        np.full(n_tbl, SEM_TABLE, dtype=np.int64),
        *obj_sems,
    ])
    inst = np.concatenate([
        np.zeros(n_tbl, dtype=np.int64),
        *obj_iids,
    ])
    bbox = np.stack(obj_bboxes) if obj_bboxes else np.zeros((0, 7))

    return dict(xyz=xyz, color=color,
                semantic_label=sem.astype(np.float64),
                instance_label=inst.astype(np.float64),
                bbox=bbox)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed",        type=int,   default=0)
    ap.add_argument("--n_objects",   type=int,   default=8)
    ap.add_argument("--shapenet_root", type=Path, default=DEFAULT_SHAPENET_ROOT,
                    help="Dataset root with <cat>/<inst>/pointcloud.npz (+ gso/). "
                         "Default: data/ShapeNet")
    ap.add_argument("--out",         default=None,
                    help="Output .npz path (default: data/synth/scene_<seed>.npz)")
    ap.add_argument("--table_w",     type=float, default=1.0)
    ap.add_argument("--table_d",     type=float, default=0.7)
    ap.add_argument("--table_h",     type=float, default=0.75)
    ap.add_argument("--table_points",type=int,   default=8000)
    ap.add_argument("--min_diam",    type=float, default=0.06)
    ap.add_argument("--max_diam",    type=float, default=0.22)
    args = ap.parse_args()

    proj_root = Path(__file__).resolve().parent.parent
    out_path = Path(args.out) if args.out else proj_root / f"data/synth/scene_{args.seed}.npz"
    if not out_path.is_absolute():
        out_path = proj_root / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"[make_synth] seed={args.seed} n_objects={args.n_objects}")
    print(f"[make_synth] out={out_path}")
    scene = make_scene(
        seed=args.seed, n_objects=args.n_objects,
        shapenet_root=args.shapenet_root,
        table_size=(args.table_w, args.table_d),
        table_height=args.table_h, table_points=args.table_points,
        obj_min_diameter=args.min_diam, obj_max_diameter=args.max_diam,
    )
    np.savez(out_path, **scene)
    print(f"[make_synth] saved: {out_path}")
    print(f"  total pts : {len(scene['xyz']):,}")
    print(f"  bbox rows : {len(scene['bbox'])}")


if __name__ == "__main__":
    main()
