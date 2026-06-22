"""
ocid_eval.py — fast evaluation with parallel SQ fitting and multi-view mode.

Speed features:
  - Parallel SQ fitting across objects (--n_workers)
  - Adaptive restarts: 1 restart if shape hint confident, 2 otherwise
  - n_lm_rounds reduced to 12 (vs 15 before, saves ~20%)

Fitter selection (--fitter):
  lm       : LM optimisation (default, runs on CPU/Mac)
  superdec : SuperDec neural fitter (requires CUDA + superdec repo)

Usage:
  python3 ocid_eval.py --data_dir /Volumes/T7/OCID-dataset/YCB10 --fit_sq --n_workers 4
  python3 ocid_eval.py --data_dir /Volumes/T7/OCID-dataset/YCB10 --fit_sq --fitter superdec \
      --superdec_dir /path/to/superdec
"""

import argparse, sys, os, time
import numpy as np
import open3d as o3d
from pathlib import Path
from PIL import Image
from typing import List, Dict, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import Counter

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from pipeline     import TabletopPerception
from superquadric import SuperquadricFitter

PERCEPTION_DIR  = os.path.dirname(os.path.abspath(__file__))
CLASSIFIER_PATH = os.path.join(PERCEPTION_DIR, "sq_shape_classifier.pkl")
if not os.path.exists(CLASSIFIER_PATH):
    CLASSIFIER_PATH = os.path.join(os.getcwd(), "sq_shape_classifier.pkl")


# ── fitter factory ────────────────────────────────────────────────────────────

def make_fitter(fitter_name: str, superdec_dir: str = None,
                superdec_checkpoint: str = 'normalized',
                exist_threshold: float = 0.3,
                n_points: int = 4096,
                device: str = None):
    if fitter_name == 'superdec':
        if not superdec_dir:
            raise ValueError("--superdec_dir required when --fitter superdec")
        import torch
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from superdec_fitter import SuperdecFitter
        return SuperdecFitter(
            superdec_dir=superdec_dir,
            checkpoint=superdec_checkpoint,
            exist_threshold=exist_threshold,
            n_points=n_points,
            device=device,
        )
    else:
        return SuperquadricFitter(n_restarts=2, n_lm_rounds=12, subsample=384)


# ── data ──────────────────────────────────────────────────────────────────────

def find_scenes(data_dir, surface=None, view=None, shape=None):
    data_dir = Path(data_dir)
    scenes   = []
    for surf in ([surface] if surface else ["floor", "table"]):
        for v in ([view] if view else ["top", "bottom"]):
            for sh in ([shape] if shape else ["mixed", "curved", "cuboid"]):
                base = data_dir / surf / v / sh
                if not base.exists():
                    continue
                for seq_dir in sorted(base.iterdir()):
                    if not seq_dir.is_dir():
                        continue
                    pcd_dir   = seq_dir / "pcd"
                    label_dir = seq_dir / "label"
                    rgb_dir   = seq_dir / "rgb"
                    if not pcd_dir.exists() or not label_dir.exists():
                        continue
                    pcds = {p.stem: p for p in pcd_dir.glob("*.pcd")  if not p.name.startswith("._")}
                    lbls = {p.stem: p for p in label_dir.glob("*.png") if not p.name.startswith("._")}
                    rgbs = {p.stem: p for p in rgb_dir.glob("*.png")   if not p.name.startswith("._")} if rgb_dir.exists() else {}
                    for stem in sorted(set(pcds) & set(lbls)):
                        scenes.append({
                            "pcd": str(pcds[stem]), "label": str(lbls[stem]),
                            "rgb": str(rgbs.get(stem, "")),
                            "stem": stem, "seq": seq_dir.name,
                            "surface": surf, "view": v, "shape": sh,
                        })
    return scenes


def load_frame(scene):
    pcd  = o3d.io.read_point_cloud(scene["pcd"])
    pts  = np.asarray(pcd.points, dtype=np.float32)
    col  = np.asarray(pcd.colors, dtype=np.float32) if pcd.has_colors() else None
    mask = np.array(Image.open(scene["label"]), dtype=np.int32)
    return pts, col, mask


def gt_instances_from_mask(pts, mask):
    h, w = mask.shape
    if len(pts) != h * w:
        return {}
    flat = mask.ravel()
    return {int(l): pts[flat == l] for l in np.unique(flat) if l != 0}


# ── matching ──────────────────────────────────────────────────────────────────

def point_set_iou(a, b, voxel=0.005):
    def vox(p): return set(map(tuple, np.round(p / voxel).astype(int).tolist()))
    va, vb = vox(a), vox(b)
    inter  = len(va & vb); union = len(va | vb)
    return inter / union if union > 0 else 0.0


def match_segments(gt, det_pts, iou_thresh=0.25):
    n_gt, n_det = len(gt), len(det_pts)
    if n_gt == 0 or n_det == 0:
        return n_gt, n_det, 0
    gt_list = list(gt.values())
    iou_mat = np.zeros((n_gt, n_det))
    for gi, g in enumerate(gt_list):
        for di, d in enumerate(det_pts):
            iou_mat[gi, di] = point_set_iou(g, d)
    mg, md = set(), set()
    for idx in np.argsort(-iou_mat.ravel()):
        gi, di = divmod(int(idx), n_det)
        if iou_mat[gi, di] < iou_thresh: break
        if gi in mg or di in md: continue
        mg.add(gi); md.add(di)
    return n_gt, n_det, len(mg)


# ── SQ fitting ────────────────────────────────────────────────────────────────

def _fit_one_lm(args):
    pts, hint, conf, base_fitter = args
    restarts = 1 if conf > 0.55 else base_fitter.n_restarts
    f = SuperquadricFitter(n_restarts=restarts,
                           n_lm_rounds=base_fitter.n_lm_rounds,
                           subsample=base_fitter.subsample)
    multi = f.fit_adaptive(pts, shape_hint=hint, l2_threshold=0.008, max_primitives=2)
    return float(np.mean([p.chamfer_l2 for p in multi.primitives]))


def _fit_one_superdec(args):
    pts, hint, conf, fitter = args
    multi = fitter.fit_adaptive(pts, shape_hint=hint)
    if not multi.primitives:
        return 0.0
    l2s = [p.chamfer_l2 for p in multi.primitives]
    return float(np.mean(l2s)) if l2s else 0.0


def fit_parallel(result, fitter, n_workers, fitter_name='lm'):
    if not result.objects:
        return []

    if fitter_name == 'superdec':
        # SuperDec: all objects in one batched forward pass
        points_list = [seg.points for seg in result.objects]
        multis = fitter.fit_batch(points_list)
        return [float(np.mean([p.chamfer_l2 for p in m.primitives])) if m.primitives else 0.0
                for m in multis]

    # LM: parallel threads
    args = [(seg.points, seg.shape_type, seg.shape_conf, fitter)
            for seg in result.objects]
    l2s  = [None] * len(args)
    with ThreadPoolExecutor(max_workers=n_workers) as ex:
        futs = {ex.submit(_fit_one_lm, a): i for i, a in enumerate(args)}
        for fut in as_completed(futs):
            l2s[futs[fut]] = fut.result()
    return l2s


# ── single-view eval ──────────────────────────────────────────────────────────

def evaluate(data_dir, surface=None, view=None, shape=None,
             max_scenes=999, iou_thresh=0.25, fit_sq=False,
             n_workers=1, multiview=False,
             fitter_name='lm', superdec_dir=None,
             superdec_checkpoint='normalized', exist_threshold=0.3,
             superdec_npoints=4096, device=None):

    scenes = find_scenes(data_dir, surface, view, shape)
    print(f"\nFound {len(scenes)} frames  (surface={surface or 'all'} "
          f"view={view or 'all'} shape={shape or 'all'})")

    if multiview:
        pairs = _pair_scenes(scenes)
        print(f"Multi-view: {len(pairs)} top+bottom pairs\n")
        _eval_multiview(pairs[:max_scenes], iou_thresh, fit_sq, n_workers,
                        fitter_name, superdec_dir, superdec_checkpoint, exist_threshold, device)
        return

    scenes = scenes[:max_scenes]
    print(f"Evaluating {len(scenes)} frames  (n_workers={n_workers}, fitter={fitter_name})\n")

    perception  = TabletopPerception(classifier_path=CLASSIFIER_PATH)
    fitter      = make_fitter(fitter_name, superdec_dir,
                              superdec_checkpoint, exist_threshold, superdec_npoints, device) if fit_sq else None
    plane_cache = {}
    rows, t_p_all, t_sq_all = [], [], []

    for i, scene in enumerate(scenes):
        try:
            pts, col, mask = load_frame(scene)
        except Exception as e:
            print(f"  [{i:3d}] ERR {e}"); continue

        t0 = time.time()
        seq_key    = (scene["surface"], scene["view"], scene["shape"], scene["seq"])
        plane_hint = plane_cache.get(seq_key)
        result     = perception.run(pts, rgb=col, plane_hint=plane_hint)
        if seq_key not in plane_cache and result.table_normal is not None:
            plane_cache[seq_key] = (result.table_normal, result.table_height)
        t_p = time.time() - t0
        t_p_all.append(t_p)

        gt = gt_instances_from_mask(pts, mask)
        if not gt:
            n_gt, n_det, n_match = 0, len(result.objects), 0
        else:
            n_gt, n_det, n_match = match_segments(gt, [o.points for o in result.objects], iou_thresh)

        p  = n_match/n_det if n_det else (1. if n_gt==0 else 0.)
        r  = n_match/n_gt  if n_gt  else (1. if n_det==0 else 0.)
        f1 = 2*p*r/(p+r)   if (p+r) else 0.

        t_sq, mean_l2 = 0., float("nan")
        if fit_sq and result.objects:
            t0sq = time.time()
            l2s  = fit_parallel(result, fitter, n_workers, fitter_name)
            t_sq = time.time() - t0sq
            mean_l2 = float(np.mean(l2s)) if l2s else float("nan")
            t_sq_all.append(t_sq)

        rows.append({"seq": scene["seq"], "surface": scene["surface"],
                     "view": scene["view"], "shape": scene["shape"],
                     "n_gt": n_gt, "n_det": n_det, "n_match": n_match,
                     "p": p, "r": r, "f1": f1,
                     "t_perc": t_p, "t_sq": t_sq, "mean_l2": mean_l2,
                     "shape_types": [o.shape_type for o in result.objects]})

        sq_str = f"  L2={mean_l2*1e3:.2f}e-3" if fit_sq and not np.isnan(mean_l2) else ""
        print(f"  [{i:3d}] {scene['surface']:5s}/{scene['view']:6s}/{scene['shape']:6s} "
              f"{scene['seq']:6s}  P={p:.2f} R={r:.2f} F1={f1:.2f}  "
              f"GT={n_gt} Det={n_det} Match={n_match}  {t_p*1000:.0f}ms{sq_str}")

    _print_summary(rows, t_p_all, t_sq_all, fit_sq)


# ── multi-view eval ───────────────────────────────────────────────────────────

def _pair_scenes(scenes):
    by_key = {}
    for sc in scenes:
        stems = sorted(p.stem for p in Path(sc["pcd"]).parent.glob("*.pcd")
                       if not Path(p).name.startswith("._"))
        rank   = stems.index(sc["stem"]) if sc["stem"] in stems else -1
        seq_num = "".join(filter(str.isdigit, sc["seq"]))
        key    = (sc["surface"], sc["shape"], seq_num, rank)
        by_key.setdefault(key, {})[sc["view"]] = sc
    return [(v["top"], v["bottom"]) for v in by_key.values()
            if "top" in v and "bottom" in v]


def _eval_multiview(pairs, iou_thresh, fit_sq, n_workers,
                    fitter_name, superdec_dir, superdec_checkpoint, exist_threshold, device=None):
    perception = TabletopPerception(classifier_path=CLASSIFIER_PATH)
    fitter     = make_fitter(fitter_name, superdec_dir,
                             superdec_checkpoint, exist_threshold, superdec_npoints, device) if fit_sq else None

    print(f"  {'#':>4}  {'surface':8} {'shape':8}  "
          f"{'F1_top':>7} {'F1_bot':>7}  "
          f"{'L2_top×e3':>10} {'L2_fused×e3':>12} {'Δ%':>6}")
    print(f"  {'-'*4}  {'-'*8} {'-'*8}  {'-'*7} {'-'*7}  {'-'*10} {'-'*12} {'-'*6}")

    rows = []
    for i, (top_sc, bot_sc) in enumerate(pairs):
        try:
            top_pts, top_col, top_mask = load_frame(top_sc)
            bot_pts, bot_col, _        = load_frame(bot_sc)
        except Exception as e:
            print(f"  [{i:3d}] ERR {e}"); continue

        top_res = perception.run(top_pts, rgb=top_col)
        bot_res = perception.run(bot_pts, rgb=bot_col)

        gt = gt_instances_from_mask(top_pts, top_mask)
        n_gt, n_dt, n_mt = match_segments(gt, [o.points for o in top_res.objects], iou_thresh)
        n_gt, n_db, n_mb = match_segments(gt, [o.points for o in bot_res.objects], iou_thresh)

        def f1(nm, nd, ng): return 2*(nm/nd)*(nm/ng)/((nm/nd)+(nm/ng)) if nm and nd and ng else 0.
        f1_t = f1(n_mt, n_dt, n_gt)
        f1_b = f1(n_mb, n_db, n_gt)

        l2_top = l2_fused = float("nan")
        if fit_sq and top_res.objects and bot_res.objects:
            top_cents = np.array([o.points.mean(0) for o in top_res.objects])
            bot_cents = np.array([o.points.mean(0) for o in bot_res.objects])
            l2s_t, l2s_f, used = [], [], set()
            for ti, to in enumerate(top_res.objects):
                dists = np.linalg.norm(bot_cents - top_cents[ti], axis=1)
                bi    = int(dists.argmin())
                if dists[bi] < 0.10 and bi not in used:
                    used.add(bi)
                    bo    = bot_res.objects[bi]
                    fused = np.vstack([to.points, bo.points])
                    if fitter_name == 'superdec':
                        l2s_t.append(_fit_one_superdec((to.points, to.shape_type, to.shape_conf, fitter)))
                        l2s_f.append(_fit_one_superdec((fused, to.shape_type, to.shape_conf, fitter)))
                    else:
                        l2s_t.append(_fit_one_lm((to.points, to.shape_type, to.shape_conf, fitter)))
                        l2s_f.append(_fit_one_lm((fused, to.shape_type, to.shape_conf, fitter)))
            if l2s_t:
                l2_top   = float(np.mean(l2s_t))
                l2_fused = float(np.mean(l2s_f))

        delta = (l2_top - l2_fused) / l2_top * 100 if not np.isnan(l2_top) else float("nan")
        rows.append({"f1_t": f1_t, "f1_b": f1_b, "l2_top": l2_top,
                     "l2_fused": l2_fused, "delta": delta,
                     "surface": top_sc["surface"], "shape": top_sc["shape"]})

        l2_str = (f"{l2_top*1e3:>10.2f} {l2_fused*1e3:>12.2f} {delta:>5.1f}%"
                  if not np.isnan(l2_top) else f"{'n/a':>10} {'n/a':>12} {'':>6}")
        print(f"  [{i:3d}]  {top_sc['surface']:8} {top_sc['shape']:8}  "
              f"{f1_t:>7.3f} {f1_b:>7.3f}  {l2_str}")

    def m(k): v=[r[k] for r in rows if not np.isnan(r[k])]; return np.mean(v) if v else float("nan")
    print(f"\n{'='*70}")
    print(f"  MULTI-VIEW SUMMARY  ({len(rows)} pairs)")
    print(f"{'='*70}")
    print(f"  Mean F1 top view     : {m('f1_t'):.3f}")
    print(f"  Mean F1 bottom view  : {m('f1_b'):.3f}")
    if fit_sq:
        print(f"  Single-view L2×1e3  : {m('l2_top'):.3f}")
        print(f"  Fused-view  L2×1e3  : {m('l2_fused'):.3f}")
        print(f"  L2 improvement      : {m('delta'):.1f}%")
    print(f"{'='*70}\n")


# ── summary ───────────────────────────────────────────────────────────────────

def _print_summary(rows, t_p_all, t_sq_all, fit_sq):
    if not rows: print("No results."); return
    def m(k): v=[r[k] for r in rows if not np.isnan(r[k])]; return np.mean(v) if v else float("nan")
    print(f"\n{'='*70}")
    print(f"  AGGREGATE RESULTS  ({len(rows)} frames)")
    print(f"{'='*70}")
    print(f"  Segmentation")
    print(f"    Mean Precision : {m('p'):.3f}")
    print(f"    Mean Recall    : {m('r'):.3f}")
    print(f"    Mean F1        : {m('f1'):.3f}")
    print(f"    Mean GT/frame  : {m('n_gt'):.1f}")
    print(f"    Mean Det/frame : {m('n_det'):.1f}")
    print(f"\n  Timing")
    print(f"    Perception     : {np.mean(t_p_all)*1000:.0f} ms/frame")
    if fit_sq and t_sq_all:
        print(f"    SQ fitting     : {np.mean(t_sq_all)*1000:.0f} ms/frame")
        print(f"    TOTAL          : {(np.mean(t_p_all)+np.mean(t_sq_all))*1000:.0f} ms/frame")
    if fit_sq:
        l2s = [r["mean_l2"] for r in rows if not np.isnan(r["mean_l2"])]
        if l2s:
            print(f"\n  SQ Fitting (Chamfer L2)")
            print(f"    Mean L2×1e3    : {np.mean(l2s)*1e3:.3f}")
            print(f"    Median L2×1e3  : {np.median(l2s)*1e3:.3f}")
            print(f"    Max L2×1e3     : {np.max(l2s)*1e3:.3f}")
    print(f"\n  Breakdown by view")
    for surf in ["floor", "table"]:
        for v in ["top", "bottom"]:
            sub = [r for r in rows if r["surface"]==surf and r["view"]==v]
            if sub: print(f"    {surf:5s}/{v:6s}: F1={np.mean([r['f1'] for r in sub]):.3f}  ({len(sub)} frames)")
    print(f"\n  Breakdown by shape type")
    for sh in ["mixed", "curved", "cuboid"]:
        sub = [r for r in rows if r["shape"]==sh]
        if sub: print(f"    {sh:8s}: F1={np.mean([r['f1'] for r in sub]):.3f}  ({len(sub)} frames)")
    all_types = [t for r in rows for t in r["shape_types"]]
    if all_types:
        counts = Counter(all_types); total = len(all_types)
        print(f"\n  Shape distribution ({total} detections)")
        for t, c in sorted(counts.items(), key=lambda x: -x[1]):
            print(f"    {t:12s}: {c:4d}  ({100*c/total:.1f}%)")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir",             required=True)
    ap.add_argument("--surface",              default=None, choices=["floor", "table"])
    ap.add_argument("--view",                 default=None, choices=["top", "bottom"])
    ap.add_argument("--shape",                default=None, choices=["mixed", "curved", "cuboid"])
    ap.add_argument("--max_scenes",           type=int,   default=999)
    ap.add_argument("--iou_thresh",           type=float, default=0.25)
    ap.add_argument("--fit_sq",               action="store_true")
    ap.add_argument("--n_workers",            type=int,   default=1)
    ap.add_argument("--multiview",            action="store_true")
    ap.add_argument("--fitter",               default='lm', choices=['lm', 'superdec'],
                    help="SQ fitter: 'lm' (default) or 'superdec'")
    ap.add_argument("--superdec_dir",         default=None,
                    help="Path to superdec repo root (required if --fitter superdec)")
    ap.add_argument("--superdec_checkpoint",  default='normalized',
                    choices=['normalized', 'shapenet'])
    ap.add_argument("--superdec_npoints",     type=int,   default=4096)
    ap.add_argument("--exist_threshold",      type=float, default=0.3,
                    help="SuperDec primitive existence threshold (default 0.3)")
    ap.add_argument("--device",               default=None,
                    help="Device for SuperDec: cuda or cpu (default: auto-detect)")
    args = ap.parse_args()

    evaluate(
        data_dir=args.data_dir, surface=args.surface, view=args.view,
        shape=args.shape, max_scenes=args.max_scenes, iou_thresh=args.iou_thresh,
        fit_sq=args.fit_sq, n_workers=args.n_workers, multiview=args.multiview,
        fitter_name=args.fitter, superdec_dir=args.superdec_dir,
        superdec_checkpoint=args.superdec_checkpoint,
        exist_threshold=args.exist_threshold,
        device=args.device,
    )
