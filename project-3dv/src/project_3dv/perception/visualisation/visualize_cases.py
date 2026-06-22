"""
visualize_cases.py
==================
Saves coloured PLY files for best and worst pipeline frames,
and an HTML report with per-frame statistics.

Usage:
  python3 visualize_cases.py --data_dir /Volumes/T7/OCID-dataset/YCB10 \
      --out_dir vis_cases --n_cases 10

Output (in out_dir/):
  good/  — top-N frames by F1
  bad/   — bottom-N frames by F1 (under-detection and over-detection)
  report.html — browsable summary
"""

import argparse, sys, os, json, time
import numpy as np
import open3d as o3d
from pathlib import Path
from PIL import Image
from typing import List, Dict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ..pipeline import TabletopPerception

CLASSIFIER_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sq_shape_classifier.pkl")
if not os.path.exists(CLASSIFIER_PATH):
    CLASSIFIER_PATH = os.path.join(os.getcwd(), "sq_shape_classifier.pkl")

# colours
PALETTE = [
    [0.9, 0.2, 0.2],  # red
    [0.2, 0.7, 0.3],  # green
    [0.2, 0.4, 0.9],  # blue
    [0.9, 0.7, 0.1],  # yellow
    [0.8, 0.3, 0.8],  # purple
    [0.1, 0.8, 0.8],  # cyan
    [0.95, 0.5, 0.1], # orange
    [0.5, 0.9, 0.5],  # light green
    [0.6, 0.4, 0.2],  # brown
    [0.9, 0.5, 0.7],  # pink
    [0.3, 0.6, 0.9],  # sky blue
    [0.7, 0.9, 0.2],  # lime
]
GT_ALPHA   = 0.6   # GT points lighter
DET_ALPHA  = 1.0
TABLE_COL  = [0.75, 0.75, 0.75]
BG_COL     = [0.15, 0.15, 0.15]


def find_scenes(data_dir, surface=None, view=None, shape=None):
    data_dir = Path(data_dir)
    scenes = []
    for surf in ([surface] if surface else ["floor", "table"]):
        for v in ([view] if view else ["top", "bottom"]):
            for sh in ([shape] if shape else ["mixed", "curved", "cuboid"]):
                base = data_dir / surf / v / sh
                if not base.exists(): continue
                for seq_dir in sorted(base.iterdir()):
                    if not seq_dir.is_dir(): continue
                    pcd_dir   = seq_dir / "pcd"
                    label_dir = seq_dir / "label"
                    if not pcd_dir.exists() or not label_dir.exists(): continue
                    pcds = {p.stem: p for p in pcd_dir.glob("*.pcd")  if not p.name.startswith("._")}
                    lbls = {p.stem: p for p in label_dir.glob("*.png") if not p.name.startswith("._")}
                    for stem in sorted(set(pcds) & set(lbls)):
                        scenes.append({"pcd": str(pcds[stem]), "label": str(lbls[stem]),
                                       "stem": stem, "seq": seq_dir.name,
                                       "surface": surf, "view": v, "shape": sh})
    return scenes


def load_frame(scene):
    pcd  = o3d.io.read_point_cloud(scene["pcd"])
    pts  = np.asarray(pcd.points, dtype=np.float32)
    col  = np.asarray(pcd.colors, dtype=np.float32) if pcd.has_colors() else None
    mask = np.array(Image.open(scene["label"]), dtype=np.int32)
    return pts, col, mask


def gt_instances(pts, mask):
    h, w = mask.shape
    if len(pts) != h * w: return {}
    flat = mask.ravel()
    return {int(l): pts[flat == l] for l in np.unique(flat) if l != 0}


def point_set_iou(a, b, voxel=0.005):
    def vox(p): return set(map(tuple, np.round(p / voxel).astype(int).tolist()))
    va, vb = vox(a), vox(b)
    inter = len(va & vb); union = len(va | vb)
    return inter / union if union > 0 else 0.0


def match_segments(gt, det_pts, iou_thresh=0.25):
    n_gt, n_det = len(gt), len(det_pts)
    if n_gt == 0 or n_det == 0: return n_gt, n_det, 0, {}
    gt_list = list(gt.values())
    iou_mat = np.zeros((n_gt, n_det))
    for gi, g in enumerate(gt_list):
        for di, d in enumerate(det_pts):
            iou_mat[gi, di] = point_set_iou(g, d)
    mg, md, matches = set(), set(), {}
    for idx in np.argsort(-iou_mat.ravel()):
        gi, di = divmod(int(idx), n_det)
        if iou_mat[gi, di] < iou_thresh: break
        if gi in mg or di in md: continue
        mg.add(gi); md.add(di)
        matches[di] = gi
    return n_gt, n_det, len(mg), matches


def save_vis_ply(pts, col, gt, det_objects, matches, out_path):
    """
    Save a coloured PLY:
      - table/background: grey
      - GT instances: coloured at 50% brightness (ground truth)
      - detected objects: full colour if matched, white if false positive
      - missed GT: red tint overlay
    """
    all_pts  = []
    all_cols = []

    # background (raw, grey)
    if col is not None:
        bg_col = col * 0.3 + 0.1
    else:
        bg_col = np.full((len(pts), 3), 0.2, dtype=np.float32)
    all_pts.append(pts)
    all_cols.append(bg_col)

    # GT instances — pale tinted
    gt_list = list(gt.values())
    for gi, g in enumerate(gt_list):
        c = np.array(PALETTE[gi % len(PALETTE)]) * 0.4 + 0.1
        cols = np.tile(c, (len(g), 1))
        all_pts.append(g)
        all_cols.append(cols.astype(np.float32))

    # detections — bright if matched, white if FP
    det_pts_list = [o.points for o in det_objects]
    for di, d in enumerate(det_pts_list):
        if di in matches:
            gi = matches[di]
            c  = np.array(PALETTE[gi % len(PALETTE)])
        else:
            c  = np.array([1.0, 1.0, 1.0])  # false positive = white
        cols = np.tile(c, (len(d), 1))
        all_pts.append(d)
        all_cols.append(cols.astype(np.float32))

    merged_pts  = np.vstack(all_pts).astype(np.float32)
    merged_cols = np.vstack(all_cols).astype(np.float32)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(merged_pts)
    pcd.colors = o3d.utility.Vector3dVector(np.clip(merged_cols, 0, 1))
    o3d.io.write_point_cloud(str(out_path), pcd)


def run_eval_and_collect(scenes, perception, max_frames=999):
    rows = []
    plane_cache = {}
    for i, scene in enumerate(scenes[:max_frames]):
        try:
            pts, col, mask = load_frame(scene)
        except: continue

        seq_key    = (scene["surface"], scene["view"], scene["shape"], scene["seq"])
        plane_hint = plane_cache.get(seq_key)
        result     = perception.run(pts, rgb=col, plane_hint=plane_hint)
        if seq_key not in plane_cache and result.table_normal is not None:
            plane_cache[seq_key] = (result.table_normal, result.table_height)

        gt = gt_instances(pts, mask)
        if not gt:
            continue

        det_pts = [o.points for o in result.objects]
        n_gt, n_det, n_match, matches = match_segments(gt, det_pts)
        p  = n_match/n_det if n_det else 0.
        r  = n_match/n_gt  if n_gt  else 0.
        f1 = 2*p*r/(p+r)   if (p+r) else 0.

        failure_mode = "ok"
        if n_det == 0:                      failure_mode = "no_detection"
        elif n_det > n_gt * 1.5:            failure_mode = "over_split"
        elif n_match < n_gt * 0.4:          failure_mode = "under_detect"
        elif n_match == n_gt and p == 1.0:  failure_mode = "perfect"

        rows.append({
            "scene": scene, "pts": pts, "col": col, "gt": gt,
            "det_objects": result.objects, "matches": matches,
            "n_gt": n_gt, "n_det": n_det, "n_match": n_match,
            "p": p, "r": r, "f1": f1,
            "failure_mode": failure_mode,
            "frame_idx": i,
        })

        print(f"  [{i:3d}] F1={f1:.2f}  GT={n_gt} Det={n_det} "
              f"Match={n_match}  [{failure_mode}]")

    return rows


def save_cases(rows, out_dir, n_cases=8):
    out_dir = Path(out_dir)
    (out_dir / "good").mkdir(parents=True, exist_ok=True)
    (out_dir / "bad"  ).mkdir(parents=True, exist_ok=True)
    (out_dir / "modes").mkdir(parents=True, exist_ok=True)

    sorted_rows = sorted(rows, key=lambda r: r["f1"])

    # worst cases
    worst = [r for r in sorted_rows if r["failure_mode"] != "perfect"][:n_cases]
    for i, r in enumerate(worst):
        sc   = r["scene"]
        name = f"bad_{i:02d}_F1{r['f1']:.2f}_{r['failure_mode']}_{sc['seq']}_{sc['stem']}.ply"
        save_vis_ply(r["pts"], r["col"], r["gt"], r["det_objects"],
                     r["matches"], out_dir / "bad" / name)
        print(f"  saved bad/{name}")

    # best cases
    best = [r for r in reversed(sorted_rows) if r["f1"] > 0.7][:n_cases]
    for i, r in enumerate(best):
        sc   = r["scene"]
        name = f"good_{i:02d}_F1{r['f1']:.2f}_{sc['seq']}_{sc['stem']}.ply"
        save_vis_ply(r["pts"], r["col"], r["gt"], r["det_objects"],
                     r["matches"], out_dir / "good" / name)
        print(f"  saved good/{name}")

    # one example per failure mode
    for mode in ["no_detection", "over_split", "under_detect", "perfect"]:
        examples = [r for r in rows if r["failure_mode"] == mode][:2]
        for i, r in enumerate(examples):
            sc   = r["scene"]
            name = f"{mode}_{i}_{sc['seq']}_{sc['stem']}.ply"
            save_vis_ply(r["pts"], r["col"], r["gt"], r["det_objects"],
                         r["matches"], out_dir / "modes" / name)

    return sorted_rows


def save_html_report(rows, out_dir):
    out_dir = Path(out_dir)
    by_mode = {}
    for r in rows:
        by_mode.setdefault(r["failure_mode"], []).append(r["f1"])

    mode_stats = {m: {"n": len(v), "mean_f1": float(np.mean(v))}
                  for m, v in by_mode.items()}

    f1s = [r["f1"] for r in rows]
    total = len(rows)

    html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>Perception Pipeline — Case Analysis</title>
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ font-family: 'JetBrains Mono', 'Fira Code', monospace;
          background: #0e0e12; color: #d4d4d8; padding: 2rem; }}
  h1 {{ color: #a78bfa; font-size: 1.4rem; margin-bottom: 0.3rem; }}
  h2 {{ color: #7dd3fc; font-size: 1rem; margin: 2rem 0 0.8rem; text-transform: uppercase;
        letter-spacing: 0.1em; }}
  .subtitle {{ color: #71717a; font-size: 0.8rem; margin-bottom: 2rem; }}
  .grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(200px, 1fr));
           gap: 1rem; margin-bottom: 2rem; }}
  .card {{ background: #18181b; border: 1px solid #27272a; border-radius: 8px;
           padding: 1rem; }}
  .card .label {{ color: #a1a1aa; font-size: 0.7rem; text-transform: uppercase;
                  letter-spacing: 0.08em; margin-bottom: 0.3rem; }}
  .card .value {{ font-size: 1.4rem; color: #f4f4f5; }}
  .card .sub {{ font-size: 0.75rem; color: #52525b; margin-top: 0.2rem; }}
  .mode-ok    {{ border-left: 3px solid #22c55e; }}
  .mode-bad   {{ border-left: 3px solid #ef4444; }}
  .mode-warn  {{ border-left: 3px solid #f59e0b; }}
  table {{ width: 100%; border-collapse: collapse; font-size: 0.8rem; }}
  th {{ color: #a78bfa; text-align: left; padding: 0.5rem;
        border-bottom: 1px solid #27272a; }}
  td {{ padding: 0.4rem 0.5rem; border-bottom: 1px solid #18181b; }}
  tr:hover td {{ background: #18181b; }}
  .f1-high {{ color: #22c55e; }}
  .f1-mid  {{ color: #f59e0b; }}
  .f1-low  {{ color: #ef4444; }}
  .bar {{ height: 4px; background: #27272a; border-radius: 2px; margin-top: 0.4rem; }}
  .bar-fill {{ height: 100%; border-radius: 2px; }}
  .legend {{ display: flex; gap: 1.5rem; font-size: 0.75rem; color: #71717a; 
             margin-bottom: 1rem; flex-wrap: wrap; }}
  .legend-item {{ display: flex; align-items: center; gap: 0.4rem; }}
  .dot {{ width: 10px; height: 10px; border-radius: 50%; }}
</style>
</head>
<body>
<h1>Perception Pipeline — Case Analysis</h1>
<p class="subtitle">OCID YCB10 · {total} frames · generated {__import__('datetime').datetime.now().strftime('%Y-%m-%d %H:%M')}</p>

<h2>Overall</h2>
<div class="grid">
  <div class="card">
    <div class="label">Mean F1</div>
    <div class="value">{np.mean(f1s):.3f}</div>
    <div class="bar"><div class="bar-fill" style="width:{np.mean(f1s)*100:.0f}%;background:#a78bfa"></div></div>
  </div>
  <div class="card">
    <div class="label">Mean Precision</div>
    <div class="value">{np.mean([r['p'] for r in rows]):.3f}</div>
  </div>
  <div class="card">
    <div class="label">Mean Recall</div>
    <div class="value">{np.mean([r['r'] for r in rows]):.3f}</div>
  </div>
  <div class="card">
    <div class="label">Frames evaluated</div>
    <div class="value">{total}</div>
  </div>
</div>

<h2>Failure Mode Breakdown</h2>
<div class="legend">
  <div class="legend-item"><div class="dot" style="background:#22c55e"></div> perfect (F1=1.0)</div>
  <div class="legend-item"><div class="dot" style="background:#7dd3fc"></div> ok (F1>0.5)</div>
  <div class="legend-item"><div class="dot" style="background:#f59e0b"></div> under_detect (low recall)</div>
  <div class="legend-item"><div class="dot" style="background:#ef4444"></div> no_detection / over_split</div>
</div>
<div class="grid">
"""
    mode_colours = {"perfect": "#22c55e", "ok": "#7dd3fc",
                    "under_detect": "#f59e0b", "no_detection": "#ef4444", "over_split": "#ef4444"}
    for mode, stats in sorted(mode_stats.items(), key=lambda x: -x[1]["n"]):
        pct = stats["n"] / total * 100
        col = mode_colours.get(mode, "#a1a1aa")
        html += f"""  <div class="card" style="border-left:3px solid {col}">
    <div class="label">{mode}</div>
    <div class="value">{stats['n']} <span style="font-size:0.9rem;color:#52525b">({pct:.0f}%)</span></div>
    <div class="sub">mean F1 = {stats['mean_f1']:.3f}</div>
    <div class="bar"><div class="bar-fill" style="width:{pct:.0f}%;background:{col}"></div></div>
  </div>
"""
    html += "</div>\n"

    # per-frame table
    html += "<h2>All Frames</h2>\n<table>\n"
    html += "<tr><th>#</th><th>Seq</th><th>Surface</th><th>Shape</th>"
    html += "<th>GT</th><th>Det</th><th>Match</th><th>P</th><th>R</th><th>F1</th><th>Mode</th></tr>\n"

    sorted_rows = sorted(rows, key=lambda r: r["f1"])
    for r in sorted_rows:
        sc = r["scene"]
        f1c = "f1-high" if r["f1"] > 0.7 else ("f1-mid" if r["f1"] > 0.4 else "f1-low")
        mc  = mode_colours.get(r["failure_mode"], "#a1a1aa")
        html += (f"<tr><td>{r['frame_idx']}</td><td>{sc['seq']}</td>"
                 f"<td>{sc['surface']}/{sc['view']}</td><td>{sc['shape']}</td>"
                 f"<td>{r['n_gt']}</td><td>{r['n_det']}</td><td>{r['n_match']}</td>"
                 f"<td>{r['p']:.2f}</td><td>{r['r']:.2f}</td>"
                 f"<td class='{f1c}'>{r['f1']:.2f}</td>"
                 f"<td style='color:{mc}'>{r['failure_mode']}</td></tr>\n")
    html += "</table>\n"

    html += """
<h2>How to view PLY files</h2>
<div class="card" style="max-width:600px">
  <div class="label">MeshLab (recommended)</div>
  <div style="margin-top:0.5rem;color:#a1a1aa;font-size:0.8rem;line-height:1.6">
    Colour key:<br>
    &nbsp;· <span style="color:#888">Grey</span> = background<br>
    &nbsp;· <span style="color:#adf">Pale tinted</span> = ground truth instance<br>
    &nbsp;· <span style="color:#7f7">Bright</span> = matched detection (same colour as GT)<br>
    &nbsp;· <span style="color:#fff">White</span> = false positive detection<br>
    &nbsp;· No bright points over a pale instance = missed object (false negative)
  </div>
</div>
</body></html>"""

    report_path = out_dir / "report.html"
    with open(report_path, "w") as f:
        f.write(html)
    print(f"\n  HTML report → {report_path}")
    print(f"  Open with: open {report_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir",   required=True)
    ap.add_argument("--out_dir",    default="vis_cases")
    ap.add_argument("--n_cases",    type=int, default=8)
    ap.add_argument("--max_frames", type=int, default=999)
    ap.add_argument("--surface",    default=None)
    ap.add_argument("--view",       default=None)
    ap.add_argument("--shape",      default=None)
    args = ap.parse_args()

    scenes = find_scenes(args.data_dir, args.surface, args.view, args.shape)
    print(f"Found {len(scenes)} frames")

    perception = TabletopPerception(classifier_path=CLASSIFIER_PATH)

    print("\nRunning pipeline...")
    rows = run_eval_and_collect(scenes, perception, args.max_frames)

    print(f"\nSaving visualisations to {args.out_dir}/...")
    sorted_rows = save_cases(rows, args.out_dir, args.n_cases)
    save_html_report(rows, args.out_dir)

    # print quick failure mode summary
    from collections import Counter
    modes = Counter(r["failure_mode"] for r in rows)
    print(f"\nFailure mode summary ({len(rows)} frames):")
    for mode, count in modes.most_common():
        mean_f1 = np.mean([r["f1"] for r in rows if r["failure_mode"] == mode])
        print(f"  {mode:20s}: {count:3d} frames  mean F1={mean_f1:.3f}") 