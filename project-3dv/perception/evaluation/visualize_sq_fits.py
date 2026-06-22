"""
visualize_sq_fits.py
====================
Visualize SuperDec superquadric fitting results on OCID frames.
Saves colored PLY files and an HTML report you can open in a browser.

Usage
-----
    cd ~/project-3dv
    python3 src/project_3dv/perception/evaluation/visualize_sq_fits.py \
        --data_dir ~/data/OCID/YCB10 \
        --superdec_dir ~/superdec \
        --out_dir ~/sq_vis \
        --n_frames 10 \
        --surface floor --view top

Output
------
    ~/sq_vis/
        frame_000/
            scene.ply          — full scene point cloud (gray)
            segments.ply       — detected segments (colored by object)
            sq_primitives.ply  — fitted SQ surfaces (colored by primitive type)
            grasp_axes.ply     — grasp approach axes as line segments
        ...
        report.html            — browsable summary
"""

import os, sys, argparse, random
import numpy as np

# ── project path ────────────────────────────────────────────────────────────
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.insert(0, os.path.join(ROOT, 'src'))
sys.path.insert(0, os.path.join(ROOT, 'src', 'project_3dv', 'perception', 'evaluation'))
sys.path.insert(0, os.path.join(ROOT, 'src', 'project_3dv', 'perception'))

from project_3dv.perception.pipeline import TabletopPerception
CLASSIFIER_PATH = os.path.join(ROOT, 'models', 'sq_shape_classifier.pkl')


# ── SQ surface sampling ──────────────────────────────────────────────────────
def sample_sq_surface(sx, sy, sz, e1, e2, tx, ty, tz, rx, ry, rz,
                       n_points=1000) -> np.ndarray:
    """Sample points uniformly on a superquadric surface (spherical product)."""
    # latitude η ∈ [-π/2, π/2], longitude ω ∈ [-π, π]
    eta = np.linspace(-np.pi / 2, np.pi / 2, int(np.sqrt(n_points)))
    omega = np.linspace(-np.pi, np.pi, int(np.sqrt(n_points)))
    ETA, OMEGA = np.meshgrid(eta, omega)
    ETA, OMEGA = ETA.ravel(), OMEGA.ravel()

    def fexp(base, exp):
        return np.sign(base) * (np.abs(base) ** exp)

    x = sx * fexp(np.cos(ETA), e1) * fexp(np.cos(OMEGA), e2)
    y = sy * fexp(np.cos(ETA), e1) * fexp(np.sin(OMEGA), e2)
    z = sz * fexp(np.sin(ETA), e1)

    pts = np.stack([x, y, z], axis=1)   # (N, 3) in canonical frame

    # rotate
    cx, sx_ = np.cos(rx), np.sin(rx)
    cy, sy_ = np.cos(ry), np.sin(ry)
    cz, sz_ = np.cos(rz), np.sin(rz)
    Rx = np.array([[1,0,0],[0,cx_,-sx_],[0,sx_,cx_]])
    Ry = np.array([[cy_,0,sy_],[0,1,0],[-sy_,0,cy_]])
    Rz = np.array([[cz_,-sz_,0],[sz_,cz_,0],[0,0,1]])
    R  = Rz @ Ry @ Rx
    pts = (R @ pts.T).T
    pts += np.array([tx, ty, tz])
    return pts.astype(np.float32)


# ── color maps ───────────────────────────────────────────────────────────────
PTYPE_COLORS = {
    'Cylinder':  [0.2, 0.6, 1.0],   # blue
    'Cuboid':    [1.0, 0.4, 0.2],   # orange
    'Ellipsoid': [0.2, 0.9, 0.4],   # green
    'Other':     [0.9, 0.8, 0.1],   # yellow
}

OBJECT_PALETTE = [
    [1.0, 0.2, 0.2], [0.2, 0.7, 1.0], [0.2, 0.9, 0.3],
    [1.0, 0.6, 0.1], [0.8, 0.2, 0.9], [0.1, 0.9, 0.9],
    [1.0, 0.9, 0.2], [0.5, 0.3, 1.0], [0.9, 0.5, 0.5],
    [0.3, 0.8, 0.5],
]


def _write_ply(path: str, points: np.ndarray, colors: np.ndarray):
    """Write colored point cloud as PLY (ASCII)."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    n = len(points)
    with open(path, 'w') as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {n}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        f.write("end_header\n")
        for i in range(n):
            r = int(np.clip(colors[i, 0] * 255, 0, 255))
            g = int(np.clip(colors[i, 1] * 255, 0, 255))
            b = int(np.clip(colors[i, 2] * 255, 0, 255))
            f.write(f"{points[i,0]:.4f} {points[i,1]:.4f} {points[i,2]:.4f} {r} {g} {b}\n")


def _write_axes_ply(path: str, origins: list, directions: list,
                    length=0.05, colors=None):
    """Write grasp approach axes as line-segment point clouds."""
    all_pts, all_cols = [], []
    for i, (o, d) in enumerate(zip(origins, directions)):
        col = colors[i] if colors else [1., 0., 0.]
        pts = np.linspace(o, o + np.array(d) * length, 20)
        all_pts.append(pts)
        all_cols.append(np.tile(col, (20, 1)))
    if all_pts:
        _write_ply(path, np.vstack(all_pts), np.vstack(all_cols))


# ── main visualizer ──────────────────────────────────────────────────────────
def visualize_frame(frame_dir: str, depth_path: str, rgb_path: str,
                    perception, fitter, selector, frame_idx: int):
    import cv2

    depth = cv2.imread(depth_path, cv2.IMREAD_ANYDEPTH).astype(np.float32) / 1000.0
    rgb   = cv2.cvtColor(cv2.imread(rgb_path), cv2.COLOR_BGR2RGB)

    # ── perception ──
    h, w = depth.shape
    fx = fy = 525.0; cx, cy = w / 2, h / 2
    xs = (np.arange(w) - cx) / fx
    ys = (np.arange(h) - cy) / fy
    XX, YY = np.meshgrid(xs, ys)
    Z  = depth
    X  = XX * Z; Y = YY * Z
    mask   = (Z > 0.1) & (Z < 2.5)
    points = np.stack([X, Y, Z], axis=-1)[mask].astype(np.float32)
    colors_scene = rgb[mask].astype(np.float32) / 255.0

    result = perception.run(points, colors_scene)

    if not result.objects:
        return None

    # ── save scene PLY ──
    os.makedirs(frame_dir, exist_ok=True)
    _write_ply(os.path.join(frame_dir, 'scene.ply'),
               points, colors_scene)

    # ── save segments PLY ──
    seg_pts, seg_cols = [], []
    for obj_id, obj in enumerate(result.objects):
        col = OBJECT_PALETTE[obj_id % len(OBJECT_PALETTE)]
        seg_pts.append(obj.points)
        seg_cols.append(np.tile(col, (len(obj.points), 1)))
    if seg_pts:
        _write_ply(os.path.join(frame_dir, 'segments.ply'),
                   np.vstack(seg_pts), np.vstack(seg_cols))

    # ── fit SQs and save primitives PLY ──
    points_list = [obj.points for obj in result.objects]
    sq_fits     = fitter.fit_batch(points_list)

    sq_pts, sq_cols = [], []
    grasp_origins, grasp_dirs, grasp_colors = [], [], []
    obj_summaries = []

    for obj_id, (obj, sq_fit) in enumerate(zip(result.objects, sq_fits)):
        obj_col = OBJECT_PALETTE[obj_id % len(OBJECT_PALETTE)]
        prim_summary = []

        for prim in sq_fit.primitives:
            ptype = getattr(prim, 'shape_type', 'Other')
            col   = PTYPE_COLORS.get(ptype, [0.5, 0.5, 0.5])

            try:
                surface_pts = sample_sq_surface(
                    prim.sx, prim.sy, prim.sz,
                    prim.e1, prim.e2,
                    prim.tx, prim.ty, prim.tz,
                    prim.rx, prim.ry, prim.rz,
                    n_points=600,
                )
                sq_pts.append(surface_pts)
                sq_cols.append(np.tile(col, (len(surface_pts), 1)))
            except Exception:
                pass

            prim_summary.append({
                'type': ptype,
                'scale': f"{prim.sx*100:.1f}×{prim.sy*100:.1f}×{prim.sz*100:.1f} cm",
                'conf': f"{prim.shape_conf:.2f}",
                'l2': f"{prim.chamfer_l2*1000:.2f}e-3",
            })

        # grasp
        if selector is not None:
            gset = selector.grasp_candidates(sq_fit, obj_id)
            best = gset.best
            if best is not None:
                grasp_origins.append(best.position)
                grasp_dirs.append(best.approach)
                grasp_colors.append(obj_col)

        obj_summaries.append({
            'id': obj_id,
            'n_pts': len(obj.points),
            'n_prims': len(sq_fit.primitives),
            'primitives': prim_summary,
        })

    if sq_pts:
        _write_ply(os.path.join(frame_dir, 'sq_primitives.ply'),
                   np.vstack(sq_pts), np.vstack(sq_cols))

    if grasp_origins:
        _write_axes_ply(os.path.join(frame_dir, 'grasp_axes.ply'),
                        grasp_origins, grasp_dirs,
                        length=0.08, colors=grasp_colors)

    return obj_summaries


def build_html_report(out_dir: str, frame_results: list):
    rows = []
    for fr in frame_results:
        if fr is None:
            continue
        fdir = fr['dir']
        fname = os.path.basename(fdir)
        rows.append(f"""
        <tr>
          <td><b>{fname}</b></td>
          <td>{fr['n_objects']}</td>
          <td>{fr['n_primitives']}</td>
          <td>
            <details><summary>show</summary>
            <pre>{fr['summary']}</pre>
            </details>
          </td>
          <td>
            <a href="{fname}/scene.ply">scene</a> |
            <a href="{fname}/segments.ply">segments</a> |
            <a href="{fname}/sq_primitives.ply">SQs</a> |
            <a href="{fname}/grasp_axes.ply">grasps</a>
          </td>
        </tr>""")

    html = f"""<!DOCTYPE html>
<html><head>
<title>SuperDec SQ Visualization</title>
<style>
body {{ font-family: monospace; padding: 20px; background: #1a1a1a; color: #eee; }}
table {{ border-collapse: collapse; width: 100%; }}
th, td {{ border: 1px solid #444; padding: 8px; text-align: left; }}
th {{ background: #333; }}
tr:nth-child(even) {{ background: #222; }}
a {{ color: #6af; }}
pre {{ font-size: 11px; max-height: 200px; overflow-y: auto; }}
</style>
</head><body>
<h1>SuperDec SQ Fitting Visualization</h1>
<p>Open PLY files in <a href="https://www.meshlab.net/">MeshLab</a> or
   <a href="https://www.open3d.org/docs/release/tutorial/visualization/visualization.html">Open3D</a>
   to view 3D results.</p>
<p><b>Color legend:</b>
  <span style="color:#39f">■ Cylinder</span>
  <span style="color:#f72">■ Cuboid</span>
  <span style="color:#3e6">■ Ellipsoid</span>
  <span style="color:#fc2">■ Other</span>
</p>
<table>
<tr>
  <th>Frame</th><th>#Objects</th><th>#Primitives</th>
  <th>Summary</th><th>PLY files</th>
</tr>
{''.join(rows)}
</table>
</body></html>"""

    with open(os.path.join(out_dir, 'report.html'), 'w') as f:
        f.write(html)
    print(f"\nReport: {os.path.join(out_dir, 'report.html')}")


# ── CLI ──────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_dir',     required=True)
    ap.add_argument('--superdec_dir', required=True)
    ap.add_argument('--out_dir',      default='~/sq_vis')
    ap.add_argument('--n_frames',     type=int, default=10)
    ap.add_argument('--surface',      default=None)
    ap.add_argument('--view',         default=None)
    ap.add_argument('--shape',        default=None)
    ap.add_argument('--checkpoint',   default='normalized')
    ap.add_argument('--exist_threshold', type=float, default=0.3)
    ap.add_argument('--n_points',     type=int, default=2048)
    ap.add_argument('--device',       default=None)
    ap.add_argument('--seed',         type=int, default=42)
    a = ap.parse_args()

    random.seed(a.seed); np.random.seed(a.seed)
    out_dir = os.path.expanduser(a.out_dir)
    os.makedirs(out_dir, exist_ok=True)

    # ── load models ──
    import torch
    device = a.device or ('cuda' if torch.cuda.is_available() else 'cpu')

    perception = TabletopPerception(classifier_path=CLASSIFIER_PATH
                                    if os.path.exists(CLASSIFIER_PATH) else None)

    sys.path.insert(0, os.path.join(ROOT, 'src', 'project_3dv', 'perception'))
    from project_3dv.perception.superdec_fitter import SuperdecFitter
    fitter = SuperdecFitter(
        superdec_dir=a.superdec_dir,
        checkpoint=a.checkpoint,
        exist_threshold=a.exist_threshold,
        n_points=a.n_points,
        device=device,
    )

    from grasp_from_sq import GraspSelector
    selector = GraspSelector(robot_base=np.array([0., 0., 0.]))

    # ── find frames ──
    sys.path.insert(0, os.path.join(ROOT, 'src', 'project_3dv',
                                    'perception', 'evaluation'))
    from ocid_eval import find_scenes, load_frame
    scenes = find_scenes(a.data_dir, a.surface, a.view, a.shape)
    if len(scenes) > a.n_frames:
        scenes = random.sample(scenes, a.n_frames)
    print(f"Visualizing {len(scenes)} frames → {out_dir}")

    frame_results = []
    for i, scene in enumerate(scenes):
        depth_path = os.path.join(scene.path, 'rgb', scene.depth_file
                                  if hasattr(scene, 'depth_file') else 'depth.png')
        rgb_path   = os.path.join(scene.path, 'rgb', scene.rgb_file
                                  if hasattr(scene, 'rgb_file') else 'rgb.png')

        # use load_frame from ocid_eval to get points + rgb correctly
        try:
            points, rgb_colors = load_frame(scene)
        except Exception as e:
            print(f"  [{i}] load error: {e}")
            frame_results.append(None)
            continue

        result = perception.run(points, rgb_colors)
        if not result.objects:
            print(f"  [{i}] no objects detected")
            frame_results.append(None)
            continue

        sq_fits = fitter.fit_batch([obj.points for obj in result.objects])

        frame_dir = os.path.join(out_dir, f'frame_{i:03d}')
        os.makedirs(frame_dir, exist_ok=True)

        # scene PLY
        scene_col = np.ones((len(points), 3)) * 0.5
        _write_ply(os.path.join(frame_dir, 'scene.ply'), points, scene_col)

        # segments PLY
        seg_pts, seg_cols = [], []
        for obj_id, obj in enumerate(result.objects):
            col = OBJECT_PALETTE[obj_id % len(OBJECT_PALETTE)]
            seg_pts.append(obj.points)
            seg_cols.append(np.tile(col, (len(obj.points), 1)))
        _write_ply(os.path.join(frame_dir, 'segments.ply'),
                   np.vstack(seg_pts), np.vstack(seg_cols))

        # SQ primitives PLY
        sq_pts, sq_cols = [], []
        grasp_origins, grasp_dirs, grasp_colors = [], [], []
        total_prims = 0
        summary_lines = []

        for obj_id, (obj, sq_fit) in enumerate(zip(result.objects, sq_fits)):
            obj_col = OBJECT_PALETTE[obj_id % len(OBJECT_PALETTE)]
            total_prims += len(sq_fit.primitives)
            summary_lines.append(f"Object {obj_id}: {len(sq_fit.primitives)} primitives")

            for prim in sq_fit.primitives:
                ptype = getattr(prim, 'shape_type', 'Other')
                col   = PTYPE_COLORS.get(ptype, [0.5, 0.5, 0.5])
                summary_lines.append(
                    f"  {ptype:10s} scale=({prim.sx:.3f},{prim.sy:.3f},{prim.sz:.3f})"
                    f" e=({prim.e1:.2f},{prim.e2:.2f}) conf={prim.shape_conf:.2f}"
                )
                try:
                    surf = sample_sq_surface(
                        prim.sx, prim.sy, prim.sz,
                        prim.e1, prim.e2,
                        prim.tx, prim.ty, prim.tz,
                        prim.rx, prim.ry, prim.rz,
                        n_points=600,
                    )
                    sq_pts.append(surf)
                    sq_cols.append(np.tile(col, (len(surf), 1)))
                except Exception:
                    pass

            gset = selector.grasp_candidates(sq_fit, obj_id)
            best = gset.best
            if best is not None:
                grasp_origins.append(best.position)
                grasp_dirs.append(best.approach)
                grasp_colors.append(obj_col)
                summary_lines.append(
                    f"  → best grasp: {best.primitive_type} "
                    f"width={best.gripper_width*100:.1f}cm score={best.score:.2f}"
                )

        if sq_pts:
            _write_ply(os.path.join(frame_dir, 'sq_primitives.ply'),
                       np.vstack(sq_pts), np.vstack(sq_cols))
        if grasp_origins:
            _write_axes_ply(os.path.join(frame_dir, 'grasp_axes.ply'),
                            grasp_origins, grasp_dirs,
                            length=0.08, colors=grasp_colors)

        print(f"  [{i:03d}] {len(result.objects)} objects, {total_prims} primitives"
              f" → {frame_dir}")
        frame_results.append({
            'dir': frame_dir,
            'n_objects': len(result.objects),
            'n_primitives': total_prims,
            'summary': '\n'.join(summary_lines),
        })

    build_html_report(out_dir, frame_results)
    print(f"\nDone. Copy results to Mac with:")
    print(f"  scp -r ubuntu@172.23.209.100:{out_dir} ~/Downloads/sq_vis")


if __name__ == '__main__':
    main()
