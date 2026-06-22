#!/usr/bin/env python3
"""
visualize_sq_fits_octscenes.py
==============================
Option A: Visual check of SuperDec SQ fitting on OCTScenes real tabletop data.
Saves colored PLY files + HTML report.

Usage:
    cd ~/project-3dv
    PYTHONPATH=src:/home/ubuntu/superdec python3 \
        src/project_3dv/perception/evaluation/visualize_sq_fits_octscenes.py \
        --data_dir ~/data/OCTScenes/128x128 \
        --out_dir ~/sq_vis_oct \
        --n_scenes 10
"""

import os, sys, argparse, random
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.insert(0, os.path.join(ROOT, 'src'))
sys.path.insert(0, '/home/ubuntu/superdec')

from project_3dv.perception.datasets.octscenes import (
    load_scene, depth_to_pointcloud, get_scene_ids
)
from project_3dv.perception.pipeline import TabletopPerception
from project_3dv.perception.superdec_fitter import SuperdecFitter

PTYPE_COLORS = {
    'Cylinder':  [0.2, 0.6, 1.0],
    'Cuboid':    [1.0, 0.4, 0.2],
    'Ellipsoid': [0.2, 0.9, 0.4],
    'Other':     [0.9, 0.8, 0.1],
}
OBJECT_PALETTE = [
    [1.0, 0.2, 0.2], [0.2, 0.7, 1.0], [0.2, 0.9, 0.3],
    [1.0, 0.6, 0.1], [0.8, 0.2, 0.9], [0.1, 0.9, 0.9],
    [1.0, 0.9, 0.2], [0.5, 0.3, 1.0],
]


def sample_sq_surface(sx, sy, sz, e1, e2, tx, ty, tz, rx, ry, rz, n_points=600):
    eta   = np.linspace(-np.pi/2, np.pi/2, int(np.sqrt(n_points)))
    omega = np.linspace(-np.pi,   np.pi,   int(np.sqrt(n_points)))
    ETA, OMEGA = np.meshgrid(eta, omega)
    ETA, OMEGA = ETA.ravel(), OMEGA.ravel()
    def fexp(b, e): return np.sign(b) * (np.abs(b) ** e)
    x = sx * fexp(np.cos(ETA), e1) * fexp(np.cos(OMEGA), e2)
    y = sy * fexp(np.cos(ETA), e1) * fexp(np.sin(OMEGA), e2)
    z = sz * fexp(np.sin(ETA), e1)
    pts = np.stack([x, y, z], axis=1)
    cx_ , sx_ = np.cos(rx), np.sin(rx)
    cy_ , sy_ = np.cos(ry), np.sin(ry)
    cz_ , sz_ = np.cos(rz), np.sin(rz)
    Rx = np.array([[1,0,0],[0,cx_,-sx_],[0,sx_,cx_]])
    Ry = np.array([[cy_,0,sy_],[0,1,0],[-sy_,0,cy_]])
    Rz = np.array([[cz_,-sz_,0],[sz_,cz_,0],[0,0,1]])
    pts = (Rz @ Ry @ Rx @ pts.T).T + np.array([tx, ty, tz])
    return pts.astype(np.float32)


def write_ply(path, points, colors):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        f.write(f"ply\nformat ascii 1.0\nelement vertex {len(points)}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n")
        for i in range(len(points)):
            r, g, b = [int(np.clip(colors[i, c] * 255, 0, 255)) for c in range(3)]
            f.write(f"{points[i,0]:.4f} {points[i,1]:.4f} {points[i,2]:.4f} {r} {g} {b}\n")


def build_html(out_dir, frame_results):
    rows = []
    for fr in frame_results:
        if fr is None:
            continue
        name = os.path.basename(fr['dir'])
        rows.append(f"""<tr>
          <td><b>{name}</b></td><td>{fr['n_objects']}</td><td>{fr['n_primitives']}</td>
          <td><details><summary>show</summary><pre>{fr['summary']}</pre></details></td>
          <td><a href="{name}/scene.ply">scene</a> | <a href="{name}/segments.ply">segs</a>
              | <a href="{name}/sq_primitives.ply">SQs</a></td></tr>""")
    html = f"""<!DOCTYPE html><html><head><title>OCTScenes SQ Vis</title>
<style>body{{font-family:monospace;padding:20px;background:#1a1a1a;color:#eee}}
table{{border-collapse:collapse;width:100%}}th,td{{border:1px solid #444;padding:8px}}
th{{background:#333}}tr:nth-child(even){{background:#222}}a{{color:#6af}}
pre{{font-size:11px;max-height:200px;overflow-y:auto}}</style></head><body>
<h1>SuperDec on OCTScenes (128x128)</h1>
<table><tr><th>Scene/Frame</th><th>#Objects</th><th>#Primitives</th>
<th>Summary</th><th>PLY</th></tr>
{''.join(rows)}</table></body></html>"""
    with open(os.path.join(out_dir, 'report.html'), 'w') as f:
        f.write(html)
    print(f"Report: {os.path.join(out_dir, 'report.html')}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_dir',     default='/home/ubuntu/data/OCTScenes/128x128')
    ap.add_argument('--out_dir',      default='/home/ubuntu/sq_vis_oct')
    ap.add_argument('--superdec_dir', default='/home/ubuntu/superdec')
    ap.add_argument('--n_scenes',     type=int, default=10)
    ap.add_argument('--frame',        type=int, default=0)
    ap.add_argument('--seed',         type=int, default=42)
    a = ap.parse_args()

    random.seed(a.seed); np.random.seed(a.seed)
    os.makedirs(a.out_dir, exist_ok=True)

    import torch
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    perception = TabletopPerception()
    fitter = SuperdecFitter(superdec_dir=a.superdec_dir, checkpoint='normalized',
                            exist_threshold=0.3, device=device)

    scene_ids = get_scene_ids(os.path.join(a.data_dir, 'segments'))
    if len(scene_ids) > a.n_scenes:
        scene_ids = random.sample(scene_ids, a.n_scenes)
    scene_ids = sorted(scene_ids)
    print(f"Running on {len(scene_ids)} scenes, frame={a.frame}, device={device}")

    frame_results = []
    for i, sid in enumerate(scene_ids):
        scene = load_scene(a.data_dir, sid, a.frame)
        pts = depth_to_pointcloud(scene['depth'], scene['K'])

        if len(pts) < 100:
            print(f"  [{i:03d}] scene {sid}: too few points ({len(pts)})")
            frame_results.append(None)
            continue

        result = perception.run(pts)
        if not result.objects:
            print(f"  [{i:03d}] scene {sid}: no objects detected")
            frame_results.append(None)
            continue

        sq_fits = fitter.fit_batch([obj.points for obj in result.objects])

        frame_dir = os.path.join(a.out_dir, f'scene_{sid:04d}_f{a.frame:02d}')
        os.makedirs(frame_dir, exist_ok=True)

        write_ply(os.path.join(frame_dir, 'scene.ply'),
                  pts, np.full((len(pts), 3), 0.5))

        seg_pts, seg_cols = [], []
        for oid, obj in enumerate(result.objects):
            col = OBJECT_PALETTE[oid % len(OBJECT_PALETTE)]
            seg_pts.append(obj.points)
            seg_cols.append(np.tile(col, (len(obj.points), 1)))
        write_ply(os.path.join(frame_dir, 'segments.ply'),
                  np.vstack(seg_pts), np.vstack(seg_cols))

        sq_pts, sq_cols = [], []
        total_prims = 0
        summary = []
        for oid, (obj, sq) in enumerate(zip(result.objects, sq_fits)):
            total_prims += len(sq.primitives)
            summary.append(f"Object {oid}: {len(sq.primitives)} primitives")
            for prim in sq.primitives:
                ptype = getattr(prim, 'shape_type', 'Other')
                col = PTYPE_COLORS.get(ptype, [0.5, 0.5, 0.5])
                summary.append(
                    f"  {ptype:10s} scale=({prim.sx:.3f},{prim.sy:.3f},{prim.sz:.3f})"
                    f" e=({prim.e1:.2f},{prim.e2:.2f}) conf={prim.shape_conf:.2f}"
                    f" l2={prim.chamfer_l2*1e3:.2f}e-3"
                )
                try:
                    surf = sample_sq_surface(prim.sx, prim.sy, prim.sz,
                                             prim.e1, prim.e2,
                                             prim.tx, prim.ty, prim.tz,
                                             prim.rx, prim.ry, prim.rz)
                    sq_pts.append(surf)
                    sq_cols.append(np.tile(col, (len(surf), 1)))
                except Exception:
                    pass

        if sq_pts:
            write_ply(os.path.join(frame_dir, 'sq_primitives.ply'),
                      np.vstack(sq_pts), np.vstack(sq_cols))

        print(f"  [{i:03d}] scene {sid}: {len(result.objects)} objs, "
              f"{total_prims} prims → {frame_dir}")
        frame_results.append({
            'dir': frame_dir,
            'n_objects': len(result.objects),
            'n_primitives': total_prims,
            'summary': '\n'.join(summary),
        })

    build_html(a.out_dir, frame_results)
    print(f"\nDone. scp -r ubuntu@172.23.209.100:{a.out_dir} ~/Downloads/sq_vis_oct")


if __name__ == '__main__':
    main()
