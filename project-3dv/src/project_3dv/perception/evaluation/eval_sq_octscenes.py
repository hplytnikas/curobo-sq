#!/usr/bin/env python3
import os, sys, argparse
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.insert(0, os.path.join(ROOT, 'src'))
sys.path.insert(0, '/home/ubuntu/superdec')

from project_3dv.perception.datasets.octscenes import load_scene, get_scene_ids
from project_3dv.perception.superdec_fitter import SuperdecFitter

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_dir',        default='/home/ubuntu/data/OCTScenes/128x128')
    ap.add_argument('--superdec_dir',    default='/home/ubuntu/superdec')
    ap.add_argument('--frame',           type=int,   default=0)
    ap.add_argument('--max_scenes',      type=int,   default=200)
    ap.add_argument('--exist_threshold', type=float, default=0.5)
    ap.add_argument('--max_extent',      type=float, default=0.35)
    a = ap.parse_args()

    import torch
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    fitter = SuperdecFitter(superdec_dir=a.superdec_dir, checkpoint='normalized',
                            exist_threshold=a.exist_threshold, device=device)

    scene_ids = get_scene_ids(os.path.join(a.data_dir, 'segments'))[:a.max_scenes]
    print(f"Evaluating {len(scene_ids)} scenes, device={device}, "
          f"exist_threshold={a.exist_threshold}, max_extent={a.max_extent}m\n")

    all_l2 = []
    per_type = {'Cylinder': [], 'Cuboid': [], 'Ellipsoid': [], 'Other': []}
    n_scenes = 0
    n_skipped = 0
    n_objects = 0

    for i, sid in enumerate(scene_ids):
        scene = load_scene(a.data_dir, sid, a.frame)
        if 'segment' not in scene:
            continue
        depth, K, seg = scene['depth'], scene['K'], scene['segment']
        h, w = depth.shape
        u, v = np.meshgrid(np.arange(w), np.arange(h))
        x = (u - K[0,2]) * depth / K[0,0]
        y = (v - K[1,2]) * depth / K[1,1]
        pts_full = np.stack([x, y, depth], axis=-1).reshape(-1, 3)
        seg_flat = seg.reshape(-1)

        obj_pts_list = []
        for lbl in np.unique(seg_flat):
            if lbl == 0: continue
            mask = (seg_flat == lbl) & (pts_full[:, 2] > 0)
            obj = pts_full[mask].astype(np.float32)
            if len(obj) < 10:
                n_skipped += 1; continue
            if (obj.max(0) - obj.min(0)).max() > a.max_extent:
                n_skipped += 1; continue
            obj_pts_list.append(obj)

        if not obj_pts_list:
            continue
        n_scenes += 1
        for sq in fitter.fit_batch(obj_pts_list):
            n_objects += 1
            for prim in sq.primitives:
                all_l2.append(prim.chamfer_l2)
                ptype = getattr(prim, 'shape_type', 'Other')
                if ptype in per_type:
                    per_type[ptype].append(prim.chamfer_l2)

        if (i + 1) % 20 == 0:
            m = np.mean(all_l2)*1e3 if all_l2 else float('nan')
            print(f"  [{i+1:3d}/{len(scene_ids)}] objects: {n_objects} | mean L2: {m:.3f}e-3 m²")

    print("\n" + "="*55)
    print("RESULTS")
    print("="*55)
    print(f"Scenes evaluated : {n_scenes}")
    print(f"Objects fitted   : {n_objects}")
    print(f"Segments skipped : {n_skipped}")
    if all_l2:
        a2 = np.array(all_l2)
        print(f"\nChamfer L2 (x1e-3 m^2):")
        print(f"  mean  : {a2.mean()*1e3:.4f}")
        print(f"  median: {np.median(a2)*1e3:.4f}")
        print(f"  std   : {a2.std()*1e3:.4f}")
        print(f"  p90   : {np.percentile(a2,90)*1e3:.4f}")
        print(f"\nPer shape type:")
        for ptype, vals in per_type.items():
            if vals:
                v = np.array(vals)
                print(f"  {ptype:10s}: n={len(v):4d} mean={v.mean()*1e3:.4f} median={np.median(v)*1e3:.4f}")
    else:
        print("No primitives fitted.")

if __name__ == '__main__':
    main()
