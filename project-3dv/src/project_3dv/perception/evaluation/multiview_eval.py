#!/usr/bin/env python3
"""
multiview_eval.py
=================
Multi-view SQ fitting evaluation on OCTScenes.

Uses real camera poses from 640x480 split with 128x128 depth.
Propagates frame-0 GT instance masks to all 60 frames using per-axis
bbox crop — clean ~9x density improvement over single-view.

Usage:
    cd ~/project-3dv
    PYTHONPATH=src:/home/ubuntu/superdec python3 \
        src/project_3dv/perception/evaluation/multiview_eval.py \
        --data_dir_128 ~/data/OCTScenes/128x128 \
        --pose_dir ~/data/OCTScenes/640x480/pose \
        --max_scenes 200
"""

import os, sys, argparse
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.insert(0, os.path.join(ROOT, 'src'))
sys.path.insert(0, '/home/ubuntu/superdec')

from project_3dv.perception.datasets.octscenes import load_scene, get_scene_ids
from project_3dv.perception.superdec_fitter import SuperdecFitter


def load_pose(pose_dir, scene_id, frame):
    return np.loadtxt(os.path.join(pose_dir, f'{scene_id:04d}_{frame:02d}.txt'))


def unproject_cam(depth, K):
    h, w = depth.shape
    u, v = np.meshgrid(np.arange(w), np.arange(h))
    z = depth
    x = (u - K[0, 2]) * z / K[0, 0]
    y = (v - K[1, 2]) * z / K[1, 1]
    return np.stack([x, y, z], axis=-1).reshape(-1, 3).astype(np.float32)


def fuse_object_multiview(data_dir_128, pose_dir, scene_id,
                           obj0_world, step=3, margin=0.03, max_extent=0.35):
    centroid  = obj0_world.mean(0)
    bbox_half = (obj0_world.max(0) - obj0_world.min(0)) / 2
    lo = centroid - bbox_half - margin
    hi = centroid + bbox_half + margin

    accumulated = [obj0_world]

    for f in range(0, 60, step):
        if f == 0:
            continue
        scene_f = load_scene(data_dir_128, scene_id, f)
        depth_f, K_f = scene_f['depth'], scene_f['K']
        T_f = load_pose(pose_dir, scene_id, f)

        pts_cam = unproject_cam(depth_f, K_f)
        pts_cam = pts_cam[pts_cam[:, 2] > 0]
        pts_world = (T_f[:3, :3] @ pts_cam.T).T + T_f[:3, 3]

        inbox = np.all((pts_world >= lo) & (pts_world <= hi), axis=1)
        nearby = pts_world[inbox]
        if len(nearby) > 5:
            accumulated.append(nearby.astype(np.float32))

    fused = np.vstack(accumulated).astype(np.float32)

    if len(fused) > 0:
        extent = (fused.max(0) - fused.min(0)).max()
        if extent > max_extent * 1.5:
            return obj0_world
    return fused


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_dir_128',    default='/home/ubuntu/data/OCTScenes/128x128')
    ap.add_argument('--pose_dir',        default='/home/ubuntu/data/OCTScenes/640x480/pose')
    ap.add_argument('--superdec_dir',    default='/home/ubuntu/superdec')
    ap.add_argument('--max_scenes',      type=int,   default=200)
    ap.add_argument('--step',            type=int,   default=3)
    ap.add_argument('--exist_threshold', type=float, default=0.5)
    ap.add_argument('--max_extent',      type=float, default=0.35)
    ap.add_argument('--margin',          type=float, default=0.03)
    a = ap.parse_args()

    import torch
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    fitter = SuperdecFitter(superdec_dir=a.superdec_dir, checkpoint='normalized',
                            exist_threshold=a.exist_threshold, device=device)

    scene_ids = get_scene_ids(os.path.join(a.data_dir_128, 'segments'))[:a.max_scenes]
    n_frames  = 60 // a.step
    print(f"Multi-view eval: {len(scene_ids)} scenes, "
          f"{n_frames} frames/scene, device={device}\n")

    all_l2 = []
    per_type = {'Cylinder': [], 'Cuboid': [], 'Ellipsoid': [], 'Other': []}
    n_scenes  = 0
    n_objects = 0
    n_skipped = 0
    pts_sv, pts_mv = [], []

    for i, sid in enumerate(scene_ids):
        scene0 = load_scene(a.data_dir_128, sid, 0)
        if 'segment' not in scene0:
            continue

        depth0, K0, seg = scene0['depth'], scene0['K'], scene0['segment']
        T0 = load_pose(a.pose_dir, sid, 0)

        pts0_cam   = unproject_cam(depth0, K0)
        pts0_world = (T0[:3, :3] @ pts0_cam.T).T + T0[:3, 3]
        seg_flat   = seg.reshape(-1)

        obj_clouds = []
        for lbl in np.unique(seg_flat):
            if lbl == 0:
                continue
            mask = (seg_flat == lbl) & (pts0_cam[:, 2] > 0)
            obj0_cam = pts0_cam[mask]
            if len(obj0_cam) < 10:
                n_skipped += 1
                continue
            extent0 = (obj0_cam.max(0) - obj0_cam.min(0)).max()
            if extent0 > a.max_extent:
                n_skipped += 1
                continue

            obj0_world = pts0_world[mask].astype(np.float32)
            obj_mv = fuse_object_multiview(
                a.data_dir_128, a.pose_dir, sid, obj0_world,
                step=a.step, margin=a.margin, max_extent=a.max_extent,
            )

            if len(obj_mv) < 10:
                n_skipped += 1
                continue

            pts_sv.append(len(obj0_world))
            pts_mv.append(len(obj_mv))
            obj_clouds.append(obj_mv)

        if not obj_clouds:
            continue

        n_scenes += 1
        sq_fits = fitter.fit_batch(obj_clouds)
        for sq in sq_fits:
            n_objects += 1
            for prim in sq.primitives:
                all_l2.append(prim.chamfer_l2)
                ptype = getattr(prim, 'shape_type', 'Other')
                if ptype in per_type:
                    per_type[ptype].append(prim.chamfer_l2)

        if (i + 1) % 20 == 0:
            m = np.mean(all_l2) * 1e3 if all_l2 else float('nan')
            density = np.mean(pts_mv) / np.mean(pts_sv) if pts_sv else 0
            print(f"  [{i+1:3d}/{len(scene_ids)}] "
                  f"objects: {n_objects} | "
                  f"mean L2: {m:.3f}e-3 | "
                  f"density: {density:.1f}x")

    print("\n" + "=" * 55)
    print("MULTI-VIEW RESULTS")
    print("=" * 55)
    print(f"Scenes evaluated : {n_scenes}")
    print(f"Objects fitted   : {n_objects}")
    print(f"Segments skipped : {n_skipped}")
    if pts_sv:
        print(f"Avg pts/obj SV   : {np.mean(pts_sv):.0f}")
        print(f"Avg pts/obj MV   : {np.mean(pts_mv):.0f}  "
              f"({np.mean(pts_mv)/np.mean(pts_sv):.1f}x density)")
    if all_l2:
        a2 = np.array(all_l2)
        print(f"\nChamfer L2 (x1e-3 m^2):")
        print(f"  mean  : {a2.mean()*1e3:.4f}")
        print(f"  median: {np.median(a2)*1e3:.4f}")
        print(f"  std   : {a2.std()*1e3:.4f}")
        print(f"  p90   : {np.percentile(a2, 90)*1e3:.4f}")
        print(f"\nPer shape type:")
        for ptype, vals in per_type.items():
            if vals:
                v = np.array(vals)
                print(f"  {ptype:10s}: n={len(v):4d} "
                      f"mean={v.mean()*1e3:.4f} "
                      f"median={np.median(v)*1e3:.4f}")
    else:
        print("No primitives fitted.")


if __name__ == '__main__':
    main()
