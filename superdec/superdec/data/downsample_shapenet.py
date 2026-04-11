import os

import numpy as np
import open3d as o3d
from tqdm import tqdm

data_root = 'data/ShapeNet'
categories = [d for d in os.listdir(data_root)if os.path.isdir(os.path.join(data_root, d))]

for c in tqdm(categories):
    category_path = os.path.join(data_root, c)
    models = [m for m in os.listdir(category_path)if os.path.isdir(os.path.join(category_path, m))]
    for m in tqdm(models):
        model_path = os.path.join(data_root, c, m)
        if os.path.exists(os.path.join(model_path, "pointcloud_4096.npz")):
            print(f"Skipping already downsampled model: {model_path}")
            continue
        pc_data = np.load(os.path.join(model_path, "pointcloud.npz"))
        points = pc_data["points"]
        normals = pc_data["normals"]
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.normals = o3d.utility.Vector3dVector(normals)
        downsampled_pcd = pcd.farthest_point_down_sample(4096)
        points = np.asarray(downsampled_pcd.points)
        normals = np.asarray(downsampled_pcd.normals)
        np.savez(os.path.join(model_path, "pointcloud_4096.npz"), points=points, normals=normals)
