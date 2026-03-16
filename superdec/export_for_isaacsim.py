import os
import torch
import numpy as np
from omegaconf import OmegaConf
import open3d as o3d

from superdec.superdec import SuperDec
from superdec.utils.predictions_handler import PredictionHandler
from superdec.data.dataloader import denormalize_outdict, denormalize_points, normalize_points
from superdec.data.transform import rotate_around_axis

# --- Config ---
checkpoints_folder = "checkpoints/normalized"
path_to_point_cloud = "examples/chair.ply"
output_dir = "/tmp/superdec_meshes"
resolution = 30
os.makedirs(output_dir, exist_ok=True)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# --- Load model ---
checkpoint = torch.load(os.path.join(checkpoints_folder, "ckpt.pt"), map_location=device, weights_only=False)
configs = OmegaConf.load(os.path.join(checkpoints_folder, "config.yaml"))
model = SuperDec(configs.superdec).to(device)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# --- Load point cloud ---
pc = o3d.io.read_point_cloud(path_to_point_cloud)
points_tmp = np.asarray(pc.points)
n_points = points_tmp.shape[0]
replace = n_points < 4096
idxs = np.random.choice(n_points, 4096, replace=replace)
points = points_tmp[idxs]
points, translation, scale = normalize_points(points)
points_tensor = torch.from_numpy(points).unsqueeze(0).to(device).float()

# --- Run inference ---
with torch.no_grad():
    outdict = model(points_tensor)
    for key in outdict:
        if isinstance(outdict[key], torch.Tensor):
            outdict[key] = outdict[key].cpu()
    outdict = denormalize_outdict(outdict, np.array([translation]), np.array([scale]), False)
    points_tensor = denormalize_points(points_tensor.cpu(), np.array([translation]), np.array([scale]), False)

# --- Export meshes ---
pred_handler = PredictionHandler.from_outdict(outdict, points_tensor, ['chair'])
combined_mesh = pred_handler.get_meshes(resolution=resolution)[0]

# Export as single combined OBJ (easiest for Isaac Sim)
combined_path = os.path.join(output_dir, "chair_superquadrics.obj")
combined_mesh.export(combined_path)
print(f"Saved combined mesh: {combined_path}")

# Export individual superquadric meshes (for cuRobo per-primitive collision)
individual_meshes = combined_mesh.split()
for i, mesh in enumerate(individual_meshes):
    path = os.path.join(output_dir, f"sq_{i}.obj")
    mesh.export(path)
print(f"Saved {len(individual_meshes)} individual superquadric meshes to {output_dir}/")
print("Done! Now load chair_superquadrics.obj into Isaac Sim.")
