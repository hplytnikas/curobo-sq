# superdec_curobo_bridge.py
import os
import torch
import numpy as np
from omegaconf import OmegaConf
import open3d as o3d

from superdec.superdec import SuperDec
from superdec.utils.predictions_handler import PredictionHandler
from superdec.data.dataloader import denormalize_outdict, denormalize_points, normalize_points
from curobo.geom.types import WorldConfig, Mesh


def superdec_to_curobo_world(
    ply_path: str,
    checkpoint_folder: str = "checkpoints/normalized",
    output_dir: str = "/tmp/superdec_meshes",
    resolution: int = 30,
) -> WorldConfig:

    os.makedirs(output_dir, exist_ok=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Load model
    checkpoint = torch.load(os.path.join(checkpoint_folder, "ckpt.pt"), map_location=device, weights_only=False)
    configs = OmegaConf.load(os.path.join(checkpoint_folder, "config.yaml"))
    model = SuperDec(configs.superdec).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # Load + preprocess point cloud
    pc = o3d.io.read_point_cloud(ply_path)
    points_tmp = np.asarray(pc.points)
    idxs = np.random.choice(len(points_tmp), 4096, replace=len(points_tmp) < 4096)
    points = points_tmp[idxs]
    points, translation, scale = normalize_points(points)
    points_tensor = torch.from_numpy(points).unsqueeze(0).to(device).float()

    # Run inference
    with torch.no_grad():
        outdict = model(points_tensor)
        for key in outdict:
            if isinstance(outdict[key], torch.Tensor):
                outdict[key] = outdict[key].cpu()
        outdict = denormalize_outdict(outdict, np.array([translation]), np.array([scale]), False)
        points_tensor = denormalize_points(points_tensor.cpu(), np.array([translation]), np.array([scale]), False)

    # Get meshes
    pred_handler = PredictionHandler.from_outdict(outdict, points_tensor, ['object'])
    # print(f"DEBUG: Network raw predictions shape: {outdict['pred_prob'].shape if 'pred_prob' in outdict else 'Unknown'}")
    combined_mesh = pred_handler.get_meshes(resolution=resolution)[0]
    # print(f"DEBUG: Combined mesh has {len(combined_mesh.vertices)} vertices and {len(combined_mesh.faces)} faces.")
    individual_meshes = combined_mesh.split()
    # print(f"DEBUG: Split combined mesh into {len(individual_meshes)} individual parts.")
    if len(individual_meshes) == 0:
        print("Trimesh split failed, using the single combined mesh instead.")
        individual_meshes = [combined_mesh]

    # Build cuRobo WorldConfig
    mesh_objects = []
    for i, mesh in enumerate(individual_meshes):
        path = os.path.join(output_dir, f"sq_{i}.obj")
        mesh.export(path)
        mesh_objects.append(
            Mesh(
                name=f"sq_{i}",
                file_path=path,
                pose=[0, 0, 0, 1, 0, 0, 0],  # identity pose — adjust if needed
            )
        )

    print(f"Created WorldConfig with {len(mesh_objects)} superquadric meshes")
    return WorldConfig(mesh=mesh_objects)
