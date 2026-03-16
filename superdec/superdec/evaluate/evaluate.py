import os
import torch
import numpy as np
import hydra
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader
from superdec.superdec import SuperDec
from superdec.utils.predictions_handler import PredictionHandler
from superdec.utils.evaluation import get_outdict, build_dataloader
from superdec.loss.loss import Loss
from superdec.data.dataloader import ShapeNet, denormalize_outdict, denormalize_points
from typing import Dict, Any
from tqdm import tqdm

class Evaluator:
    """
    Evaluates the reconstruction quality of a model by computing Chamfer distances
    and the average number of primitives on a given dataset.
    """
    def __init__(self, device: str, cfg: DictConfig, dataloader: DataLoader, mesh_resolution: int = 100):
        self.device = device
        self.cfg = cfg
        self.dataloader = dataloader
        self.mesh_resolution = mesh_resolution
        
        ckp_path = os.path.join(cfg.checkpoints_folder, cfg.checkpoint_file)
        config_path = os.path.join(cfg.checkpoints_folder, cfg.config_file)
        
        if not os.path.isfile(ckp_path):
            raise FileNotFoundError(f"Checkpoint not found: {ckp_path}")
        checkpoint = torch.load(ckp_path, map_location=device, weights_only=False)
        with open(config_path) as f:
            configs = OmegaConf.load(f)

        self.model = SuperDec(configs.superdec).to(device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()

    def evaluate(self) -> Dict[str, float]:
        """
        Runs evaluation and returns metrics.
        Returns:
            dict: {'mean_chamfer_l1', 'mean_chamfer_l2', 'avg_num_primitives'}
        """
        count = 0
        with torch.no_grad():
            for j, batch in tqdm(enumerate(self.dataloader)):
                points = batch['points'].to(self.device).float()  
                if 'normals' in batch.keys(): 
                    normals = batch['normals'].to(self.device).float()
                else:
                    normals = None
                batch['translation'] = batch['translation'].to(self.device)
                batch['scale'] = batch['scale'].to(self.device)
                names = batch.get('model_id', np.arange(points.shape[0]))

                outdict = self.model(points)

                outdict = denormalize_outdict(outdict, batch['translation'], batch['scale'])
                points = denormalize_points(points, batch['translation'], batch['scale'])

                pred_handler = PredictionHandler.from_outdict(outdict, points, names)
                pred_meshes = pred_handler.get_meshes(resolution=self.mesh_resolution, colors=False)
                
                gt_pcs = points  # (B, N, 3)
                exist = outdict['exist'].cpu().numpy()  # (B, P)
                
                for i, mesh in enumerate(pred_meshes):
                    if mesh is None: 
                        continue
                    pc_pred, idx = mesh.sample(gt_pcs.shape[1], return_index = True)
                    normals_pred = mesh.face_normals[idx] # get predicted normals
                    gt_pc = gt_pcs[i].cpu().numpy()
                    if normals is not None:
                        gt_normal = normals[i].cpu().numpy()
                    else:
                        gt_normal = None
                    out_dict_cur = get_outdict(pc_pred, normals_pred, gt_pc, gt_normal)
                    out_dict_cur['num_primitives'] = (exist[i] > 0.5).sum()
                    if i == 0 and j == 0:
                        out_dict = out_dict_cur
                    else:
                        for k in out_dict.keys():
                            out_dict[k] += out_dict_cur[k]
                    count += 1
                
        for k in out_dict.keys():
            out_dict[k] = out_dict[k] / count
        avg_chamfer_l1 = out_dict['chamfer-L1']
        avg_chamfer_l2 = out_dict['chamfer-L2']
        avg_num_primitives = out_dict['num_primitives']
        return {
            'mean_chamfer_l1': avg_chamfer_l1,
            'mean_chamfer_l2': avg_chamfer_l2,
            'avg_num_primitives': avg_num_primitives
        }


def main(cfg: DictConfig) -> None:
    """
    Main evaluation entrypoint. Loads config, runs evaluation, prints results.
    """
    print("\n========== SuperDec Evaluation ==========")
    print("Config:\n" + OmegaConf.to_yaml(cfg))
    device = cfg.get('device', 'cuda')
    mesh_resolution = cfg.evaluation.resolution
    # Dataloader
    dataloader = build_dataloader(cfg)
    # dataset = ShapeNet(split=cfg.dataloader.split, cfg=cfg)
    # dataloader = DataLoader(dataset, batch_size=cfg.dataloader.batch_size, shuffle=False, num_workers=cfg.dataloader.num_workers)
    # # Evaluator
    evaluator = Evaluator(
        device=device,
        cfg=cfg,
        dataloader=dataloader,
        mesh_resolution=mesh_resolution
    )
    # Evaluate
    print(f"\nEvaluating with mesh resolution: {mesh_resolution}\n")
    results = evaluator.evaluate()
    print("\n----- Evaluation Results -----")
    for k, v in results.items():
        print(f"{k:>25}: {v:.6f}")
    print("\nEvaluation complete.\n")

if __name__ == "__main__":
    @hydra.main(version_base=None, config_path="../../configs", config_name="eval")
    def run_main(cfg: DictConfig):
        main(cfg)
    run_main()