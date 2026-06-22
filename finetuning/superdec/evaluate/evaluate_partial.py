import os
import torch
import numpy as np
import hydra
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader
from superdec.superdec import SuperDec
from superdec.utils.predictions_handler_extended import PredictionHandler
from superdec.utils.evaluation import get_outdict, build_dataloader
from superdec.data.dataloader import denormalize_outdict, denormalize_points
from typing import Dict
from tqdm import tqdm


class PartialEvaluator:
    """
    Partial-view fit quality: the model is fed the OCCLUDED point cloud
    (batch['points'], produced when trainer.force_occlusions=true), but the
    Chamfer distance is measured against the FULL, UNOCCLUDED cloud
    (batch['unoccluded_points']). This rewards completing unseen geometry and is
    the fair test for the partial-view-robustness fine-tunes, unlike the stock
    Evaluator which scores against the (partial) input and would penalize
    completion.
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
        count = 0
        out_dict = None
        with torch.no_grad():
            for j, batch in tqdm(enumerate(self.dataloader)):
                points = batch['points'].to(self.device).float()        # occluded input
                gt_full = batch['unoccluded_points'].to(self.device).float()  # full cloud (GT)
                gt_full_normals = batch['unoccluded_normals'].to(self.device).float()
                batch['translation'] = batch['translation'].to(self.device)
                batch['scale'] = batch['scale'].to(self.device)
                names = batch.get('model_id', np.arange(points.shape[0]))

                outdict = self.model(points)

                outdict = denormalize_outdict(outdict, batch['translation'], batch['scale'])
                gt_full = denormalize_points(gt_full, batch['translation'], batch['scale'])

                pred_handler = PredictionHandler.from_outdict(outdict, gt_full, names)
                pred_meshes = pred_handler.get_meshes(resolution=self.mesh_resolution, colors=False)

                exist = outdict['exist'].cpu().numpy()  # (B, P)

                for i, mesh in enumerate(pred_meshes):
                    if mesh is None:
                        continue
                    pc_pred, idx = mesh.sample(gt_full.shape[1], return_index=True)
                    normals_pred = mesh.face_normals[idx]
                    gt_pc = gt_full[i].cpu().numpy()
                    gt_normal = gt_full_normals[i].cpu().numpy()
                    out_dict_cur = get_outdict(pc_pred, normals_pred, gt_pc, gt_normal)
                    out_dict_cur['num_primitives'] = (exist[i] > 0.5).sum()
                    if out_dict is None:
                        out_dict = out_dict_cur
                    else:
                        for k in out_dict.keys():
                            out_dict[k] += out_dict_cur[k]
                    count += 1

        for k in out_dict.keys():
            out_dict[k] = out_dict[k] / count
        return {
            'mean_chamfer_l1': out_dict['chamfer-L1'],
            'mean_chamfer_l2': out_dict['chamfer-L2'],
            'f-score-1pct': out_dict['f-score'],
            'avg_num_primitives': out_dict['num_primitives'],
        }


def main(cfg: DictConfig) -> None:
    print("\n========== SuperDec Partial-View Evaluation ==========")
    print("Config:\n" + OmegaConf.to_yaml(cfg))
    # Fix seeds so every model sees the same random occlusions (fair comparison).
    seed = int(cfg.get('seed', 42))
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = cfg.get('device', 'cuda')
    mesh_resolution = cfg.evaluation.resolution
    dataloader = build_dataloader(cfg)
    evaluator = PartialEvaluator(
        device=device,
        cfg=cfg,
        dataloader=dataloader,
        mesh_resolution=mesh_resolution,
    )
    print(f"\nEvaluating (partial input -> full GT) with mesh resolution: {mesh_resolution}\n")
    results = evaluator.evaluate()
    print("\n----- Partial-View Evaluation Results -----")
    for k, v in results.items():
        print(f"{k:>25}: {v:.6f}")
    print("\nEvaluation complete.\n")


if __name__ == "__main__":
    @hydra.main(version_base=None, config_path="../../configs", config_name="eval")
    def run_main(cfg: DictConfig):
        main(cfg)
    run_main()
