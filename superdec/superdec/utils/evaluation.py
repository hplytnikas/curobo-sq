import torch.nn as nn
import trimesh
import numpy as np
from scipy.spatial import KDTree
from superdec.data.dataloader import ShapeNet, ScenesDataset
from torch.utils.data import DataLoader

def build_dataloader(cfg):
    if cfg.dataloader.dataset == 'shapenet':
        ds = ShapeNet(split=cfg.shapenet.split, cfg=cfg)
    elif cfg.dataloader.dataset == 'scenes_dataset':
        ds = ScenesDataset(cfg=cfg)
    else:
        raise ValueError(f"Unsupported dataset {cfg.dataloader.dataset}")

    dl = DataLoader(
        ds, batch_size=cfg.dataloader.batch_size, shuffle=False,
        num_workers=cfg.dataloader.num_workers, pin_memory=True
    )
    return dl

def get_threshold_percentage(dist, thresholds):
    ''' Evaluates a point cloud.

    Args:
        dist (numpy array): calculated distance
        thresholds (numpy array): threshold values for the F-score calculation
    '''
    in_threshold = [
        (dist <= t).mean() for t in thresholds
    ]
    return in_threshold

def distance_p2p(points_src, normals_src, points_tgt, normals_tgt): # from convolutional occupancy network
    ''' Computes minimal distances of each point in points_src to points_tgt.

    Args:
        points_src (numpy array): source points
        normals_src (numpy array): source normals
        points_tgt (numpy array): target points
        normals_tgt (numpy array): target normals
    '''
    kdtree = KDTree(points_tgt)
    dist, idx = kdtree.query(points_src)

    if normals_src is not None and normals_tgt is not None:
        normals_src = \
            normals_src / np.linalg.norm(normals_src, axis=-1, keepdims=True)
        normals_tgt = \
            normals_tgt / np.linalg.norm(normals_tgt, axis=-1, keepdims=True)

        normals_dot_product = (normals_tgt[idx] * normals_src).sum(axis=-1)
        # Handle normals that point into wrong direction gracefully
        # (mostly due to mehtod not caring about this in generation)
        normals_dot_product = np.abs(normals_dot_product)
    else:
        normals_dot_product = np.array(
            [np.nan] * points_src.shape[0], dtype=np.float32)
    return dist, normals_dot_product

def get_outdict(pc_gt, normals_gt, pc_pred, normals_pred):
    thresholds = np.linspace(1./1000, 1, 1000)
    completeness, completeness_normals = distance_p2p(pc_gt, normals_gt, pc_pred, normals_pred)
    recall = get_threshold_percentage(completeness, thresholds)
    completeness2 = completeness**2
    
    completeness = completeness.mean()
    completeness2 = completeness2.mean()
    completeness_normals = completeness_normals.mean()
    
    # Accuracy: how far are th points of the predicted pointcloud
    # from the target pointcloud
    accuracy, accuracy_normals = distance_p2p(pc_pred, normals_pred, pc_gt, normals_gt)
    precision = get_threshold_percentage(accuracy, thresholds)
    accuracy2 = accuracy**2
    
    accuracy = accuracy.mean()
    accuracy2 = accuracy2.mean()
    accuracy_normals = accuracy_normals.mean()
    
    # Chamfer distance
    chamferL2 = 0.5 * (completeness2 + accuracy2)
    normals_correctness = (
        0.5 * completeness_normals + 0.5 * accuracy_normals
    )
    chamferL1 = 0.5 * (completeness + accuracy)
    F = [
        2 * precision[i] * recall[i] / (precision[i] + precision[i] + 0.000001)
        for i in range(len(precision))
    ]

    
    out_dict_cur = {
            'completeness': completeness,
            'accuracy': accuracy,
            'normals completeness': completeness_normals,
            'normals accuracy': accuracy_normals,
            'normals': normals_correctness,
            'completeness2': completeness2,
            'accuracy2': accuracy2,
            'chamfer-L2': chamferL2,
            'chamfer-L1': chamferL1,
            'f-score': F[9], # threshold = 1.0%
            'f-score-15': F[14], # threshold = 1.5%
            'f-score-20': F[19], # threshold = 2.0%
        }
    return out_dict_cur