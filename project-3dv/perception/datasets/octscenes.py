import numpy as np
from pathlib import Path
from PIL import Image


def load_intrinsics(path: str) -> np.ndarray:
    return np.loadtxt(path)


def load_depth(path: str, scale: float = 1000.0) -> np.ndarray:
    img = Image.open(path)
    return np.array(img, dtype=np.float32) / scale


def load_segment(path: str) -> np.ndarray:
    img = Image.open(path)
    return np.array(img, dtype=np.uint8)


def get_scene_ids(segments_dir: str) -> list:
    ids = set()
    for range_dir in Path(segments_dir).iterdir():
        if range_dir.is_dir():
            for f in range_dir.glob("*.png"):
                ids.add(int(f.stem.split("_")[0]))
    return sorted(ids)


def load_scene(data_dir: str, scene_id: int, frame: int = 0):
    data_dir = Path(data_dir)
    name = f"{scene_id:04d}_{frame:02d}"
    depth = load_depth(data_dir / "depth" / f"{name}.png")
    K = load_intrinsics(data_dir / "intrinsic" / f"{name}.txt")
    result = {"depth": depth, "K": K, "scene_id": scene_id, "frame": frame}
    for range_dir in (data_dir / "segments").iterdir():
        seg_path = range_dir / f"{name}.png"
        if seg_path.exists():
            result["segment"] = load_segment(seg_path)
            break
    return result


def depth_to_pointcloud(depth: np.ndarray, K: np.ndarray) -> np.ndarray:
    h, w = depth.shape
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    u, v = np.meshgrid(np.arange(w), np.arange(h))
    x = (u - cx) * depth / fx
    y = (v - cy) * depth / fy
    pts = np.stack([x, y, depth], axis=-1).reshape(-1, 3)
    return pts[pts[:, 2] > 0]


def turntable_pose(frame_idx: int, n_frames: int = 60,
                   radius: float = 0.77, height: float = 0.0) -> np.ndarray:
    """Camera-to-world matrix assuming turntable capture at 6deg/frame."""
    angle = 2 * np.pi * frame_idx / n_frames
    pos = np.array([radius * np.sin(angle), height, radius * np.cos(angle)])
    forward = -pos / np.linalg.norm(pos)
    up = np.array([0., 1., 0.])
    right = np.cross(forward, up); right /= np.linalg.norm(right)
    up = np.cross(right, forward)
    T = np.eye(4)
    T[:3, :3] = np.stack([right, up, forward], axis=1)
    T[:3,  3] = pos
    return T


def load_scene_multiview(data_dir: str, scene_id: int,
                         step: int = 6,
                         radius: float = 0.77,
                         crop_radius: float = 0.35,
                         y_min: float = -0.05,
                         y_max: float = 0.35) -> np.ndarray:
    """
    Fuse multiple frames of a scene into one world-space point cloud.

    Parameters
    ----------
    step        : sample every `step` frames (step=6 → 10 frames from 60)
    radius      : turntable camera radius in metres
    crop_radius : keep only points within this XZ radius of origin
    y_min/y_max : height band to keep (removes table surface and ceiling)

    Returns
    -------
    (N, 3) float32 array in world coordinates
    """
    all_pts = []
    for f in range(0, 60, step):
        scene = load_scene(data_dir, scene_id, f)
        depth, K = scene['depth'], scene['K']
        h, w = depth.shape
        u, v = np.meshgrid(np.arange(w), np.arange(h))
        z = depth
        x = (u - K[0, 2]) * z / K[0, 0]
        y = (v - K[1, 2]) * z / K[1, 1]
        pts_cam = np.stack([x, y, z], axis=-1).reshape(-1, 3)
        pts_cam = pts_cam[pts_cam[:, 2] > 0].astype(np.float32)
        T = turntable_pose(f, radius=radius)
        pts_world = (T[:3, :3] @ pts_cam.T).T + T[:3, 3]
        all_pts.append(pts_world)

    pts = np.vstack(all_pts)
    xz = np.sqrt(pts[:, 0] ** 2 + pts[:, 2] ** 2)
    mask = (xz < crop_radius) & (pts[:, 1] > y_min) & (pts[:, 1] < y_max)
    return pts[mask].astype(np.float32)


# ── 640x480 loader with real poses ──────────────────────────────────────────

_K_640 = None

def load_intrinsics_640(data_dir_640: str) -> np.ndarray:
    global _K_640
    if _K_640 is None:
        _K_640 = np.loadtxt(os.path.join(data_dir_640, 'intrinsic_640x480.txt'))
    return _K_640

def load_pose_640(data_dir_640: str, scene_id: int, frame: int) -> np.ndarray:
    """Load 4x4 camera-to-world pose matrix."""
    path = os.path.join(data_dir_640, 'pose', f'{scene_id:04d}_{frame:02d}.txt')
    return np.loadtxt(path)

def load_depth_640(data_dir_640: str, scene_id: int, frame: int,
                   scale: float = 1000.0) -> np.ndarray:
    path = os.path.join(data_dir_640, 'depth', f'{scene_id:04d}_{frame:02d}.png')
    img = Image.open(path)
    return np.array(img, dtype=np.float32) / scale

def load_segment_640(data_dir_640: str, scene_id: int, frame: int = 0) -> np.ndarray:
    """Segments only exist for annotated test scenes."""
    path = os.path.join(data_dir_640, 'segment', f'{scene_id:04d}_{frame:02d}.png')
    if not os.path.exists(path):
        return None
    return np.array(Image.open(path), dtype=np.uint8)

def load_scene_multiview_640(data_dir_640: str, scene_id: int,
                              step: int = 3,
                              crop_radius: float = 0.5,
                              y_min: float = -0.1,
                              y_max: float = 0.5) -> np.ndarray:
    """
    Fuse multiple 640x480 frames using real camera poses.
    Returns world-space point cloud cropped around scene origin.
    """
    import os
    K = load_intrinsics_640(data_dir_640)
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    all_pts = []
    for f in range(0, 60, step):
        depth_path = os.path.join(data_dir_640, 'depth', f'{scene_id:04d}_{f:02d}.png')
        if not os.path.exists(depth_path):
            continue
        depth = load_depth_640(data_dir_640, scene_id, f)
        T = load_pose_640(data_dir_640, scene_id, f)
        h, w = depth.shape
        u, v = np.meshgrid(np.arange(w), np.arange(h))
        z = depth
        x = (u - cx) * z / fx
        y = (v - cy) * z / fy
        pts_cam = np.stack([x, y, z], axis=-1).reshape(-1, 3).astype(np.float32)
        pts_cam = pts_cam[pts_cam[:, 2] > 0]
        pts_world = (T[:3, :3] @ pts_cam.T).T + T[:3, 3]
        all_pts.append(pts_world)

    if not all_pts:
        return np.zeros((0, 3), dtype=np.float32)

    pts = np.vstack(all_pts).astype(np.float32)
    xz = np.sqrt(pts[:, 0] ** 2 + pts[:, 2] ** 2)
    mask = (xz < crop_radius) & (pts[:, 1] > y_min) & (pts[:, 1] < y_max)
    return pts[mask]
