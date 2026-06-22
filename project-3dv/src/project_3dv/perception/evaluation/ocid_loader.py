"""
ocid_loader.py

Handles both OCID layouts:
  ARID20  → rgb/*.png, depth/*.png, label/*.png   (timestamped, with extension)
  ARID10/YCB10 → rgb/0001, depth/0001, label/0001 (numeric, NO extension)

Core fix: _sorted_files() accepts any file, with or without extension.
Frames matched positionally by sort order — guaranteed to correspond.
"""

import sys, numpy as np
from pathlib import Path
from dataclasses import dataclass, field
from typing import Iterator, List, Optional

def _cv2():
    import cv2; return cv2

def _o3d():
    import open3d as o3d; return o3d

sys.path.insert(0, str(Path(__file__).parent))
from pipeline import TabletopPerception, PerceptionResult

ASUS_XTION = dict(
    fx=570.3422241210938, fy=570.3422241210938,
    cx=319.5, cy=239.5, width=640, height=480, depth_scale=1000.0,
)


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

@dataclass
class SegmentationScore:
    n_gt_objects: int
    n_detected:   int
    mean_iou:     float
    matched:      int
    threshold:    float = 0.5

    def __str__(self):
        return (
            f"GT: {self.n_gt_objects}  Detected: {self.n_detected}  "
            f"Matched(IoU>{self.threshold}): {self.matched}/{self.n_gt_objects}  "
            f"Mean IoU: {self.mean_iou:.3f}"
        )


def _iou(a, b):
    return float(np.logical_and(a,b).sum()) / float(np.logical_or(a,b).sum() + 1e-9)


def evaluate_against_labels(label_image, depth_image, result,
                             intrinsics=ASUS_XTION, iou_threshold=0.5):
    if label_image is None:
        return SegmentationScore(0, len(result.objects), 0.0, 0, iou_threshold)
    fx, fy, cx, cy = intrinsics['fx'], intrinsics['fy'], intrinsics['cx'], intrinsics['cy']
    H, W = label_image.shape
    gt_ids = np.unique(label_image)
    gt_ids = gt_ids[gt_ids > 0]
    det_masks = []
    for obj in result.objects:
        pts = obj.points
        Z = pts[:, 2]
        valid = Z > 0.1
        u = np.clip(np.round(fx * pts[valid,0] / Z[valid] + cx).astype(int), 0, W-1)
        v = np.clip(np.round(fy * pts[valid,1] / Z[valid] + cy).astype(int), 0, H-1)
        mask = np.zeros((H, W), dtype=bool)
        mask[v, u] = True
        det_masks.append(mask)
    ious = []
    for gt_id in gt_ids:
        gt_mask = label_image == gt_id
        if gt_mask.sum() < 50:
            continue
        best = max((_iou(gt_mask, m) for m in det_masks), default=0.0)
        ious.append(best)
    if not ious:
        return SegmentationScore(0, len(result.objects), 0.0, 0, iou_threshold)
    return SegmentationScore(
        n_gt_objects=len(ious), n_detected=len(result.objects),
        mean_iou=float(np.mean(ious)),
        matched=sum(1 for x in ious if x >= iou_threshold),
        threshold=iou_threshold,
    )


# ---------------------------------------------------------------------------
# Image loading — PIL fallback for macOS 16-bit PNG and extensionless files
# ---------------------------------------------------------------------------

def _load_img(path: Path, flags=None) -> Optional[np.ndarray]:
    """Load image via cv2, falling back to PIL. Handles extensionless files."""
    cv2 = _cv2()
    args = [str(path)] + ([flags] if flags is not None else [])
    img = cv2.imread(*args)
    if img is not None:
        return img
    # PIL fallback — works for extensionless PNG/16-bit files on macOS
    try:
        from PIL import Image
        return np.array(Image.open(str(path)))
    except Exception as e:
        print(f"  [warn] Could not load {path.name}: {e}")
        return None


def _sorted_files(directory: Optional[Path]) -> List[Path]:
    """
    Return sorted list of FILES in directory.
    Works for both .png files (ARID20) and extensionless files (ARID10/YCB10).
    Excludes subdirectories and hidden files.
    """
    if directory is None or not directory.exists():
        return []
    return sorted(p for p in directory.iterdir()
                  if p.is_file() and not p.name.startswith('.'))


# ---------------------------------------------------------------------------
# Scene
# ---------------------------------------------------------------------------

@dataclass
class OCIDScene:
    rgb_path:   Path
    depth_path: Path
    label_path: Optional[Path]
    pcd_path:   Optional[Path]
    subset:     str
    location:   str
    view:       str
    seq_id:     str
    frame_id:   str
    pipe:       TabletopPerception = field(repr=False, default=None)

    def load_depth(self) -> Optional[np.ndarray]:
        cv2 = _cv2()
        d = _load_img(self.depth_path, cv2.IMREAD_ANYDEPTH)
        if d is None:
            return None
        return d.astype(np.float32) / ASUS_XTION['depth_scale']

    def load_rgb(self) -> Optional[np.ndarray]:
        cv2 = _cv2()
        img = _load_img(self.rgb_path)
        if img is None:
            return None
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB) if img.ndim == 3 else img

    def load_label(self) -> Optional[np.ndarray]:
        if self.label_path is None or not self.label_path.exists():
            return None
        cv2 = _cv2()
        return _load_img(self.label_path, cv2.IMREAD_ANYDEPTH)

    def depth_to_pointcloud(self, depth, intrinsics=ASUS_XTION, max_depth=2.0):
        fx, fy, cx, cy = intrinsics['fx'], intrinsics['fy'], intrinsics['cx'], intrinsics['cy']
        H, W = depth.shape
        v_idx, u_idx = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')
        valid = (depth > 0.1) & (depth < max_depth)
        X = (u_idx[valid] - cx) * depth[valid] / fx
        Y = (v_idx[valid] - cy) * depth[valid] / fy
        return np.stack([X, Y, depth[valid]], axis=1).astype(np.float32)

    def load_pointcloud_from_pcd(self):
        if self.pcd_path is None or not self.pcd_path.exists():
            return None
        pcd = _o3d().io.read_point_cloud(str(self.pcd_path))
        return np.asarray(pcd.points, dtype=np.float32)

    def get_points(self, prefer_pcd=False):
        if prefer_pcd:
            pts = self.load_pointcloud_from_pcd()
            if pts is not None:
                return pts
        depth = self.load_depth()
        return self.depth_to_pointcloud(depth) if depth is not None else None

    def run_perception(self, prefer_pcd=False):
        pts = self.get_points(prefer_pcd=prefer_pcd)
        if pts is None or len(pts) == 0:
            return None
        return self.pipe.run(pts)

    def evaluate(self, result, iou_threshold=0.5):
        depth = self.load_depth()
        if depth is None:
            return SegmentationScore(0, 0, 0.0, 0, iou_threshold)
        return evaluate_against_labels(self.load_label(), depth, result,
                                       iou_threshold=iou_threshold)

    def n_gt_objects(self) -> int:
        label = self.load_label()
        if label is None:
            return 0
        return int((np.unique(label) > 0).sum())


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------

class OCIDLoader:
    """
    Loads OCID scenes robustly across all three subsets.

    ARID20  — best for this project (20 objects, tabletop, richest)
    ARID10  — 10 objects, mixed shapes
    YCB10   — 10 YCB objects, most relevant to robotics benchmarks

    Quick start:
        loader = OCIDLoader("/Volumes/T7/OCID-dataset")
        loader.inspect_dataset()
        for scene in loader.iter_scenes("ARID20", "table", "bottom"):
            result = scene.run_perception()
            print(result.summary())
    """

    def __init__(self, root: str, pipe: TabletopPerception = None):
        self.root = Path(root)
        if not self.root.exists():
            raise FileNotFoundError(f"OCID root not found: {self.root}")
        self.pipe = pipe or TabletopPerception(
            voxel_size=0.005, cluster_eps=0.030,
            cluster_min_points=30, min_object_points=40,
            plane_dist_threshold=0.015,
        )

    # ------------------------------------------------------------------
    # Inspection
    # ------------------------------------------------------------------

    def inspect_dataset(self):
        """Print what's on disk. Run this first on a new machine."""
        print(f"\nInspecting OCID at: {self.root}")
        print("=" * 60)
        top = sorted(p.name for p in self.root.iterdir() if p.is_dir())
        print(f"Top-level folders: {top}\n")
        for subset in top:
            sp = self.root / subset
            for loc in sorted(p.name for p in sp.iterdir() if p.is_dir()):
                for view in sorted(p.name for p in (sp/loc).iterdir() if p.is_dir()):
                    vp = sp / loc / view
                    seqs = sorted(p for p in vp.iterdir() if p.is_dir())
                    print(f"  {subset}/{loc}/{view}  ({len(seqs)} sequences)")
                    if seqs:
                        s0 = seqs[0]
                        print(f"    First seq: '{s0.name}'")
                        for sub in sorted(p for p in s0.iterdir() if p.is_dir()):
                            files = _sorted_files(sub)
                            if files:
                                exts = set(f.suffix for f in files)
                                print(f"      {sub.name}/  {len(files)} files  "
                                      f"exts={exts}  e.g. '{files[0].name}'")
                            else:
                                all_items = list(sub.iterdir())
                                print(f"      {sub.name}/  {len(all_items)} items "
                                      f"(no readable files found)")
        print("=" * 60)

    # ------------------------------------------------------------------
    # Iteration
    # ------------------------------------------------------------------

    def iter_scenes(self, subset="ARID20", location="table", view="bottom",
                    max_scenes=None, min_objects=0) -> Iterator[OCIDScene]:
        subset, location, view = self._resolve(subset, location, view)
        base = self.root / subset / location / view
        print(f"Loading scenes from: {base}")
        if not base.exists():
            raise FileNotFoundError(
                f"Path not found: {base}\n"
                f"Run loader.inspect_dataset() to see what's available."
            )
        count = 0
        for top_dir in sorted(p for p in base.iterdir() if p.is_dir()):
            # ARID20: base/seq01/rgb/  — seq dirs directly under base
            # ARID10/YCB10: base/box/seq05/rgb/ — extra category level
            has_rgb = (top_dir / "rgb").is_dir() or (top_dir / "RGB").is_dir()
            seq_dirs = [top_dir] if has_rgb else sorted(
                p for p in top_dir.iterdir() if p.is_dir())
            for seq_dir in seq_dirs:
                for scene in self._iter_frames(seq_dir, subset, location, view):
                    if min_objects > 0 and scene.n_gt_objects() < min_objects:
                        continue
                    # store category name for grouping (ARID10/YCB10 only)
                    scene.category = top_dir.name if not has_rgb else ""
                    yield scene
                    count += 1
                    if max_scenes and count >= max_scenes:
                        return

    def load_sequence(self, subset, location, view, seq_id) -> List[OCIDScene]:
        subset, location, view = self._resolve(subset, location, view)
        seq_dir = self.root / subset / location / view / seq_id
        if not seq_dir.exists():
            raise FileNotFoundError(f"Sequence not found: {seq_dir}")
        return list(self._iter_frames(seq_dir, subset, location, view))

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _resolve(self, subset, location, view):
        """Case-insensitive folder matching."""
        def match(parent, target):
            if parent.exists():
                for p in parent.iterdir():
                    if p.is_dir() and p.name.lower() == target.lower():
                        return p.name
            return target
        subset   = match(self.root, subset)
        location = match(self.root / subset, location)
        view     = match(self.root / subset / location, view)
        return subset, location, view

    def _find_subdir(self, parent, candidates):
        for name in candidates:
            p = parent / name
            if p.is_dir():
                return p
        return None

    def _iter_frames(self, seq_dir, subset, location, view) -> Iterator[OCIDScene]:
        rgb_dir   = self._find_subdir(seq_dir, ["rgb",   "RGB",   "color"])
        depth_dir = self._find_subdir(seq_dir, ["depth", "Depth"])
        label_dir = self._find_subdir(seq_dir, ["label", "Label", "labels"])
        pcd_dir   = self._find_subdir(seq_dir, ["pcd",   "PCD"])

        if rgb_dir is None or depth_dir is None:
            return

        # _sorted_files handles both .png and extensionless files
        rgb_files   = _sorted_files(rgb_dir)
        depth_files = _sorted_files(depth_dir)
        label_files = _sorted_files(label_dir) if label_dir else []
        pcd_files   = _sorted_files(pcd_dir)   if pcd_dir   else []

        if not rgb_files or not depth_files:
            return

        # match by position — guaranteed correspondence within each sequence
        n = min(len(rgb_files), len(depth_files))
        for i in range(n):
            yield OCIDScene(
                rgb_path   = rgb_files[i],
                depth_path = depth_files[i],
                label_path = label_files[i] if i < len(label_files) else None,
                pcd_path   = pcd_files[i]   if i < len(pcd_files)   else None,
                subset=subset, location=location, view=view,
                seq_id=seq_dir.name, frame_id=rgb_files[i].stem,

                pipe=self.pipe,
            )

    # ------------------------------------------------------------------
    # Benchmark
    # ------------------------------------------------------------------

    def benchmark(self, subset="ARID20", location="table", view="bottom",
                  max_scenes=50, iou_threshold=0.5, verbose=True) -> dict:
        scores = []
        for i, scene in enumerate(
            self.iter_scenes(subset, location, view, max_scenes=max_scenes)
        ):
            result = scene.run_perception()
            if result is None:
                continue
            score = scene.evaluate(result, iou_threshold)
            scores.append(score)
            if verbose:
                print(f"  [{i+1:3d}] {scene.seq_id}/{scene.frame_id} | {score}")
        if not scores:
            print("No scenes processed. Run inspect_dataset() to debug.")
            return {}
        summary = {
            'n_scenes':   len(scores),
            'mean_iou':   round(float(np.mean([s.mean_iou for s in scores])), 4),
            'match_rate': round(float(np.mean(
                [s.matched / max(s.n_gt_objects, 1) for s in scores])), 4),
        }
        if verbose:
            print(f"\nmean_iou={summary['mean_iou']}  "
                  f"match_rate={summary['match_rate']}  "
                  f"n={summary['n_scenes']}")
        return summary