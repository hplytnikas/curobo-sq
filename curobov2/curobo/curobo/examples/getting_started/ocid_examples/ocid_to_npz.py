"""
ocid_to_npz.py
==============
Convert OCID dataset frames to the NPZ format expected by motion_planning_sq.py.

Self-contained — uses only numpy and PIL (no cv2, no project_3dv imports).

Output NPZ keys
---------------
    xyz            : (N, 3) float32  — point coordinates in camera frame (metres)
    color          : (N, 3) uint8    — RGB colours 0-255
    instance_label : (N,)  int64     — 0 = background/table (skipped by planner),
                                       1..K = object instances

OCID dataset structure
----------------------
    <root>/<subset>/<location>/<view>/<seq>/rgb/
                                            depth/    (16-bit PNG, mm)
                                            label/    (8 or 16-bit PNG, instance IDs)

    Files within each subdir are matched positionally by sorted name.

Camera (ASUS Xtion, all OCID subsets)
--------------------------------------
    fx = fy = 570.342,  cx = 319.5,  cy = 239.5,  depth_scale = 1000 (mm → m)

Usage
-----
Single frame:
    python ocid_to_npz.py --ocid /mnt/seagate/code/3dv/OCID-dataset --out scene.npz

Multiple frames:
    python ocid_to_npz.py --ocid /mnt/seagate/code/3dv/OCID-dataset \\
        --subset ARID20 --location table --view bottom \\
        --n 6 --out_dir /home/haroldas/3DV/ocid_npz/

Then run motion planning:
    conda run -n 3dv python \\
        curobov2/curobo/curobo/examples/getting_started/motion_planning_sq.py \\
        --scenes_dir /home/haroldas/3DV/ocid_npz/ --show_pointcloud --visualize

Coordinate system note
----------------------
OCID clouds are in camera frame: X right, Y down, Z into scene.
motion_planning_sq.py applies --scene_translation / --scene_quat_wxyz to place the
scene in the robot workspace. Tune those flags if objects appear in the wrong place.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterator, List, Optional

import numpy as np
from PIL import Image

# ── Camera intrinsics (ASUS Xtion, fixed for all OCID subsets) ─────────────────
FX = FY = 570.3422241210938
CX, CY = 319.5, 239.5
DEPTH_SCALE = 1000.0   # raw pixel value → metres


# ── Data structures ─────────────────────────────────────────────────────────────

class OCIDFrame:
    def __init__(self, rgb_path: Path, depth_path: Path,
                 label_path: Optional[Path], seq_id: str, frame_id: str):
        self.rgb_path   = rgb_path
        self.depth_path = depth_path
        self.label_path = label_path
        self.seq_id     = seq_id
        self.frame_id   = frame_id

    def load_rgb(self) -> np.ndarray:
        return np.array(Image.open(self.rgb_path).convert("RGB"), dtype=np.uint8)

    def load_depth(self) -> np.ndarray:
        return np.array(Image.open(self.depth_path), dtype=np.float32) / DEPTH_SCALE

    def load_label(self) -> Optional[np.ndarray]:
        if self.label_path is None or not self.label_path.exists():
            return None
        return np.array(Image.open(self.label_path), dtype=np.int64)

    def n_gt_objects(self) -> int:
        label = self.load_label()
        if label is None:
            return 0
        return int((np.unique(label) > 0).sum())


# ── Directory walking ────────────────────────────────────────────────────────────

def _sorted_files(directory: Optional[Path]) -> List[Path]:
    if directory is None or not directory.exists():
        return []
    return sorted(p for p in directory.iterdir()
                  if p.is_file() and not p.name.startswith('.'))


def _find_subdir(parent: Path, candidates: List[str]) -> Optional[Path]:
    for name in candidates:
        p = parent / name
        if p.is_dir():
            return p
    return None


def _iter_frames(seq_dir: Path) -> Iterator[OCIDFrame]:
    rgb_dir   = _find_subdir(seq_dir, ["rgb",   "RGB",   "color"])
    depth_dir = _find_subdir(seq_dir, ["depth", "Depth"])
    label_dir = _find_subdir(seq_dir, ["label", "Label", "labels"])

    if rgb_dir is None or depth_dir is None:
        return

    rgb_files   = _sorted_files(rgb_dir)
    depth_files = _sorted_files(depth_dir)
    label_files = _sorted_files(label_dir) if label_dir else []

    for i in range(min(len(rgb_files), len(depth_files))):
        yield OCIDFrame(
            rgb_path   = rgb_files[i],
            depth_path = depth_files[i],
            label_path = label_files[i] if i < len(label_files) else None,
            seq_id     = seq_dir.name,
            frame_id   = rgb_files[i].stem,
        )


def iter_scenes(ocid_root: Path, subset: str, location: str, view: str,
                max_scenes: int = 1, min_objects: int = 0) -> Iterator[OCIDFrame]:
    base = ocid_root / subset / location / view
    if not base.exists():
        raise FileNotFoundError(
            f"Path not found: {base}\n"
            f"Available top-level folders: {[p.name for p in ocid_root.iterdir() if p.is_dir()]}"
        )
    print(f"Loading scenes from: {base}")
    count = 0
    for top_dir in sorted(p for p in base.iterdir() if p.is_dir()):
        has_rgb = (top_dir / "rgb").is_dir() or (top_dir / "RGB").is_dir()
        seq_dirs = [top_dir] if has_rgb else sorted(
            p for p in top_dir.iterdir() if p.is_dir())
        for seq_dir in seq_dirs:
            for frame in _iter_frames(seq_dir):
                if min_objects > 0 and frame.n_gt_objects() < min_objects:
                    continue
                yield frame
                count += 1
                if count >= max_scenes:
                    return


# ── Conversion ───────────────────────────────────────────────────────────────────

def frame_to_arrays(frame: OCIDFrame,
                    max_depth: float = 2.0) -> Optional[dict]:
    depth = frame.load_depth()
    rgb   = frame.load_rgb()
    label = frame.load_label()

    H, W = depth.shape
    v_idx, u_idx = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')
    valid = (depth > 0.1) & (depth < max_depth)

    Z = depth[valid]
    X = (u_idx[valid] - CX) * Z / FX
    Y = (v_idx[valid] - CY) * Z / FY
    xyz   = np.stack([X, Y, Z], axis=1).astype(np.float32)
    color = rgb[valid].astype(np.uint8)

    if label is not None:
        instance_label = label[valid].astype(np.int64)
    else:
        instance_label = np.ones(len(xyz), dtype=np.int64)

    return dict(xyz=xyz, color=color, instance_label=instance_label)


def export_frame(frame: OCIDFrame, out_path: Path, max_depth: float = 2.0) -> bool:
    arrays = frame_to_arrays(frame, max_depth=max_depth)
    if arrays is None:
        print(f"  [skip] {frame.seq_id}/{frame.frame_id} — load failed")
        return False

    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(str(out_path), **arrays)

    n     = len(arrays['xyz'])
    n_obj = int((np.unique(arrays['instance_label']) > 0).sum())
    print(f"  Saved {out_path.name}  ({n:,} pts, {n_obj} object instance(s))")
    return True


# ── CLI ──────────────────────────────────────────────────────────────────────────

def _parse() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Export OCID frames to NPZ for motion_planning_sq.py",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--ocid",     required=True,
                   help="Root directory of the OCID dataset")
    p.add_argument("--subset",   default="ARID20",
                   choices=["ARID20", "ARID10", "YCB10"],
                   help="OCID subset (default: ARID20)")
    p.add_argument("--location", default="table",
                   choices=["table", "floor"],
                   help="Surface type (default: table)")
    p.add_argument("--view",     default="bottom",
                   choices=["top", "bottom"],
                   help="Camera viewpoint (default: bottom)")
    p.add_argument("--min_objects", type=int, default=2,
                   help="Skip frames with fewer GT object instances (default: 2)")
    p.add_argument("--max_depth",   type=float, default=2.0,
                   help="Discard points beyond this depth in metres (default: 2.0)")
    p.add_argument("--n", type=int, default=1,
                   help="Number of frames to export (default: 1)")

    grp = p.add_mutually_exclusive_group(required=True)
    grp.add_argument("--out",     type=str,
                     help="Output path for a single NPZ file")
    grp.add_argument("--out_dir", type=str,
                     help="Output directory for multiple NPZ files")

    p.add_argument("--inspect", action="store_true",
                   help="Print dataset layout and exit")
    return p.parse_args()


def _inspect(ocid_root: Path) -> None:
    print(f"\nInspecting OCID at: {ocid_root}")
    print("=" * 60)
    for subset in sorted(p.name for p in ocid_root.iterdir() if p.is_dir()):
        sp = ocid_root / subset
        for loc in sorted(p.name for p in sp.iterdir() if p.is_dir()):
            for view in sorted(p.name for p in (sp / loc).iterdir() if p.is_dir()):
                vp = sp / loc / view
                seqs = sorted(p for p in vp.iterdir() if p.is_dir())
                print(f"  {subset}/{loc}/{view}  ({len(seqs)} sequences)")
                if seqs:
                    s0 = seqs[0]
                    print(f"    First seq: '{s0.name}'")
                    for sub in sorted(p for p in s0.iterdir() if p.is_dir()):
                        files = _sorted_files(sub)
                        if files:
                            print(f"      {sub.name}/  {len(files)} files  "
                                  f"e.g. '{files[0].name}'")
    print("=" * 60)


def main() -> None:
    args = _parse()
    ocid_root = Path(args.ocid)

    if args.inspect:
        _inspect(ocid_root)
        return

    if args.n > 1 and args.out is not None:
        print("Use --out_dir when exporting more than one frame.")
        sys.exit(1)

    frames = iter_scenes(
        ocid_root,
        subset=args.subset,
        location=args.location,
        view=args.view,
        max_scenes=args.n,
        min_objects=args.min_objects,
    )

    exported = 0
    for frame in frames:
        if args.out:
            out_path = Path(args.out)
        else:
            fname = f"{args.subset}_{frame.seq_id}_{frame.frame_id}.npz"
            out_path = Path(args.out_dir) / fname

        if export_frame(frame, out_path, max_depth=args.max_depth):
            exported += 1

    if exported == 0:
        print("No frames exported. Try --inspect to see what's available, "
              "or lower --min_objects.")
    else:
        print(f"\nDone — {exported} frame(s) exported.")
        if args.out:
            print(f"\nRun motion planning with:")
            print(f"  conda run -n 3dv python \\")
            print(f"    curobov2/curobo/curobo/examples/getting_started/motion_planning_sq.py \\")
            print(f"    --ply_path {args.out}")
        else:
            print(f"\nRun motion planning with scene dropdown:")
            print(f"  conda run -n 3dv python \\")
            print(f"    curobov2/curobo/curobo/examples/getting_started/motion_planning_sq.py \\")
            print(f"    --scenes_dir {args.out_dir} --show_pointcloud --visualize")


if __name__ == "__main__":
    main()
