#!/usr/bin/env python3
"""
visualise_own_scan.py
=====================
Interactive Open3D visualisation of own-scan SuperDec results.

Usage — interactive:
  python scripts/visualise_own_scan.py \
    --cloud  outputs/own_scan_singleframe2/fused_cloud.npz \
    --results outputs/own_scan_singleframe2/results.npz

Usage — non-interactive PNG export:
  python scripts/visualise_own_scan.py \
    --cloud  outputs/own_scan_singleframe2/fused_cloud.npz \
    --results outputs/own_scan_singleframe2/results.npz \
    --no-interactive --mode overlay \
    --output outputs/own_scan_singleframe2/viz_overlay.png

Key bindings (interactive):
  1  raw RGB cloud
  2  segments (each segment a different colour)
  3  SQ primitive meshes
  4  overlay (grey cloud + coloured meshes)
  S  save screenshot
  Q  quit
"""

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import numpy as np
from scipy.spatial.transform import Rotation

import open3d as o3d

# ── colour palette (tab10) ────────────────────────────────────────────────────
_TAB10 = np.array([
    [0.122, 0.467, 0.706],
    [1.000, 0.498, 0.055],
    [0.173, 0.627, 0.173],
    [0.839, 0.153, 0.157],
    [0.580, 0.404, 0.741],
    [0.549, 0.337, 0.294],
    [0.890, 0.467, 0.761],
    [0.498, 0.498, 0.498],
    [0.737, 0.741, 0.133],
    [0.090, 0.745, 0.812],
], dtype=np.float64)


def _seg_colour(i: int) -> np.ndarray:
    return _TAB10[i % len(_TAB10)]


# ── SQ mesh ───────────────────────────────────────────────────────────────────

def _sq_mesh(prim: dict, n: int = 30) -> o3d.geometry.TriangleMesh:
    """Build an Open3D TriangleMesh for one superquadric primitive."""
    u = np.linspace(-np.pi / 2, np.pi / 2, n)
    v = np.linspace(-np.pi,      np.pi,     n * 2)

    def se(x, e):
        return np.sign(x) * (np.abs(x) + 1e-8) ** e

    U, V = np.meshgrid(u, v)   # shape: (2n, n)
    X = prim["sx"] * se(np.cos(U), prim["e1"]) * se(np.cos(V), prim["e2"])
    Y = prim["sy"] * se(np.cos(U), prim["e1"]) * se(np.sin(V), prim["e2"])
    Z = prim["sz"] * se(np.sin(U), prim["e1"]) * np.ones_like(V)

    # rotation
    if "rotation_matrix" in prim and prim["rotation_matrix"] is not None:
        R = np.array(prim["rotation_matrix"], dtype=np.float64).reshape(3, 3)
    else:
        R = Rotation.from_euler("xyz",
                                [prim["rx"], prim["ry"], prim["rz"]]).as_matrix()

    pts_flat = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=1)  # (2n*n, 3)
    pts_world = pts_flat @ R.T + np.array([prim["tx"], prim["ty"], prim["tz"]])

    # triangle connectivity: grid (n_v rows × n_u cols), v is periodic
    n_v, n_u = n * 2, n
    tris = []
    for iv in range(n_v):
        for iu in range(n_u - 1):
            iv1 = (iv + 1) % n_v
            v00 = iv  * n_u + iu
            v01 = iv  * n_u + (iu + 1)
            v10 = iv1 * n_u + iu
            v11 = iv1 * n_u + (iu + 1)
            tris.append([v00, v10, v01])
            tris.append([v10, v11, v01])

    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices  = o3d.utility.Vector3dVector(pts_world)
    mesh.triangles = o3d.utility.Vector3iVector(np.array(tris, dtype=np.int32))
    mesh.compute_vertex_normals()
    return mesh


# ── geometry builders ─────────────────────────────────────────────────────────

def _geom_cloud(pts: np.ndarray, colors: np.ndarray | None) -> list:
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts.astype(np.float64))
    if colors is not None:
        pcd.colors = o3d.utility.Vector3dVector(
            colors.astype(np.float64) / 255.0)
    else:
        pcd.paint_uniform_color([0.6, 0.6, 0.6])
    return [pcd]


def _geom_segments(pts: np.ndarray,
                   seg_pts_list: list | None) -> list:
    """Colour each segment; remaining background pts grey."""
    geoms = []
    if not seg_pts_list:
        return _geom_cloud(pts, None)

    # background: all pts not in any segment
    all_seg = np.concatenate(seg_pts_list, axis=0) if seg_pts_list else np.empty((0, 3))
    bg_pcd = o3d.geometry.PointCloud()
    bg_pcd.points = o3d.utility.Vector3dVector(pts.astype(np.float64))
    bg_pcd.paint_uniform_color([0.7, 0.7, 0.7])
    geoms.append(bg_pcd)

    for si, seg in enumerate(seg_pts_list):
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(seg.astype(np.float64))
        pcd.paint_uniform_color(_seg_colour(si).tolist())
        geoms.append(pcd)

    return geoms


def _geom_primitives(prim_list: list | None) -> list:
    """One mesh per primitive, coloured by segment index."""
    if not prim_list:
        return []
    geoms = []
    for prim in prim_list:
        mesh = _sq_mesh(prim)
        c = _seg_colour(prim.get("seg_idx", 0))
        mesh.paint_uniform_color(c.tolist())
        geoms.append(mesh)
    return geoms


def _geom_overlay(pts: np.ndarray, prim_list: list | None) -> list:
    """Grey cloud + coloured primitive meshes."""
    bg_pcd = o3d.geometry.PointCloud()
    bg_pcd.points = o3d.utility.Vector3dVector(pts.astype(np.float64))
    bg_pcd.paint_uniform_color([0.75, 0.75, 0.75])
    geoms = [bg_pcd]
    if prim_list:
        geoms.extend(_geom_primitives(prim_list))
    return geoms


# ── matplotlib helpers ────────────────────────────────────────────────────────

def _mpl_set_limits(ax, pts: np.ndarray, margin: float = 0.1):
    if len(pts) == 0:
        return
    c = pts.mean(0)
    r = max((pts.max(0) - pts.min(0)).max() / 2 * (1 + margin), 0.01)
    ax.set_xlim(c[0] - r, c[0] + r)
    ax.set_ylim(c[1] - r, c[1] + r)
    ax.set_zlim(c[2] - r, c[2] + r)
    ax.set_box_aspect([1, 1, 1])
    ax.set_xticklabels([]); ax.set_yticklabels([]); ax.set_zticklabels([])
    ax.set_xlabel("X", fontsize=7); ax.set_ylabel("Y", fontsize=7)
    ax.set_zlabel("Z", fontsize=7)


def _mpl_wireframe(ax, prim: dict, color, alpha: float = 0.5, lw: float = 0.8):
    u = np.linspace(-np.pi / 2, np.pi / 2, 20)
    v = np.linspace(-np.pi, np.pi, 40)
    def se(x, e): return np.sign(x) * (np.abs(x) + 1e-8) ** e
    U, V = np.meshgrid(u, v)
    X = prim["sx"] * se(np.cos(U), prim["e1"]) * se(np.cos(V), prim["e2"])
    Y = prim["sy"] * se(np.cos(U), prim["e1"]) * se(np.sin(V), prim["e2"])
    Z = prim["sz"] * se(np.sin(U), prim["e1"]) * np.ones_like(V)
    if "rotation_matrix" in prim and prim["rotation_matrix"] is not None:
        R = np.array(prim["rotation_matrix"], dtype=np.float64).reshape(3, 3)
    else:
        R = Rotation.from_euler("xyz", [prim["rx"], prim["ry"], prim["rz"]]).as_matrix()
    body  = np.stack([X.ravel(), Y.ravel(), Z.ravel()], 1)
    world = body @ R.T + np.array([prim["tx"], prim["ty"], prim["tz"]])
    Xw = world[:, 0].reshape(X.shape)
    Yw = world[:, 1].reshape(Y.shape)
    Zw = world[:, 2].reshape(Z.shape)
    ax.plot_wireframe(Xw, Yw, Zw, rstride=2, cstride=2,
                      color=color, alpha=alpha, linewidth=lw)


# ── offscreen render (matplotlib) ─────────────────────────────────────────────

def _render_offscreen(mode: str,
                      pts: np.ndarray,
                      colors: np.ndarray | None,
                      seg_pts_list: list | None,
                      prim_list: list | None,
                      out_path: Path,
                      elev: float = 45,
                      azim: float = -60):
    """Render one mode to PNG using matplotlib 3D (works headless)."""
    fig = plt.figure(figsize=(8, 7))
    ax  = fig.add_subplot(1, 1, 1, projection="3d")

    if mode == "cloud":
        c = colors.astype(np.float32) / 255.0 if colors is not None else "steelblue"
        # subsample for speed
        idx = np.random.default_rng(0).choice(len(pts),
                                               min(len(pts), 50_000), replace=False)
        ax.scatter(pts[idx, 0], pts[idx, 1], pts[idx, 2],
                   c=c[idx] if isinstance(c, np.ndarray) else c,
                   s=0.4, linewidths=0, rasterized=True)
        ax.set_title("Raw cloud (RGB)", fontsize=10)

    elif mode == "segments":
        ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2],
                   c="0.8", s=0.2, linewidths=0, rasterized=True)
        if seg_pts_list:
            for si, seg in enumerate(seg_pts_list):
                c = _seg_colour(si).tolist()
                ax.scatter(seg[:, 0], seg[:, 1], seg[:, 2],
                           c=[c], s=1.5, linewidths=0, rasterized=True)
        ax.set_title(f"Segments ({len(seg_pts_list or [])})", fontsize=10)

    elif mode == "primitives":
        if prim_list:
            for prim in prim_list:
                c = _seg_colour(prim.get("seg_idx", 0)).tolist()
                _mpl_wireframe(ax, prim, color=c)
        ax.set_title(f"SQ primitives ({len(prim_list or [])})", fontsize=10)

    elif mode == "overlay":
        idx = np.random.default_rng(0).choice(len(pts),
                                               min(len(pts), 30_000), replace=False)
        ax.scatter(pts[idx, 0], pts[idx, 1], pts[idx, 2],
                   c="0.75", s=0.3, linewidths=0, alpha=0.4, rasterized=True)
        if prim_list:
            for prim in prim_list:
                c = _seg_colour(prim.get("seg_idx", 0)).tolist()
                _mpl_wireframe(ax, prim, color=c)
        ax.set_title(f"Overlay — cloud + {len(prim_list or [])} primitives",
                     fontsize=10)

    elif mode == "topdown":
        idx = np.random.default_rng(0).choice(len(pts),
                                               min(len(pts), 30_000), replace=False)
        ax.scatter(pts[idx, 0], pts[idx, 1], pts[idx, 2],
                   c="0.75", s=0.3, linewidths=0, alpha=0.4, rasterized=True)
        if seg_pts_list:
            for si, seg in enumerate(seg_pts_list):
                c = _seg_colour(si).tolist()
                ax.scatter(seg[:, 0], seg[:, 1], seg[:, 2],
                           c=[c], s=2.0, linewidths=0, rasterized=True)
        if prim_list:
            for prim in prim_list:
                c = _seg_colour(prim.get("seg_idx", 0)).tolist()
                _mpl_wireframe(ax, prim, color=c, alpha=0.7)
        elev, azim = 89, -90   # override to bird's-eye
        ax.set_title("Top-down view", fontsize=10)

    # ── axis limits ──────────────────────────────────────────────────────────────
    if mode == "cloud":
        _mpl_set_limits(ax, pts)
    elif seg_pts_list:
        fg = np.concatenate([np.array(s) for s in seg_pts_list], axis=0)
        fg_center = fg.mean(0)
        fg_range  = (fg.max(0) - fg.min(0)).max() * 0.7
        ax.set_xlim(fg_center[0] - fg_range, fg_center[0] + fg_range)
        ax.set_ylim(fg_center[1] - fg_range, fg_center[1] + fg_range)
        ax.set_zlim(fg_center[2] - fg_range, fg_center[2] + fg_range)
    else:
        _mpl_set_limits(ax, pts)

    ax.view_init(elev=elev, azim=azim)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved → {out_path}")


# ── interactive visualiser ────────────────────────────────────────────────────

def _run_interactive(pts: np.ndarray,
                     colors: np.ndarray | None,
                     seg_pts_list: list | None,
                     prim_list: list | None,
                     out_dir: Path):
    print("=== Own Scan Visualiser ===")
    print("  1 — raw cloud")
    print("  2 — segments")
    print("  3 — SQ primitives")
    print("  4 — overlay")
    print("  S — save screenshot")
    print("  Q — quit")

    mode_geoms = {
        1: _geom_cloud(pts, colors),
        2: _geom_segments(pts, seg_pts_list),
        3: _geom_primitives(prim_list) or _geom_cloud(pts, None),
        4: _geom_overlay(pts, prim_list),
    }

    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window(window_name="Own Scan — SuperDec", width=1280, height=960)

    current_geoms = []

    def _set_mode(mode_idx: int):
        nonlocal current_geoms
        for g in current_geoms:
            vis.remove_geometry(g, reset_bounding_box=False)
        current_geoms = mode_geoms[mode_idx]
        first = True
        for g in current_geoms:
            vis.add_geometry(g, reset_bounding_box=first)
            first = False
        vis.poll_events()
        vis.update_renderer()
        print(f"  mode {mode_idx}")

    # initialise with mode 1
    for g in mode_geoms[1]:
        vis.add_geometry(g)
    current_geoms = mode_geoms[1]

    vis.register_key_callback(ord("1"), lambda v: _set_mode(1))
    vis.register_key_callback(ord("2"), lambda v: _set_mode(2))
    vis.register_key_callback(ord("3"), lambda v: _set_mode(3))
    vis.register_key_callback(ord("4"), lambda v: _set_mode(4))

    def _save_screenshot(v):
        p = out_dir / "screenshot.png"
        v.capture_screen_image(str(p), do_render=True)
        print(f"  screenshot saved → {p}")

    vis.register_key_callback(ord("S"), _save_screenshot)
    vis.register_key_callback(ord("Q"), lambda v: v.close())

    vis.run()
    vis.destroy_window()


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description="Visualise own-scan SuperDec results with Open3D.")
    ap.add_argument("--cloud",   required=True,
                    help="Path to fused_cloud.npz (keys: pts, colors)")
    ap.add_argument("--results", default=None,
                    help="Path to results.npz (keys: seg_pts, primitives); "
                         "optional — cloud-only if omitted")
    ap.add_argument("--mode", default="overlay",
                    choices=["cloud", "segments", "primitives", "overlay", "topdown"],
                    help="Display mode for --no-interactive (default: overlay)")
    ap.add_argument("--elev", type=float, default=45,
                    help="Matplotlib elevation angle in degrees (default 45)")
    ap.add_argument("--azim", type=float, default=-60,
                    help="Matplotlib azimuth angle in degrees (default -60)")
    ap.add_argument("--no-interactive", action="store_true",
                    help="Render to PNG and exit (no window)")
    ap.add_argument("--output", default=None,
                    help="Output PNG path for --no-interactive "
                         "(default: <cloud_dir>/viz_<mode>.png)")
    args = ap.parse_args()

    # ── load cloud ────────────────────────────────────────────────────────────
    cloud_path = Path(args.cloud)
    data = np.load(cloud_path)
    pts    = data["pts"].astype(np.float32)
    colors = data["colors"] if "colors" in data else None
    print(f"cloud: {len(pts):,} pts  colors={'yes' if colors is not None else 'no'}")

    # ── load results ──────────────────────────────────────────────────────────
    seg_pts_list = None
    prim_list    = None

    if args.results and Path(args.results).exists():
        res = np.load(args.results, allow_pickle=True)
        seg_pts_list = list(res["seg_pts"])   # list of (N_i, 3) arrays
        prim_list    = list(res["primitives"])  # list of dicts
        n_prims = sum(1 for p in prim_list if p)
        print(f"results: {len(seg_pts_list)} segments, {n_prims} primitives")
    else:
        print("results.npz not found — cloud-only (modes 3/4 will show raw cloud)")

    # ── output path ───────────────────────────────────────────────────────────
    out_dir = cloud_path.parent

    # ── dispatch ──────────────────────────────────────────────────────────────
    if args.no_interactive:
        out_path = Path(args.output) if args.output else \
            out_dir / f"viz_{args.mode}.png"
        _render_offscreen(args.mode, pts, colors, seg_pts_list, prim_list, out_path,
                          elev=args.elev, azim=args.azim)
    else:
        _run_interactive(pts, colors, seg_pts_list, prim_list, out_dir)


if __name__ == "__main__":
    main()
