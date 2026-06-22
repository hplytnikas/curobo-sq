"""
visualize_sq.py — Visualize superquadric fits on OCID scenes.

Shows:
  1. Raw point cloud coloured by detected object
  2. Fitted superquadric surfaces overlaid
  3. Side-by-side: input vs SQ representation
  4. Per-object fit quality (Chamfer L2)

Saves figures to outputs/ — no display needed (works headless).

Usage:
    python3 visualize_sq.py --ocid /Volumes/T7/OCID-dataset --scene 10
    python3 visualize_sq.py --ocid /Volumes/T7/OCID-dataset --scene 10 --interactive
"""

import argparse, sys, os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from project_3dv.perception.evaluation.ocid_loader import OCIDLoader, ASUS_XTION
from pipeline import TabletopPerception
from superquadric import SuperquadricFitter, MultiSQFit
from superdec_utils import Superquadrics, Scene, sq_fits_to_npz

# ── Colour palette ────────────────────────────────────────────────────────────
PALETTE = [
    '#e63946', '#457b9d', '#2a9d8f', '#e76f51',
    '#8338ec', '#f4a261', '#06d6a0', '#ffd166',
    '#118ab2', '#ef476f', '#06a77d', '#d62246',
]


def _colour(i):
    return PALETTE[i % len(PALETTE)]


def _hex_to_rgb(h):
    h = h.lstrip('#')
    return tuple(int(h[i:i+2], 16)/255 for i in (0, 2, 4))


# ── Project points to image ───────────────────────────────────────────────────

def project_points(pts, H=480, W=640):
    fx, fy = ASUS_XTION['fx'], ASUS_XTION['fy']
    cx, cy = ASUS_XTION['cx'], ASUS_XTION['cy']
    Z = pts[:, 2]
    valid = Z > 0
    u = np.round(pts[valid, 0] * fx / Z[valid] + cx).astype(int)
    v = np.round(pts[valid, 1] * fy / Z[valid] + cy).astype(int)
    ok = (u >= 0) & (u < W) & (v >= 0) & (v < H)
    return v[ok], u[ok]


# ── Figure 1: RGB + detection overlay ────────────────────────────────────────

def fig_detection_overlay(scene, result, save_path):
    rgb   = scene.load_rgb()
    label = scene.load_label()

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.patch.set_facecolor('#1a1a2e')

    # RGB
    axes[0].imshow(rgb)
    axes[0].set_title('RGB Input', color='white', fontsize=11)
    axes[0].axis('off')

    # GT label
    if label is not None:
        label_vis = np.zeros((*label.shape, 3))
        ids = np.unique(label); ids = ids[ids > 0]
        for i, lid in enumerate(ids):
            mask = label == lid
            if mask.sum() < label.size * 0.05:  # skip background
                rgb_c = _hex_to_rgb(_colour(i))
                label_vis[mask] = rgb_c
        axes[1].imshow(rgb)
        axes[1].imshow(label_vis, alpha=0.6)
        axes[1].set_title(f'GT Labels ({len(ids)} objects)', color='white', fontsize=11)
    else:
        axes[1].imshow(rgb)
        axes[1].set_title('GT Labels (unavailable)', color='white', fontsize=11)
    axes[1].axis('off')

    # Detections
    overlay = np.zeros((480, 640, 3))
    for i, obj in enumerate(result.objects):
        v, u = project_points(obj.points)
        rgb_c = _hex_to_rgb(_colour(i))
        overlay[v, u] = rgb_c

    axes[2].imshow(rgb)
    axes[2].imshow(overlay, alpha=0.7)
    patches = [mpatches.Patch(color=_colour(i), label=f'Obj {obj.id} ({obj.n_points}pt)')
               for i, obj in enumerate(result.objects)]
    axes[2].legend(handles=patches, loc='upper right', fontsize=8,
                   facecolor='#1a1a2e', labelcolor='white')
    axes[2].set_title(f'Detections ({len(result.objects)} objects)', color='white', fontsize=11)
    axes[2].axis('off')

    plt.suptitle(f'{scene.seq_id}/{scene.frame_id}  |  GT={scene.n_gt_objects()}  Det={len(result.objects)}',
                 color='white', fontsize=12, y=1.01)
    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='#1a1a2e')
    plt.close()
    print(f"Saved: {save_path}")


# ── Figure 2: 3D point cloud + SQ surfaces ───────────────────────────────────


def fig_3d_sq(result, sq_multis, save_path, n_surface=800):
    """
    3D visualization using matplotlib scatter.
    The publication-quality figure is the PLY file — open in MeshLab or open3d interactively.
    """
    fig = plt.figure(figsize=(18, 6))
    fig.patch.set_facecolor('#1a1a2e')

    ax1 = fig.add_subplot(131, projection='3d')
    ax1.set_facecolor('#1a1a2e')
    for i, obj in enumerate(result.objects):
        pts = obj.points
        idx = np.random.choice(len(pts), min(300, len(pts)), replace=False)
        p = pts[idx]
        ax1.scatter(p[:, 0], p[:, 2], -p[:, 1], c=_colour(i), s=2, alpha=0.6)
    ax1.set_title('Detected Segments', color='white', fontsize=10)
    _style_3d(ax1)

    ax2 = fig.add_subplot(132, projection='3d')
    ax2.set_facecolor('#1a1a2e')
    for i, multi in enumerate(sq_multis):
        for sq in multi.primitives:
            surf = sq.surface_points(n_surface)
            idx = np.random.choice(len(surf), min(200, len(surf)), replace=False)
            s = surf[idx]
            ax2.scatter(s[:, 0], s[:, 2], -s[:, 1], c=_colour(i), s=1, alpha=0.4)
    ax2.set_title('Superquadric Surfaces', color='white', fontsize=10)
    _style_3d(ax2)

    ax3 = fig.add_subplot(133, projection='3d')
    ax3.set_facecolor('#1a1a2e')
    for i, (obj, multi) in enumerate(zip(result.objects, sq_multis)):
        pts = obj.points
        idx = np.random.choice(len(pts), min(200, len(pts)), replace=False)
        p = pts[idx]
        ax3.scatter(p[:, 0], p[:, 2], -p[:, 1], c=_colour(i), s=1, alpha=0.3)
        for sq in multi.primitives:
            surf = sq.surface_points(400)
            idx2 = np.random.choice(len(surf), min(150, len(surf)), replace=False)
            s = surf[idx2]
            ax3.scatter(s[:, 0], s[:, 2], -s[:, 1], c=_colour(i), s=3, alpha=0.9)
    ax3.set_title('Overlay: Points + SQ', color='white', fontsize=10)
    _style_3d(ax3)

    plt.suptitle('3D Superquadric Decomposition', color='white', fontsize=12, y=1.01)
    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='#1a1a2e')
    plt.close()
    print(f"Saved: {save_path}")


def _style_3d(ax):
    ax.set_xlabel('X', color='gray', fontsize=7)
    ax.set_ylabel('Z', color='gray', fontsize=7)
    ax.set_zlabel('Y', color='gray', fontsize=7)
    ax.tick_params(colors='gray', labelsize=6)
    for pane in [ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane]:
        pane.fill = False
        pane.set_edgecolor('#333355')
    ax.grid(True, alpha=0.1)


# ── Figure 3: per-object fit quality bar chart ────────────────────────────────

def fig_fit_quality(result, sq_multis, save_path):
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    fig.patch.set_facecolor('#1a1a2e')

    obj_labels, l2_vals, n_prims, shapes = [], [], [], []
    for i, (obj, multi) in enumerate(zip(result.objects, sq_multis)):
        mean_l2 = np.mean([sq.chamfer_l2 * 1e3 for sq in multi.primitives])
        for j, sq in enumerate(multi.primitives):
            label = f'Obj{obj.id}.{j}'
            obj_labels.append(label)
            l2_vals.append(sq.chamfer_l2 * 1e3)
            n_prims.append(len(multi))
            shape = ("cuboid" if min(sq.e1,sq.e2)<0.3
                     else "ellipsoid" if max(sq.e1,sq.e2)>0.8
                     else "cylinder")
            shapes.append(shape)

    x = np.arange(len(obj_labels))
    colours = [_colour(int(l.split('.')[0].replace('Obj',''))) for l in obj_labels]

    # L2 bar chart
    ax = axes[0]
    ax.set_facecolor('#0f0f23')
    bars = ax.bar(x, l2_vals, color=colours, alpha=0.85, edgecolor='white', lw=0.5)
    ax.axhline(10, color='#e63946', lw=1.5, ls='--', label='Split threshold (10e-3)')
    ax.set_xticks(x); ax.set_xticklabels(obj_labels, rotation=45, ha='right',
                                           fontsize=8, color='white')
    ax.set_ylabel('Chamfer L2 (×10⁻³)', color='white', fontsize=10)
    ax.set_title('Fit Quality per Primitive', color='white', fontsize=11)
    ax.tick_params(colors='white'); ax.legend(fontsize=8)
    ax.spines[['top','right']].set_visible(False)
    for sp in ax.spines.values(): sp.set_color('#333355')

    # Shape type pie
    ax2 = axes[1]
    ax2.set_facecolor('#0f0f23')
    from collections import Counter
    counts = Counter(shapes)
    wedge_colours = {'cuboid': '#e63946', 'ellipsoid': '#457b9d', 'cylinder': '#2a9d8f'}
    ax2.pie(counts.values(), labels=counts.keys(),
            colors=[wedge_colours.get(s, '#8338ec') for s in counts],
            autopct='%1.0f%%', textprops={'color': 'white', 'fontsize': 10},
            wedgeprops={'edgecolor': '#1a1a2e', 'linewidth': 2})
    ax2.set_title('Shape Distribution', color='white', fontsize=11)

    plt.suptitle('Superquadric Fit Analysis', color='white', fontsize=12)
    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='#1a1a2e')
    plt.close()
    print(f"Saved: {save_path}")


# ── Figure 4: SQ parameter space ─────────────────────────────────────────────

def fig_sq_parameter_space(sq_multis, save_path):
    """
    Plot all fitted SQs in (e1, e2) shape space.
    Shows what shape family each object belongs to.
    """
    fig, ax = plt.subplots(figsize=(7, 6))
    fig.patch.set_facecolor('#1a1a2e')
    ax.set_facecolor('#0f0f23')

    # background regions
    ax.fill_between([0.1, 0.5], [0.1, 0.1], [0.5, 0.5],
                    alpha=0.08, color='#e63946', label='Cuboid region')
    ax.fill_between([0.8, 2.0], [0.8, 0.8], [2.0, 2.0],
                    alpha=0.08, color='#457b9d', label='Ellipsoid region')
    ax.axvline(1.0, color='#333355', lw=0.5, ls='--')
    ax.axhline(1.0, color='#333355', lw=0.5, ls='--')

    # annotate corners
    for (xe, ye, label) in [(0.1,0.1,'cuboid'), (1,1,'ellipsoid'),
                             (2,2,'star'), (0.1,2,'disk'), (2,0.1,'spindle')]:
        ax.text(xe, ye, label, color='#aaaaaa', fontsize=8, ha='center')

    for i, multi in enumerate(sq_multis):
        for j, sq in enumerate(multi.primitives):
            ax.scatter(sq.e1, sq.e2, c=_colour(i), s=120,
                       edgecolors='white', lw=1.0, zorder=5)
            ax.annotate(f'  Obj{i}.{j}', (sq.e1, sq.e2),
                        color=_colour(i), fontsize=8)

    ax.set_xlabel('e₁ (vertical roundness)', color='white', fontsize=11)
    ax.set_ylabel('e₂ (horizontal roundness)', color='white', fontsize=11)
    ax.set_xlim(0.05, 2.1); ax.set_ylim(0.05, 2.1)
    ax.set_title('Shape Parameter Space (e₁, e₂)', color='white', fontsize=12)
    ax.tick_params(colors='white')
    ax.legend(fontsize=9, facecolor='#1a1a2e', labelcolor='white')
    for sp in ax.spines.values(): sp.set_color('#333355')

    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='#1a1a2e')
    plt.close()
    print(f"Saved: {save_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ocid',       required=True)
    parser.add_argument('--scene',      type=int, default=10,
                        help='Scene index in subset/table/view sequence')
    parser.add_argument('--subset',     default='ARID20')
    parser.add_argument('--view',       default='top')
    parser.add_argument('--out',        default=None,
                        help='Output directory (default: perception/outputs/)')
    parser.add_argument('--n_restarts', type=int, default=5)
    parser.add_argument('--n_lm',      type=int, default=30)
    parser.add_argument('--saddle',     type=float, default=0.25,
                        help='Saddle depth for cluster splitting (0.15=aggressive, 0.35=conservative)')
    parser.add_argument('--ply_res',    type=int, default=20,
                        help='PLY mesh resolution (higher=smoother, slower)')
    args = parser.parse_args()

    out_dir = Path(args.out) if args.out else (
        Path(__file__).parent / 'outputs')
    out_dir.mkdir(parents=True, exist_ok=True)

    loader = OCIDLoader(args.ocid)
    pipe   = TabletopPerception(saddle_depth=args.saddle, split_min_points=1000)
    fitter = SuperquadricFitter(n_restarts=args.n_restarts, n_lm_rounds=args.n_lm)

    scenes = list(loader.iter_scenes(args.subset, 'table', args.view))
    if args.scene >= len(scenes):
        print(f"Scene index {args.scene} out of range (max {len(scenes)-1})")
        return
    scene = scenes[args.scene]
    print(f"\nVisualizing: {scene.seq_id}/{scene.frame_id}")
    print(f"  GT objects : {scene.n_gt_objects()}")

    depth  = scene.load_depth()
    pts    = scene.depth_to_pointcloud(depth)
    result = pipe.run(pts)
    print(f"  Detections : {len(result.objects)}")

    # fit superquadrics
    sq_multis   = []
    flat_fits   = []   # all primitives flattened — for PLY + Scene
    total_prims = 0
    for obj in result.objects:
        multi = fitter.fit_adaptive(obj.points, l2_threshold=0.010)
        sq_multis.append(multi)
        flat_fits.extend(multi.primitives)
        total_prims += len(multi)
        for j, sq in enumerate(multi.primitives):
            shape = ("cuboid"   if min(sq.e1, sq.e2) < 0.3
                     else "ellipsoid" if max(sq.e1, sq.e2) > 0.8
                     else "cylinder")
            print(f"  Obj {obj.id}.{j}: {shape:<10} "
                  f"scale={sq.scale.round(3)}  "
                  f"e=[{sq.e1:.2f},{sq.e2:.2f}]  "
                  f"L2={sq.chamfer_l2*1e3:.2f}e-3  "
                  f"pts={obj.n_points}  "
                  f"margin={sq.collision_margin*100:.0f}cm")
    print(f"  Total SQ primitives: {total_prims}")

    tag = f"scene{args.scene:03d}"

    # ── Matplotlib figures (same as before) ──────────────────────────────────
    fig_detection_overlay(scene, result, out_dir / f"sq_detection_{tag}.png")
    fig_3d_sq(result, sq_multis, out_dir / f"sq_3d_{tag}.png")
    fig_fit_quality(result, sq_multis, out_dir / f"sq_quality_{tag}.png")
    fig_sq_parameter_space(sq_multis, out_dir / f"sq_params_{tag}.png")

    # ── SUPERDEC PLY mesh export ─────────────────────────────────────────────
    # Save .npz in SUPERDEC format (Person 2 / CuRobo handoff)
    npz_path = out_dir / f"sq_{tag}.npz"
    sq_fits_to_npz(flat_fits, str(npz_path))

    # Export coloured PLY mesh — open in MeshLab for paper figures
    ply_path = out_dir / f"sq_{tag}.ply"
    sq_obj   = Superquadrics.from_fits(flat_fits)
    sq_obj.save_ply(str(ply_path), resolution=args.ply_res)

    # ── Scene SDF demo — what CuRobo uses ────────────────────────────────────
    sq_scene = Scene.from_fits(flat_fits)
    # sample a few query points from the scene and show signed distances
    sample_pts = pts[np.random.choice(len(pts), min(5, len(pts)), replace=False)]
    sd = sq_scene.get_signed_distance(sample_pts)
    print(f"\n  SDF demo (5 random scene points):")
    for i, (p, d) in enumerate(zip(sample_pts, sd)):
        inside = "INSIDE" if d < 0 else "outside"
        print(f"    pt {i}: [{p[0]:.3f},{p[1]:.3f},{p[2]:.3f}]  sd={d:.4f}  ({inside})")

    print(f"\nAll outputs saved to {out_dir}/")
    print(f"  PLY mesh : {ply_path.name}  ← open in MeshLab")
    print(f"  NPZ file : {npz_path.name}  ← share with Person 1 & Person 2")


if __name__ == '__main__':
    main()