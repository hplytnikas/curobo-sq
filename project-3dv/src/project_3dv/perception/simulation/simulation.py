"""
simulation.py

Visualises the tabletop perception pipeline output on real OCID data.
No VLM/CLIP — pure geometry: depth → point cloud → segmented objects.

Run:
    python3 -m project_3dv.perception.simulation --ocid /Volumes/T7/OCID-dataset
    python3 -m project_3dv.perception.simulation --synthetic
"""

from __future__ import annotations
import argparse, os, sys, warnings
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pipeline import TabletopPerception, PerceptionResult

# consistent colours per object id
OBJ_COLOURS = ['#e63946','#457b9d','#2a9d8f','#e76f51','#8338ec',
                '#fb8500','#06d6a0','#264653','#f4a261','#a8dadc']

def obj_colour(i): return OBJ_COLOURS[i % len(OBJ_COLOURS)]


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def plot_results(result: PerceptionResult, output_dir: str, scene_id: str = ""):
    os.makedirs(output_dir, exist_ok=True)
    fig1 = _plot_topdown(result, scene_id)
    fig2 = _plot_object_extents(result, scene_id)
    fig3 = _plot_pipeline_diagram()

    fig1.savefig(f"{output_dir}/fig1_segmentation.png",  dpi=180, bbox_inches='tight')
    fig2.savefig(f"{output_dir}/fig2_object_extents.png", dpi=180, bbox_inches='tight')
    fig3.savefig(f"{output_dir}/fig3_pipeline.png",       dpi=180, bbox_inches='tight')
    plt.close('all')
    print(f"  Saved 3 figures to {output_dir}/")


def _plot_topdown(result: PerceptionResult, scene_id: str):
    """Top-down XZ view of detected objects (X = right, Z = depth into scene)."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    fig.patch.set_facecolor('#f8f9fa')

    # left: XZ scatter (top-down)
    ax = axes[0]
    ax.set_facecolor('#f0f0f0')
    for i, obj in enumerate(result.objects):
        col = obj_colour(i)
        pts = obj.points
        ax.scatter(pts[:, 0], pts[:, 2], s=1, c=col, alpha=0.4, rasterized=True)
        cx, cz = obj.centroid[0], obj.centroid[2]
        ax.scatter(cx, cz, s=80, c=col, edgecolors='white', linewidths=1.2, zorder=5)
        ext = obj.bbox_extent
        ax.text(cx + 0.01, cz + 0.01,
                f"obj {i}\n{ext[0]:.2f}×{ext[2]:.2f}m",
                fontsize=7, color=col)
    ax.set_xlabel("X (m)", fontsize=9)
    ax.set_ylabel("Z / depth (m)", fontsize=9)
    ax.set_title(f"Top-down view — {len(result.objects)} object(s) detected", fontsize=10)
    ax.grid(alpha=0.3, linestyle='--')
    ax.set_aspect('equal')

    # right: XY scatter (front view, height on Y)
    ax2 = axes[1]
    ax2.set_facecolor('#f0f0f0')
    for i, obj in enumerate(result.objects):
        col = obj_colour(i)
        pts = obj.points
        ax2.scatter(pts[:, 0], pts[:, 1], s=1, c=col, alpha=0.4, rasterized=True)
        cx, cy = obj.centroid[0], obj.centroid[1]
        ax2.scatter(cx, cy, s=80, c=col, edgecolors='white', linewidths=1.2, zorder=5)
    ax2.set_xlabel("X (m)", fontsize=9)
    ax2.set_ylabel("Y (m)", fontsize=9)
    ax2.set_title("Front view (height)", fontsize=10)
    ax2.grid(alpha=0.3, linestyle='--')
    ax2.set_aspect('equal')

    title = f"Tabletop Perception — OCID {scene_id}" if scene_id else "Tabletop Perception"
    plt.suptitle(title, fontsize=12, y=1.01)
    plt.tight_layout()
    return fig


def _plot_object_extents(result: PerceptionResult, scene_id: str):
    """Bar chart: per-object bounding box extents (dx, dy, dz)."""
    fig, ax = plt.subplots(figsize=(max(6, len(result.objects)*1.5 + 2), 4))
    fig.patch.set_facecolor('#f8f9fa')

    if not result.objects:
        ax.text(0.5, 0.5, "No objects detected", ha='center', va='center',
                transform=ax.transAxes, fontsize=12, color='gray')
        ax.set_title("Object Extents", fontsize=11)
        return fig

    n   = len(result.objects)
    x   = np.arange(n)
    w   = 0.25
    dxs = [o.bbox_extent[0] for o in result.objects]
    dys = [o.bbox_extent[1] for o in result.objects]
    dzs = [o.bbox_extent[2] for o in result.objects]

    ax.bar(x - w, dxs, w, label='dx (width)',  color='#457b9d', alpha=0.85)
    ax.bar(x,     dys, w, label='dy (height)', color='#2a9d8f', alpha=0.85)
    ax.bar(x + w, dzs, w, label='dz (depth)',  color='#e63946', alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels([f"obj {o.id}\n({o.n_points} pts)" for o in result.objects], fontsize=8)
    ax.set_ylabel("Extent (m)", fontsize=9)
    ax.set_title("Bounding Box Extents per Detected Object\n"
                 "(input to superquadric fitter for scale initialisation)", fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(axis='y', alpha=0.3)
    ax.spines[['top','right']].set_visible(False)
    plt.tight_layout()
    return fig


def _plot_pipeline_diagram():
    """System architecture figure."""
    fig, ax = plt.subplots(figsize=(12, 3))
    ax.set_xlim(0, 12); ax.set_ylim(0, 3); ax.axis('off')
    fig.patch.set_facecolor('#f8f9fa')

    boxes = [
        (1.0,  "RGB-D\nCamera",       "#dee2e6"),
        (3.2,  "Tabletop\nPerception", "#adb5bd"),
        (5.4,  "SQ Primitive\nFitter", "#ffa94d"),
        (7.6,  "CuRobo\nGPU Planner", "#da77f2"),
        (9.8,  "Robot\nExecution",    "#a9e34b"),
    ]
    arrow_labels = [
        (2.1, "point cloud"),
        (4.3, "segments\n+ bbox"),
        (6.5, "SQ params\n+ SDFs"),
        (8.7, "trajectory"),
    ]

    for i, (x, label, col) in enumerate(boxes):
        ax.add_patch(mpatches.FancyBboxPatch(
            (x-0.9, 0.8), 1.8, 1.4, boxstyle="round,pad=0.08",
            facecolor=col, edgecolor='#343a40', linewidth=1.5, zorder=2))
        ax.text(x, 1.5, label, ha='center', va='center',
                fontsize=9, fontweight='bold', zorder=3)
        if i < len(boxes)-1:
            ax.annotate("", xy=(boxes[i+1][0]-0.9, 1.5), xytext=(x+0.9, 1.5),
                        arrowprops=dict(arrowstyle="-|>", lw=1.5, color='#343a40'), zorder=4)

    for x, lbl in arrow_labels:
        ax.text(x, 0.55, lbl, ha='center', va='top', fontsize=7.5,
                color='#495057', style='italic')

    # bracket showing "this work"
    ax.annotate("", xy=(4.3, 0.2), xytext=(2.1, 0.2),
                arrowprops=dict(arrowstyle="-", lw=1.5, color='#e63946'))
    ax.text(3.2, 0.05, "← this work →", ha='center', fontsize=8, color='#e63946')

    ax.set_title("System Architecture", fontsize=11, pad=6)
    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Synthetic scene
# ---------------------------------------------------------------------------

def run_synthetic(output_dir: str):
    from project_3dv.perception.simulation.synthetic_scene import generate_tabletop_scene
    print("Generating synthetic scene...")
    pts, labels_gt = generate_tabletop_scene(n_objects=4, noise_std=0.003)
    pipe   = TabletopPerception()
    result = pipe.run(pts)
    print(result.summary())
    plot_results(result, output_dir, scene_id="synthetic")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ocid",      default=None)
    parser.add_argument("--synthetic", action="store_true")
    parser.add_argument("--subset",    default="ARID20")
    parser.add_argument("--view",      default="top")
    parser.add_argument("--out", default=os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "outputs"))
    args = parser.parse_args()

    if args.ocid and not args.synthetic:
        print("OCID mode...")
        from project_3dv.perception.evaluation.ocid_loader import OCIDLoader
        loader = OCIDLoader(args.ocid)
        scene  = next(loader.iter_scenes(args.subset, "table", args.view, min_objects=5))
        print(f"Using scene: {scene.seq_id}/{scene.frame_id}")

        depth  = scene.load_depth()
        pts    = scene.depth_to_pointcloud(depth)
        pipe   = TabletopPerception()
        result = pipe.run(pts)
        print(result.summary())
        plot_results(result, args.out, scene_id=f"{scene.seq_id}/{scene.frame_id}")
    else:
        run_synthetic(args.out)