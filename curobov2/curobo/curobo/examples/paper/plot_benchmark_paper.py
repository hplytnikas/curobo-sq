"""Paper figures for the SQ-vs-mesh benchmark (clean, two-panel style).

Reads ``eval_out/results.csv`` and writes two figures:

  1. paper_planning_time.png  — total planning time per representation: ONE line per
     representation (mean over the 4 legs of each scene) with the std shown as a
     shaded band.  Uses ``plan_wall_s`` (true end-to-end planning time).

  2. paper_collisions_bar.png — ground-truth point-cloud collisions: a grouped bar
     chart, one group per scene, one bar per representation.

Notes on the two planning-time columns in the CSV:
  * plan_wall_s          end-to-end wall-clock around plan_pose (CUDA-synced) — the
                         real planning time, used here.
  * plan_solver_total_s  curobo's internally reported solve time (result.total_time,
                         IK + trajopt across attempts); excludes Python/launch overhead.

Run::

    conda run -n 3dv python plot_benchmark_paper.py
    conda run -n 3dv python plot_benchmark_paper.py --x objects   # x-axis = #objects
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

OUT_DIR = Path(__file__).resolve().parent / "eval_out"
RESULTS_CSV = OUT_DIR / "results.csv"

# Plot order + styling (only representations present in the CSV are drawn).
REP_ORDER = ["superquadrics", "mesh", "pointcloud", "shapenet_mesh"]
REP_STYLE = {
    "superquadrics": dict(color="tab:blue", label="superquadrics (SDF)"),
    "mesh": dict(color="tab:red", label="mesh (SQ tessellation)"),
    "pointcloud": dict(color="tab:green", label="point cloud (voxel-surface mesh)"),
    "shapenet_mesh": dict(color="tab:purple", label="mesh (original ShapeNet)"),
}

# Which collision count to show in the bar chart: number of trajectory waypoints
# (robot configurations) that were in true collision, summed over the scene's legs.
COLLISION_COUNT_COL = "n_steps_collide"


def _present_reps(df: pd.DataFrame) -> list:
    return [r for r in REP_ORDER if r in set(df["representation"].unique())]


def _scene_axis(df: pd.DataFrame) -> Tuple[list, dict, list]:
    """Return (sorted object counts, {n_objects: categorical index}, tick labels).

    Tick labels show object count with the superquadric primitive count in brackets,
    e.g. ``50\\n(311)``.
    """
    objs = sorted(df["n_objects"].unique())
    idx = {o: i for i, o in enumerate(objs)}
    labels = []
    for o in objs:
        sub = df[df["n_objects"] == o]
        sq = sub[sub["representation"] == "superquadrics"]["n_primitives"]
        prim = int(sq.iloc[0]) if len(sq) else int(sub["n_primitives"].max())
        labels.append(f"{o}\n({prim})")
    return objs, idx, labels


def plot_planning_time(df: pd.DataFrame) -> None:
    """One line per representation: mean planning time, std as a shaded band.

    x-axis = number of objects (with #superquadric-primitives in brackets).
    """
    ok = df[df["plan_success"] == 1]
    objs, idx, labels = _scene_axis(df)
    fig, ax = plt.subplots(figsize=(max(7, 1.1 * len(objs)), 5))
    for rep in _present_reps(df):
        sub = (
            ok[ok["representation"] == rep]
            .groupby("n_objects")["plan_wall_s"]
            .agg(["mean", "std"])
            .reset_index()
            .sort_values("n_objects")
        )
        if sub.empty:
            continue
        x = np.array([idx[o] for o in sub["n_objects"]])
        mean = sub["mean"].to_numpy()
        std = sub["std"].fillna(0.0).to_numpy()
        color = REP_STYLE[rep]["color"]
        ax.plot(x, mean, marker="o", color=color, label=REP_STYLE[rep]["label"])
        ax.fill_between(x, mean - std, mean + std, color=color, alpha=0.20)

    ax.set_xlabel("number of objects (number of superquadric primitives)")
    ax.set_ylabel("planning time per leg [s]")
    ax.set_title("Planning time (mean ± std over legs)")
    ax.set_xticks(np.arange(len(objs)))
    ax.set_xticklabels(labels)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    out = OUT_DIR / "paper_planning_time.png"
    fig.savefig(out, dpi=150)
    print(f"wrote {out}")


def plot_collisions_bar(df: pd.DataFrame) -> None:
    """Grouped bar chart: ground-truth collisions per scene, one bar per representation."""
    reps = _present_reps(df)
    objs, idx, labels = _scene_axis(df)
    # total collisions per (scene, representation), summed over legs
    agg = (
        df.groupby(["n_objects", "representation"])[COLLISION_COUNT_COL]
        .sum()
        .reset_index()
    )
    pivot = agg.pivot(index="n_objects", columns="representation", values=COLLISION_COUNT_COL)
    pivot = pivot.reindex(index=objs)

    fig, ax = plt.subplots(figsize=(max(7, 1.2 * len(objs)), 5))
    n_reps = len(reps)
    width = 0.8 / max(n_reps, 1)
    x = np.arange(len(objs))
    for i, rep in enumerate(reps):
        vals = pivot[rep].to_numpy() if rep in pivot.columns else np.zeros(len(objs))
        vals = np.nan_to_num(vals)
        offset = (i - (n_reps - 1) / 2.0) * width
        ax.bar(x + offset, vals, width=width,
               color=REP_STYLE[rep]["color"], label=REP_STYLE[rep]["label"])

    ax.set_xlabel("number of objects (number of superquadric primitives)")
    ax.set_ylabel("# trajectory waypoints in true collision\n(summed over 4 legs)")
    ax.set_title("Ground-truth point-cloud collisions per scene")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    out = OUT_DIR / "paper_collisions_bar.png"
    fig.savefig(out, dpi=150)
    print(f"wrote {out}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--csv", type=str, default=str(RESULTS_CSV))
    args = parser.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.exists():
        raise SystemExit(f"No results at {csv_path}. Run the benchmark first.")
    df = pd.read_csv(csv_path)

    plot_planning_time(df)
    plot_collisions_bar(df)


if __name__ == "__main__":
    main()
