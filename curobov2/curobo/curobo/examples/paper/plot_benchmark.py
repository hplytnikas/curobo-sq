"""Plot the SQ-vs-mesh benchmark results produced by ``benchmark_sq_vs_mesh.py``.

Reads ``eval_out/results.csv`` and writes three comparison figures plus a printed
summary table:

  * objects_vs_planning_time.png  — planning time (wall + solver) vs. #objects
  * objects_vs_motion_time.png    — time-optimal motion time & playback vs. #objects
  * objects_vs_collision.png      — ground-truth collision rate / penetration vs. #objects

Run::

    conda run -n 3dv python plot_benchmark.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

OUT_DIR = Path(__file__).resolve().parent / "eval_out"
RESULTS_CSV = OUT_DIR / "results.csv"

REP_STYLE = {
    "superquadrics": dict(color="tab:blue", marker="o", label="superquadrics (SDF)"),
    "mesh": dict(color="tab:red", marker="s", label="mesh (SQ tessellation)"),
    "pointcloud": dict(color="tab:green", marker="^", label="point cloud (voxel-surface mesh)"),
    "shapenet_mesh": dict(color="tab:purple", marker="D", label="mesh (original ShapeNet)"),
}

# Plot against number of collision primitives (per-SQ for sq/mesh, per-object for
# shapenet_mesh) rather than number of objects.
X_COL = "n_primitives"
X_LABEL = "number of collision primitives in scene"


def _agg(df: pd.DataFrame, value: str) -> pd.DataFrame:
    """Mean + std of ``value`` per (X_COL, representation), successful legs only."""
    ok = df[df["plan_success"] == 1]
    g = ok.groupby(["representation", X_COL])[value]
    return g.agg(["mean", "std", "count"]).reset_index()


def _line(ax, df, value, rep, **extra):
    sub = _agg(df, value)
    sub = sub[sub["representation"] == rep].sort_values(X_COL)
    if sub.empty:
        return
    style = dict(REP_STYLE[rep])
    style.update(extra)
    ax.errorbar(sub[X_COL], sub["mean"], yerr=sub["std"].fillna(0.0),
                capsize=3, **style)


def plot_planning_time(df: pd.DataFrame, log_x: bool) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    for rep in REP_STYLE:
        _line(ax, df, "plan_wall_s", rep)
        _line(ax, df, "plan_solver_total_s", rep,
              linestyle="--", marker="x", label=f"{rep} (solver)")
    ax.set_xlabel(X_LABEL)
    ax.set_ylabel("planning time per leg [s]")
    ax.set_title("Planning time: superquadrics vs. mesh")
    if log_x:
        ax.set_xscale("log")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    out = OUT_DIR / "objects_vs_planning_time.png"
    fig.savefig(out, dpi=150)
    print(f"wrote {out}")


def plot_motion_time(df: pd.DataFrame, log_x: bool) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    for rep in REP_STYLE:
        _line(ax, df, "motion_time_s", rep)
        _line(ax, df, "playback_s", rep,
              linestyle="--", marker="x", label=f"{rep} (playback)")
    ax.set_xlabel(X_LABEL)
    ax.set_ylabel("trajectory time per leg [s]")
    ax.set_title("Motion time (time-optimal) & playback: superquadrics vs. mesh")
    if log_x:
        ax.set_xscale("log")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    out = OUT_DIR / "objects_vs_motion_time.png"
    fig.savefig(out, dpi=150)
    print(f"wrote {out}")


def plot_collision(df: pd.DataFrame, log_x: bool) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    for rep in REP_STYLE:
        _line(ax1, df, "frac_in_collision", rep)
        _line(ax2, df, "max_penetration_m", rep)
    ax1.set_xlabel(X_LABEL)
    ax1.set_ylabel("fraction of trajectory in true collision")
    ax1.set_title("Ground-truth collision rate")
    ax2.set_xlabel(X_LABEL)
    ax2.set_ylabel("max penetration into point cloud [m]")
    ax2.set_title("Ground-truth penetration depth")
    for ax in (ax1, ax2):
        if log_x:
            ax.set_xscale("log")
        ax.grid(True, alpha=0.3)
        ax.legend()
    fig.tight_layout()
    out = OUT_DIR / "objects_vs_collision.png"
    fig.savefig(out, dpi=150)
    print(f"wrote {out}")


def print_summary(df: pd.DataFrame) -> None:
    ok = df[df["plan_success"] == 1]
    summary = ok.groupby(["n_objects", "representation"]).agg(
        plan_wall_s=("plan_wall_s", "mean"),
        motion_time_s=("motion_time_s", "mean"),
        frac_in_collision=("frac_in_collision", "mean"),
        max_penetration_m=("max_penetration_m", "max"),
        success_rate=("plan_success", "mean"),
        n_legs=("leg", "count"),
    )
    # success rate counted over all legs (including failures)
    total = df.groupby(["n_objects", "representation"])["plan_success"].mean()
    summary["success_rate"] = total
    pd.set_option("display.width", 160)
    pd.set_option("display.max_columns", 20)
    print("\n=== Summary (means over successful legs; success_rate over all legs) ===")
    print(summary.round(4))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--csv", type=str, default=str(RESULTS_CSV))
    parser.add_argument("--linear-x", action="store_true",
                        help="Use a linear x-axis (default is log).")
    args = parser.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.exists():
        raise SystemExit(f"No results at {csv_path}. Run the benchmark first.")
    df = pd.read_csv(csv_path)
    log_x = not args.linear_x

    plot_planning_time(df, log_x)
    plot_motion_time(df, log_x)
    plot_collision(df, log_x)
    print_summary(df)


if __name__ == "__main__":
    main()
