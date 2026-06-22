"""Plot the mesh-fidelity sweep produced by ``benchmark_sq_vs_mesh.py benchmark --fidelity``.

Reads ``eval_out/results_fidelity.csv`` (analytic superquadric baseline rows plus mesh
rows at increasing tessellation resolutions) and writes two figures plus a summary:

  * fidelity_vs_planning_time.png — planning time vs. mesh fidelity: the analytic SQ
                                    baseline is flat, the mesh cost rises with fidelity.
  * fidelity_vs_speedup.png       — speedup (mesh planning time / SQ planning time) vs.
                                    mesh fidelity; the hypothesis is that this grows.

The speedup is computed per (scene, leg) by dividing the mesh planning time at each
fidelity level by that scene's superquadric baseline, then averaged across scenes/legs.

Run::

    conda run -n 3dv python plot_fidelity.py
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
RESULTS_CSV = OUT_DIR / "results_fidelity.csv"

# Which planning-time column to use as the cost metric.
TIME_COL = "plan_wall_s"
# Tessellation resolution N -> triangles per superquadric (~2*(N-1)**2), for annotation.
X_COL = "mesh_resolution"
X_LABEL = "mesh tessellation resolution N  (triangles/SQ ~ 2(N-1)²)"


def _load(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df = df[df["plan_success"] == 1].copy()
    if df.empty:
        raise SystemExit("No successful legs in the fidelity results.")
    return df


def _baseline_per_scene(df: pd.DataFrame) -> pd.Series:
    """Mean superquadric planning time per scene (the fidelity-independent baseline)."""
    sq = df[df["representation"] == "superquadrics"]
    if sq.empty:
        raise SystemExit("No superquadric baseline rows found.")
    return sq.groupby("scene")[TIME_COL].mean()


def _mesh_with_speedup(df: pd.DataFrame, baseline: pd.Series) -> pd.DataFrame:
    """Mesh rows annotated with per-scene SQ baseline and speedup (mesh / SQ)."""
    mesh = df[df["representation"] == "mesh"].copy()
    if mesh.empty:
        raise SystemExit("No mesh rows found.")
    mesh["sq_baseline_s"] = mesh["scene"].map(baseline)
    mesh = mesh.dropna(subset=["sq_baseline_s"])
    mesh["speedup"] = mesh[TIME_COL] / mesh["sq_baseline_s"]
    return mesh


def plot_planning_time(df: pd.DataFrame, baseline: pd.Series, mesh: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))

    # Mesh planning time vs. fidelity (mean +/- std across scenes/legs).
    g = mesh.groupby(X_COL)[TIME_COL].agg(["mean", "std", "count"]).reset_index()
    ax.errorbar(g[X_COL], g["mean"], yerr=g["std"].fillna(0.0), capsize=3,
                color="tab:red", marker="s", label="mesh (SQ tessellation)")

    # Analytic SQ baseline: a single fidelity-independent value (horizontal line + band).
    sq_mean = float(baseline.mean())
    sq_std = float(baseline.std(ddof=0)) if len(baseline) > 1 else 0.0
    ax.axhline(sq_mean, color="tab:blue", linestyle="--", label="superquadrics (SDF, analytic)")
    if sq_std > 0.0:
        ax.axhspan(sq_mean - sq_std, sq_mean + sq_std, color="tab:blue", alpha=0.12)

    ax.set_xlabel(X_LABEL)
    ax.set_ylabel("planning time per leg [s]")
    ax.set_title("Planning cost vs. mesh fidelity")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    out = OUT_DIR / "fidelity_vs_planning_time.png"
    fig.savefig(out, dpi=150)
    print(f"wrote {out}")


def plot_speedup(mesh: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    g = mesh.groupby(X_COL)["speedup"].agg(["mean", "std", "count"]).reset_index()
    ax.errorbar(g[X_COL], g["mean"], yerr=g["std"].fillna(0.0), capsize=3,
                color="tab:purple", marker="o", label="mesh / superquadrics")
    ax.axhline(1.0, color="gray", linestyle=":", label="parity (equal cost)")

    ax.set_xlabel(X_LABEL)
    ax.set_ylabel("planning-time ratio  (mesh / superquadric)")
    ax.set_title("Superquadric speedup grows with mesh fidelity")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    out = OUT_DIR / "fidelity_vs_speedup.png"
    fig.savefig(out, dpi=150)
    print(f"wrote {out}")


def print_summary(baseline: pd.Series, mesh: pd.DataFrame) -> None:
    summary = mesh.groupby(X_COL).agg(
        n_triangles=("n_triangles", "mean"),
        mesh_plan_s=(TIME_COL, "mean"),
        sq_plan_s=("sq_baseline_s", "mean"),
        speedup=("speedup", "mean"),
        speedup_std=("speedup", "std"),
        n_legs=("leg", "count"),
    )
    pd.set_option("display.width", 160)
    pd.set_option("display.max_columns", 20)
    print(f"\nSuperquadric baseline (mean over scenes/legs): {baseline.mean():.4f} s")
    print("\n=== Mesh fidelity sweep (means over successful legs) ===")
    print(summary.round(4))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--csv", type=str, default=str(RESULTS_CSV))
    args = parser.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.exists():
        raise SystemExit(
            f"No fidelity results at {csv_path}. Run "
            "`benchmark_sq_vs_mesh.py benchmark --fidelity` first."
        )

    df = _load(csv_path)
    baseline = _baseline_per_scene(df)
    mesh = _mesh_with_speedup(df, baseline)

    plot_planning_time(df, baseline, mesh)
    plot_speedup(mesh)
    print_summary(baseline, mesh)


if __name__ == "__main__":
    main()
