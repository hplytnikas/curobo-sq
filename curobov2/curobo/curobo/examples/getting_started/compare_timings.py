"""Compare timing logs produced by motion_planning_sq.py.

Usage
-----
# Compare the two most recent runs (one SQ, one mesh) automatically:
    python compare_timings.py

# Specify files explicitly:
    python compare_timings.py \
        --sq_log  /path/to/superquadrics_20240101_120000.json \
        --mesh_log /path/to/mesh_20240101_120100.json

# Print a table without showing the plot:
    python compare_timings.py --no_plot

Workflow
--------
1. Run motion_planning_sq.py with --world_representation superquadrics --auto_cube_targets '...'
2. Run motion_planning_sq.py with --world_representation mesh         --auto_cube_targets '...'
3. Run this script.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

LOG_DIR = Path("/home/haroldas/3DV/logs/curobov2/timing")


def _latest_for_rep(rep: str) -> Optional[Path]:
    candidates = sorted(LOG_DIR.glob(f"{rep}_*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0] if candidates else None


def _load(path: Path) -> dict:
    with open(path) as fh:
        return json.load(fh)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare SQ vs mesh timing logs")
    parser.add_argument("--sq_log", type=str, default=None, help="Path to superquadrics timing JSON")
    parser.add_argument("--mesh_log", type=str, default=None, help="Path to mesh timing JSON")
    parser.add_argument("--no_plot", action="store_true", help="Print table only, skip matplotlib plot")
    return parser.parse_args()


def _fmt(v) -> str:
    if v is None:
        return "N/A"
    return f"{float(v):.4f}s"


def _print_summary(sq: dict, mesh: dict) -> None:
    keys = [
        ("superdec_inference_s", "SuperDec inference"),
        ("scene_build_s",        "Scene build"),
        ("planner_init_s",       "Planner init"),
        ("warmup_s",             "Warmup"),
        ("plan_mean_s",          "Plan mean (auto targets)"),
        ("plan_min_s",           "Plan min  (auto targets)"),
        ("plan_max_s",           "Plan max  (auto targets)"),
        ("plan_total_s",         "Plan total (auto targets)"),
        ("pose_plan_s",          "Pose plan (single)"),
        ("grasp_plan_s",         "Grasp plan (single)"),
        ("total_s",              "Total wall time"),
    ]

    col_w = 32
    print()
    print(f"{'Metric':<{col_w}}  {'Superquadrics':>16}  {'Mesh':>16}  {'Ratio (SQ/mesh)':>16}")
    print("-" * (col_w + 55))
    for key, label in keys:
        sq_v = sq.get(key)
        mesh_v = mesh.get(key)
        if sq_v is None and mesh_v is None:
            continue
        try:
            ratio = f"{float(sq_v) / float(mesh_v):.3f}x" if sq_v is not None and mesh_v not in (None, 0) else "N/A"
        except (TypeError, ZeroDivisionError):
            ratio = "N/A"
        print(f"{label:<{col_w}}  {_fmt(sq_v):>16}  {_fmt(mesh_v):>16}  {ratio:>16}")
    print()

    # Per-target breakdown
    sq_plans = sq.get("plans", [])
    mesh_plans = mesh.get("plans", [])
    if sq_plans or mesh_plans:
        print(f"{'Target':>3}  {'SQ plan':>12}  {'SQ ok':>6}  {'Mesh plan':>12}  {'Mesh ok':>6}  {'Ratio':>8}")
        print("-" * 58)
        n = max(len(sq_plans), len(mesh_plans))
        for i in range(n):
            sq_r = sq_plans[i] if i < len(sq_plans) else {}
            mesh_r = mesh_plans[i] if i < len(mesh_plans) else {}
            sq_t = sq_r.get("plan_s")
            mesh_t = mesh_r.get("plan_s")
            try:
                ratio = f"{float(sq_t) / float(mesh_t):.3f}x" if sq_t is not None and mesh_t not in (None, 0) else "N/A"
            except (TypeError, ZeroDivisionError):
                ratio = "N/A"
            xyz = sq_r.get("target_xyz") or mesh_r.get("target_xyz") or "?"
            sq_ok = "Y" if sq_r.get("success") else "N"
            mesh_ok = "Y" if mesh_r.get("success") else "N"
            print(
                f"{i:>3}  {_fmt(sq_t):>12}  {sq_ok:>6}  {_fmt(mesh_t):>12}  {mesh_ok:>6}  {ratio:>8}"
                f"  {xyz}"
            )
        print()


def _plot(sq: dict, mesh: dict) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("matplotlib not available — skipping plot")
        return

    sq_plans = sq.get("plans", [])
    mesh_plans = mesh.get("plans", [])
    n = max(len(sq_plans), len(mesh_plans))

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("SQ vs Mesh planning times", fontsize=13)

    # --- Left: per-target bar chart ---
    ax = axes[0]
    if n > 0:
        x = np.arange(n)
        sq_t = [sq_plans[i]["plan_s"] if i < len(sq_plans) else float("nan") for i in range(n)]
        mesh_t = [mesh_plans[i]["plan_s"] if i < len(mesh_plans) else float("nan") for i in range(n)]
        w = 0.35
        ax.bar(x - w / 2, sq_t, w, label="Superquadrics", color="#2196F3")
        ax.bar(x + w / 2, mesh_t, w, label="Mesh", color="#FF5722")
        ax.set_xlabel("Target index")
        ax.set_ylabel("Planning time (s)")
        ax.set_title("Per-target planning time")
        ax.set_xticks(x)
        ax.legend()
    else:
        ax.text(0.5, 0.5, "No per-target data", ha="center", va="center", transform=ax.transAxes)
        ax.set_title("Per-target planning time")

    # --- Right: overall breakdown bar chart ---
    ax2 = axes[1]
    phase_keys = [
        ("superdec_inference_s", "SuperDec\ninference"),
        ("scene_build_s",        "Scene\nbuild"),
        ("planner_init_s",       "Planner\ninit"),
        ("warmup_s",             "Warmup"),
        ("plan_total_s",         "Total\nplanning"),
    ]
    labels = []
    sq_vals: list[float] = []
    mesh_vals: list[float] = []
    for key, lbl in phase_keys:
        sv = sq.get(key)
        mv = mesh.get(key)
        if sv is not None or mv is not None:
            labels.append(lbl)
            sq_vals.append(float(sv) if sv is not None else 0.0)
            mesh_vals.append(float(mv) if mv is not None else 0.0)

    if labels:
        x2 = np.arange(len(labels))
        w = 0.35
        ax2.bar(x2 - w / 2, sq_vals, w, label="Superquadrics", color="#2196F3")
        ax2.bar(x2 + w / 2, mesh_vals, w, label="Mesh", color="#FF5722")
        ax2.set_xticks(x2)
        ax2.set_xticklabels(labels, fontsize=8)
        ax2.set_ylabel("Time (s)")
        ax2.set_title("Pipeline phase breakdown")
        ax2.legend()

    plt.tight_layout()
    out_path = LOG_DIR / "comparison.png"
    plt.savefig(out_path, dpi=150)
    print(f"Plot saved → {out_path}")
    try:
        plt.show()
    except Exception:
        pass


def main() -> None:
    args = _parse_args()

    sq_path = Path(args.sq_log) if args.sq_log else _latest_for_rep("superquadrics")
    mesh_path = Path(args.mesh_log) if args.mesh_log else _latest_for_rep("mesh")

    if sq_path is None or not sq_path.exists():
        print(f"No superquadrics log found in {LOG_DIR}. Run motion_planning_sq.py first.", file=sys.stderr)
        sys.exit(1)
    if mesh_path is None or not mesh_path.exists():
        print(f"No mesh log found in {LOG_DIR}. Run motion_planning_sq.py with --world_representation mesh first.", file=sys.stderr)
        sys.exit(1)

    print(f"SQ log:   {sq_path}")
    print(f"Mesh log: {mesh_path}")

    sq = _load(sq_path)
    mesh = _load(mesh_path)

    _print_summary(sq, mesh)

    if not args.no_plot:
        _plot(sq, mesh)


if __name__ == "__main__":
    main()
