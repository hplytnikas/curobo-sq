"""Object-size scaling figure: planning time vs. number of primitives.

Reads the sofa benchmark CSV (``eval_out/results_sofa.csv`` by default) and
plots planning time against the number of collision primitives in the scene,
with one line per representation (superquadrics vs. mesh).  This is the figure
that shows how the analytic superquadric SDF scales versus the mesh
tessellation as the scene grows.

Kept separate from ``plot_benchmark_paper.py`` (whose x-axis is the number of
objects) so the scaling experiment can plot against primitive count directly.

Run::

    conda run -n 3dv python plot_sofa_scaling.py
    conda run -n 3dv python plot_sofa_scaling.py --csv eval_out/results_sofa.csv
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
RESULTS_SOFA_CSV = OUT_DIR / "results_sofa.csv"

# Only the two representations the sofa benchmark runs; drawn in this order.
REP_ORDER = ["superquadrics", "mesh"]
REP_STYLE = {
    "superquadrics": dict(color="tab:blue", label="superquadrics (SDF)"),
    "mesh": dict(color="tab:red", label="mesh (SQ tessellation)"),
}


def plot_time_vs_primitives(df: pd.DataFrame, out: Path) -> None:
    """One line per representation: mean planning time vs #primitives, std band."""
    ok = df[df["plan_success"] == 1]
    if ok.empty:
        ok = df  # nothing succeeded — fall back so the figure is still drawn

    fig, ax = plt.subplots(figsize=(7, 5))
    for rep in REP_ORDER:
        sub = (
            ok[ok["representation"] == rep]
            .groupby("n_primitives")["plan_wall_s"]
            .agg(["mean", "std"])
            .reset_index()
            .sort_values("n_primitives")
        )
        if sub.empty:
            continue
        x = sub["n_primitives"].to_numpy()
        mean = sub["mean"].to_numpy()
        std = sub["std"].fillna(0.0).to_numpy()
        color = REP_STYLE[rep]["color"]
        ax.plot(x, mean, marker="o", color=color, label=REP_STYLE[rep]["label"])
        ax.fill_between(x, mean - std, mean + std, color=color, alpha=0.20)

    ax.set_xlabel("number of collision primitives")
    ax.set_ylabel("planning time per target [s]")
    ax.set_title("Planning time vs. scene complexity (mean ± std over targets)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    print(f"wrote {out}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--csv", type=str, default=str(RESULTS_SOFA_CSV))
    parser.add_argument("--out", type=str, default=str(OUT_DIR / "sofa_time_vs_primitives.png"))
    args = parser.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.exists():
        raise SystemExit(f"No results at {csv_path}. Run benchmark_sofa_scaling.py first.")
    df = pd.read_csv(csv_path)

    plot_time_vs_primitives(df, Path(args.out))


if __name__ == "__main__":
    main()
