"""Rebalance the pseudo-GT filter: keep top X% of each category by IoU
(instead of a global IoU threshold that biases toward easy categories).

Reads:
    data/output_npz/exp_tt/iou/train_metrics.csv
Writes (next to the npz so the dataloader auto-finds it via _load_filter):
    data/output_npz/exp_tt/iou_balanced/train.npz          (symlink to original)
    data/output_npz/exp_tt/iou_balanced/train_metrics.csv  (per-category top-X%)

Usage:
    python scripts/rebalance_pseudo_gt.py --top_frac 0.5

Then point the student config's gt_train_path at the new dir.
"""
import argparse
import csv
import os
from collections import defaultdict


CATS = {
    "02876657": "bottle", "02880940": "bowl", "03624134": "knife",
    "03642806": "laptop", "03797390": "mug", "gso": "gso",
}


def category_for(name, ids_by_cat):
    for cat, ids in ids_by_cat.items():
        if name in ids:
            return cat
    return "unknown"


def load_split_ids(data_root, split):
    out = {}
    for cat in CATS:
        path = os.path.join(data_root, cat, f"{split}.lst")
        if os.path.exists(path):
            out[cat] = set(l.strip() for l in open(path) if l.strip())
    return out


def rebalance(metrics_csv_in, metrics_csv_out, ids_by_cat, top_frac, min_iou):
    rows_by_cat = defaultdict(list)
    with open(metrics_csv_in) as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        for row in reader:
            cat = category_for(row["name"], ids_by_cat)
            rows_by_cat[cat].append(row)

    keep = []
    print(f"{'cat':<8} {'n':>5} {'kept':>5} {'kept%':>7} {'min_iou_kept':>14}")
    for cat in sorted(rows_by_cat):
        rows = rows_by_cat[cat]
        rows.sort(key=lambda r: float(r["iou"]), reverse=True)
        # Top-X% but enforce floor on absolute IoU
        n_top = max(1, int(round(len(rows) * top_frac)))
        cand = rows[:n_top]
        cand = [r for r in cand if float(r["iou"]) >= min_iou]
        keep += cand
        cname = CATS.get(cat, cat)
        if cand:
            min_kept = min(float(r["iou"]) for r in cand)
        else:
            min_kept = float("nan")
        print(f"{cname:<8} {len(rows):>5} {len(cand):>5} "
              f"{100.0 * len(cand)/max(len(rows),1):>6.1f}% "
              f"{min_kept:>14.3f}")

    os.makedirs(os.path.dirname(metrics_csv_out), exist_ok=True)
    with open(metrics_csv_out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in keep:
            w.writerow(r)
    print(f"\nWrote {len(keep)} rows -> {metrics_csv_out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", default="data/ShapeNet")
    ap.add_argument("--src_dir", default="data/output_npz/exp_tt/iou")
    ap.add_argument("--dst_dir", default="data/output_npz/exp_tt/iou_balanced")
    ap.add_argument("--top_frac", type=float, default=0.5,
                    help="Keep this fraction of each category, ranked by IoU.")
    ap.add_argument("--min_iou", type=float, default=0.40,
                    help="Reject shapes below this IoU even if they're in the top X%%.")
    args = ap.parse_args()

    os.makedirs(args.dst_dir, exist_ok=True)
    # Symlink the .npz files (same content, pre-filter is via the metrics CSV)
    for split in ("train", "val"):
        src = os.path.abspath(os.path.join(args.src_dir, f"{split}.npz"))
        dst = os.path.join(args.dst_dir, f"{split}.npz")
        if os.path.exists(dst) or os.path.islink(dst):
            os.remove(dst)
        os.symlink(src, dst)
        print(f"Symlinked {dst} -> {src}")

    # Per-split rebalance
    for split in ("train", "val"):
        in_csv = os.path.join(args.src_dir, f"{split}_metrics.csv")
        out_csv = os.path.join(args.dst_dir, f"{split}_metrics.csv")
        if not os.path.exists(in_csv):
            print(f"[skip] no {in_csv}")
            continue
        ids_by_cat = load_split_ids(args.data_root, split)
        print(f"\n=== {split} ===")
        rebalance(in_csv, out_csv, ids_by_cat,
                  top_frac=args.top_frac, min_iou=args.min_iou)


if __name__ == "__main__":
    main()
