"""
sq_shape_library.py
===================
Builds a superquadric shape prior library from the 3D Warehouse objects
and trains a classifier that maps SQ parameters -> shape category.

Two modes:
  build   — fit SQs to all full models (_0.ply), save library
  train   — train + evaluate shape classifier from library
  predict — classify a single point cloud

Naming convention in the dataset:
  bowl_1_0.ply   →  category=bowl, model_id=1, view_id=0 (full model)
  bowl_1_1.ply   →  category=bowl, model_id=1, view_id=1 (partial view)

Category → SQ shape type mapping (confirmed from extent analysis):
  bowl       → Ellipsoid   (flat, Z << XY)
  cap        → Ellipsoid   (flat disc)
  soda_can   → Cylinder    (upright, Z > XY, circular XY)
  cereal_box → Cuboid      (tall box, low X, high Z)
  coffee_mug → Other       (cylinder + handle, irregular)

Usage:
  python3 sq_shape_library.py --obj_dir /Volumes/T7/objects_3dwarehouse/pc --mode build
  python3 sq_shape_library.py --obj_dir /Volumes/T7/objects_3dwarehouse/pc --mode train
"""

import argparse
import numpy as np
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional
import json


# ─────────────────────────────────────────────────────────────────────────────
# Shape type mapping
# ─────────────────────────────────────────────────────────────────────────────

SMALL_OBJECT_CATEGORIES = ["bowl", "cap", "cereal_box", "coffee_mug", "soda_can"]

CATEGORY_TO_SQ_TYPE = {
    "bowl":       "Ellipsoid",
    "cap":        "Ellipsoid",
    "soda_can":   "Cylinder",
    "cereal_box": "Cuboid",
    "coffee_mug": "Other",
}

SQ_TYPE_TO_INT = {"Ellipsoid": 0, "Cylinder": 1, "Cuboid": 2, "Other": 3}
INT_TO_SQ_TYPE = {v: k for k, v in SQ_TYPE_TO_INT.items()}


# ─────────────────────────────────────────────────────────────────────────────
# Geometric features — same ones used in your pipeline's splitting logic,
# but extended for shape classification
# ─────────────────────────────────────────────────────────────────────────────

def extract_shape_features(pts: np.ndarray) -> Dict[str, float]:
    """
    Extract geometric features for SQ shape classification.
    Works on normalised (unit-scale) or real-world point clouds.
    All features are scale-invariant ratios.
    """
    if len(pts) < 10:
        return {}

    pts = pts.astype(np.float64)
    centred = pts - pts.mean(axis=0)
    extents = pts.max(axis=0) - pts.min(axis=0)
    sorted_ext = np.sort(extents)[::-1]   # [largest, mid, smallest]

    # ── axis ratios (scale-invariant) ────────────────────────────────────────
    e1, e2, e3 = sorted_ext
    r12 = e1 / (e2 + 1e-9)   # elongation along principal axis
    r13 = e1 / (e3 + 1e-9)   # flatness
    r23 = e2 / (e3 + 1e-9)   # secondary flatness

    # ── PCA in 3D ────────────────────────────────────────────────────────────
    cov = (centred.T @ centred) / len(centred)
    eigvals, eigvecs = np.linalg.eigh(cov)
    eigvals = np.sort(eigvals)[::-1]   # descending
    total_var = eigvals.sum() + 1e-9
    pca_r12 = eigvals[0] / (eigvals[1] + 1e-9)
    pca_r13 = eigvals[0] / (eigvals[2] + 1e-9)
    pca_explained_1 = eigvals[0] / total_var
    pca_explained_2 = (eigvals[0] + eigvals[1]) / total_var

    # ── XY circularity (key for cylinder vs box) ─────────────────────────────
    # project to plane perpendicular to longest axis and measure circularity
    xy = centred[:, :2]
    xy_dists = np.linalg.norm(xy, axis=1)
    xy_mean_r = xy_dists.mean()
    xy_std_r  = xy_dists.std()
    circularity_xy = 1.0 - (xy_std_r / (xy_mean_r + 1e-9))

    # also check in XZ and YZ planes
    xz = centred[:, [0, 2]]
    xz_dists = np.linalg.norm(xz, axis=1)
    circularity_xz = 1.0 - (xz_dists.std() / (xz_dists.mean() + 1e-9))

    yz = centred[:, [1, 2]]
    yz_dists = np.linalg.norm(yz, axis=1)
    circularity_yz = 1.0 - (yz_dists.std() / (yz_dists.mean() + 1e-9))

    max_circularity = max(circularity_xy, circularity_xz, circularity_yz)

    # ── surface density uniformity (bowl is open, can is closed) ─────────────
    # split into top/bottom halves along Z and compare point densities
    z = centred[:, 2]
    z_mid   = z.mean()
    top_frac = (z > z_mid).mean()    # fraction of points in top half
    # for a bowl (open top): top_frac << 0.5
    # for a can (closed):    top_frac ≈ 0.5

    # ── vertical profile (histogram of heights) ───────────────────────────────
    hist, _ = np.histogram(z, bins=20)
    hist = hist.astype(float) / (hist.sum() + 1e-9)
    hist_std  = hist.std()    # low = uniform (cylinder/box), high = peaked (bowl)
    hist_skew = float(((hist - hist.mean()) ** 3).mean() / (hist.std() ** 3 + 1e-9))

    # ── convex hull vs bbox volume ratio ─────────────────────────────────────
    try:
        from scipy.spatial import ConvexHull
        hull = ConvexHull(centred)
        bbox_vol = float(np.prod(extents + 1e-9))
        hull_fill = hull.volume / bbox_vol
    except Exception:
        hull_fill = 0.5

    feats = {
        # axis ratios
        "ext_r12":          float(r12),
        "ext_r13":          float(r13),
        "ext_r23":          float(r23),
        # PCA
        "pca_r12":          float(pca_r12),
        "pca_r13":          float(pca_r13),
        "pca_explained_1":  float(pca_explained_1),
        "pca_explained_2":  float(pca_explained_2),
        # circularity
        "circularity_xy":   float(circularity_xy),
        "circularity_xz":   float(circularity_xz),
        "circularity_yz":   float(circularity_yz),
        "max_circularity":  float(max_circularity),
        # vertical profile
        "top_half_frac":    float(top_frac),
        "height_hist_std":  float(hist_std),
        "height_hist_skew": float(hist_skew),
        # volume
        "hull_fill":        float(hull_fill),
    }
    return feats


# ─────────────────────────────────────────────────────────────────────────────
# Build shape library
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ShapeEntry:
    category:    str
    sq_type:     str
    model_id:    str
    view_id:     int
    features:    Dict[str, float]
    n_points:    int


def build_library(obj_dir: str,
                  full_only: bool = False,
                  save_path: str = "sq_shape_library.json") -> List[ShapeEntry]:
    """
    Iterate over all small-object PLYs and extract shape features.

    full_only=True  → only process _0.ply (complete CAD models)
    full_only=False → process all views (partial scans, more realistic)
    """
    import open3d as o3d

    obj_dir = Path(obj_dir)
    library = []
    skipped = 0

    for cat in SMALL_OBJECT_CATEGORIES:
        cat_dir = obj_dir / cat
        if not cat_dir.exists():
            print(f"  [build] Missing category dir: {cat_dir}")
            continue

        plys = sorted(p for p in cat_dir.glob("*.ply")
                      if not p.name.startswith("._"))
        if full_only:
            plys = [p for p in plys if p.stem.endswith("_0")]

        sq_type = CATEGORY_TO_SQ_TYPE[cat]
        print(f"  {cat:15s} ({sq_type:10s}): {len(plys)} files")

        for p in plys:
            pts = np.asarray(o3d.io.read_point_cloud(str(p)).points,
                             dtype=np.float32)
            if len(pts) < 20:
                skipped += 1
                continue

            # parse model_id and view_id from filename
            # e.g. bowl_1_0  → model_id="1", view_id=0
            parts   = p.stem.split("_")
            view_id  = int(parts[-1])
            model_id = "_".join(parts[1:-1])

            feats = extract_shape_features(pts)
            if not feats:
                skipped += 1
                continue

            library.append(ShapeEntry(
                category=cat,
                sq_type=sq_type,
                model_id=model_id,
                view_id=view_id,
                features=feats,
                n_points=len(pts),
            ))

    print(f"\n  Library built: {len(library)} entries  ({skipped} skipped)")

    # save as JSON for portability
    data = [
        dict(category=e.category, sq_type=e.sq_type,
             model_id=e.model_id, view_id=e.view_id,
             n_points=e.n_points, features=e.features)
        for e in library
    ]
    with open(save_path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"  Saved → {save_path}")

    return library


# ─────────────────────────────────────────────────────────────────────────────
# Train and evaluate shape classifier
# ─────────────────────────────────────────────────────────────────────────────

def train_classifier(library_path: str = "sq_shape_library.json",
                     save_path:    str = "sq_shape_classifier.pkl"):
    """
    Train a RandomForest shape classifier from the library.
    Input:  geometric features (scale-invariant)
    Output: SQ type (Ellipsoid / Cylinder / Cuboid / Other)

    Also prints per-class accuracy and feature importances.
    """
    try:
        from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
        from sklearn.model_selection import StratifiedKFold, cross_val_score
        from sklearn.preprocessing import LabelEncoder
        from sklearn.metrics import classification_report
        import joblib
    except ImportError:
        print("Run: pip install scikit-learn joblib")
        return

    with open(library_path) as f:
        data = json.load(f)

    feat_names = sorted(data[0]["features"].keys())
    X = np.array([[e["features"][k] for k in feat_names] for e in data])
    y = np.array([SQ_TYPE_TO_INT[e["sq_type"]]            for e in data])

    print(f"\n  Dataset: {len(X)} examples")
    for sq_int, sq_name in INT_TO_SQ_TYPE.items():
        print(f"    {sq_name:10s}: {(y==sq_int).sum()}")

    # train RandomForest
    clf = RandomForestClassifier(
        n_estimators=300,
        max_depth=8,
        min_samples_leaf=3,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )

    # stratified 5-fold CV
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(clf, X, y, cv=cv, scoring="f1_macro")
    print(f"\n  5-fold CV F1 (macro): {scores.mean():.3f} ± {scores.std():.3f}")

    # fit on full data
    clf.fit(X, y)
    y_pred = clf.predict(X)
    print(f"\n  Train classification report:")
    print(classification_report(
        y, y_pred,
        target_names=[INT_TO_SQ_TYPE[i] for i in sorted(INT_TO_SQ_TYPE)],
    ))

    # feature importances
    print("  Feature importances (top 10):")
    pairs = sorted(zip(feat_names, clf.feature_importances_), key=lambda x: -x[1])
    for name, imp in pairs[:10]:
        bar = "█" * int(imp * 40)
        print(f"    {name:25s}  {imp:.3f}  {bar}")

    joblib.dump((clf, feat_names, SQ_TYPE_TO_INT, INT_TO_SQ_TYPE), save_path)
    print(f"\n  Saved → {save_path}")
    return clf, feat_names


# ─────────────────────────────────────────────────────────────────────────────
# Inference — classify a single point cloud
# ─────────────────────────────────────────────────────────────────────────────

def classify_shape(pts: np.ndarray,
                   classifier_path: str = "sq_shape_classifier.pkl") -> Tuple[str, float]:
    """
    Classify a point cloud into an SQ shape type.
    Returns (sq_type_string, confidence).

    Drop-in replacement for your current threshold-based classifier.
    Usage in pipeline:
        sq_type, conf = classify_shape(object_segment.points)
    """
    import joblib
    clf, feat_names, _, int_to_type = joblib.load(classifier_path)
    feats = extract_shape_features(pts)
    if not feats:
        return "Other", 0.0
    x = np.array([[feats[k] for k in feat_names]])
    proba = clf.predict_proba(x)[0]
    pred  = int(clf.predict(x)[0])
    return int_to_type[pred], float(proba.max())


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--obj_dir", required=True,
                        help="Path to objects_3dwarehouse/pc/")
    parser.add_argument("--mode", default="build",
                        choices=["build", "train", "both"],
                        help="build: extract features | train: fit classifier | "
                             "both: do both in sequence")
    parser.add_argument("--full_only", action="store_true",
                        help="Use only _0.ply full models (faster, cleaner)")
    parser.add_argument("--library",    default="sq_shape_library.json")
    parser.add_argument("--classifier", default="sq_shape_classifier.pkl")
    args = parser.parse_args()

    if args.mode in ("build", "both"):
        print(f"\nBuilding shape library from {args.obj_dir} ...")
        build_library(args.obj_dir,
                      full_only=args.full_only,
                      save_path=args.library)

    if args.mode in ("train", "both"):
        print(f"\nTraining classifier from {args.library} ...")
        train_classifier(library_path=args.library,
                         save_path=args.classifier)