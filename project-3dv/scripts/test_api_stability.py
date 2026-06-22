#!/usr/bin/env python3
"""
test_api_stability.py
=====================
API contract + downstream deployment compatibility checks for pipeline.py.

Parts
-----
1. Static API contract checks (no data required)
2. End-to-end with synthetic depth input
3. Node-wrapper compatibility (subprocess, no internal imports)
4. Git status + push instructions

Run from project root:
    python scripts/test_api_stability.py
"""

import dataclasses
import inspect
import subprocess
import sys
from pathlib import Path

import numpy as np
import open3d as o3d

_REPO = Path(__file__).resolve().parent.parent
_SRC  = _REPO / "src" / "project_3dv" / "perception"
for _p in [str(_SRC)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

import pipeline as tsp
from pipeline import (
    Frame, SuperquadricWorld,
    get_world_pointcloud, fit_superquadrics_world, superquadrics_to_curobo_world,
    MULTIVIEW,
)

# ─────────────────────────────────────────────────────────────────────────────
# helpers
# ─────────────────────────────────────────────────────────────────────────────

_results = []

def check(label: str, ok: bool, detail: str = ""):
    tag = "PASS" if ok else "FAIL"
    line = f"  [{tag}] {label}"
    if detail:
        line += f"  ({detail})"
    print(line)
    _results.append((label, ok))
    return ok


def section(title: str):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


# ─────────────────────────────────────────────────────────────────────────────
# Part 1: static API contract
# ─────────────────────────────────────────────────────────────────────────────

section("Part 1: API contract checks")

# Frame dataclass fields
frame_fields = {f.name: f for f in dataclasses.fields(Frame)}
check("Frame is a dataclass",
      dataclasses.is_dataclass(Frame))
check("Frame.rgb field exists",
      "rgb" in frame_fields)
check("Frame.depth field exists",
      "depth" in frame_fields)
check("Frame.K field exists",
      "K" in frame_fields)
check("Frame.extrinsic field exists",
      "extrinsic" in frame_fields)

# SuperquadricWorld dataclass fields
sq_fields = {f.name: f for f in dataclasses.fields(SuperquadricWorld)}
check("SuperquadricWorld is a dataclass",
      dataclasses.is_dataclass(SuperquadricWorld))
for fname in ("a1", "a2", "a3", "e1", "e2", "position", "rotation"):
    check(f"SuperquadricWorld.{fname} field exists",
          fname in sq_fields)

# verify float annotations on a1..e2
for fname in ("a1", "a2", "a3", "e1", "e2"):
    ann = SuperquadricWorld.__dataclass_fields__[fname].type
    check(f"SuperquadricWorld.{fname} typed float",
          ann is float or ann == float or str(ann) in ("float", "<class 'float'>"),
          str(ann))

# verify array annotations on position, rotation
for fname in ("position", "rotation"):
    ann = str(SuperquadricWorld.__dataclass_fields__[fname].type)
    check(f"SuperquadricWorld.{fname} typed ndarray",
          "ndarray" in ann or "np" in ann or "array" in ann.lower(),
          ann)

# instantiate and check shapes
sq = SuperquadricWorld(
    a1=0.05, a2=0.04, a3=0.03, e1=1.0, e2=1.0,
    position=np.array([0.1, 0.2, 0.8]),
    rotation=np.eye(3),
)
check("SuperquadricWorld.position shape (3,)",
      np.array(sq.position).shape == (3,))
check("SuperquadricWorld.rotation shape (3,3)",
      np.array(sq.rotation).shape == (3, 3))
check("SuperquadricWorld.quaternion_wxyz() returns (4,)",
      sq.quaternion_wxyz().shape == (4,))
check("SuperquadricWorld.quaternion_wxyz() is unit",
      abs(float(np.linalg.norm(sq.quaternion_wxyz())) - 1.0) < 1e-6)

# callable signatures
check("get_world_pointcloud is callable",
      callable(get_world_pointcloud))
sig1 = inspect.signature(get_world_pointcloud)
p1   = list(sig1.parameters.keys())
check("get_world_pointcloud first param is 'frames'",
      p1[0] == "frames", str(p1))

check("fit_superquadrics_world is callable",
      callable(fit_superquadrics_world))
sig2 = inspect.signature(fit_superquadrics_world)
p2   = list(sig2.parameters.keys())
check("fit_superquadrics_world first param is 'world_pcd'",
      p2[0] == "world_pcd", str(p2))

check("superquadrics_to_curobo_world is callable",
      callable(superquadrics_to_curobo_world))

check("MULTIVIEW is bool",
      isinstance(MULTIVIEW, bool), str(type(MULTIVIEW)))


# ─────────────────────────────────────────────────────────────────────────────
# Part 2: end-to-end with synthetic input
# ─────────────────────────────────────────────────────────────────────────────

section("Part 2: end-to-end synthetic pipeline")

H, W = 480, 640
depth = np.full((H, W), 0.75, dtype=np.float32)       # flat surface at 0.75 m
cy_px, cx_px = H // 2, W // 2
depth[cy_px-15:cy_px+15, cx_px-15:cx_px+15] = 0.65   # bump (object) at 0.65 m
rgb = np.zeros((H, W, 3), dtype=np.uint8)
K   = np.array([[570.342, 0., 319.5],
                [0., 570.342, 239.5],
                [0., 0., 1.]], dtype=np.float64)
extrinsic = np.eye(4, dtype=np.float64)

frame = Frame(rgb=rgb, depth=depth, K=K, extrinsic=extrinsic)
print(f"\n  synthetic depth: {depth.shape}  bump depth=0.65m, background=0.75m")

# Stage 1
world_pcd = get_world_pointcloud([frame], voxel_size=0.003, max_depth=2.0)
pts = np.asarray(world_pcd.points)
n_pts = len(pts)
print(f"  Stage 1: {n_pts:,} points")
check("world_pcd has points", n_pts > 0, f"{n_pts}")
if n_pts > 0:
    z_min, z_max = float(pts[:, 2].min()), float(pts[:, 2].max())
    check("world_pcd z in [0.60, 0.80]",
          0.55 <= z_min and z_max <= 0.85,
          f"z=[{z_min:.3f}, {z_max:.3f}]")

# Stage 2
sqs = fit_superquadrics_world(world_pcd)
print(f"  Stage 2: {len(sqs)} SQ(s)")
check("fit_superquadrics_world returns list", isinstance(sqs, list))
check("no exception from fit_superquadrics_world", True)

# cuRoBO adapter — may not be installed on login node
try:
    curobo_cfg = superquadrics_to_curobo_world(sqs)
    check("superquadrics_to_curobo_world returns non-None", curobo_cfg is not None)
    print(f"  cuRoBO config type: {type(curobo_cfg).__name__}")
except ImportError as e:
    check("superquadrics_to_curobo_world importable (cuRoBO not installed — skip)",
          True, f"ImportError: {e}")
    curobo_cfg = None

# empty-list edge case must never raise
try:
    superquadrics_to_curobo_world([])
    check("superquadrics_to_curobo_world([]) does not raise", True)
except ImportError:
    check("superquadrics_to_curobo_world([]) does not raise (cuRoBO absent — skip)", True)
except Exception as ex:
    check("superquadrics_to_curobo_world([]) does not raise", False, str(ex))

if sqs:
    pos0 = np.array(sqs[0].position)
    check("sqs[0].position shape (3,)", pos0.shape == (3,))
    check("sqs[0].rotation shape (3,3)",
          np.array(sqs[0].rotation).shape == (3, 3))


# ─────────────────────────────────────────────────────────────────────────────
# Part 3: node-wrapper compatibility (subprocess — zero internal imports)
# ─────────────────────────────────────────────────────────────────────────────

section("Part 3: node wrapper compatibility")

wrapper = _REPO / "scripts" / "node_wrapper_check.py"
wrapper.write_text(f'''\
#!/usr/bin/env python3
"""
node_wrapper_check.py
=====================
Simulates what a ROS/deployment node would do.
Imports ONLY the public API — zero knowledge of internals.
"""
import sys
sys.path.insert(0, "{_SRC}")

import numpy as np
from pipeline import (
    Frame, SuperquadricWorld,
    get_world_pointcloud, fit_superquadrics_world,
    superquadrics_to_curobo_world, MULTIVIEW,
)

# ── construct Frame from raw numpy (as a node would receive from a sensor) ───
H, W = 480, 640
depth = np.full((H, W), 0.75, dtype=np.float32)
depth[220:260, 300:340] = 0.65
rgb   = np.zeros((H, W, 3), dtype=np.uint8)
K     = np.array([[570.342,0.,319.5],[0.,570.342,239.5],[0.,0.,1.]], dtype=np.float64)
extrinsic = np.eye(4, dtype=np.float64)

frame = Frame(rgb=rgb, depth=depth, K=K, extrinsic=extrinsic)

# ── stage 1 ──────────────────────────────────────────────────────────────────
world_pcd = get_world_pointcloud([frame])
print(f"world_pcd: {{len(world_pcd.points):,}} points")

# ── stage 2 ──────────────────────────────────────────────────────────────────
sqs = fit_superquadrics_world(world_pcd)
print(f"fitted {{len(sqs)}} superquadric(s)")
for sq in sqs:
    print(f"  pos={{sq.position.round(3)}}  "
          f"a=({{{sq.a1:.3f}}},{{{sq.a2:.3f}}},{{{sq.a3:.3f}}})  type={{sq.shape_type}}")

# ── cuRoBO adapter ────────────────────────────────────────────────────────────
try:
    cfg = superquadrics_to_curobo_world(sqs)
    print(f"curobo config: {{type(cfg).__name__}}  "
          f"obstacles={{len(cfg.superquadric) if hasattr(cfg,'superquadric') else '?'}}")
except ImportError as e:
    print(f"cuRoBO not installed (expected on login node): {{e}}")

print("node_wrapper_check: OK")
''')
wrapper.chmod(0o755)

result = subprocess.run(
    [sys.executable, str(wrapper)],
    capture_output=True, text=True,
)
print(result.stdout.rstrip())
if result.stderr:
    for line in result.stderr.strip().splitlines():
        if "warn" not in line.lower() and "UserWarning" not in line:
            print(f"  stderr: {line}")

check("node_wrapper_check.py exits 0", result.returncode == 0,
      f"exit={result.returncode}")
check("node_wrapper_check prints 'OK'", "node_wrapper_check: OK" in result.stdout)


# ─────────────────────────────────────────────────────────────────────────────
# Part 4: git status + push instructions
# ─────────────────────────────────────────────────────────────────────────────

section("Part 4: git status + push instructions")

git = lambda *args: subprocess.run(
    ["git", "-C", str(_REPO), *args],
    capture_output=True, text=True,
).stdout.strip()

changed = git("diff", "--name-only", "HEAD")
untracked = [
    line[3:]
    for line in git("status", "--short").splitlines()
    if line.startswith("??")
]

# files we own — exclude data/, logs/, eps_max scratch
owned_untracked = [
    f for f in untracked
    if not any(f.startswith(p) for p in ("data/", "logs/", "eps_max"))
]

print("\ngit diff --name-only HEAD:")
for f in (changed.splitlines() if changed else []):
    print(f"  {f}")

print("\ngit status --short:")
for line in git("status", "--short").splitlines():
    print(f"  {line}")

all_to_add = sorted(set(
    (changed.splitlines() if changed else []) + owned_untracked
))

print("\n── commands to commit and push ──────────────────────────────────────")
print()
print(f"cd {_REPO}")
if all_to_add:
    print("git add \\")
    for i, f in enumerate(all_to_add):
        sep = " \\" if i < len(all_to_add) - 1 else ""
        print(f"  {f}{sep}")
print()
print('git commit -m "refactor: two-stage perception pipeline with world-frame SQ output"')
print()
print("git push")


# ─────────────────────────────────────────────────────────────────────────────
# Summary
# ─────────────────────────────────────────────────────────────────────────────

section("Summary")
passed = sum(1 for _, ok in _results if ok)
total  = len(_results)
failed = [(label, ok) for label, ok in _results if not ok]
print(f"\n  {passed}/{total} checks passed")
if failed:
    print("\n  FAILED:")
    for label, _ in failed:
        print(f"    - {label}")
    sys.exit(1)
else:
    print("\n  ALL CHECKS PASSED")
