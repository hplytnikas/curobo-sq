"""
Regression test for the quaternion ordering bug in pack_env_sq.

pack_env_sq copies the stored [qx,qy,qz,qw] tensor directly into SQData
which declares fields as `float qw, qx, qy, qz`.  Before the fix the
components were shuffled, so the kernel evaluated the SDF in the wrong
local frame for any non-identity orientation.

Key discriminating probe (Test 1):
  SQ with long axis (sz=0.20) rotated 90° around X → local-Z aligns to world -Y.
  A sphere at world (0, 0.15, 0) lands at local z=-0.15, inside the long axis.
      WITH FIX:    local-Z frame → inside → COLLISION
      WITHOUT FIX: wrong frame (90° around Z) → local x=0.15, outside short axis → no collision

Run with:
  PATH=/usr/local/cuda-12.8/bin:/usr/bin:$PATH ~/isaacsim/python.sh tests/test_sq_rotation.py
"""

import sys, math
sys.path.insert(0, "src")

import torch
from curobo.geom.types import Superquadric, WorldConfig
from curobo.geom.sdf.world import WorldPrimitiveCollision, WorldCollisionConfig, CollisionQueryBuffer
from curobo.types.math import Pose
from curobo.types.base import TensorDeviceType

tensor_args = TensorDeviceType(device=torch.device("cuda:0"), dtype=torch.float32)
PASS = "\033[92mPASS\033[0m"
FAIL = "\033[91mFAIL\033[0m"
results = []

def check(name, condition, extra=""):
    tag = PASS if condition else FAIL
    print(f"  [{tag}] {name}{(' — ' + extra) if extra else ''}")
    results.append((name, condition))


def make_world(world_cfg):
    cfg = WorldCollisionConfig(
        tensor_args=tensor_args,
        world_model=world_cfg,
        cache={"obb": 0, "superquadric": 8},
    )
    return WorldPrimitiveCollision(cfg)


def sq_sdf(world, sphere_xyz, sphere_r=0.005):
    """CuRobo ESDF query. Returns positive = collision, negative = clearance."""
    q = CollisionQueryBuffer.initialize_from_shape(
        (1, 1, 1, 4), tensor_args, world.collision_types
    )
    sph = tensor_args.to_device(
        [[sphere_xyz[0], sphere_xyz[1], sphere_xyz[2], sphere_r]]
    ).view(1, 1, 1, 4)
    weight   = tensor_args.to_device([1.0])
    act_dist = tensor_args.to_device([0.0])
    env_idx  = tensor_args.to_device([0]).to(torch.int32)
    world.get_sphere_distance(sph, q, weight, act_dist,
                              env_query_idx=env_idx, compute_esdf=True)
    return q.superquadric_collision_buffer.distance_buffer.item()


# ──────────────────────────────────────────────────────────────────────────────
# Test 1: rotation correctness — elongated SQ rotated 90° around X
#
# SQ params:  sx=sy=0.05 (short),  sz=0.20 (long),  eps=[0.5, 0.5]
# Rotation:   qw=cos45°, qx=sin45°, qy=0, qz=0  →  90° around X
# Effect:     local-Z (long, 0.20) → world -Y
#             local-Y (short, 0.05) → world +Z
#
# Discriminating probes (sphere_r=0.005 to avoid lower-bound clamping):
#   (0,  0.15, 0):  world +Y → local -Z → inside long axis (0.15 < 0.20)  → COLLISION
#   (0, -0.15, 0):  world -Y → local +Z → inside long axis (0.15 < 0.20)  → COLLISION
#   (0.15, 0,  0):  world +X → local +X → outside short axis (0.15 > 0.05) → no collision
#   (0,  0, 0.15):  world +Z → local +Y → outside short axis (0.15 > 0.05) → no collision
#
# Without the quaternion fix the kernel incorrectly used 90°-around-Z, mapping:
#   (0, 0.15, 0) → local (0.15, 0, 0) = outside short X axis  → would report no collision (wrong)
# ──────────────────────────────────────────────────────────────────────────────
print("\n=== Test 1: elongated SQ rotated 90° around X ===")
qw90x = math.cos(math.pi / 4)
qx90x = math.sin(math.pi / 4)
sq1 = Superquadric(
    name="sq_rot90x",
    pose=[0.0, 0.0, 0.0,  qw90x, qx90x, 0.0, 0.0],
    radii=[0.05, 0.05, 0.20],
    eps=[0.5, 0.5],
)
w1 = make_world(WorldConfig(superquadric=[sq1]))

d_pos_y  = sq_sdf(w1, [ 0.00,  0.15,  0.00])   # inside long (−Y)
d_neg_y  = sq_sdf(w1, [ 0.00, -0.15,  0.00])   # inside long (+Y mapped to −Z)
d_pos_x  = sq_sdf(w1, [ 0.15,  0.00,  0.00])   # outside short
d_pos_z  = sq_sdf(w1, [ 0.00,  0.00,  0.15])   # outside short

print(f"  d(Y=+0.15, inside long axis) = {d_pos_y:+.4f}   expect > 0 (collision)")
print(f"  d(Y=-0.15, inside long axis) = {d_neg_y:+.4f}   expect > 0 (collision)")
print(f"  d(X=+0.15, outside short)    = {d_pos_x:+.4f}   expect < 0 (no collision)")
print(f"  d(Z=+0.15, outside short)    = {d_pos_z:+.4f}   expect < 0 (no collision)")

check("+Y=0.15 is inside long axis (collision)",  d_pos_y > 0, f"d={d_pos_y:+.4f}")
check("-Y=0.15 is inside long axis (collision)",  d_neg_y > 0, f"d={d_neg_y:+.4f}")
check("+X=0.15 is outside short axis",            d_pos_x < 0, f"d={d_pos_x:+.4f}")
check("+Z=0.15 is outside short axis",            d_pos_z < 0, f"d={d_pos_z:+.4f}")


# ──────────────────────────────────────────────────────────────────────────────
# Test 2: same shape, identity rotation — baseline / sanity
#
# No rotation: long axis is world-Z.
#   (0, 0, 0.15): inside long Z → COLLISION
#   (0, 0.15, 0): outside short Y → no collision
# ──────────────────────────────────────────────────────────────────────────────
print("\n=== Test 2: same SQ, identity rotation ===")
sq2 = Superquadric(
    name="sq_identity",
    pose=[0.0, 0.0, 0.0,  1.0, 0.0, 0.0, 0.0],
    radii=[0.05, 0.05, 0.20],
    eps=[0.5, 0.5],
)
w2 = make_world(WorldConfig(superquadric=[sq2]))

d2_z = sq_sdf(w2, [0.00, 0.00,  0.15])   # inside long Z
d2_y = sq_sdf(w2, [0.00, 0.15,  0.00])   # outside short Y

print(f"  d(Z=+0.15, inside long axis)  = {d2_z:+.4f}   expect > 0 (collision)")
print(f"  d(Y=+0.15, outside short axis)= {d2_y:+.4f}   expect < 0 (no collision)")

check("identity: Z=0.15 inside long axis (collision)",   d2_z > 0, f"d={d2_z:+.4f}")
check("identity: Y=0.15 outside short axis",             d2_y < 0, f"d={d2_y:+.4f}")


# ──────────────────────────────────────────────────────────────────────────────
# Test 3: 90° around Y — long axis (Z→X in world frame)
#   (0.15, 0, 0): inside long X → COLLISION
#   (0, 0, 0.15): outside short Z → no collision
# ──────────────────────────────────────────────────────────────────────────────
print("\n=== Test 3: elongated SQ rotated 90° around Y ===")
qw90y = math.cos(math.pi / 4)
qy90y = math.sin(math.pi / 4)
sq3 = Superquadric(
    name="sq_rot90y",
    pose=[0.0, 0.0, 0.0,  qw90y, 0.0, qy90y, 0.0],
    radii=[0.05, 0.05, 0.20],
    eps=[0.5, 0.5],
)
w3 = make_world(WorldConfig(superquadric=[sq3]))

d3_x = sq_sdf(w3, [ 0.15, 0.00, 0.00])   # long axis after 90°-Y
d3_z = sq_sdf(w3, [ 0.00, 0.00, 0.15])   # short axis

print(f"  d(X=+0.15, inside long axis)  = {d3_x:+.4f}   expect > 0 (collision)")
print(f"  d(Z=+0.15, outside short axis)= {d3_z:+.4f}   expect < 0 (no collision)")

check("90°Y: X=0.15 inside long axis (collision)", d3_x > 0, f"d={d3_x:+.4f}")
check("90°Y: Z=0.15 outside short axis",           d3_z < 0, f"d={d3_z:+.4f}")


# ──────────────────────────────────────────────────────────────────────────────
# Test 4: pose update — translate SQ by [0.5, 0, 0]
#   Sphere at (0.5, 0, 0.15) should be inside long axis after translation.
#   Sphere at (0.0, 0.0, 0.15) should now be outside (SQ moved away).
# ──────────────────────────────────────────────────────────────────────────────
print("\n=== Test 4: pose update (translate X+0.5, identity orientation) ===")
sq4 = Superquadric(
    name="movable",
    pose=[0.0, 0.0, 0.0,  1.0, 0.0, 0.0, 0.0],
    radii=[0.10, 0.10, 0.10],
    eps=[1.0, 1.0],
)
w4 = make_world(WorldConfig(superquadric=[sq4]))

d4_before_inside  = sq_sdf(w4, [0.0, 0.0, 0.0])   # at origin → inside
d4_before_outside = sq_sdf(w4, [0.5, 0.0, 0.0])   # far from origin → outside

# translate to (0.5, 0, 0)
new_pose = Pose(
    position=tensor_args.to_device([0.5, 0.0, 0.0]),
    quaternion=tensor_args.to_device([1.0, 0.0, 0.0, 0.0]),
)
w4.update_superquadric_pose(w_obj_pose=new_pose, name="movable")

d4_after_new_ctr  = sq_sdf(w4, [0.5, 0.0, 0.0])   # now at new center → inside
d4_after_old_ctr  = sq_sdf(w4, [0.0, 0.0, 0.0])   # old center → outside

print(f"  before move: d(origin)  = {d4_before_inside:+.4f}   expect > 0 (inside)")
print(f"  before move: d(X=0.5)   = {d4_before_outside:+.4f}   expect < 0 (outside)")
print(f"  after  move: d(X=0.5)   = {d4_after_new_ctr:+.4f}   expect > 0 (inside new center)")
print(f"  after  move: d(origin)  = {d4_after_old_ctr:+.4f}   expect < 0 (outside)")

check("before move: origin is inside",          d4_before_inside  > 0)
check("before move: X=0.5 is outside",          d4_before_outside < 0)
check("after  move: new center is inside",       d4_after_new_ctr  > 0)
check("after  move: old center is outside",      d4_after_old_ctr  < 0)


# ──────────────────────────────────────────────────────────────────────────────
# Test 5: sharp SQ (eps≈0.25) — first-order approx degenerates outside short axis.
#
# sq_0 from real scene: sx=0.09, sy=0.37, sz=0.35, eps1=0.25, identity rotation.
# A sphere of r=0.05 at (0.37, 0, 0):
#   true SDF ≈ 0.37 - 0.09 - 0.05 = 0.23 (outside)
#   first-order approx returns ≈ -0.004 (false collision — outside short axis)
# The local-AABB lower bound: dx_box = 0.37-0.09 = 0.28, lb_box = 0.28-0.05 = 0.23
#   → fmaxf(sdf_approx, lb_box) = 0.23 → correctly reports outside (no collision)
# ──────────────────────────────────────────────────────────────────────────────
print("\n=== Test 5: sharp SQ (eps=0.25) — sphere outside short axis ===")
sq5 = Superquadric(
    name="sq_sharp",
    pose=[0.0, 0.0, 0.0,  1.0, 0.0, 0.0, 0.0],
    radii=[0.09, 0.37, 0.35],
    eps=[0.25, 0.25],
)
w5 = make_world(WorldConfig(superquadric=[sq5]))

# sphere at (0.37, 0, 0): just at the short-axis boundary, should be outside
d5_short = sq_sdf(w5, [0.37, 0.0, 0.0], sphere_r=0.05)
# sphere inside along long Y axis
d5_long  = sq_sdf(w5, [0.0, 0.20, 0.0], sphere_r=0.05)

print(f"  d(X=0.37, outside short X=0.09) = {d5_short:+.4f}   expect < 0 (no collision)")
print(f"  d(Y=0.20, inside long  Y=0.37)  = {d5_long:+.4f}   expect > 0 (collision)")

check("sharp SQ: X=0.37 is outside short axis", d5_short < 0, f"d={d5_short:+.4f}")
check("sharp SQ: Y=0.20 is inside long axis",   d5_long  > 0, f"d={d5_long:+.4f}")


# ──────────────────────────────────────────────────────────────────────────────
print("\n=== Summary ===")
n_pass = sum(1 for _, ok in results if ok)
n_fail = sum(1 for _, ok in results if not ok)
print(f"  {n_pass}/{len(results)} passed")
if n_fail:
    for name, ok in results:
        if not ok:
            print(f"  [FAIL] {name}")
    sys.exit(1)
else:
    print("  All tests passed.")
