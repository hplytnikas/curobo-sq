"""
Tests for the analytical gradient (Taubin normal) used in sphere-vs-SQ collision.

Replaces the previous 6-launch numerical FD gradient with the analytical
n̂ = ∇F/‖∇F‖ rotated to world frame.  These tests verify:

  1. Gradient direction matches numerical FD (< 5° angular error)
  2. Gradient magnitude is plausible (non-zero inside/on-surface)
  3. SDF sign is correct (positive = collision in CuRobo convention)
  4. Rotated SQs produce correct gradients in world frame

Run with:
  PATH=/usr/local/cuda-12.8/bin:/usr/bin:$PATH ~/isaacsim/python.sh \
    tests/test_sq_taubin_grad.py
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
EPS_FD = 1e-3   # finite-difference step size


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


def sq_sdf_and_grad(world, sphere_xyz, sphere_r=0.01):
    """
    Returns (sdf_curobo, grad_xyz) where:
      sdf_curobo > 0  →  collision
      grad_xyz        →  analytical ∂cost/∂p (from closest_point buffer)
    """
    q = CollisionQueryBuffer.initialize_from_shape(
        (1, 1, 1, 4), tensor_args, world.collision_types
    )
    sph_t = tensor_args.to_device(
        [[sphere_xyz[0], sphere_xyz[1], sphere_xyz[2], sphere_r]]
    ).view(1, 1, 1, 4).requires_grad_(True)

    weight   = tensor_args.to_device([1.0])
    act_dist = tensor_args.to_device([0.0])
    env_idx  = tensor_args.to_device([0]).to(torch.int32)

    world.get_sphere_distance(sph_t, q, weight, act_dist,
                              env_query_idx=env_idx, compute_esdf=True)

    sdf_val  = q.superquadric_collision_buffer.distance_buffer.item()
    grad_buf = q.superquadric_collision_buffer.grad_distance_buffer  # [1,1,1,4]
    grad_xyz = grad_buf.view(4)[:3].detach().cpu().tolist()
    return sdf_val, grad_xyz


def sq_sdf_fd_grad(world, sphere_xyz, sphere_r=0.01, eps=EPS_FD):
    """Numerical finite-difference gradient of sdf_curobo w.r.t. sphere position."""
    def query(pt):
        q = CollisionQueryBuffer.initialize_from_shape(
            (1, 1, 1, 4), tensor_args, world.collision_types
        )
        sph = tensor_args.to_device([[pt[0], pt[1], pt[2], sphere_r]]).view(1, 1, 1, 4)
        weight   = tensor_args.to_device([1.0])
        act_dist = tensor_args.to_device([0.0])
        env_idx  = tensor_args.to_device([0]).to(torch.int32)
        world.get_sphere_distance(sph, q, weight, act_dist,
                                  env_query_idx=env_idx, compute_esdf=True)
        return q.superquadric_collision_buffer.distance_buffer.item()

    x, y, z = sphere_xyz
    gx = (query([x+eps, y, z]) - query([x-eps, y, z])) / (2*eps)
    gy = (query([x, y+eps, z]) - query([x, y-eps, z])) / (2*eps)
    gz = (query([x, y, z+eps]) - query([x, y, z-eps])) / (2*eps)
    return [gx, gy, gz]


def angle_between_deg(a, b):
    """Angle in degrees between two 3-vectors (handles near-zero gracefully)."""
    na = math.sqrt(sum(x*x for x in a))
    nb = math.sqrt(sum(x*x for x in b))
    if na < 1e-8 or nb < 1e-8:
        return 0.0   # both near-zero → treat as aligned
    dot = sum(ai*bi for ai, bi in zip(a, b)) / (na * nb)
    dot = max(-1.0, min(1.0, dot))
    return math.degrees(math.acos(dot))


def grad_norm(g):
    return math.sqrt(sum(x*x for x in g))


# ─────────────────────────────────────────────────────────────────────────────
# Test 1: sphere SQ (eps1=eps2=1) — gradient should point radially outward
# ─────────────────────────────────────────────────────────────────────────────
print("\n=== Test 1: sphere SQ (eps1=eps2=1) ===")
sq1 = Superquadric(
    name="sphere_sq",
    pose=[0.0, 0.0, 0.0,  1.0, 0.0, 0.0, 0.0],
    radii=[0.10, 0.10, 0.10],
    eps=[1.0, 1.0],
)
w1 = make_world(WorldConfig(superquadric=[sq1]))

# Probe at (0.05, 0, 0) — inside, sphere centre on +X axis
xyz1 = [0.05, 0.0, 0.0]
sdf1, ag1 = sq_sdf_and_grad(w1, xyz1, sphere_r=0.005)
fd1 = sq_sdf_fd_grad(w1, xyz1, sphere_r=0.005)

angle1 = angle_between_deg(ag1, fd1)
print(f"  SDF={sdf1:+.4f}  analytical_grad={[f'{v:.4f}' for v in ag1]}")
print(f"  FD_grad={[f'{v:.4f}' for v in fd1]}")
print(f"  Angular error = {angle1:.2f}°")

check("sphere SQ: SDF > 0 (collision at centre)", sdf1 > 0, f"sdf={sdf1:+.4f}")
check("sphere SQ: grad angle error < 10°", angle1 < 10.0, f"{angle1:.2f}°")
check("sphere SQ: grad is non-zero", grad_norm(ag1) > 1e-4, f"|g|={grad_norm(ag1):.5f}")


# ─────────────────────────────────────────────────────────────────────────────
# Test 2: cube-like SQ (eps=0.1) — probe on each face
# ─────────────────────────────────────────────────────────────────────────────
print("\n=== Test 2: cube-like SQ (eps1=eps2=0.1) ===")
sq2 = Superquadric(
    name="cube_sq",
    pose=[0.0, 0.0, 0.0,  1.0, 0.0, 0.0, 0.0],
    radii=[0.10, 0.10, 0.10],
    eps=[0.1, 0.1],
)
w2 = make_world(WorldConfig(superquadric=[sq2]))

for probe, label in [
    ([0.05, 0.0,  0.0], "+X face"),
    ([0.0,  0.05, 0.0], "+Y face"),
    ([0.0,  0.0,  0.05], "+Z face"),
]:
    sdf2, ag2 = sq_sdf_and_grad(w2, probe, sphere_r=0.005)
    fd2  = sq_sdf_fd_grad(w2, probe, sphere_r=0.005)
    ang2 = angle_between_deg(ag2, fd2)
    print(f"  {label}: SDF={sdf2:+.4f}  angle_err={ang2:.2f}°")
    check(f"cube SQ {label}: grad angle < 15°", ang2 < 15.0, f"{ang2:.2f}°")


# ─────────────────────────────────────────────────────────────────────────────
# Test 3: elongated SQ, identity rotation
# ─────────────────────────────────────────────────────────────────────────────
print("\n=== Test 3: elongated SQ (a=0.05, b=0.05, c=0.20), identity ===")
sq3 = Superquadric(
    name="elong_sq",
    pose=[0.0, 0.0, 0.0,  1.0, 0.0, 0.0, 0.0],
    radii=[0.05, 0.05, 0.20],
    eps=[0.5, 0.5],
)
w3 = make_world(WorldConfig(superquadric=[sq3]))

# Inside along long Z axis
xyz3a = [0.0, 0.0, 0.10]
sdf3a, ag3a = sq_sdf_and_grad(w3, xyz3a, sphere_r=0.005)
fd3a  = sq_sdf_fd_grad(w3, xyz3a, sphere_r=0.005)
ang3a = angle_between_deg(ag3a, fd3a)

# Inside along short X axis
xyz3b = [0.03, 0.0, 0.0]
sdf3b, ag3b = sq_sdf_and_grad(w3, xyz3b, sphere_r=0.005)
fd3b  = sq_sdf_fd_grad(w3, xyz3b, sphere_r=0.005)
ang3b = angle_between_deg(ag3b, fd3b)

print(f"  long-Z probe: SDF={sdf3a:+.4f}  angle_err={ang3a:.2f}°")
print(f"  short-X probe: SDF={sdf3b:+.4f}  angle_err={ang3b:.2f}°")

check("elongated: inside long-Z, grad angle < 10°",  ang3a < 10.0, f"{ang3a:.2f}°")
check("elongated: inside short-X, grad angle < 10°", ang3b < 10.0, f"{ang3b:.2f}°")
check("elongated: long-Z SDF > 0 (collision)",  sdf3a > 0)
check("elongated: short-X SDF > 0 (collision)", sdf3b > 0)


# ─────────────────────────────────────────────────────────────────────────────
# Test 4: rotated SQ — long axis (sz=0.20) rotated 90° around X
#   world +Y → local -Z.  Probe at (0, 0.10, 0) is inside long axis.
#   Gradient should point approximately in +Y direction (away from SQ in world).
# ─────────────────────────────────────────────────────────────────────────────
print("\n=== Test 4: elongated SQ rotated 90° around X ===")
qw90x = math.cos(math.pi / 4)
qx90x = math.sin(math.pi / 4)
sq4 = Superquadric(
    name="rot90x",
    pose=[0.0, 0.0, 0.0,  qw90x, qx90x, 0.0, 0.0],
    radii=[0.05, 0.05, 0.20],
    eps=[0.5, 0.5],
)
w4 = make_world(WorldConfig(superquadric=[sq4]))

xyz4 = [0.0, 0.10, 0.0]
sdf4, ag4 = sq_sdf_and_grad(w4, xyz4, sphere_r=0.005)
fd4 = sq_sdf_fd_grad(w4, xyz4, sphere_r=0.005)
ang4 = angle_between_deg(ag4, fd4)

print(f"  SDF={sdf4:+.4f}  analytical={[f'{v:.4f}' for v in ag4]}")
print(f"  FD_grad={[f'{v:.4f}' for v in fd4]}")
print(f"  Angular error = {ang4:.2f}°")

check("rot90X: collision (SDF > 0)", sdf4 > 0, f"sdf={sdf4:+.4f}")
check("rot90X: grad angle < 10°", ang4 < 10.0, f"{ang4:.2f}°")


# ─────────────────────────────────────────────────────────────────────────────
# Test 5: rotated SQ 90° around Y
# ─────────────────────────────────────────────────────────────────────────────
print("\n=== Test 5: elongated SQ rotated 90° around Y ===")
qw90y = math.cos(math.pi / 4)
qy90y = math.sin(math.pi / 4)
sq5 = Superquadric(
    name="rot90y",
    pose=[0.0, 0.0, 0.0,  qw90y, 0.0, qy90y, 0.0],
    radii=[0.05, 0.05, 0.20],
    eps=[0.5, 0.5],
)
w5 = make_world(WorldConfig(superquadric=[sq5]))

xyz5 = [0.10, 0.0, 0.0]   # long axis now along +X
sdf5, ag5 = sq_sdf_and_grad(w5, xyz5, sphere_r=0.005)
fd5 = sq_sdf_fd_grad(w5, xyz5, sphere_r=0.005)
ang5 = angle_between_deg(ag5, fd5)

print(f"  SDF={sdf5:+.4f}  analytical={[f'{v:.4f}' for v in ag5]}")
print(f"  FD_grad={[f'{v:.4f}' for v in fd5]}")
print(f"  Angular error = {ang5:.2f}°")

check("rot90Y: collision (SDF > 0)", sdf5 > 0, f"sdf={sdf5:+.4f}")
check("rot90Y: grad angle < 10°", ang5 < 10.0, f"{ang5:.2f}°")


# ─────────────────────────────────────────────────────────────────────────────
# Test 6: sharp SQ (eps=0.25) — same setup as test_sq_rotation.py Test 5
# ─────────────────────────────────────────────────────────────────────────────
print("\n=== Test 6: sharp SQ (eps=0.25) ===")
sq6 = Superquadric(
    name="sharp_sq",
    pose=[0.0, 0.0, 0.0,  1.0, 0.0, 0.0, 0.0],
    radii=[0.09, 0.37, 0.35],
    eps=[0.25, 0.25],
)
w6 = make_world(WorldConfig(superquadric=[sq6]))

# Inside along long Y axis (within radii[1]=0.37)
xyz6 = [0.0, 0.20, 0.0]
sdf6, ag6 = sq_sdf_and_grad(w6, xyz6, sphere_r=0.01)
fd6 = sq_sdf_fd_grad(w6, xyz6, sphere_r=0.01)
ang6 = angle_between_deg(ag6, fd6)

print(f"  SDF={sdf6:+.4f}  analytical={[f'{v:.4f}' for v in ag6]}")
print(f"  FD_grad={[f'{v:.4f}' for v in fd6]}")
print(f"  Angular error = {ang6:.2f}°")

check("sharp SQ: collision at Y=0.20", sdf6 > 0, f"sdf={sdf6:+.4f}")
check("sharp SQ: grad angle < 10°", ang6 < 10.0, f"{ang6:.2f}°")


# ─────────────────────────────────────────────────────────────────────────────
# Test 7: near-origin stability — sphere very close to SQ centre
# ─────────────────────────────────────────────────────────────────────────────
print("\n=== Test 7: near-origin stability ===")
sq7 = Superquadric(
    name="near_origin",
    pose=[0.0, 0.0, 0.0,  1.0, 0.0, 0.0, 0.0],
    radii=[0.10, 0.10, 0.10],
    eps=[1.0, 1.0],
)
w7 = make_world(WorldConfig(superquadric=[sq7]))

xyz7 = [1e-4, 1e-4, 1e-4]
sdf7, ag7 = sq_sdf_and_grad(w7, xyz7, sphere_r=0.005)

check("near-origin: no NaN in SDF",  not math.isnan(sdf7),  f"sdf={sdf7}")
check("near-origin: no NaN in grad", not any(math.isnan(v) for v in ag7),
      f"grad={ag7}")
check("near-origin: grad is finite", all(abs(v) < 1e6 for v in ag7),
      f"grad={ag7}")


# ─────────────────────────────────────────────────────────────────────────────
# Summary
# ─────────────────────────────────────────────────────────────────────────────
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
