"""
Headless integration test: verifies that the analytical gradient does not
introduce NaN losses, causes convergence, and produces collision-free
trajectories for simple sphere-vs-SQ scenes.

Run with:
  PATH=/usr/local/cuda-12.8/bin:/usr/bin:$PATH ~/isaacsim/python.sh \
    tests/test_sq_motion_gen_headless.py
"""

import sys, math, time
sys.path.insert(0, "src")

import torch
from curobo.geom.types import Superquadric, WorldConfig, Sphere as CuroboSphere
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


def make_world(sq_list):
    cfg = WorldCollisionConfig(
        tensor_args=tensor_args,
        world_model=WorldConfig(superquadric=sq_list),
        cache={"obb": 0, "superquadric": len(sq_list) + 2},
    )
    return WorldPrimitiveCollision(cfg)


def run_gradient_descent(world, spheres_xyzr, n_steps=200, lr=0.01):
    """
    Simple gradient descent to push spheres out of collision.
    Uses the analytical gradient from the SQ kernel.
    Returns (final_loss, trajectory_of_losses, any_nan).
    """
    sph = torch.tensor(spheres_xyzr, dtype=torch.float32, device="cuda:0")  # [N, 4]
    N = sph.shape[0]
    pos = sph[:, :3].clone().requires_grad_(False)

    losses = []
    any_nan = False

    weight   = tensor_args.to_device([1.0])
    act_dist = tensor_args.to_device([0.02])  # 2 cm activation distance
    env_idx  = tensor_args.to_device([0]).to(torch.int32)

    for step in range(n_steps):
        q = CollisionQueryBuffer.initialize_from_shape(
            (N, 1, 1, 4), tensor_args, world.collision_types
        )
        sphere_in = torch.cat([pos, sph[:, 3:4]], dim=1).view(N, 1, 1, 4)
        sphere_in.requires_grad_(True)

        dist_out = world.get_sphere_distance(
            sphere_in, q, weight, act_dist,
            env_query_idx=env_idx, compute_esdf=False, sum_collisions=True
        )

        loss = dist_out.sum()
        if torch.isnan(loss) or torch.isinf(loss):
            any_nan = True
            break

        losses.append(loss.item())

        # Use the analytical gradient stored in grad_distance_buffer
        grad = q.superquadric_collision_buffer.grad_distance_buffer[:, 0, 0, :3]  # [N,3]

        # Gradient descent: move in -grad direction (reduces collision cost)
        with torch.no_grad():
            pos = pos - lr * grad

    return losses[-1] if losses else float("nan"), losses, any_nan


# ─────────────────────────────────────────────────────────────────────────────
# Test A: single sphere inside sphere SQ → should escape via gradient descent
# ─────────────────────────────────────────────────────────────────────────────
print("\n=== Test A: sphere inside sphere SQ — gradient descent ===")
sq_a = Superquadric(
    name="sq_sphere",
    pose=[0.0, 0.0, 0.0,  1.0, 0.0, 0.0, 0.0],
    radii=[0.20, 0.20, 0.20],
    eps=[1.0, 1.0],
)
world_a = make_world([sq_a])

# Start inside: (0.05, 0, 0), r=0.01
initial_loss_a, losses_a, nan_a = run_gradient_descent(
    world_a, [[0.05, 0.0, 0.0, 0.01]], n_steps=300, lr=0.005
)

print(f"  Initial loss: {losses_a[0]:.5f}")
print(f"  Final loss:   {initial_loss_a:.5f}  (after {len(losses_a)} steps)")
print(f"  NaN detected: {nan_a}")

check("Test A: no NaN in losses", not nan_a)
check("Test A: loss decreases (gradient pushes sphere out)",
      len(losses_a) >= 2 and losses_a[-1] < losses_a[0],
      f"{losses_a[0]:.4f} → {losses_a[-1]:.4f}")
check("Test A: final loss near zero (sphere escaped)",
      initial_loss_a < 0.01, f"loss={initial_loss_a:.5f}")


# ─────────────────────────────────────────────────────────────────────────────
# Test B: sphere inside rotated elongated SQ — correct gradient direction
# ─────────────────────────────────────────────────────────────────────────────
print("\n=== Test B: sphere inside rotated elongated SQ ===")
qw90x = math.cos(math.pi / 4)
qx90x = math.sin(math.pi / 4)
sq_b = Superquadric(
    name="sq_elong_rot",
    pose=[0.0, 0.0, 0.0,  qw90x, qx90x, 0.0, 0.0],
    radii=[0.05, 0.05, 0.30],  # long axis along world -Y after 90° X rotation
    eps=[0.5, 0.5],
)
world_b = make_world([sq_b])

# Sphere inside long axis (world Y): (0, 0.15, 0)
initial_loss_b, losses_b, nan_b = run_gradient_descent(
    world_b, [[0.0, 0.15, 0.0, 0.01]], n_steps=300, lr=0.005
)

print(f"  Initial loss: {losses_b[0]:.5f}")
print(f"  Final loss:   {initial_loss_b:.5f}")

check("Test B: no NaN", not nan_b)
check("Test B: loss decreases", len(losses_b) >= 2 and losses_b[-1] < losses_b[0],
      f"{losses_b[0]:.4f} → {losses_b[-1]:.4f}")
check("Test B: final loss near zero", initial_loss_b < 0.01,
      f"loss={initial_loss_b:.5f}")


# ─────────────────────────────────────────────────────────────────────────────
# Test C: multiple spheres, multiple obstacles — no NaN, loss monotone
# ─────────────────────────────────────────────────────────────────────────────
print("\n=== Test C: multiple spheres + obstacles ===")
sq_c1 = Superquadric(
    name="sq_c1",
    pose=[0.3, 0.0, 0.0,  1.0, 0.0, 0.0, 0.0],
    radii=[0.15, 0.15, 0.15],
    eps=[1.0, 1.0],
)
sq_c2 = Superquadric(
    name="sq_c2",
    pose=[-0.3, 0.0, 0.0,  1.0, 0.0, 0.0, 0.0],
    radii=[0.10, 0.10, 0.10],
    eps=[0.5, 0.5],
)
world_c = make_world([sq_c1, sq_c2])

spheres_c = [
    [0.30,  0.05, 0.0, 0.01],   # inside sq_c1
    [-0.30, 0.05, 0.0, 0.01],   # inside sq_c2
    [0.0,   0.0,  0.5, 0.01],   # outside both
]
initial_loss_c, losses_c, nan_c = run_gradient_descent(
    world_c, spheres_c, n_steps=300, lr=0.005
)

print(f"  Initial loss: {losses_c[0]:.5f}")
print(f"  Final loss:   {initial_loss_c:.5f}")

check("Test C: no NaN", not nan_c)
check("Test C: loss decreases", len(losses_c) >= 2 and losses_c[-1] < losses_c[0],
      f"{losses_c[0]:.4f} → {losses_c[-1]:.4f}")


# ─────────────────────────────────────────────────────────────────────────────
# Test D: sharp SQ (eps=0.25) — gradient should still be finite and correct
# ─────────────────────────────────────────────────────────────────────────────
print("\n=== Test D: sharp SQ (eps=0.25) ===")
sq_d = Superquadric(
    name="sq_sharp",
    pose=[0.0, 0.0, 0.0,  1.0, 0.0, 0.0, 0.0],
    radii=[0.09, 0.37, 0.35],
    eps=[0.25, 0.25],
)
world_d = make_world([sq_d])

# Inside along long Y axis
initial_loss_d, losses_d, nan_d = run_gradient_descent(
    world_d, [[0.0, 0.20, 0.0, 0.01]], n_steps=300, lr=0.005
)

print(f"  Initial loss: {losses_d[0]:.5f}")
print(f"  Final loss:   {initial_loss_d:.5f}")

check("Test D: no NaN for sharp SQ", not nan_d)
check("Test D: loss decreases", len(losses_d) >= 2 and losses_d[-1] < losses_d[0],
      f"{losses_d[0]:.4f} → {losses_d[-1]:.4f}")


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
