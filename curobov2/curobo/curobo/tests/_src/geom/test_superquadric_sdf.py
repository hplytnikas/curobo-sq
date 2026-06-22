# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""Tests for superquadric SDF integration in curobov2.

Run with Isaac Sim Python:
    PATH=/usr/local/cuda-12.8/bin:/usr/bin:$PATH ~/isaacsim/python.sh \
        curobov2/curobo/curobo/tests/_src/geom/test_superquadric_sdf.py

Or with pytest (if warp is available in the environment):
    pytest curobov2/curobo/curobo/tests/_src/geom/test_superquadric_sdf.py -v
"""

import torch
import warp as wp

wp.init()

from curobo._src.geom.data.data_superquadric import SuperquadricData, SuperquadricDataWarp
from curobo._src.geom.data.data_scene import SceneData
from curobo._src.geom.types import SceneCfg, Superquadric
from curobo._src.types.device_cfg import DeviceCfg


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def make_device_cfg():
    return DeviceCfg(device=DEVICE)


# =============================================================================
# Unit tests
# =============================================================================


def test_superquadric_data_create():
    """SuperquadricData.create_cache allocates correct tensors."""
    dcfg = make_device_cfg()
    data = SuperquadricData.create_cache(max_n=4, num_envs=2, device_cfg=dcfg)
    assert data.params.shape == (2, 4, 8)
    assert data.inv_pose.shape == (2, 4, 8)
    assert data.enable.shape == (2, 4)
    assert data.max_n == 4
    assert data.num_envs == 2
    print("PASS: test_superquadric_data_create")


def test_superquadric_scene_cfg():
    """SceneCfg accepts Superquadric obstacles and stores them correctly."""
    sq = Superquadric(
        name="sq0",
        pose=[0.5, 0.0, 0.5, 1.0, 0.0, 0.0, 0.0],
        radii=[0.1, 0.15, 0.2],
        shape=[0.5, 0.5],
    )
    scene = SceneCfg(superquadric=[sq])
    assert len(scene.superquadric) == 1
    assert len(scene.objects) == 1
    print("PASS: test_superquadric_scene_cfg")


def test_superquadric_data_from_scene_cfg():
    """SuperquadricData.from_scene_cfg loads poses and params correctly."""
    dcfg = make_device_cfg()
    sq = Superquadric(
        name="sq0",
        pose=[0.0, 0.0, 0.5, 1.0, 0.0, 0.0, 0.0],
        radii=[0.1, 0.15, 0.2],
        shape=[0.6, 0.8],
    )
    scene = SceneCfg(superquadric=[sq])
    data = SuperquadricData.from_scene_cfg(scene, dcfg)

    assert data.get_active_count(0) == 1
    assert abs(data.params[0, 0, 0].item() - 0.1) < 1e-5  # sx
    assert abs(data.params[0, 0, 1].item() - 0.15) < 1e-5  # sy
    assert abs(data.params[0, 0, 2].item() - 0.2) < 1e-5   # sz
    assert abs(data.params[0, 0, 3].item() - 0.6) < 1e-5   # eps1
    assert abs(data.params[0, 0, 4].item() - 0.8) < 1e-5   # eps2
    print("PASS: test_superquadric_data_from_scene_cfg")


def test_to_warp_conversion():
    """SuperquadricData.to_warp returns a valid SuperquadricDataWarp struct."""
    dcfg = make_device_cfg()
    sq = Superquadric(
        name="test_sq",
        pose=[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
        radii=[0.1, 0.1, 0.1],
        shape=[1.0, 1.0],
    )
    scene = SceneCfg(superquadric=[sq])
    data = SuperquadricData.from_scene_cfg(scene, dcfg)
    warp_data = data.to_warp()

    # Warp structs produce StructInstance subclasses — check by attribute presence
    assert hasattr(warp_data, "params"), "Warp struct missing params field"
    assert hasattr(warp_data, "inv_pose"), "Warp struct missing inv_pose field"
    assert warp_data.max_n == data.max_n
    assert warp_data.num_envs == data.num_envs
    print("PASS: test_to_warp_conversion")


def test_scene_data_with_superquadrics():
    """SceneData correctly creates and stores superquadric data."""
    dcfg = make_device_cfg()
    sqs = [
        Superquadric(
            name=f"sq{i}",
            pose=[float(i) * 0.3, 0.0, 0.5, 1.0, 0.0, 0.0, 0.0],
            radii=[0.1, 0.1, 0.1],
            shape=[1.0, 1.0],
        )
        for i in range(3)
    ]
    scene_cfg = SceneCfg(superquadric=sqs)
    scene_data = SceneData.from_scene_cfg(scene_cfg, dcfg)

    assert scene_data.has_superquadrics()
    assert scene_data.superquadrics.get_active_count(0) == 3
    valid = scene_data.get_valid_data()
    assert any(isinstance(d, SuperquadricData) for d in valid)
    print("PASS: test_scene_data_with_superquadrics")


def _run_sdf_kernel_on_sphere(
    sphere_pos: list,
    sphere_r: float,
    sq_pose: list,
    sq_radii: list,
    sq_shape: list,
) -> float:
    """Launch the collision kernel for a single sphere vs single SQ and return distance."""
    from curobo._src.geom.collision.wp_autograd import SphereObstacleCollision
    from curobo._src.geom.collision.buffer_collision import CollisionBuffer

    dcfg = make_device_cfg()
    sq = Superquadric(name="sq", pose=sq_pose, radii=sq_radii, shape=sq_shape)
    scene_cfg = SceneCfg(superquadric=[sq])
    scene_data = SceneData.from_scene_cfg(scene_cfg, dcfg)

    # batch=1, horizon=1, num_spheres=1
    sphere_t = torch.tensor(
        [[[[sphere_pos[0], sphere_pos[1], sphere_pos[2], sphere_r]]]],
        dtype=dcfg.dtype,
        device=dcfg.device,
        requires_grad=True,
    )

    buf = CollisionBuffer.from_shape(sphere_t.shape[:3], dcfg)
    weight = torch.tensor([1.0], device=dcfg.device, dtype=dcfg.dtype)
    act_dist = torch.tensor([0.0], device=dcfg.device, dtype=dcfg.dtype)
    env_idx = torch.zeros(1, dtype=torch.int32, device=dcfg.device)

    dist = SphereObstacleCollision.apply(
        sphere_t, buf, scene_data, weight, act_dist,
        torch.tensor([1.0], device=dcfg.device),
        env_idx, False, False,
    )
    return float(dist[0, 0, 0].item())


def test_sphere_outside_ellipsoid():
    """Sphere far outside an ellipsoid → cost should be zero (no penetration)."""
    cost = _run_sdf_kernel_on_sphere(
        sphere_pos=[2.0, 0.0, 0.0],
        sphere_r=0.05,
        sq_pose=[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
        sq_radii=[0.1, 0.1, 0.1],
        sq_shape=[1.0, 1.0],
    )
    assert cost == 0.0, f"Expected 0 cost for sphere outside SQ, got {cost}"
    print("PASS: test_sphere_outside_ellipsoid")


def test_sphere_inside_ellipsoid():
    """Sphere at center of ellipsoid → positive cost (penetrating)."""
    cost = _run_sdf_kernel_on_sphere(
        sphere_pos=[0.0, 0.0, 0.0],
        sphere_r=0.01,
        sq_pose=[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
        sq_radii=[0.2, 0.2, 0.2],
        sq_shape=[1.0, 1.0],
    )
    assert cost > 0.0, f"Expected positive cost for sphere inside SQ, got {cost}"
    print(f"PASS: test_sphere_inside_ellipsoid  (cost={cost:.4f})")


def test_sphere_surface_ellipsoid():
    """Sphere just touching ellipsoid surface → cost ~0."""
    # Place sphere at x=sx + sphere_r (exactly touching)
    sx = 0.15
    r = 0.02
    cost = _run_sdf_kernel_on_sphere(
        sphere_pos=[sx + r, 0.0, 0.0],
        sphere_r=r,
        sq_pose=[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
        sq_radii=[sx, sx, sx],
        sq_shape=[1.0, 1.0],
    )
    # Cost should be near zero (sphere just touching surface)
    assert cost < 0.01, f"Expected near-zero cost at surface, got {cost}"
    print(f"PASS: test_sphere_surface_ellipsoid  (cost={cost:.6f})")


def test_boxy_superquadric():
    """Box-like SQ (small eps) with sphere inside → should detect penetration."""
    cost = _run_sdf_kernel_on_sphere(
        sphere_pos=[0.05, 0.05, 0.05],
        sphere_r=0.01,
        sq_pose=[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
        sq_radii=[0.15, 0.15, 0.15],
        sq_shape=[0.2, 0.2],  # boxy
    )
    assert cost > 0.0, f"Expected positive cost for sphere inside boxy SQ, got {cost}"
    print(f"PASS: test_boxy_superquadric  (cost={cost:.4f})")


def test_translated_superquadric():
    """SQ at non-zero pose. Sphere near its center should be in collision."""
    cost = _run_sdf_kernel_on_sphere(
        sphere_pos=[0.5, 0.0, 0.3],
        sphere_r=0.01,
        sq_pose=[0.5, 0.0, 0.3, 1.0, 0.0, 0.0, 0.0],
        sq_radii=[0.1, 0.1, 0.1],
        sq_shape=[1.0, 1.0],
    )
    assert cost > 0.0, f"Expected positive cost for sphere inside translated SQ, got {cost}"
    print(f"PASS: test_translated_superquadric  (cost={cost:.4f})")


def test_gradient_flows():
    """Gradients should flow through the collision kernel."""
    dcfg = make_device_cfg()
    sq = Superquadric(
        name="sq",
        pose=[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
        radii=[0.2, 0.2, 0.2],
        shape=[1.0, 1.0],
    )
    scene_cfg = SceneCfg(superquadric=[sq])
    scene_data = SceneData.from_scene_cfg(scene_cfg, dcfg)

    from curobo._src.geom.collision.wp_autograd import SphereObstacleCollision
    from curobo._src.geom.collision.buffer_collision import CollisionBuffer

    sphere_t = torch.tensor(
        [[[[0.0, 0.0, 0.0, 0.01]]]],
        dtype=dcfg.dtype,
        device=dcfg.device,
        requires_grad=True,
    )
    buf = CollisionBuffer.from_shape(sphere_t.shape[:3], dcfg)
    weight = torch.tensor([1.0], device=dcfg.device, dtype=dcfg.dtype)
    act_dist = torch.tensor([0.0], device=dcfg.device, dtype=dcfg.dtype)
    env_idx = torch.zeros(1, dtype=torch.int32, device=dcfg.device)

    dist = SphereObstacleCollision.apply(
        sphere_t, buf, scene_data, weight, act_dist,
        torch.tensor([1.0], device=dcfg.device),
        env_idx, False, True,
    )
    loss = dist.sum()
    loss.backward()
    grad = sphere_t.grad
    assert grad is not None, "Gradient did not flow back to sphere positions"
    grad_norm = float(grad.norm().item())
    assert grad_norm > 0, f"Gradient is zero: {grad}"
    print(f"PASS: test_gradient_flows  (|grad|={grad_norm:.4f})")


def run_all():
    """Run all tests in sequence."""
    print("\n=== Superquadric SDF Integration Tests ===\n")
    test_superquadric_data_create()
    test_superquadric_scene_cfg()
    test_superquadric_data_from_scene_cfg()
    test_to_warp_conversion()
    test_scene_data_with_superquadrics()
    test_sphere_outside_ellipsoid()
    test_sphere_inside_ellipsoid()
    test_sphere_surface_ellipsoid()
    test_boxy_superquadric()
    test_translated_superquadric()
    test_gradient_flows()
    print("\n=== All tests passed! ===\n")


if __name__ == "__main__":
    run_all()
