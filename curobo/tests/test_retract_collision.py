"""Diagnose which robot spheres at retract pose genuinely collide with the scene SQs.

Uses the 8 SQ parameters captured from the headless diagnostic run and checks them
against the Franka retract-pose spheres via the ESDF kernel.

Run with:
  PATH=/usr/local/cuda-12.8/bin:/usr/bin:$PATH ~/isaacsim/python.sh tests/test_retract_collision.py
"""

import sys
sys.path.insert(0, "src")

import torch
import numpy as np
from curobo.geom.types import Superquadric, WorldConfig
from curobo.geom.sdf.world import WorldPrimitiveCollision, WorldCollisionConfig, CollisionQueryBuffer
from curobo.types.math import Pose
from curobo.types.base import TensorDeviceType
from curobo.types.state import JointState
from curobo.util_file import get_robot_configs_path, join_path, load_yaml
from curobo.wrap.reacher.motion_gen import MotionGen, MotionGenConfig

tensor_args = TensorDeviceType(device=torch.device("cuda:0"), dtype=torch.float32)

# ── SQ parameters from headless diagnostic (raw Python tensor layout):
# [sx, sy, sz, eps1, eps2, cx, cy, cz, qx, qy, qz, qw]
# These are the scene SQs AFTER scene transform and tolerance=0.01m shrink.
# Note: OpenGJK and CuRobo's Python storage both use [qx,qy,qz,qw] (scalar last).
SCENE_SQS = [
    # name      sx      sy      sz     eps1    eps2     cx       cy       cz     qx      qy      qz      qw
    ("sq_0",  0.0885, 0.3706, 0.3536, 0.2486, 0.3636, -0.2314, -0.6889,  0.0274, -0.5060, -0.4957, -0.4999,  0.4983),
    ("sq_3",  0.0957, 0.2982, 0.0603, 0.1444, 0.3041, -0.2661, -1.0000,  0.2137, -0.4968, -0.4446, -0.5520,  0.5007),
    ("sq_4",  0.1847, 0.0269, 0.0176, 0.1170, 0.2451,  0.0875, -1.0097, -0.1646, -0.4736, -0.4965, -0.5505,  0.4756),
    ("sq_5",  0.3053, 0.0257, 0.0167, 0.1133, 0.2149, -0.5566, -0.9134, -0.0423, -0.5050, -0.4706, -0.5366,  0.4853),
    ("sq_7",  0.2519, 0.0194, 0.0247, 0.1188, 0.2483, -0.5489, -0.4471, -0.1060, -0.4783, -0.5111, -0.5013,  0.5086),
    ("sq_8",  0.0845, 0.3163, 0.0559, 0.1338, 0.3207, -0.2620, -0.3534,  0.2184, -0.5052, -0.5533, -0.4438,  0.4916),
    ("sq_14", 0.1612, 0.0319, 0.0130, 0.1109, 0.2143,  0.0919, -0.3741, -0.1554, -0.4839, -0.5188, -0.5253,  0.4698),
    ("sq_15", 0.2149, 0.0476, 0.3758, 0.2819, 0.3816, -0.6097, -0.6793,  0.4319, -0.4239, -0.5655, -0.5720,  0.4163),
]


def sq_sdf_single(world, sphere_xyz, sphere_r=0.005):
    """ESDF query against a single-SQ world. Returns positive = collision."""
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


def build_single_sq_world(row):
    name, sx, sy, sz, eps1, eps2, cx, cy, cz, qx, qy, qz, qw = row
    sq = Superquadric(
        name=name,
        pose=[cx, cy, cz, qw, qx, qy, qz],  # CuRobo Pose: [x,y,z, qw,qx,qy,qz]
        radii=[sx, sy, sz],
        eps=[eps1, eps2],
    )
    cfg = WorldCollisionConfig(
        tensor_args=tensor_args,
        world_model=WorldConfig(superquadric=[sq]),
        cache={"obb": 0, "superquadric": 1},
    )
    return WorldPrimitiveCollision(cfg)


def get_retract_spheres():
    """Get robot sphere positions at retract configuration."""
    robot_cfg = load_yaml(join_path(get_robot_configs_path(), "franka.yml"))["robot_cfg"]
    joint_names = robot_cfg["kinematics"]["cspace"]["joint_names"]
    default_config = robot_cfg["kinematics"]["cspace"]["retract_config"]

    from curobo.geom.sdf.world import CollisionCheckerType
    # Build a minimal world (just a table) to get the motion gen
    table_world = WorldConfig.from_dict({
        "cuboid": {"table": {"dims": [2.0, 2.0, 0.2], "pose": [0, 0, -0.12, 1, 0, 0, 0]}}
    })
    motion_gen_config = MotionGenConfig.load_from_robot_config(
        robot_cfg,
        table_world,
        tensor_args,
        collision_checker_type=CollisionCheckerType.PRIMITIVE,
        num_ik_seeds=1,
        num_trajopt_seeds=1,
        num_graph_seeds=0,
        interpolation_steps=100,
        use_cuda_graph=False,
        collision_cache={"obb": 1},
    )
    mg = MotionGen(motion_gen_config)

    retract_q = tensor_args.to_device(default_config).view(1, -1)
    retract_js = JointState.from_position(retract_q, joint_names=mg.kinematics.joint_names)
    spheres = mg.kinematics.get_robot_as_spheres(retract_js.position)

    # Flatten nested list
    flat_spheres = []
    if isinstance(spheres, (list, tuple)):
        for batch_item in spheres:
            if isinstance(batch_item, (list, tuple)):
                flat_spheres.extend(batch_item)
            elif hasattr(batch_item, 'position'):
                flat_spheres.append(batch_item)

    result = []
    for s in flat_spheres:
        if hasattr(s, 'position') and float(s.radius) > 0:
            result.append((float(s.position[0]), float(s.position[1]), float(s.position[2]), float(s.radius)))
    return result


print("Loading Franka retract spheres...", flush=True)
retract_spheres = get_retract_spheres()
print(f"  Found {len(retract_spheres)} active spheres.", flush=True)

print("\n=== Robot spheres at retract pose ===")
for i, (x, y, z, r) in enumerate(retract_spheres):
    print(f"  sphere[{i:2d}]  ({x:7.4f}, {y:7.4f}, {z:7.4f})  r={r:.4f}")

print("\n=== Per-SQ collision check (ESDF, positive = collision) ===")
print(f"  {'SQ':10s}  {'min_sdf':>10s}  {'worst_sphere':>14s}  {'sphere_pos':>30s}")

colliding_pairs = []

for row in SCENE_SQS:
    name = row[0]
    world = build_single_sq_world(row)
    min_d = float('inf')
    worst = -1
    worst_xyz = None
    for i, (x, y, z, r) in enumerate(retract_spheres):
        d = sq_sdf_single(world, [x, y, z], sphere_r=r)
        if d < min_d:
            min_d = d
            worst = i
            worst_xyz = (x, y, z, r)

    flag = "  *** COLLISION ***" if min_d > 0 else ""
    if min_d > 0:
        colliding_pairs.append((name, min_d, worst, worst_xyz))
    wx, wy, wz, wr = worst_xyz if worst_xyz else (0,0,0,0)
    print(f"  {name:10s}  {min_d:+10.4f}  sphere[{worst:2d}] r={wr:.3f}  "
          f"({wx:7.4f},{wy:7.4f},{wz:7.4f}){flag}")

print("\n=== Summary ===")
if not colliding_pairs:
    print("  No genuine collisions found with ESDF kernel. The issue is in the GJK path.")
else:
    print(f"  {len(colliding_pairs)} genuine collision(s):")
    for name, d, sph_idx, (x, y, z, r) in colliding_pairs:
        print(f"    {name}: sdf={d:+.4f}m  sphere[{sph_idx}] at ({x:.4f},{y:.4f},{z:.4f}) r={r:.4f}")
    print()
    max_penetration = max(d for _, d, _, _ in colliding_pairs)
    print(f"  Max penetration depth: {max_penetration:.4f}m")
    print(f"  Required tolerance to clear: ~{max_penetration:.4f}m")
