"""Test GJK path (sphere_superquadric_clpt) with actual retract spheres and scene SQs.

Checks whether the local-AABB lower bound fix in superquadric_distance_kernel.cu
correctly prevents false-positive collisions for spheres far from the SQ surface.

Run with:
  PATH=/usr/local/cuda-12.8/bin:/usr/bin:$PATH ~/isaacsim/python.sh tests/test_gjk_retract.py
"""

import sys
sys.path.insert(0, "src")

import torch
from curobo.geom.types import Superquadric, WorldConfig
from curobo.geom.sdf.world import WorldPrimitiveCollision, WorldCollisionConfig, CollisionQueryBuffer
from curobo.types.base import TensorDeviceType

tensor_args = TensorDeviceType(device=torch.device("cuda:0"), dtype=torch.float32)

# Same SQs from the headless diagnostic (after tolerance=0.01m shrink)
# Layout for _superquadric_to_tensor: pose=[cx,cy,cz, qw,qx,qy,qz], radii=[sx,sy,sz], eps=[eps1,eps2]
SCENE_SQS = [
    ("sq_0",  0.0885, 0.3706, 0.3536, 0.2486, 0.3636, -0.2314, -0.6889,  0.0274, -0.5060, -0.4957, -0.4999,  0.4983),
    ("sq_3",  0.0957, 0.2982, 0.0603, 0.1444, 0.3041, -0.2661, -1.0000,  0.2137, -0.4968, -0.4446, -0.5520,  0.5007),
    ("sq_4",  0.1847, 0.0269, 0.0176, 0.1170, 0.2451,  0.0875, -1.0097, -0.1646, -0.4736, -0.4965, -0.5505,  0.4756),
    ("sq_5",  0.3053, 0.0257, 0.0167, 0.1133, 0.2149, -0.5566, -0.9134, -0.0423, -0.5050, -0.4706, -0.5366,  0.4853),
    ("sq_7",  0.2519, 0.0194, 0.0247, 0.1188, 0.2483, -0.5489, -0.4471, -0.1060, -0.4783, -0.5111, -0.5013,  0.5086),
    ("sq_8",  0.0845, 0.3163, 0.0559, 0.1338, 0.3207, -0.2620, -0.3534,  0.2184, -0.5052, -0.5533, -0.4438,  0.4916),
    ("sq_14", 0.1612, 0.0319, 0.0130, 0.1109, 0.2143,  0.0919, -0.3741, -0.1554, -0.4839, -0.5188, -0.5253,  0.4698),
    ("sq_15", 0.2149, 0.0476, 0.3758, 0.2819, 0.3816, -0.6097, -0.6793,  0.4319, -0.4239, -0.5655, -0.5720,  0.4163),
]

# Retract sphere positions from test_retract_collision.py output
# (x, y, z, r)
RETRACT_SPHERES = [
    (0.0000,  0.0000,  0.0850, 0.0340),
    (-0.1000,  0.0000,  0.0850, 0.0340),
    (0.0000, -0.0800,  0.3330, 0.0590),
    (0.0000, -0.0300,  0.3330, 0.0640),
    (0.0000,  0.0000,  0.2130, 0.0640),
    (0.0000,  0.0000,  0.1630, 0.0640),
    (0.0000,  0.0300,  0.3330, 0.0590),
    (0.0000,  0.0800,  0.3330, 0.0590),
    (-0.1156, -0.0000,  0.3651, 0.0590),
    (-0.1638, -0.0000,  0.3785, 0.0590),
    (-0.2467, -0.0000,  0.4015, 0.0540),
    (-0.2081, -0.0000,  0.3908, 0.0640),
    (-0.2831,  0.0600,  0.4946, 0.0560),
    (-0.2831,  0.0200,  0.4946, 0.0560),
    (-0.2824, -0.0200,  0.4970, 0.0560),
    (-0.2824, -0.0600,  0.4970, 0.0560),
    (-0.2229, -0.0000,  0.6060, 0.0590),
    (-0.2555, -0.0000,  0.5933, 0.0560),
    (0.0456,  0.0300,  0.7131, 0.0540),
    (0.0456,  0.0820,  0.7131, 0.0540),
    (-0.1595, -0.0000,  0.6333, 0.0540),
    (-0.1222,  0.0520,  0.6478, 0.0440),
    (-0.0813,  0.0800,  0.6530, 0.0260),
    (-0.0533,  0.0850,  0.6639, 0.0260),
    (-0.0253,  0.0900,  0.6748, 0.0260),
    (0.0026,  0.0950,  0.6856, 0.0260),
    (-0.0885,  0.0800,  0.6717, 0.0260),
    (-0.0606,  0.0850,  0.6825, 0.0260),
    (-0.0326,  0.0900,  0.6934, 0.0260),
    (-0.0046,  0.0950,  0.7043, 0.0260),
    (0.0456, -0.0090,  0.7131, 0.0540),
    (0.1359, -0.0000,  0.7305, 0.0490),
    (0.1289, -0.0000,  0.6962, 0.0490),
    (0.1259, -0.0000,  0.6815, 0.0490),
    (0.1179, -0.0000,  0.6270, 0.0490),
    (0.1355, -0.0400,  0.6132, 0.0280),
    (0.1551, -0.0200,  0.6092, 0.0280),
    (0.1542, -0.0600,  0.6043, 0.0240),
    (0.1738, -0.0400,  0.6004, 0.0240),
    (0.0566,  0.0530,  0.5914, 0.0270),
    (0.0774,  0.0318,  0.5872, 0.0270),
    (0.0982,  0.0106,  0.5830, 0.0270),
    (0.1190, -0.0106,  0.5788, 0.0270),
    (0.1398, -0.0318,  0.5746, 0.0270),
    (0.1606, -0.0530,  0.5704, 0.0270),
    (0.0492,  0.0566,  0.5725, 0.0260),
    (0.0734,  0.0318,  0.5676, 0.0260),
    (0.0942,  0.0106,  0.5634, 0.0260),
    (0.1150, -0.0106,  0.5592, 0.0260),
    (0.1358, -0.0318,  0.5550, 0.0260),
    (0.1601, -0.0566,  0.5501, 0.0260),
    (0.0462,  0.0566,  0.5578, 0.0260),
    (0.0705,  0.0318,  0.5529, 0.0260),
    (0.0912,  0.0106,  0.5487, 0.0260),
    (0.1120, -0.0106,  0.5445, 0.0260),
    (0.1328, -0.0318,  0.5403, 0.0260),
    (0.1571, -0.0566,  0.5354, 0.0260),
    (0.1251, -0.0354,  0.4843, 0.0150),
    (0.1376, -0.0424,  0.5103, 0.0150),
    (0.0558,  0.0354,  0.4984, 0.0150),
    (0.0544,  0.0424,  0.5272, 0.0150),
]

print("=== GJK path (WorldPrimitiveCollision) test ===\n")
print("Building world with all 8 SQs...", flush=True)

# Build SQ objects with CuRobo pose convention [cx,cy,cz, qw,qx,qy,qz]
sq_objs = []
for name, sx, sy, sz, eps1, eps2, cx, cy, cz, qx, qy, qz, qw in SCENE_SQS:
    sq_objs.append(Superquadric(
        name=name,
        pose=[cx, cy, cz, qw, qx, qy, qz],
        radii=[sx, sy, sz],
        eps=[eps1, eps2],
    ))

cfg = WorldCollisionConfig(
    tensor_args=tensor_args,
    world_model=WorldConfig(superquadric=sq_objs),
    cache={"obb": 0, "superquadric": len(sq_objs)},
)
world = WorldPrimitiveCollision(cfg)

# Build sphere tensor [n_spheres, 4] = [x, y, z, r]
spheres_list = [[x, y, z, r] for x, y, z, r in RETRACT_SPHERES]
sphere_tensor = tensor_args.to_device(spheres_list).unsqueeze(0).unsqueeze(0)  # [1,1,n,4]

print(f"Querying {len(RETRACT_SPHERES)} retract spheres against {len(sq_objs)} SQs...", flush=True)

q = CollisionQueryBuffer.initialize_from_shape(
    sphere_tensor.shape, tensor_args, world.collision_types
)

weight = tensor_args.to_device([1.0])
act_dist = tensor_args.to_device([0.0])
env_idx = tensor_args.to_device([0]).to(torch.int32)

world.get_sphere_distance(sphere_tensor, q, weight, act_dist,
                          env_query_idx=env_idx, compute_esdf=True)

# esdf values [n_spheres]: positive = collision (CuRobo convention)
if q.superquadric_collision_buffer is not None:
    esdf_vals = q.superquadric_collision_buffer.distance_buffer.view(-1)
    print(f"\nESDF values per sphere (positive = collision):")
    n_collide = 0
    for i, v in enumerate(esdf_vals.tolist()):
        x, y, z, r = RETRACT_SPHERES[i]
        flag = "  *** COLLISION ***" if v > 0 else ""
        print(f"  sphere[{i:2d}] ({x:7.4f},{y:7.4f},{z:7.4f}) r={r:.3f}  sdf={v:+.4f}{flag}")
        if v > 0:
            n_collide += 1
    print(f"\n  {n_collide}/{len(RETRACT_SPHERES)} spheres report collision via GJK path")
else:
    print("  No superquadric collision buffer found (GJK path not available or no SQs loaded)")

print("\nDone.")
