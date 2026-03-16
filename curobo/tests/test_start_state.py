"""Reproduce the check_start_state failure using the same code path.

Builds a minimal MotionGen with the SQ world and checks the retract pose,
then prints the raw collision cost values from the rollout.

Run with:
  PATH=/usr/local/cuda-12.8/bin:/usr/bin:$PATH ~/isaacsim/python.sh tests/test_start_state.py
"""

import sys
sys.path.insert(0, "src")

import torch
import numpy as np
from curobo.geom.types import Cuboid, Superquadric, WorldConfig
from curobo.geom.sdf.world import WorldCollisionConfig, CollisionCheckerType
from curobo.types.base import TensorDeviceType
from curobo.types.state import JointState
from curobo.util_file import get_robot_configs_path, get_world_configs_path, join_path, load_yaml
from curobo.wrap.reacher.motion_gen import MotionGen, MotionGenConfig, MotionGenStatus

tensor_args = TensorDeviceType(device=torch.device("cuda:0"), dtype=torch.float32)

# Same SQs from the headless diagnostic (after tolerance=0.01m shrink)
# [name, sx, sy, sz, eps1, eps2, cx, cy, cz, qx, qy, qz, qw]
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

print("Building SQ world config...", flush=True)

# Table OBB (same as simulation: collision_table.yml + pose[2] -= 0.02)
table_cuboid = Cuboid(
    name="table",
    dims=[5.0, 5.0, 0.2],
    pose=[0.0, 0.0, -0.12, 1, 0, 0, 0],  # [x,y,z, qw,qx,qy,qz]
)

# SQ objects (CuRobo pose convention [cx,cy,cz, qw,qx,qy,qz])
sq_objs = [
    Superquadric(
        name=name,
        pose=[cx, cy, cz, qw, qx, qy, qz],
        radii=[sx, sy, sz],
        eps=[eps1, eps2],
    )
    for name, sx, sy, sz, eps1, eps2, cx, cy, cz, qx, qy, qz, qw in SCENE_SQS
]

collision_world = WorldConfig(cuboid=[table_cuboid], superquadric=sq_objs)

print("Building MotionGen...", flush=True)
robot_cfg = load_yaml(join_path(get_robot_configs_path(), "franka.yml"))["robot_cfg"]
motion_gen_config = MotionGenConfig.load_from_robot_config(
    robot_cfg,
    collision_world,
    tensor_args,
    collision_checker_type=CollisionCheckerType.PRIMITIVE,
    num_ik_seeds=1,
    num_trajopt_seeds=1,
    num_graph_seeds=0,
    interpolation_steps=100,
    use_cuda_graph=False,
    collision_cache={"obb": 1, "superquadric": len(sq_objs)},
)
mg = MotionGen(motion_gen_config)

print("Checking retract pose...", flush=True)
retract_q = mg.get_retract_config().view(1, -1)
retract_js = JointState.from_position(retract_q, joint_names=mg.kinematics.joint_names)

# Direct rollout to get the raw constraint value
from curobo.geom.sdf.world import CollisionQueryBuffer
joint_pos_3d = retract_q.unsqueeze(1)  # [1, 1, n_dof]
metrics = mg.rollout_fn.rollout_constraint(joint_pos_3d, use_batch_env=False)
print(f"  raw constraint value: {metrics.constraint.item():.6f}  (0 = feasible)")
print(f"  feasible: {metrics.feasible.item()}")

# Inspect SQ params inside MotionGen's world collision checker
wc = mg.rollout_fn.primitive_collision_constraint.world_coll_checker
print(f"  n_env_sq: {wc._env_n_superquadrics}")
sq_params_np = wc._sq_tensor_list[0][0].cpu().numpy()  # [n_obs, 12]
print(f"  sq_params shape: {wc._sq_tensor_list[0].shape}")
print("  SQ params [sx,sy,sz, eps1,eps2, cx,cy,cz, qx,qy,qz,qw]:")
for i in range(sq_params_np.shape[0]):
    p = sq_params_np[i]
    if wc._sq_tensor_list[1][0, i]:
        print(f"    [{i}] sx={p[0]:.4f} sy={p[1]:.4f} sz={p[2]:.4f} eps={p[3]:.3f},{p[4]:.3f} c=({p[5]:.3f},{p[6]:.3f},{p[7]:.3f}) q=({p[8]:.3f},{p[9]:.3f},{p[10]:.3f},{p[11]:.3f})")

# Get actual robot sphere positions from kinematics
retract_spheres_t = mg.kinematics.get_robot_as_spheres(retract_q)
if isinstance(retract_spheres_t, (list, tuple)):
    flat = []
    for bi in retract_spheres_t:
        if isinstance(bi, (list, tuple)): flat.extend(bi)
        else: flat.append(bi)
    print(f"  robot spheres (first 5):")
    for i, s in enumerate(flat[:5]):
        if hasattr(s, 'position'):
            print(f"    [{i}] ({float(s.position[0]):.4f},{float(s.position[1]):.4f},{float(s.position[2]):.4f}) r={float(s.radius):.4f}")

# Direct SQ distance check (non-ESDF, same path as constraint)
from curobo.curobolib.geom import SdfSphereSuperquadric
wc = mg.rollout_fn.primitive_collision_constraint.world_coll_checker
sq_params = wc._sq_tensor_list[0]   # [1, n_obs, 12]
sq_enable = wc._sq_tensor_list[1]   # [1, n_obs]
n_env_sq = wc._env_n_superquadrics  # [1]
n_obs = sq_params.shape[1]
b, h, n, _ = robot_spheres.shape
buf = CollisionQueryBuffer.initialize_from_shape(robot_spheres.shape, tensor_args, wc.collision_types)
weight = tensor_args.to_device([1.0])
act_dist = tensor_args.to_device([0.0])
dist = SdfSphereSuperquadric.apply(
    robot_spheres,
    buf.superquadric_collision_buffer.distance_buffer,
    buf.superquadric_collision_buffer.grad_distance_buffer,
    buf.superquadric_collision_buffer.sparsity_index_buffer,
    weight, act_dist,
    sq_params, sq_enable, n_env_sq,
    wc._env_n_superquadrics,  # env_query_idx
    n_obs, b, h, n,
    True, False,  # compute_distance, use_batch_env
    False, True, False,  # return_loss, sum_collisions, compute_esdf
)
print(f"  SQ cost (non-ESDF, sum): {dist.max().item():.6f}  (0 = no collision)")
dist_esdf = SdfSphereSuperquadric.apply(
    robot_spheres,
    buf.superquadric_collision_buffer.distance_buffer,
    buf.superquadric_collision_buffer.grad_distance_buffer,
    buf.superquadric_collision_buffer.sparsity_index_buffer,
    weight, act_dist,
    sq_params, sq_enable, n_env_sq,
    wc._env_n_superquadrics,
    n_obs, b, h, n,
    True, False, False, True, True,
)
print(f"  SQ SDF (ESDF): {dist_esdf.max().item():.6f}  (positive = collision)")

valid, status = mg.check_start_state(retract_js)
print(f"  check_start_state: valid={valid}  status={status}")

if not valid:
    print("\n--- Checking OBB-only (no SQ) ---")
    obb_only_world = WorldConfig(cuboid=[table_cuboid])
    motion_gen_config2 = MotionGenConfig.load_from_robot_config(
        robot_cfg,
        obb_only_world,
        tensor_args,
        collision_checker_type=CollisionCheckerType.PRIMITIVE,
        num_ik_seeds=1,
        num_trajopt_seeds=1,
        num_graph_seeds=0,
        interpolation_steps=100,
        use_cuda_graph=False,
        collision_cache={"obb": 1},
    )
    mg2 = MotionGen(motion_gen_config2)
    valid2, status2 = mg2.check_start_state(retract_js)
    print(f"  OBB-only check: valid={valid2}  status={status2}")

    print("\n--- Checking each SQ individually ---")
    for sq in sq_objs:
        sq_world = WorldConfig(cuboid=[table_cuboid], superquadric=[sq])
        mg_cfg_sq = MotionGenConfig.load_from_robot_config(
            robot_cfg,
            sq_world,
            tensor_args,
            collision_checker_type=CollisionCheckerType.PRIMITIVE,
            num_ik_seeds=1, num_trajopt_seeds=1, num_graph_seeds=0,
            interpolation_steps=100, use_cuda_graph=False,
            collision_cache={"obb": 1, "superquadric": 1},
        )
        mg_sq = MotionGen(mg_cfg_sq)
        v, s = mg_sq.check_start_state(retract_js)
        flag = "  *** COLLISION ***" if not v else ""
        print(f"  {sq.name:10s}: valid={v}  status={s}{flag}")

print("\nDone.")
