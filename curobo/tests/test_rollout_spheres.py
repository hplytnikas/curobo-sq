"""Check exactly what sphere positions the rollout uses when checking the retract pose.

Run with:
  PATH=/usr/local/cuda-12.8/bin:/usr/bin:$PATH ~/isaacsim/python.sh tests/test_rollout_spheres.py
"""
import sys
sys.path.insert(0, "src")
import torch
from curobo.geom.types import Cuboid, Superquadric, WorldConfig
from curobo.geom.sdf.world import CollisionCheckerType
from curobo.types.base import TensorDeviceType
from curobo.types.state import JointState
from curobo.util_file import get_robot_configs_path, join_path, load_yaml
from curobo.wrap.reacher.motion_gen import MotionGen, MotionGenConfig

tensor_args = TensorDeviceType(device=torch.device("cuda:0"), dtype=torch.float32)

SCENE_SQS = [
    ("sq_0",  0.0885, 0.3706, 0.3536, 0.2486, 0.3636, -0.2314, -0.6889,  0.0274, -0.5060, -0.4957, -0.4999,  0.4983),
]

table_cuboid = Cuboid(name="table", dims=[5.0, 5.0, 0.2], pose=[0.0, 0.0, -0.12, 1, 0, 0, 0])
sq_objs = [Superquadric(name=n, pose=[cx, cy, cz, qw, qx, qy, qz], radii=[sx,sy,sz], eps=[e1,e2])
           for n,sx,sy,sz,e1,e2,cx,cy,cz,qx,qy,qz,qw in SCENE_SQS]
collision_world = WorldConfig(cuboid=[table_cuboid], superquadric=sq_objs)

robot_cfg = load_yaml(join_path(get_robot_configs_path(), "franka.yml"))["robot_cfg"]
mg_cfg = MotionGenConfig.load_from_robot_config(
    robot_cfg, collision_world, tensor_args,
    collision_checker_type=CollisionCheckerType.PRIMITIVE,
    num_ik_seeds=1, num_trajopt_seeds=1, num_graph_seeds=0,
    interpolation_steps=100, use_cuda_graph=False,
    collision_cache={"obb": 1, "superquadric": 1},
)
mg = MotionGen(mg_cfg)

retract_q = mg.get_retract_config().view(1, -1)
retract_js = JointState.from_position(retract_q, joint_names=mg.kinematics.joint_names)

# Get rollout horizon
horizon = mg.rollout_fn.dynamics_model.horizon
print(f"Rollout horizon: {horizon}")

# Build act_seq with retract pose at all timesteps
act_seq = retract_q.unsqueeze(1).expand(-1, horizon, -1).contiguous()
print(f"act_seq shape: {act_seq.shape}")

# Run forward dynamics to get sphere positions
state = mg.rollout_fn.dynamics_model.forward(mg.rollout_fn.start_state, act_seq)
robot_spheres = state.robot_spheres  # [1, horizon, n_spheres, 4]
print(f"robot_spheres shape: {robot_spheres.shape}")

# Show first timestep spheres
sph_t0 = robot_spheres[0, 0]  # [n_spheres, 4]
print(f"First timestep spheres (first 5):")
for i in range(min(5, sph_t0.shape[0])):
    s = sph_t0[i]
    print(f"  [{i}] ({s[0]:.4f},{s[1]:.4f},{s[2]:.4f}) r={s[3]:.4f}")

# Show last timestep spheres
sph_tn = robot_spheres[0, -1]
print(f"Last timestep spheres (first 5):")
for i in range(min(5, sph_tn.shape[0])):
    s = sph_tn[i]
    print(f"  [{i}] ({s[0]:.4f},{s[1]:.4f},{s[2]:.4f}) r={s[3]:.4f}")

# Check if all timesteps are the same
max_diff = (robot_spheres[0] - robot_spheres[0, 0:1]).abs().max().item()
print(f"Max variation across timesteps: {max_diff:.6f}")

# Now run the constraint check on just these spheres
print("\n--- Constraint check ---")
valid, status = mg.check_start_state(retract_js)
print(f"check_start_state: valid={valid} status={status}")

# Check OBB collision only (temporarily disable SQ)
mg.update_world(WorldConfig(cuboid=[table_cuboid]))
valid_obb, status_obb = mg.check_start_state(retract_js)
print(f"OBB-only: valid={valid_obb} status={status_obb}")

# Restore SQ world
mg.update_world(collision_world)

# Direct collision check: ESDF vs cost path
from curobo.geom.sdf.world import CollisionQueryBuffer
from curobo.curobolib.geom import SdfSphereSuperquadric

wc = mg.rollout_fn.primitive_collision_constraint.world_coll_checker
sph_t = robot_spheres  # [1, 1, 65, 4]

buf = CollisionQueryBuffer.initialize_from_shape(sph_t.shape, tensor_args, wc.collision_types)
weight = tensor_args.to_device([1.0])
act_dist = tensor_args.to_device([0.0])

# ESDF path (compute_esdf=True)
dist_esdf = wc.get_sphere_distance(sph_t, buf, weight, act_dist, compute_esdf=True)
print(f"\nESDF path (compute_esdf=True): max={dist_esdf.max().item():.4f} min={dist_esdf.min().item():.4f}")
print(f"  (positive = collision)")

# Cost path (compute_esdf=False, sum_collisions=True)
buf2 = CollisionQueryBuffer.initialize_from_shape(sph_t.shape, tensor_args, wc.collision_types)
dist_cost = wc.get_sphere_distance(sph_t, buf2, weight, act_dist, compute_esdf=False, sum_collisions=True)
print(f"\nCost path (compute_esdf=False): max={dist_cost.max().item():.4f} min={dist_cost.min().item():.4f}")
print(f"  (0 = no collision)")

# Check what the constraint weight is
print(f"\nconstraint weight: {mg.rollout_fn.primitive_collision_constraint.weight}")
print(f"constraint activation_distance: {mg.rollout_fn.primitive_collision_constraint.activation_distance}")
print(f"constraint classify: {mg.rollout_fn.primitive_collision_constraint.classify}")

# Check each OBB and SQ separately
print("\n--- OBB-only ESDF check ---")
from curobo.geom.types import WorldConfig as WC2
from curobo.geom.sdf.world import WorldPrimitiveCollision, WorldCollisionConfig as WCC
wc_obb_only = WorldPrimitiveCollision(WCC(
    tensor_args=tensor_args,
    world_model=WC2(cuboid=[table_cuboid]),
    cache={"obb": 1, "superquadric": 0}
))
buf_obb = CollisionQueryBuffer.initialize_from_shape(sph_t.shape, tensor_args, wc_obb_only.collision_types)
dist_obb = wc_obb_only.get_sphere_distance(sph_t, buf_obb, weight, act_dist, compute_esdf=True)
print(f"OBB ESDF: max={dist_obb.max().item():.4f} min={dist_obb.min().item():.4f}")

print("\n--- SQ-only ESDF check ---")
wc_sq_only = WorldPrimitiveCollision(WCC(
    tensor_args=tensor_args,
    world_model=WC2(superquadric=sq_objs),
    cache={"obb": 0, "superquadric": 1}
))
buf_sq = CollisionQueryBuffer.initialize_from_shape(sph_t.shape, tensor_args, wc_sq_only.collision_types)
dist_sq = wc_sq_only.get_sphere_distance(sph_t, buf_sq, weight, act_dist, compute_esdf=True)
print(f"SQ ESDF: max={dist_sq.max().item():.4f} min={dist_sq.min().item():.4f}")

# The max across all spheres
sdf_sq_per_sphere = dist_sq.view(-1)
print(f"SQ: {(sdf_sq_per_sphere > 0).sum().item()} of {sdf_sq_per_sphere.numel()} spheres in collision")
