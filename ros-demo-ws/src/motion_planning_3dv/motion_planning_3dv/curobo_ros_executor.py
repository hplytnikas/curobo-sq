import argparse
import rclpy
from dataclasses import dataclass
from typing import List, Optional
from rclpy.node import Node
from rclpy.action import ActionClient, ActionServer
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor

from geometry_msgs.msg import PoseStamped, Point
from sensor_msgs.msg import JointState as RosJointState
from control_msgs.action import FollowJointTrajectory, GripperCommand
from trajectory_msgs.msg import JointTrajectoryPoint
from builtin_interfaces.msg import Duration
from visualization_msgs.msg import Marker, MarkerArray
from superquadric_interfaces.msg import SuperquadricArray
from superquadric_interfaces.action import Move, Grasp, Release

import torch
import numpy as np
import time

import curobo.runtime as runtime
from curobo.motion_planner import MotionPlanner, MotionPlannerCfg
from curobo.types import GoalToolPose, JointState as CuroboJointState
from curobo._src.types.pose import Pose as CuroboPose
from curobo._src.geom.types import SceneCfg, Superquadric
from curobo._src.collision.attachment_manager import AttachmentManager
from curobo._src.types.device_cfg import DeviceCfg

from motion_planning_3dv.sq_utils import sample_superquadric_mesh

@dataclass
class Trajectory:
    """CPU-side clean container holding pure host arrays for execution."""
    positions: np.ndarray    # Shape: (num_states, num_joints)
    velocities: np.ndarray   # Shape: (num_states, num_joints)
    joint_names: List[str]

class CuroboPlannerNode(Node):
    def __init__(self, robot_cfg="franka.yml", scene_cfg="collision_test.yml"):
        super().__init__('curobo_planner_node')
        if not self.has_parameter('use_sim_time'):
            self.declare_parameter('use_sim_time', True)
        else:
            from rclpy.parameter import Parameter
            self.set_parameters([Parameter('use_sim_time', Parameter.Type.BOOL, True)])
        self.timescale = 1.0
        self.max_dynamic_obstacles = 100

        self.cb_group = ReentrantCallbackGroup()

        self.ground_plane = Superquadric(
            name="ground_plane",
            pose=[0.0, 0.0, -0.05, 1.0, 0.0, 0.0, 0.0], # Placed slightly below Z=0
            radii=[2.0, 2.0, 0.05],                     # Wide 4x4m area, thin 0.1m thickness
            shape=[1.0, 1.0]                            # Cube-like (sharp edges)
        )

        # Pre-allocate the cache by filling the scene with "hidden" objects
        initial_sqs = []
        for i in range(self.max_dynamic_obstacles):
            initial_sqs.append(
                Superquadric(
                    name=f"dynamic_sq_{i}",
                    pose=[0.0, 0.0, -10.0, 1.0, 0.0, 0.0, 0.0],  # Hidden far underground
                    radii=[0.01, 0.01, 0.01],
                    shape=[1.0, 1.0]
                )
            )

        initial_sqs[0] = self.ground_plane

        self.scene_config = SceneCfg(
            superquadric=initial_sqs
        )

        self.is_holding_object = False

        # Initialize cuRobo Motion Planner
        self.get_logger().info("Initializing cuRobo Motion Planner...")
        config = MotionPlannerCfg.create(
            robot=robot_cfg,
            scene_model=scene_cfg,
        )

        # Override scene_model config with our explicitly declared scene
        config.scene_collision_cfg.scene_model = self.scene_config


        # configure planner and attachment manager
        self.planner = MotionPlanner(config)
        self.planner.warmup(enable_graph=True, num_warmup_iterations=5)
        self.get_logger().info("cuRobo warmup complete. Ready for goal poses.")

        self.device_config = DeviceCfg(device=torch.device("cuda"))

        self.ee_link_name = self.planner.tool_frames[0]

        self.attachment_manager = AttachmentManager(
            kinematics=self.planner.kinematics,
            scene_collision=self.planner.scene_collision_checker,
            device_cfg=self.device_config
        )

        # ROS 2 Interfaces
        self._action_server = ActionServer(
            self,
            Move,
            'move_arm',
            execute_callback=self.execute_move,
            callback_group=self.cb_group
        )

        self._grasp_server = ActionServer(
            self, Grasp, 'grasp_object',
            execute_callback=self.execute_grasp,
            callback_group=self.cb_group
        )

        self._release_server = ActionServer(
            self, Release, 'release_object',
            execute_callback=self.execute_release,
            callback_group=self.cb_group
        )

        self._action_client = ActionClient(
            self,
            FollowJointTrajectory,
            '/joint_trajectory_controller/follow_joint_trajectory',
            callback_group=self.cb_group
        )

        self._gripper_action_client = ActionClient(
            self,
            GripperCommand,
            '/gripper_action_controller/gripper_cmd',
            callback_group=self.cb_group
        )

        self.joint_state_sub = self.create_subscription(
            RosJointState,
            'joint_states',
            self.joint_state_callback,
            10,
            callback_group=self.cb_group
        )

        self.scene_sub = self.create_subscription(
            SuperquadricArray,
            'scene_superquadrics',
            self.scene_update_callback,
            10,
            callback_group=self.cb_group
        )

        # Marker publisher for RViz visualization
        self.marker_pub = self.create_publisher(
            MarkerArray,
            'curobo_scene_markers',
            10
        )

        self.attached_spheres_marker_pub = self.create_publisher(
            MarkerArray,
            'curobo_attached_spheres_markers',
            10
        )

        # State caching & translation map
        self.current_joint_state = None
        self.arm_joint_map = {
            "panda_joint1": "fp3_joint1",
            "panda_joint2": "fp3_joint2",
            "panda_joint3": "fp3_joint3",
            "panda_joint4": "fp3_joint4",
            "panda_joint5": "fp3_joint5",
            "panda_joint6": "fp3_joint6",
            "panda_joint7": "fp3_joint7",
        }

        self.gripper_joint_map = {
            "panda_finger_joint1": "fp3_finger_joint1"
        }

        self.reverse_total_map = {
            **{v: k for k, v in self.arm_joint_map.items()},
            **{v: k for k, v in self.gripper_joint_map.items()}
        }

        # Publish the static markers once on startup
        self.publish_scene_markers()

    def scene_update_callback(self, msg: SuperquadricArray):
        """Dynamically updates the cuRobo collision world using a pre-allocated cache pool."""
        self.get_logger().info(f"Received scene update with {len(msg.superquadrics)} superquadrics.")

        sq_list = [self.ground_plane]

        # Map the active objects from the ROS message
        for i, sq_msg in enumerate(msg.superquadrics):
            if i + 1 >= self.max_dynamic_obstacles:
                self.get_logger().warn(f"Exceeded max obstacles ({self.max_dynamic_obstacles})! Ignoring extras.")
                break

            curobo_sq = Superquadric(
                name=f"dynamic_sq_{i}",
                pose=[
                    sq_msg.pose.position.x,
                    sq_msg.pose.position.y,
                    sq_msg.pose.position.z,
                    sq_msg.pose.orientation.w,
                    sq_msg.pose.orientation.x,
                    sq_msg.pose.orientation.y,
                    sq_msg.pose.orientation.z
                ],
                radii=[sq_msg.x, sq_msg.y, sq_msg.z],
                shape=[sq_msg.e1, sq_msg.e2]
            )
            sq_list.append(curobo_sq)

        # Pad the remainder of the array up to max_dynamic_obstacles This keeps
        # the underlying tensor size static so the CUDA graph doesn't break!
        for i in range(len(sq_list), self.max_dynamic_obstacles):
            sq_list.append(
                Superquadric(
                    name=f"dynamic_sq_{i}",
                    pose=[0.0, 0.0, -10.0, 1.0, 0.0, 0.0, 0.0], # Hide unused ones out of bounds
                    radii=[0.01, 0.01, 0.01],
                    shape=[1.0, 1.0]
                )
            )

        # Update the stored configuration and inject into the planner
        self.scene_config = SceneCfg(
            superquadric=sq_list
        )
        self.planner.update_world(self.scene_config)

        # Update RViz visualization
        self.publish_scene_markers()

    def publish_scene_markers(self):
        """Translates stored cuRobo scene shapes into highly accurate 3D Mesh Markers."""
        marker_array = MarkerArray()

        if self.scene_config.superquadric is not None:
            for idx, sq in enumerate(self.scene_config.superquadric):
                marker = Marker()
                marker.header.frame_id = "world"
                marker.header.stamp = self.get_clock().now().to_msg()
                marker.ns = "accurate_superquadrics"
                marker.id = idx
                marker.type = Marker.TRIANGLE_LIST
                marker.action = Marker.ADD

                # Geometry Base Frame Origin and Orientation
                marker.pose.position.x = float(sq.pose[0])
                marker.pose.position.y = float(sq.pose[1])
                marker.pose.position.z = float(sq.pose[2])
                marker.pose.orientation.w = float(sq.pose[3])
                marker.pose.orientation.x = float(sq.pose[4])
                marker.pose.orientation.y = float(sq.pose[5])
                marker.pose.orientation.z = float(sq.pose[6])

                marker.scale.x = 1.0
                marker.scale.y = 1.0
                marker.scale.z = 1.0

                marker.color.r = 0.1
                marker.color.g = 0.7
                marker.color.b = 0.9
                marker.color.a = 0.5

                marker.points = sample_superquadric_mesh(sq.radii, sq.shape, grid_res=32)
                marker_array.markers.append(marker)

        if marker_array.markers:
            self.marker_pub.publish(marker_array)

    def publish_attached_spheres_markers(self):
        """Extracts local attached object spheres and maps them to the physical tf tree hand link."""
        if not self.is_holding_object:
            return

        # Access the underlying kinematics parameters holding the link-local allocations
        kparams = self.attachment_manager.kinematics_params
        try:
            link_sphere_idx = kparams.get_sphere_index_from_link_name("attached_object")
        except Exception as e:
            self.get_logger().error(f"Failed to find attached_object link index: {e}")
            return

        local_spheres = kparams.link_spheres[0, link_sphere_idx, :]

        self.get_logger().error(f"Trying to publish: {local_spheres}")

        marker_array = MarkerArray()
        sphere_counter = 0

        for i in range(local_spheres.shape[0]):
            radius = float(local_spheres[i, 3])

            # Filter out empty or unallocated padding slots
            if radius <= 0.0:
                continue

            marker = Marker()
            marker.header.frame_id = "fp3_hand"
            marker.frame_locked = True
            marker.ns = "attached_collision_spheres_tf"
            marker.id = sphere_counter
            marker.type = Marker.SPHERE
            marker.action = Marker.ADD

            marker.pose.position.x = float(local_spheres[i, 0])
            marker.pose.position.y = float(local_spheres[i, 1])
            marker.pose.position.z = float(local_spheres[i, 2])
            marker.pose.orientation.w = 1.0

            marker.scale.x = radius * 2.0
            marker.scale.y = radius * 2.0
            marker.scale.z = radius * 2.0

            marker.color.r = 1.0
            marker.color.g = 0.4
            marker.color.b = 0.0
            marker.color.a = 0.65

            marker_array.markers.append(marker)
            sphere_counter += 1

        if marker_array.markers:
            self.attached_spheres_marker_pub.publish(marker_array)

    def delete_attached_spheres_markers(self):
        """Publishes a DELETEALL command to wipe out all active attached collision markers."""
        marker_array = MarkerArray()

        marker = Marker()

        marker.header.frame_id = "fp3_hand"
        marker.ns = "attached_collision_spheres_tf"

        marker.action = Marker.DELETEALL

        marker_array.markers.append(marker)

        self.attached_spheres_marker_pub.publish(marker_array)
        self.get_logger().info("Sent command to clear all attached collision sphere markers.")

    def joint_state_callback(self, msg: RosJointState):
        """Cache current joint states for the motion planner."""
        curobo_positions = {}
        for name, pos in zip(msg.name, msg.position):
            if name in self.reverse_total_map:
                curobo_positions[self.reverse_total_map[name]] = pos

        if len(curobo_positions) == len(self.reverse_total_map):
            pos_list = []
            valid = True
            for name in self.planner.joint_names:
                if name in curobo_positions:
                    pos_list.append(curobo_positions[name])
                else:
                    valid = False
                    break

            if valid:
                self.current_joint_state = CuroboJointState.from_position(
                    torch.tensor([pos_list], device="cuda", dtype=torch.float32),
                    joint_names=self.planner.joint_names,
                )

    def to_cpu_plan(self, curobo_interpolated_trajectory):
        """Helper to cleanly unload GPU resources into our standard host dataclass structure."""
        pos = curobo_interpolated_trajectory.position.detach().cpu().numpy().squeeze()
        vel = curobo_interpolated_trajectory.velocity.detach().cpu().numpy().squeeze()

        # Ensure consistent dimensions
        if pos.ndim == 1:
            pos = np.expand_dims(pos, axis=0)
        if vel.ndim == 1:
            vel = np.expand_dims(vel, axis=0)

        return Trajectory(
            positions=pos,
            velocities=vel,
            joint_names=list(curobo_interpolated_trajectory.joint_names)
        )

    def _get_current_state(self):
        """Fetch either the current robot state or the default joint configuratiuon if unavailable"""
        if self.current_joint_state is not None:
            return self.current_joint_state
        self.get_logger().warn("No /joint_states received yet, using default joint state.")
        return CuroboJointState.from_position(
            self.planner.default_joint_state.position.unsqueeze(0),
            joint_names=self.planner.joint_names,
        )

    def _msg_to_goal_pose(self, msg):
        """datatype coversion helper"""
        pos = [msg.pose.position.x, msg.pose.position.y, msg.pose.position.z]
        quat = [msg.pose.orientation.w, msg.pose.orientation.x, msg.pose.orientation.y, msg.pose.orientation.z]
        return GoalToolPose(
            tool_frames=self.planner.tool_frames,
            position=torch.tensor([[[[[*pos]]]]], device="cuda", dtype=torch.float32),
            quaternion=torch.tensor([[[[[*quat]]]]], device="cuda", dtype=torch.float32),
        )

    async def execute_move(self, goal_handle):
        """move the robotic arm to the target pose"""
        msg = goal_handle.request.target_pose
        self.get_logger().info("Received standard Move action.")
        self.publish_scene_markers()

        # plan trajectory
        q_start = self._get_current_state()
        goal_pose = self._msg_to_goal_pose(msg)
        action_result = Move.Result()

        result = self.planner.plan_pose(goal_pose, q_start)
        if result is None or result.success is None or not result.success.any():
            self.get_logger().error("✗ Move Planning failed.")
            goal_handle.abort()
            action_result.success = False
            return action_result

        plan = self.to_cpu_plan(result.get_interpolated_plan())
        dt = self.planner.trajopt_solver.config.interpolation_dt

        gripper_target_pos = 0.0 if self.is_holding_object else 0.04

        # forward to controller
        exec_success = await self.execute_trajectory_ros2(plan, dt, gripper_target_pos, timescale=self.timescale)

        if not exec_success:
            goal_handle.abort()
            action_result.success = False
            return action_result

        goal_handle.succeed()
        action_result.success = True
        return action_result

    async def execute_grasp(self, goal_handle):
        """execute a three-phase grasping maneuver"""
        action_result = Grasp.Result()

        if self.is_holding_object:
            self.get_logger().warn("Rejecting Grasp: System is already holding an object.")
            goal_handle.abort()
            action_result.success = False
            return action_result

        msg = goal_handle.request.target_pose
        sq_obstacles_msg = goal_handle.request.obstacles

        self.get_logger().info(f"Received Grasp action for {len(sq_obstacles_msg)} objects.")
        self.publish_scene_markers()

        # plan trajectory
        q_start = self._get_current_state()
        goal_pose = self._msg_to_goal_pose(msg)

        results = self.planner.plan_grasp(
            goal_pose, q_start,
            plan_approach_to_grasp=True,
            plan_grasp_to_lift=True,
            grasp_lift_in_tool_frame=True,
        )

        if results is None or results.success is None or not results.success.any():
            self.get_logger().error("✗ Grasp Planning failed.")
            goal_handle.abort()
            action_result.success = False
            return action_result

        plans_to_execute = [
            self.to_cpu_plan(results.approach_interpolated_trajectory),
            self.to_cpu_plan(results.grasp_interpolated_trajectory),
            self.to_cpu_plan(results.lift_interpolated_trajectory)
        ]

        # Exact middle-trajectory extraction for grasped object attachment
        grasp_positions = plans_to_execute[1].positions
        if grasp_positions.ndim == 3:
            grasp_final_pos_np = grasp_positions[:, -1, :]
        else:
            grasp_final_pos_np = grasp_positions[-1, :]
            grasp_final_pos_np = np.expand_dims(grasp_final_pos_np, axis=0)

        grasp_final_pos_np = grasp_final_pos_np[..., :8]

        q_grasp_attachment = CuroboJointState.from_position(
            torch.tensor([grasp_final_pos_np], device="cuda", dtype=torch.float32),
            joint_names=self.planner.joint_names
        )

        world_objects_pose_offset = CuroboPose(
            position=torch.tensor(
                [[msg.pose.position.x, msg.pose.position.y, msg.pose.position.z]],
                device="cuda", dtype=torch.float32
            ),
            quaternion=torch.tensor(
                [[msg.pose.orientation.w, msg.pose.orientation.x, msg.pose.orientation.y, msg.pose.orientation.z]],
                device="cuda", dtype=torch.float32
            )
        )

        dt = self.planner.trajopt_solver.config.interpolation_dt

        # execute plans
        for phase_idx, plan in enumerate(plans_to_execute):
            gripper_target_pos = 0.035 if phase_idx == 0 else 0.0
            exec_success = await self.execute_trajectory_ros2(plan, dt, gripper_target_pos, timescale=self.timescale)

            if not exec_success:
                self.get_logger().error(f"✗ Trajectory execution failed during phase {phase_idx + 1}.")
                goal_handle.abort()
                action_result.success = False
                return action_result

        # determine relative position of grasped objects
        curobo_obstacles = []
        for sq in sq_obstacles_msg:
            curobo_obstacles.append(
                Superquadric(
                    name=f"attached_sq_{len(curobo_obstacles)}",
                    pose=[sq.pose.position.x, sq.pose.position.y, sq.pose.position.z,
                          sq.pose.orientation.w, sq.pose.orientation.x, sq.pose.orientation.y, sq.pose.orientation.z],
                    radii=[sq.x, sq.y, sq.z],
                    shape=[sq.e1, sq.e2]
                )
            )

        # attach grasped object to robot model
        self.attachment_manager.attach(
            joint_states=q_grasp_attachment,
            obstacles=curobo_obstacles,
            link_name="attached_object",
            disable_obstacle_names=None
        )

        self.is_holding_object = True
        self.publish_attached_spheres_markers()

        self.get_logger().info("✓ Grasp completed and object safely attached to kinematic model.")
        goal_handle.succeed()
        action_result.success = True
        return action_result

    async def execute_release(self, goal_handle):
        """release the currently grasped object"""
        action_result = Release.Result()

        if not self.is_holding_object:
            self.get_logger().warn("Rejecting Release: Not holding any object.")
            goal_handle.abort()
            action_result.success = False
            return action_result

        self.get_logger().info("Received Release action. Opening gripper...")

        # Execute release
        exec_success = await self._actuate_gripper_async(target_pos=0.04)
        if not exec_success:
            self.get_logger().error("✗ Failed to actuate gripper open.")
            goal_handle.abort()
            action_result.success = False
            return action_result

        # Detach from robot model
        self.attachment_manager.detach(link_name="attached_object")
        self.is_holding_object = False
        self.delete_attached_spheres_markers()

        self.get_logger().info("✓ Object released and detached from kinematic model.")
        goal_handle.succeed()
        action_result.success = True
        return action_result

    async def execute_trajectory_ros2(self, interpolated_plan, dt, gripper_pos, timescale=1.0):
        """execute a cuRobo plan through ros2_control"""
        pos = interpolated_plan.positions
        vel = interpolated_plan.velocities / timescale
        curobo_names = interpolated_plan.joint_names

        arm_goal = FollowJointTrajectory.Goal()
        arm_indices = []

        for i, name in enumerate(curobo_names):
            if name in self.arm_joint_map:
                arm_indices.append(i)
                arm_goal.trajectory.joint_names.append(self.arm_joint_map[name])

        num_states = pos.shape[0]
        last_kept_idx = 0
        ros_point_counter = 0

        # convert plan to ros2 control trajectories
        for i in range(num_states):
            if i > 0 and i < (num_states - 1):
                if np.allclose(pos[i, arm_indices], pos[last_kept_idx, arm_indices], atol=1e-6):
                    continue
            last_kept_idx = i

            t_sec = ros_point_counter * dt * timescale
            duration = Duration(sec=int(t_sec), nanosec=int((t_sec - int(t_sec)) * 1e9))
            ros_point_counter += 1

            if arm_indices:
                pt = JointTrajectoryPoint()
                pt.positions = [float(pos[i, idx]) for idx in arm_indices]
                pt.velocities = [float(vel[i, idx]) for idx in arm_indices]
                pt.time_from_start = duration
                arm_goal.trajectory.points.append(pt)

        # forward plan to controller nodes
        if arm_indices:
            if not self._action_client.wait_for_server(timeout_sec=2.0):
                self.get_logger().error('Arm action server not available!')
                return False

            self.get_logger().info("Sending trajectory goal to the arm...")
            send_goal_future = await self._action_client.send_goal_async(arm_goal)

            if not send_goal_future.accepted:
                self.get_logger().error('Arm goal rejected by controller.')
                return False

            arm_result = await send_goal_future.get_result_async()
            if arm_result.result.error_code != 0:
                self.get_logger().error(f'Arm tracking failed with error code: {arm_result.result.error_code}')
                return False
            self.get_logger().info("Arm movement complete.")

        return await self._actuate_gripper_async(gripper_pos)

    async def _actuate_gripper_async(self, target_pos: float):
        """Helper to actuate the gripper independently of arm motions."""
        if not self._gripper_action_client.wait_for_server(timeout_sec=2.0):
            self.get_logger().error('Parallel Gripper action server not available!')
            return False

        gripper_goal = GripperCommand.Goal()
        gripper_goal.command.position = float(target_pos)
        gripper_goal.command.max_effort = float(100)

        self.get_logger().info(f"Sending gripper command: position={target_pos}...")
        gripper_send_future = await self._gripper_action_client.send_goal_async(gripper_goal)

        if not gripper_send_future.accepted:
            self.get_logger().error('Gripper goal rejected by controller.')
            return False

        gripper_result = await gripper_send_future.get_result_async()

        if not gripper_result.result.reached_goal and not gripper_result.result.stalled:
            self.get_logger().error('Gripper failed to complete action target successfully.')
            return False

        self.get_logger().info("Gripper actuation complete.")
        time.sleep(1)
        return True


def main(args=None):
    rclpy.init(args=args)
    parser = argparse.ArgumentParser(description="cuRobo ROS 2 Motion Planner Node")
    parser.add_argument("--robot", type=str, default="franka.yml", help="Robot config file")
    parser.add_argument("--scene", type=str, default="collision_test.yml", help="Scene config file")

    parsed_args, _ = parser.parse_known_args()

    node = CuroboPlannerNode(robot_cfg=parsed_args.robot, scene_cfg=parsed_args.scene)

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Shutting down node...")
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

if __name__ == '__main__':
    main()
