#!/usr/bin/env python3

import math
import sys
from time import sleep

import rclpy
from rclpy.action import ActionClient
from rclpy.node import Node

from geometry_msgs.msg import PoseStamped, Pose
from std_srvs.srv import Trigger
from superquadric_interfaces.msg import SceneObject
from superquadric_interfaces.srv import DeleteObject, InsertObject, ListObjects

from scipy.spatial.transform import Rotation as R
import numpy as np

# Import the newly separated Action APIs
from superquadric_interfaces.action import Move, Grasp, Release

def pose_to_matrix(position, orientation):
    """Converts ROS position and orientation into a 4x4 homogeneous transform matrix."""
    T = np.eye(4)
    # Extract translation
    T[:3, 3] = [position.x, position.y, position.z]
    # Extract rotation quaternion [x, y, z, w]
    q = [orientation.x, orientation.y, orientation.z, orientation.w]
    T[:3, :3] = R.from_quat(q).as_matrix()
    return T

def pos_quat_to_matrix(position, quat):
    """Converts ROS position and orientation into a 4x4 homogeneous transform matrix."""
    T = np.eye(4)
    # Extract translation
    T[:3, 3] = position
    # Extract rotation quaternion [x, y, z, w]
    T[:3, :3] = R.from_quat(quat).as_matrix()
    return T

def matrix_to_pose(T):
    """Converts a 4x4 homogeneous transform matrix back into a ROS Pose message."""
    pose = Pose()
    # Set translation
    pose.position.x = T[0, 3]
    pose.position.y = T[1, 3]
    pose.position.z = T[2, 3]
    # Set rotation
    q = R.from_matrix(T[:3, :3]).as_quat()  # Returns [x, y, z, w]
    pose.orientation.x = q[0]
    pose.orientation.y = q[1]
    pose.orientation.z = q[2]
    pose.orientation.w = q[3]
    return pose


def action_feedback_callback(feedback_msg):
    """Optional: Global function to handle live action feedback."""
    # Note: Depending on the action, feedback fields may vary (e.g., current_phase vs distance_to_goal)
    print(f"[Feedback] Update received from action server.")


def send_move_goal(node, client, position, orientation, description):
    """Helper function to execute standard free-space motions."""
    logger = node.get_logger()
    logger.info(f"Starting Action: {description}")

    goal = Move.Goal()
    goal.target_pose.header.frame_id = "world"
    goal.target_pose.pose.position.x, goal.target_pose.pose.position.y, goal.target_pose.pose.position.z = position
    goal.target_pose.pose.orientation.x, goal.target_pose.pose.orientation.y, goal.target_pose.pose.orientation.z, goal.target_pose.pose.orientation.w = orientation

    send_goal_future = client.send_goal_async(goal, feedback_callback=action_feedback_callback)
    rclpy.spin_until_future_complete(node, send_goal_future)

    goal_handle = send_goal_future.result()
    if not goal_handle.accepted:
        logger.error(f"Action Failed: '{description}' goal rejected by server.")
        return False

    result_future = goal_handle.get_result_async()
    rclpy.spin_until_future_complete(node, result_future)

    if not result_future.result().result.success:
        logger.error(f"Action Failed during execution: {result_future.result().result.message}")
        return False

    logger.info(f"Finished Action: {description}")
    return True


def send_grasp_goal(node, client, position, orientation, obstacles, description):
    """Helper function to execute the full approach-grasp-lift-attach sequence."""
    logger = node.get_logger()
    logger.info(f"Starting Grasp Action: {description}")

    goal = Grasp.Goal()
    goal.target_pose.header.frame_id = "world"
    goal.target_pose.pose.position.x, goal.target_pose.pose.position.y, goal.target_pose.pose.position.z = position
    goal.target_pose.pose.orientation.x, goal.target_pose.pose.orientation.y, goal.target_pose.pose.orientation.z, goal.target_pose.pose.orientation.w = orientation

    # Pass the geometry payload for the AttachmentManager
    goal.obstacles = obstacles

    send_goal_future = client.send_goal_async(goal, feedback_callback=action_feedback_callback)
    rclpy.spin_until_future_complete(node, send_goal_future)

    goal_handle = send_goal_future.result()
    if not goal_handle.accepted:
        logger.error(f"Grasp Action Failed: '{description}' goal rejected.")
        return False

    result_future = goal_handle.get_result_async()
    rclpy.spin_until_future_complete(node, result_future)

    if not result_future.result().result.success:
        logger.error(f"Grasp Action Failed during execution: {result_future.result().result.message}")
        return False

    logger.info(f"Finished Grasp Action: {description}")
    return True


def send_release_goal(node, client, description):
    """Helper function to open the gripper and flush the kinematic attachment tree."""
    logger = node.get_logger()
    logger.info(f"Starting Release Action: {description}")

    goal = Release.Goal() # Empty goal request

    send_goal_future = client.send_goal_async(goal)
    rclpy.spin_until_future_complete(node, send_goal_future)

    goal_handle = send_goal_future.result()
    if not goal_handle.accepted:
        logger.error(f"Release Action Failed: '{description}' goal rejected.")
        return False

    result_future = goal_handle.get_result_async()
    rclpy.spin_until_future_complete(node, result_future)

    if not result_future.result().result.success:
        logger.error(f"Release Action Failed during execution: {result_future.result().result.message}")
        return False

    logger.info(f"Finished Release Action: {description}")
    return True


def main(args=None):
    rclpy.init(args=args)

    node = rclpy.create_node('script_node')
    logger = node.get_logger()

    logger.info("=== Starting Procedural Sequence ===")

    # Initialize Service Clients
    update_service = node.create_client(Trigger, '/scene_manager/request_update')
    list_service = node.create_client(ListObjects, '/scene_manager/list_objects')
    add_service = node.create_client(InsertObject, '/scene_manager/insert_object')
    delete_service = node.create_client(DeleteObject, '/scene_manager/delete_object')

    # Initialize Action Clients
    move_client = ActionClient(node, Move, '/move_arm')
    grasp_client = ActionClient(node, Grasp, '/grasp_object')
    release_client = ActionClient(node, Release, '/release_object')

    # Verify all connections are alive before starting
    logger.info("Waiting for servers and services to come online...")
    services = [update_service, list_service, add_service, delete_service]
    for srv in services:
        if not srv.wait_for_service(timeout_sec=20.0):
            logger.error(f"Service {srv.srv_name} not available. Exiting.")
            sys.exit(1)

    if not move_client.wait_for_server(timeout_sec=20.0):
        logger.error("Action server /move_arm not available. Exiting.")
        sys.exit(1)

    if not grasp_client.wait_for_server(timeout_sec=20.0):
        logger.error("Action server /grasp_object not available. Exiting.")
        sys.exit(1)

    if not release_client.wait_for_server(timeout_sec=20.0):
        logger.error("Action server /release_object not available. Exiting.")
        sys.exit(1)

    # =========================================================================
    # STEP 1: Update Scene via Perception
    # =========================================================================
    logger.info("Step 1: Update Scene from perception")
    future = update_service.call_async(Trigger.Request())
    rclpy.spin_until_future_complete(node, future)
    logger.info(f"Step 1 Complete. Response: {future.result().message}")

    sleep(5)

    # =========================================================================
    # STEP 2: Load Scene and Find Relevant Object
    # =========================================================================
    logger.info("Step 2: Load Scene and find relevant object")
    future = list_service.call_async(ListObjects.Request())
    rclpy.spin_until_future_complete(node, future)
    object_list = future.result().objects

    def compare_obj_to_point(xyz, obj):
        p = [obj.pose.position.x, obj.pose.position.y, obj.pose.position.z]
        return math.dist(xyz, p)

    target_point = [0.5, 0.0, 0.32]
    closest_object = min(object_list, key=lambda x: compare_obj_to_point(target_point, x))
    logger.info(f"Step 2 Complete. Found closest object ID: {closest_object.id}")

    sleep(5)

    # =========================================================================
    # STEP 3: Grasping Object (Delegates kinematic attachment to the planner)
    # =========================================================================
    grasp_position = (0.5, 0.0, 0.34)
    grasp_orientation = (0.7, 0.7, 0.0, 0.0)

    world_to_ee = pos_quat_to_matrix(grasp_position, grasp_orientation)
    world_to_obj = pose_to_matrix(closest_object.pose.position, closest_object.pose.orientation)

    ee_to_obj = np.linalg.inv(world_to_ee) @ world_to_obj

    offset_obstacles = []
    for sq in closest_object.superquadrics:
        # Create a new local instance of the Superquadric message
        local_sq = type(sq)()
        local_sq.x = sq.x
        local_sq.y = sq.y
        local_sq.z = sq.z
        local_sq.e1 = sq.e1
        local_sq.e2 = sq.e2

        obj_to_sq = pose_to_matrix(sq.pose.position, sq.pose.orientation)

        ee_to_sq = ee_to_obj @ obj_to_sq

        # Copy original orientation
        local_sq.pose = matrix_to_pose(ee_to_sq)

        offset_obstacles.append(local_sq)

    success = send_grasp_goal(
        node=node,
        client=grasp_client,
        position=grasp_position,
        orientation=grasp_orientation,
        obstacles=offset_obstacles, # Pass the transformed frame geometry
        description="Step 3: Grasping object with offset layout coordinates"
    )

    # =========================================================================
    # STEP 4: Delete Grasped Object From Global Scene Manager
    # =========================================================================
    # The planner node handles its internal collision attachment, but the global
    # perception scene still needs to delete the static copy.
    logger.info("Step 4: Delete grasped object from global scene")
    req = DeleteObject.Request()
    req.id = closest_object.id

    future = delete_service.call_async(req)
    rclpy.spin_until_future_complete(node, future)
    logger.info("Step 4 Complete.")

    # =========================================================================
    # STEP 5: Move Around (Maintains attached geometry)
    # =========================================================================
    success = send_move_goal(
        node=node,
        client=move_client,
        position=(0.5, 0.3, 0.5),
        orientation=(1.0, 0.0, 0.0, 0.0),
        description="Step 5: Move around"
    )

    if not success:
        node.destroy_node()
        rclpy.shutdown()
        return

    release_position = (0.7, -0.3, 0.36)
    release_orientation = (0.7, 0.7, 0.0, 0.0)

    world_to_ee = pos_quat_to_matrix(release_position, release_orientation)

    success = send_move_goal(
        node=node,
        client=move_client,
        position=release_position,
        orientation=release_orientation,
        description="Step 5: Move around"
    )

    if not success:
        node.destroy_node()
        rclpy.shutdown()
        return

    # =========================================================================
    # STEP 6: Release Object via Action Server
    # =========================================================================
    success = send_release_goal(
        node=node,
        client=release_client,
        description="Step 6: Open gripper and detach object kinematics"
    )
    if not success:
        node.destroy_node()
        rclpy.shutdown()
        return

    safe_position = (release_position[0], release_position[1], release_position[2] + 0.1)

    success = send_move_goal(
        node=node,
        client=move_client,
        position=safe_position,
        orientation=release_orientation,
        description="Move up"
    )

    if not success:
        node.destroy_node()
        rclpy.shutdown()
        return

    # =========================================================================
    # STEP 7: Re-insert Object with Preserved & Updated Orientation
    # =========================================================================
    logger.info("Step 7: Constructing SceneObject message and updating scene...")

    world_to_obj = world_to_ee @ ee_to_obj

    new_scene_object = SceneObject()
    new_scene_object.header = closest_object.header
    new_scene_object.id = closest_object.id
    new_scene_object.pose = matrix_to_pose(world_to_obj)
    new_scene_object.min = closest_object.min
    new_scene_object.max = closest_object.max
    new_scene_object.superquadrics = closest_object.superquadrics

    insert_req = InsertObject.Request()
    insert_req.object = new_scene_object

    insert_future = add_service.call_async(insert_req)
    rclpy.spin_until_future_complete(node, insert_future)

    if insert_future.result() and insert_future.result().success:
        logger.info("Step 7 Complete: Scene manager successfully updated with the dropped SceneObject.")
    else:
        logger.error("Step 7 Failed: Scene manager rejected object insertion request.")

    # =========================================================================
    # STEP 8: Return Home
    # =========================================================================
    success = send_move_goal(
        node=node,
        client=move_client,
        position=(0.3, 0.0, 0.6),
        orientation=(1.0, 0.0, 0.0, 0.0),
        description="Step 8: Returning home"
    )

    # =========================================================================
    # CLEANUP
    # =========================================================================
    logger.info("=== All Sequence Steps Finished Successfully ===")
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
