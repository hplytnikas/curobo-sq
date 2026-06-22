import rclpy
from rclpy.node import Node

# Import standard and custom message/service types
from std_msgs.msg import Header
# Note: Replace 'your_custom_interfaces' with your actual package name
from superquadric_interfaces.msg import SceneObject, Scene, SuperquadricArray
from superquadric_interfaces.srv import ListObjects, InsertObject, DeleteObject
from std_srvs.srv import Trigger
from geometry_msgs.msg import Pose

from scipy.spatial.transform import Rotation as R
import numpy as np


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

class SceneManager(Node):
    def __init__(self):
        super().__init__('scene_manager')

        # --- State ---
        # Dictionary mapping object ID to SceneObject
        self.scene_objects = {}
        self.update_requested = False

        # --- Publishers ---
        self.flattened_sq_pub = self.create_publisher(
            SuperquadricArray,
            '/scene_superquadrics',
            10
        )

        # --- Subscribers ---
        self.scene_sub = self.create_subscription(
            Scene,
            '/scene',
            self.incoming_objects_callback,
            10
        )

        # --- Services ---
        self.srv_request_update = self.create_service(
            Trigger, '~/request_update', self.request_update_cb)
        self.srv_list_objects = self.create_service(
            ListObjects, '~/list_objects', self.list_objects_cb)
        self.srv_insert_object = self.create_service(
            InsertObject, '~/insert_object', self.insert_object_cb)
        self.srv_delete_object = self.create_service(
            DeleteObject, '~/delete_object', self.delete_object_cb)

        # --- Timers ---
        # Publish the flattened superquadrics at 10 Hz
        self.publish_timer = self.create_timer(0.1, self.publish_flattened_scene)

        self.get_logger().info("Scene Manager initialized.")

    # ==========================================
    # Callbacks
    # ==========================================

    def incoming_objects_callback(self, msg: Scene):
        """Consumes incoming arrays ONLY if an update was requested."""
        if not self.update_requested:
            return

        self.scene_objects = {}

        # Process the new objects (overwrite existing ones with the same ID)
        for obj in msg.objects:
            self.scene_objects[obj.id] = obj, obj.pose
            self.get_logger().info(f"Consumed object {obj.id} from topic.")

        # Reset the flag after fulfilling the request
        self.update_requested = False
        self.get_logger().info(f"Update consumed. Tracking {len(self.scene_objects)} objects total.")

    def request_update_cb(self, request, response):
        """Flags the node to consume the next incoming Scene message."""
        self.update_requested = True
        response.success = True
        self.get_logger().info("Update requested. Waiting for next message...")
        return response

    def list_objects_cb(self, request, response):
        """Returns the current state of the scene."""
        response.objects = [obj for obj, pose in self.scene_objects.values()]
        return response

    def insert_object_cb(self, request, response):
        """Inserts or overwrites a single object in the scene."""
        obj = request.object
        self.scene_objects[obj.id] = obj, obj.pose
        response.success = True
        self.get_logger().info(f"Inserted single object with ID: {obj.id}")
        return response

    def delete_object_cb(self, request, response):
        """Deletes an object by ID if it exists."""
        if request.id in self.scene_objects:
            del self.scene_objects[request.id]
            response.success = True
            self.get_logger().info(f"Deleted object with ID: {request.id}")
        else:
            response.success = False
            self.get_logger().warn(f"Failed to delete object {request.id}: Not found.")
        return response

    # ==========================================
    # Timer Callbacks
    # ==========================================

    def publish_flattened_scene(self):
        """Periodically flattens all superquadrics in the scene and publishes them."""
        out_msg = SuperquadricArray()

        flattened_list = []
        for obj, pose in self.scene_objects.values():
            # Extend the flattened list with the superquadrics of this object
            for sq in obj.superquadrics:
                from copy import deepcopy
                from scipy.spatial.transform import Rotation as R
                sq_copy = deepcopy(sq)

                world_to_obj = pose_to_matrix(pose.position, pose.orientation)
                obj_to_sq = pose_to_matrix(sq.pose.position, sq.pose.orientation)

                sq_copy.pose = matrix_to_pose(world_to_obj @ obj_to_sq)

                flattened_list.append(sq_copy)

        out_msg.superquadrics = flattened_list
        self.flattened_sq_pub.publish(out_msg)


def main(args=None):
    rclpy.init(args=args)
    node = SceneManager()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

if __name__ == '__main__':
    main()
