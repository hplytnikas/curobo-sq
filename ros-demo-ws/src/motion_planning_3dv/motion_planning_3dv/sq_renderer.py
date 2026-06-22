import rclpy
from rclpy.node import Node
import numpy as np

from geometry_msgs.msg import Point
from visualization_msgs.msg import Marker, MarkerArray
from superquadric_interfaces.msg import SuperquadricArray

from motion_planning_3dv.sq_utils import sample_superquadric_mesh

class SuperquadricVisualizer(Node):
    def __init__(self):
        super().__init__('superquadric_visualizer')

        # Subscribe to your custom array topic
        self.subscription = self.create_subscription(
            SuperquadricArray,
            'scene_superquadrics',
            self.array_callback,
            10
        )

        # Publisher for RViz markers
        self.marker_pub = self.create_publisher(MarkerArray, 'superquadric_markers', 10)

        # Mesh resolution (higher = smoother but slower to compute)
        self.resolution = 20
        self.get_logger().info("Superquadric Visualizer initialized.")

    def array_callback(self, msg):
        marker_array = MarkerArray()

        # Optional: Issue a DELETEALL marker first to clear previous shapes if the array shrank
        clear_marker = Marker()
        clear_marker.action = Marker.DELETEALL
        marker_array.markers.append(clear_marker)

        # Generate a marker for each superquadric in the array
        for i, sq in enumerate(msg.superquadrics):
            marker = self.create_triangle_list_marker(sq, marker_id=i)
            marker_array.markers.append(marker)

        self.marker_pub.publish(marker_array)

    def create_triangle_list_marker(self, sq, marker_id):
        marker = Marker()
        marker.header = sq.header
        marker.ns = "superquadrics"
        marker.id = marker_id
        marker.type = Marker.TRIANGLE_LIST
        marker.action = Marker.ADD

        # RViz handles the 6D transform; our mesh points remain local
        marker.pose = sq.pose
        print(sq)
        # Scale MUST be 1.0 because the x/y/z are baked directly into our mesh vertices
        marker.scale.x = 1.0
        marker.scale.y = 1.0
        marker.scale.z = 1.0

        # Color the superquadric (Blue, semi-transparent)
        marker.color.r = 0.0
        marker.color.g = 0.5
        marker.color.b = 1.0
        marker.color.a = 1.0

        # Generate the mesh points
        marker.points = sample_superquadric_mesh(
            [sq.x, sq.y, sq.z], [sq.e1, sq.e2]
        )

        return marker

def main(args=None):
    rclpy.init(args=args)
    node = SuperquadricVisualizer()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
