#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
import numpy as np

class DepthToMmNumPyConverter(Node):
    def __init__(self):
        super().__init__('depth_to_mm_numpy_converter')

        # Subscribe to Gazebo's meter-based depth topic (32FC1)
        self.subscription = self.create_subscription(
            Image,
            '/camera/depth',
            self.listener_callback,
            10
        )

        # Publish the new millimeter-based depth topic (16UC1)
        self.publisher_ = self.create_publisher(
            Image,
            '/camera/depth/image',
            10
        )

    def listener_callback(self, msg):
        try:
            # Interpret raw byte data as a flat NumPy float32 array
            # msg.data is a buffer of uint8; frombuffer views it as float32
            depth_in_meters = np.frombuffer(msg.data, dtype=np.float32).copy()

            # Clean up invalid/infinite values safely before multiplying
            depth_in_meters = np.nan_to_num(depth_in_meters, nan=0.0, posinf=10.0, neginf=0.0)

            # Scale to millimeters and cast to 16-bit Unsigned Integer
            depth_in_mm = (depth_in_meters * 1000.0).astype(np.uint16)

            # Construct the new ROS 2 Image Message
            out_msg = Image()
            out_msg.header = msg.header        # Preserve timestamps and tf frame_id
            out_msg.height = msg.height        # Keep original image height
            out_msg.width = msg.width          # Keep original image width
            out_msg.encoding = '16UC1'         # Update encoding metadata
            out_msg.is_bigendian = msg.is_bigendian

            # 16UC1 uses 2 bytes per pixel, so step is width * 2 bytes
            out_msg.step = msg.width * 2

            # Convert NumPy array back to raw bytes for transmission
            out_msg.data = depth_in_mm.tobytes()

            # Publish the modified image
            self.publisher_.publish(out_msg)

        except Exception as e:
            self.get_logger().error(f'Failed to convert depth using NumPy: {e}')

def main(args=None):
    rclpy.init(args=args)
    node = DepthToMmNumPyConverter()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
