#!/usr/bin/env python3

from typing import List, Sequence, Tuple

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy

# Standard Sensor and Geometry Messages
from sensor_msgs.msg import Image, PointCloud2, PointField
from geometry_msgs.msg import TransformStamped

# TF2 imports for transforming perception data into your target frame
from tf2_ros import TransformException
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener

# Placeholders for your future custom message layout
# Once defined, uncomment and update these lines:
from superquadric_interfaces.msg import Superquadric, SuperquadricArray, SceneObject, Scene

from scipy.spatial.transform import Rotation as SciRotation

from project_3dv.perception.pipeline import depth_to_pointcloud, remove_table, segment_instances_dual, preprocess_pointcloud, SuperquadricWorld

from project_3dv.perception.pipeline import Frame, get_world_pointcloud, fit_superquadrics_world

from project_3dv.perception.superdec_fitter import SuperdecFitter

import torch
from superdec.superdec import SuperDec
from omegaconf import OmegaConf

import numpy as np

import traceback

import os

# these converters are necessary because we use a venv's python which provides numpy>=2

def ros_depth_to_numpy(msg: Image) -> np.ndarray:
    """
    Converts a ROS 2 Image message to a numpy array without cv_bridge.
    """
    # Determine dtype based on encoding
    # '16UC1' is 16-bit unsigned integer (2 bytes per pixel)
    # '32FC1' is 32-bit float (4 bytes per pixel)
    if msg.encoding == '16UC1':
        dtype = np.uint16
    elif msg.encoding == '32FC1':
        dtype = np.float32
    else:
        raise ValueError(f"Unsupported encoding: {msg.encoding}")

    # Create numpy array from the raw data buffer
    depth_array = np.frombuffer(msg.data, dtype=dtype)
    depth_array = depth_array.reshape((msg.height, msg.width))

    # Handle row padding (stride)
    if msg.step != msg.width * np.dtype(dtype).itemsize:
        itemsize = np.dtype(dtype).itemsize
        depth_array = np.array([
            depth_array[i, :msg.width]
            for i in range(msg.height)
        ])

    return depth_array

def numpy_to_pointcloud2(points: np.ndarray, frame_id: str, stamp=None) -> PointCloud2:
    """
    points: (N, 3) numpy array of floats
    frame_id: string
    """

    points = points.astype(np.float32)

    # Define fields
    fields = [
        PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
        PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
        PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
    ]

    # Create message
    msg = PointCloud2()
    msg.header.frame_id = frame_id
    if stamp:
        msg.header.stamp = stamp

    msg.height = 1
    msg.width = points.shape[0]
    msg.fields = fields
    msg.is_bigendian = False
    msg.point_step = 12  # 3 fields * 4 bytes
    msg.row_step = msg.point_step * msg.width
    msg.is_dense = True
    msg.data = points.tobytes()

    return msg

def numpys_to_pointcloud2(segments: list, frame_id: str, stamp=None) -> PointCloud2:
    """convert list of (N, 3) numpy arrays to a single pointcloud, where intensity indicates which array the point stems for."""
    # compute pointcloud size
    total_points = sum(s.shape[0] for s in segments)

    data = np.zeros(total_points, dtype=[
        ('x', 'f4'), ('y', 'f4'), ('z', 'f4'), ('intensity', 'f4')
    ])

    # populate array
    current_idx = 0
    for i, seg in enumerate(segments):
        n = seg.shape[0]
        data['x'][current_idx:current_idx+n] = seg[:, 0]
        data['y'][current_idx:current_idx+n] = seg[:, 1]
        data['z'][current_idx:current_idx+n] = seg[:, 2]
        # Assign the segment index as the intensity value
        data['intensity'][current_idx:current_idx+n] = float(i)
        current_idx += n


    # assemble message
    fields = [
        PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
        PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
        PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
        PointField(name='intensity', offset=12, datatype=PointField.FLOAT32, count=1),
    ]

    msg = PointCloud2()
    msg.header.frame_id = frame_id
    if stamp:
        msg.header.stamp = stamp

    msg.height = 1
    msg.width = total_points
    msg.fields = fields
    msg.is_bigendian = False
    msg.point_step = 16  # 4 fields * 4 bytes
    msg.row_step = msg.point_step * total_points
    msg.is_dense = True
    msg.data = data.tobytes()

    return msg

def transform_to_matrix(transform: TransformStamped) -> np.ndarray:
    """
    Converts a ROS 2 TransformStamped message into a 4x4 homogeneous matrix.
    """
    t = transform.transform.translation
    q = transform.transform.rotation

    x, y, z, w = q.x, q.y, q.z, q.w

    rot_matrix = np.array([
        [1 - 2*y**2 - 2*z**2, 2*x*y - 2*z*w,     2*x*z + 2*y*w],
        [2*x*y + 2*z*w,     1 - 2*x**2 - 2*z**2, 2*y*z - 2*x*w],
        [2*x*z - 2*y*w,     2*y*z + 2*x*w,     1 - 2*x**2 - 2*y**2]
    ])

    matrix = np.eye(4)
    matrix[:3, :3] = rot_matrix
    matrix[:3, 3] = [t.x, t.y, t.z]

    return matrix

def transform_numpy(points: np.ndarray, transform: TransformStamped) -> np.ndarray:
    """transform an (N, 3) numpy array/pointcloud."""
    # 1. Get the transformation matrix from the helper
    matrix = transform_to_matrix(transform)

    # 2. Pad points to (N, 4) for homogeneous multiplication
    points_h = np.hstack([points, np.ones((points.shape[0], 1))])

    # 3. Perform the transformation: (N, 4) @ (4, 4)^T
    transformed = points_h @ matrix.T

    # 4. Return as (N, 3)
    return transformed[:, :3]

def matrix_to_quat(R: np.ndarray) -> np.ndarray:
    """
    Converts a 3x3 rotation matrix to a quaternion [x, y, z, w].
    """
    tr = np.trace(R)

    if tr > 0:
        S = np.sqrt(tr + 1.0) * 2
        w = 0.25 * S
        x = (R[2, 1] - R[1, 2]) / S
        y = (R[0, 2] - R[2, 0]) / S
        z = (R[1, 0] - R[0, 1]) / S
    elif (R[0, 0] > R[1, 1]) and (R[0, 0] > R[2, 2]):
        S = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        w = (R[2, 1] - R[1, 2]) / S
        x = 0.25 * S
        y = (R[0, 1] + R[1, 0]) / S
        z = (R[0, 2] + R[2, 0]) / S
    elif R[1, 1] > R[2, 2]:
        S = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        w = (R[0, 2] - R[2, 0]) / S
        x = (R[0, 1] + R[1, 0]) / S
        y = 0.25 * S
        z = (R[1, 2] + R[2, 1]) / S
    else:
        S = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
        w = (R[1, 0] - R[0, 1]) / S
        x = (R[0, 2] + R[2, 0]) / S
        y = (R[1, 2] + R[2, 1]) / S
        z = 0.25 * S

    return np.array([x, y, z, w])

# helpers from the pipeline package

def _normalize_points(points: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    translation = points.mean(axis=0)
    centered = points - translation
    scale = float(2.0 * np.max(np.abs(centered)))
    normalized = centered / scale
    return normalized, translation, scale

def _denormalize_outdict(outdict, translation: np.ndarray, scale: float, z_up: bool = False):
    scale_arr = np.asarray([[scale]], dtype=np.float32)
    translation_arr = translation.reshape(1, 1, 3)
    outdict["scale"] = outdict["scale"] * scale_arr[:, :, None]
    outdict["trans"] = outdict["trans"] * scale_arr[:, :, None] + translation_arr
    return outdict

def _denormalize_points(points: torch.Tensor, translation: np.ndarray, scale: float, z_up: bool = False):
    scale_t = torch.tensor(scale, dtype=points.dtype, device=points.device).view(1, 1, 1)
    translation_t = torch.tensor(translation, dtype=points.dtype, device=points.device).view(1, 1, 3)
    return points * scale_t + translation_t

def _apply_scene_transform(
    translation: Sequence[float] | np.ndarray,
    rotation_matrix: np.ndarray,
    scene_translation: np.ndarray,
    scene_quat_wxyz: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    scene_quat_xyzw = np.array(
        [scene_quat_wxyz[1], scene_quat_wxyz[2], scene_quat_wxyz[3], scene_quat_wxyz[0]], dtype=np.float32
    )
    scene_rot = SciRotation.from_quat(scene_quat_xyzw)
    primitive_rot = SciRotation.from_matrix(rotation_matrix)
    new_translation = scene_rot.apply(np.asarray(translation, dtype=np.float32)) + scene_translation
    new_rotation = (scene_rot * primitive_rot).as_matrix()
    return new_translation, new_rotation

class SuperquadricFitterNode(Node):

    def __init__(self):
        super().__init__('superquadric_fitter_node')

        self.declare_parameter('target_frame', 'world')
        self.declare_parameter('use_lidar', False)
        self.declare_parameter('timer_period_seconds', 0.1)

        self.target_frame = self.get_parameter('target_frame').get_parameter_value().string_value
        self.use_lidar = self.get_parameter('use_lidar').get_parameter_value().bool_value
        timer_period = self.get_parameter('timer_period_seconds').get_parameter_value().double_value

        sensor_qos = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            depth=10
        )

        # configure perception pipeline
        checkpoint_folder = "/home/vision/Downloads/concave_checkpoint/"
        self.fitter = SuperdecFitter("/home/vision/Downloads/superdec/", checkpoint_folder)

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        checkpoint = torch.load(os.path.join(checkpoint_folder, "ckpt.pt"), map_location=self.device, weights_only=False)
        configs = OmegaConf.load(os.path.join(checkpoint_folder, "config.yaml"))
        self.model = SuperDec(configs.superdec).to(self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()

        # ROS2 Interfaces
        if self.use_lidar:
            self.lidar_sub = self.create_subscription(
                PointCloud2,
                '/camera/depth/color/points',
                self.pointcloud_callback,
                sensor_qos
            )
            self.get_logger().info("Superquadric node listening to PointCloud2.")
        else:
            self.camera_sub = self.create_subscription(
                Image,
                '/camera/depth/image',
                self.image_callback,
                sensor_qos
            )
            self.get_logger().info("Superquadric node listening to Image stream.")

        self.sq_publisher = self.create_publisher(
            SuperquadricArray,
            '/raw_superquadrics',
            10
        )

        self.scene_publisher = self.create_publisher(
            Scene,
            '/scene',
            10
        )

        self.dbg_pc_publisher = self.create_publisher(
            PointCloud2,
            '/dbg/depth_pc',
            10
        )

        self.dbg_obj_pc_publisher = self.create_publisher(
            PointCloud2,
            '/dbg/obj_pc',
            10
        )

        self.dbg_tbl_pc_publisher = self.create_publisher(
            PointCloud2,
            '/dbg/tbl_pc',
            10
        )

        self.dbg_inst_pc_publisher = self.create_publisher(
            PointCloud2,
            '/dbg/inst_pc',
            10
        )

        self.dbg_prepr_pc_publisher = self.create_publisher(
            PointCloud2,
            '/dbg/prepr_pc',
            10
        )

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.processing_timer = self.create_timer(timer_period, self.process_and_fit)

        # cache
        self.latest_sensor_data = None
        self.latest_header = None

    def pointcloud_callback(self, msg: PointCloud2):
        """Cache incoming pointclouds."""
        self.latest_sensor_data = msg
        self.latest_header = msg.header
        raise NotImplemented()

    def image_callback(self, msg: Image):
        self.latest_header = msg.header

        frame = Frame(ros_depth_to_numpy(msg).astype(np.float32, copy=True) / 1000,
                      np.asarray([[525.0, 0.0, 319.5],
                                  [0.0, 525.0, 239.5],
                                  [0.0, 0.0, 1]]))
        self.latest_sensor_data = frame

    def process_and_fit(self):
        """Main estimation loop running on its own timer."""
        if self.latest_sensor_data is None:
            # Silently wait until data arrives
            return

        source_frame = self.latest_header.frame_id

        # fetch transform
        try:
            transform: TransformStamped = self.tf_buffer.lookup_transform(
                target_frame=self.target_frame,
                source_frame=source_frame,
                time=rclpy.time.Time()
            )
        except TransformException as ex:
            self.get_logger().warning(f"Could not transform {source_frame} to {self.target_frame}: {ex}")
            return

        self.get_logger().info(f"Processing frame data from {source_frame} in {self.target_frame} context...", throttle_duration_sec=2.0)

        # compute pointcloud
        if isinstance(self.latest_sensor_data, Frame):
            self.latest_sensor_data.extrinsic = transform_to_matrix(transform)
            self.latest_sensor_data = get_world_pointcloud([self.latest_sensor_data], max_depth=1000000)
            print(self.latest_sensor_data)
            self.dbg_pc_publisher.publish(numpy_to_pointcloud2(np.asarray(self.latest_sensor_data.points), "world"))

        # fit
        computed_superquadrics, scene = self.fit_superquadrics(self.latest_sensor_data, transform)

        # publish
        if computed_superquadrics:
            output_msg = SuperquadricArray()
            output_msg.superquadrics = computed_superquadrics
            self.sq_publisher.publish(output_msg)
            pass

        self.scene_publisher.publish(scene)

    def fit_superquadrics(self, data, transform):
        """
        run the later stages of the perception pipeline
        """

        # removal
        obj_pts, table_normal, table_height, table_pts, _ = remove_table(np.asarray(data.points))

        self.dbg_obj_pc_publisher.publish(numpy_to_pointcloud2(obj_pts, "world"))
        self.dbg_tbl_pc_publisher.publish(numpy_to_pointcloud2(table_pts, "world"))

        # segmentation
        instances = segment_instances_dual(obj_pts)

        self.dbg_inst_pc_publisher.publish(numpys_to_pointcloud2(instances, "world"))

        sq_list = []

        scene = Scene()

        for i, inst in enumerate(instances):
            scene_obj = SceneObject()

            # fit
            sample_size = min(4096, len(inst))
            sample_idx = np.random.choice(len(inst), sample_size, replace=len(inst) < sample_size)
            points = inst[sample_idx]
            points, translation, scale = _normalize_points(points)
            points_tensor = torch.from_numpy(points).unsqueeze(0).to(self.device).float()

            scene_obj.pose.position.x = float(translation[0])
            scene_obj.pose.position.y = float(translation[1])
            scene_obj.pose.position.z = float(translation[2])

            with torch.no_grad():
                outdict = self.model(points_tensor)
                for key, value in outdict.items():
                    if isinstance(value, torch.Tensor):
                        outdict[key] = value.cpu()
                outdict = _denormalize_outdict(outdict, np.asarray(translation, dtype=np.float32), scale, False)
                points_tensor = _denormalize_points(
                    points_tensor.cpu(), np.asarray(translation, dtype=np.float32), scale, False
                )

            primitive_count = int(outdict["scale"].shape[1])
            for idx in range(primitive_count):
                if float(outdict["exist"][0, idx]) <= 0.5:
                    continue

                scale = np.asarray(outdict["scale"][0, idx], dtype=np.float32) * 1
                exponents = np.asarray(outdict["shape"][0, idx], dtype=np.float32)
                rotation = np.asarray(outdict["rotate"][0, idx], dtype=np.float32)
                translation = np.asarray(outdict["trans"][0, idx], dtype=np.float32)

                transformed_translation, transformed_rotation = _apply_scene_transform(
                    translation.tolist(), rotation, [0, 0, 0], [1, 0, 0, 0]
                )

                # populate base message
                sq = Superquadric()
                sq.header.frame_id = self.target_frame
                sq.pose.position.x = float(translation[0] - scene_obj.pose.position.x)
                sq.pose.position.y = float(translation[1] - scene_obj.pose.position.y)
                sq.pose.position.z = float(translation[2] - scene_obj.pose.position.z)
                sq.x = float(scale[0])
                sq.y = float(scale[1])
                sq.z = float(scale[2])
                sq.e1 = float(exponents[0])
                sq.e2 = float(exponents[1])
                quat = matrix_to_quat(transformed_rotation)
                sq.pose.orientation.x = float(quat[0])
                sq.pose.orientation.y = float(quat[1])
                sq.pose.orientation.z = float(quat[2])
                sq.pose.orientation.w = float(quat[3])

                scene_obj.superquadrics.append(sq)
                sq_list.append(sq)

            # populate scene message
            scene.objects.append(scene_obj)
            scene_obj.id = i
            scene_obj.header.frame_id = 'world'
            scene_obj.min.x = 0.0
            scene_obj.min.y = 0.0
            scene_obj.min.z = 0.0
            scene_obj.max.x = 0.0
            scene_obj.max.y = 0.0
            scene_obj.max.z = 0.0

        return sq_list, scene

def main(args=None):
    rclpy.init(args=args)
    node = SuperquadricFitterNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
