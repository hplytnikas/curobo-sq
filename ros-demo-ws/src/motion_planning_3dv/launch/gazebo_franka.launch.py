import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, SetEnvironmentVariable, OpaqueFunction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
from launch.substitutions import Command, FindExecutable, PathJoinSubstitution, LaunchConfiguration
from launch_ros.substitutions import FindPackageShare
from launch_ros.parameter_descriptions import ParameterValue
import xacro

def launch_setup(context, *args, **kwargs):
    # 1. Path to your custom Controllers
    # We resolve the substitution to a string for the replace function
    initial_controllers_path = os.path.join(
        get_package_share_directory("motion_planning_3dv"), "config", "controllers.yaml"
    )

    # 2. Get URDF via xacro (Manual processing for string manipulation)
    franka_wrapper_path = os.path.join(
        get_package_share_directory("motion_planning_3dv"), "config", "franka_wrapper.urdf.xacro"
    )

    # Process xacro with your specific mappings
    robot_description_raw = xacro.process_file(
        franka_wrapper_path,
        mappings={"hand": "true", "ros2_control": "true", "gazebo": "true"}
    ).toxml()

    # 3. THE REPLACEMENT
    # Surgical strike on the hardcoded path
    old_path = os.path.join(
        get_package_share_directory("franka_gazebo_bringup"), "config", "franka_gazebo_controllers.yaml"
    )
    robot_description_fixed = robot_description_raw.replace(old_path, initial_controllers_path)

    if initial_controllers_path in robot_description_fixed:
        print("SUCCESS: Custom controllers injected into URDF.")
    else:
        print("FAILURE: Custom controllers NOT found in URDF.")

    # 4. Re-wrap into the expected node parameters
    robot_description = {"robot_description": robot_description_fixed}

    use_sim_time_value = LaunchConfiguration("use_sim_time").perform(context)

    # 5. Node Definitions (Preserving your original structure/names)
    gz_spawn_entity = Node(
        package="ros_gz_sim",
        executable="create",
        output="screen",
        arguments=[
            "-topic", "robot_description",
            "-name", "franka_fp3",
            "-allow_renaming", "true",
        ],
        # Note: Gazebo plugin reads from URDF, but we keep your param pass for safety
        parameters=[initial_controllers_path],
    )

    robot_state_pub_node = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        output="both",
        parameters=[robot_description, {"use_sim_time": use_sim_time_value.lower() == 'true'}],
    )

    return [gz_spawn_entity, robot_state_pub_node]

def generate_launch_description():
    # 1. Declare Launch Arguments
    use_sim_time = DeclareLaunchArgument("use_sim_time", default_value="true")

    # Environment Setup
    pkg_share = get_package_share_directory('franka_description')
    local_share = get_package_share_directory('motion_planning_3dv')
    model_path = os.path.dirname(pkg_share)
    world_path = os.path.join(local_share, 'config', 'sim_world.sdf')
    set_model_path = SetEnvironmentVariable(name='GZ_SIM_RESOURCE_PATH', value=f"{model_path}:{local_share}")

    # 4. Include Gazebo Launch
    gazebo_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            PathJoinSubstitution([
                FindPackageShare("ros_gz_sim"), "launch", "gz_sim.launch.py"
            ])
        ]),
        launch_arguments={"gz_args": f"-r {world_path}"}.items(),
    )

    # 7. Spawners (Standard)
    joint_traj_spawner = Node(
        package="controller_manager",
        executable="spawner",
        arguments=["joint_trajectory_controller"],
    )

    grip_traj_spawner = Node(
        package="controller_manager",
        executable="spawner",
        arguments=["gripper_action_controller"],
    )

    joint_state_broadcaster_spawner = Node(
        package="controller_manager",
        executable="spawner",
        arguments=["joint_state_broadcaster"],
    )

    ros_gz_bridge = Node(
        package="ros_gz_bridge",
        executable="parameter_bridge",
        parameters=[{
            'config_file': os.path.join(
                get_package_share_directory('motion_planning_3dv'),
                'config',
                'bridge_config.yaml'
            )
        }],
        output="screen"
    )

    convert_to_mm_node = Node(
        package='motion_planning_3dv',
        executable='depth_image_m_to_mm', # Converts 32FC1 (meters) to 16UC1 (mm)
        name='depth_to_mm_converter',
    )

    static_tf_node = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        arguments=[
            '--x', '0.1',
            '--z', '0.5',       # x, y, z translation
            '--pitch', '0.7',   # yaw, pitch, roll (matching Gazebo pitch, adjusted for ROS convention)
            '--frame-id', 'world',
            '--child-frame-id', 'camera_link'
        ]
    )

    static_tf_camera_optical = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='tf_camera_link_to_optical',
        arguments=[
            '--roll', '-1.5708', '--yaw', '-1.5708',
            '--frame-id', 'camera_link',
            '--child-frame-id', 'depth_camera/camera_link/depth_camera'
        ]
    )

    return LaunchDescription([
        use_sim_time,
        set_model_path,
        gazebo_launch,
        # This replaces the static gz_spawn_entity and robot_state_pub_node
        OpaqueFunction(function=launch_setup),
        joint_traj_spawner,
        grip_traj_spawner,
        joint_state_broadcaster_spawner,
        ros_gz_bridge,
        static_tf_node,
        static_tf_camera_optical,
        convert_to_mm_node
    ])
