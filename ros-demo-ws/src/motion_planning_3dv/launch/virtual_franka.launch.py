from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch_ros.actions import Node
from launch.substitutions import Command, FindExecutable, PathJoinSubstitution, LaunchConfiguration
from launch_ros.substitutions import FindPackageShare
from launch_ros.parameter_descriptions import ParameterValue

def generate_launch_description():
    # 1. Declare Launch Arguments
    # This allows you to run: ros2 launch <package> <file> use_fake_hardware:=false robot_ip:=172.16.0.2
    use_fake_hardware_arg = DeclareLaunchArgument(
        "use_fake_hardware",
        default_value="true",
        description="Start robot with fake hardware mirroring command to state."
    )

    robot_ip_arg = DeclareLaunchArgument(
        "robot_ip",
        default_value="dont-care",
        description="Hostname or IP address of the robot."
    )

    # 2. Get URDF via xacro with arguments passed in
    robot_description_content = Command([
        PathJoinSubstitution([FindExecutable(name="xacro")]),
        " ",
        PathJoinSubstitution(
            [FindPackageShare("franka_description"), "robots", "fp3", "fp3.urdf.xacro"]
        ),
        " use_fake_hardware:=", LaunchConfiguration("use_fake_hardware"),
        " hand:=", "true",
        " ros2_control:=", "true",
        " robot_ip:=", LaunchConfiguration("robot_ip"),
    ])
    robot_description = {"robot_description": ParameterValue(robot_description_content, value_type=str)}

    # 3. Controller Manager Configuration
    initial_controllers = PathJoinSubstitution(
        [FindPackageShare("motion_planning_3dv"), "config", "controllers.yaml"]
    )

    # 4. The Controller Manager Node
    control_node = Node(
        package="controller_manager",
        executable="ros2_control_node",
        parameters=[robot_description, initial_controllers],
        output="both",
    )

    # 5. Robot State Publisher (TF)
    robot_state_pub_node = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        output="both",
        parameters=[robot_description],
    )

    # 6. Spawner for the Joint Trajectory Controller
    joint_traj_spawner = Node(
        package="controller_manager",
        executable="spawner",
        arguments=["joint_trajectory_controller", "-c", "/controller_manager"],
    )

    # 7. Spawner for Joint State Broadcaster (Needed for TF updates)
    joint_state_broadcaster_spawner = Node(
        package="controller_manager",
        executable="spawner",
        arguments=["joint_state_broadcaster", "-c", "/controller_manager"],
    )

    return LaunchDescription([
        use_fake_hardware_arg,
        robot_ip_arg,
        control_node,
        robot_state_pub_node,
        joint_traj_spawner,
        joint_state_broadcaster_spawner,
    ])
