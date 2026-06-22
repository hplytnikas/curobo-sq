#! /usr/bin/env python3

import argparse
import sys
from pathlib import Path
from typing import List, Optional

import curobo.runtime as runtime
from curobo.motion_planner import MotionPlanner, MotionPlannerCfg
from curobo.types import ContentPath, GoalToolPose, JointState, Pose
from curobo._src.geom.types import Cuboid, Mesh, SceneCfg, Superquadric
from curobo.viewer import ViserVisualizer
import time

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from control_msgs.action import FollowJointTrajectory
from trajectory_msgs.msg import JointTrajectoryPoint
from builtin_interfaces.msg import Duration

import torch

import csv
import threading

class TrajectoryActionClient(Node):
    def __init__(self):
        super().__init__('curobo_trajectory_executor')
        self._action_client = ActionClient(
            self,
            FollowJointTrajectory,
            '/joint_trajectory_controller/follow_joint_trajectory'
        )

    def send_goal(self, interpolated_plan, dt, timescale):
        goal_msg = FollowJointTrajectory.Goal()

        # 1. Configuration: Master Dictionary / Translation Map
        # Key: cuRobo name, Value: ROS 2 Controller name
        # Only joints in this dictionary will be sent to the controller.
        joint_map = {
            "panda_joint1": "fp3_joint1",
            "panda_joint2": "fp3_joint2",
            "panda_joint3": "fp3_joint3",
            "panda_joint4": "fp3_joint4",
            "panda_joint5": "fp3_joint5",
            "panda_joint6": "fp3_joint6",
            "panda_joint7": "fp3_joint7",
        }

        # 2. Extract and Prepare Data
        pos = interpolated_plan.position.detach().cpu().numpy().squeeze()
        vel = interpolated_plan.velocity.detach().cpu().numpy().squeeze() / timescale
        curobo_names = interpolated_plan.joint_names

        # Identify indices of joints that exist in our master dictionary
        valid_indices = []
        target_names = []

        for i, name in enumerate(curobo_names):
            if name in joint_map:
                valid_indices.append(i)
                target_names.append(joint_map[name])

        # Set the translated names in the message
        goal_msg.trajectory.joint_names = target_names

        # 3. Build Trajectory Points
        for i in range(pos.shape[0]):
            point = JointTrajectoryPoint()

            # Extract only the valid joint values using our indices
            point.positions = [float(pos[i, idx]) for idx in valid_indices]
            point.velocities = [float(vel[i, idx]) for idx in valid_indices]

            # Calculate timing
            t_sec = i * dt * timescale
            point.time_from_start = Duration(
                sec=int(t_sec),
                nanosec=int((t_sec - int(t_sec)) * 1e9)
            )
            goal_msg.trajectory.points.append(point)

        # 4. Dispatch
        self.get_logger().info(f'Sending goal with joints: {target_names}')
        self._action_client.wait_for_server()
        return self._action_client.send_goal_async(goal_msg)

def execute_trajectory_ros2(interpolated_plan, dt, timescale = 1):
    """Initializes ROS 2, sends the goal, and shuts down."""
    if not rclpy.ok():
        rclpy.init()

    executor_node = TrajectoryActionClient()

    # Spin in a background thread to handle action callbacks
    thread = threading.Thread(target=rclpy.spin, args=(executor_node,), daemon=True)
    thread.start()

    try:
        future = executor_node.send_goal(interpolated_plan, dt, timescale)
        # Wait for the server to accept the goal
        while not future.done():
            time.sleep(0.1)

        goal_handle = future.result()
        if not goal_handle.accepted:
            print("Goal rejected by controller")
            return

        print("Goal accepted, executing...")
    finally:
        executor_node.destroy_node()
        rclpy.shutdown()
        thread.join()


def export_curobo_to_csv(interpolated_plan, filename="trajectory.csv"):
    # 1. Extract tensors and move to CPU
    # cuRobo JointState tensors are typically (1, 1, W, J)
    pos = interpolated_plan.position.detach().cpu().numpy().squeeze()
    vel = interpolated_plan.velocity.detach().cpu().numpy().squeeze()
    acc = interpolated_plan.acceleration.detach().cpu().numpy().squeeze()
    names = interpolated_plan.joint_names

    num_waypoints = pos.shape[0]
    num_joints = pos.shape[1]

    # 2. Prepare Headers
    # Format: pos_joint1, vel_joint1, acc_joint1, pos_joint2...
    headers = names

    # 3. Write to CSV
    with open(filename, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(headers)

        for i in range(num_waypoints):
            row = []
            for j in range(num_joints):
                row.append(pos[i, j])
                # row.append(vel[i, j])
                # row.append(acc[i, j])
            writer.writerow(row)

    print(f"Successfully saved {num_waypoints} waypoints to {filename}")

def _get_output_dir() -> Path:
    """Return the example output directory, creating it if needed."""
    out = Path(runtime.cache_dir) / "examples" / "motion_planning"
    out.mkdir(parents=True, exist_ok=True)
    return out


def _plot_trajectory(
    positions: torch.Tensor,
    joint_names: List[str],
    dt: float,
    save_path: str,
    title: str = "Joint Trajectory",
    phase_boundaries: Optional[List[int]] = None,
    phase_labels: Optional[List[str]] = None,
):
    """Plot joint positions over time and save to *save_path*.

    Args:
        positions: Joint positions tensor of shape ``(timesteps, n_joints)``.
        joint_names: Label for each joint.
        dt: Time step between waypoints (seconds).
        save_path: Output file path (e.g. ``"trajectory.pdf"``).
        title: Plot title.
        phase_boundaries: Timestep indices where a new phase starts.
        phase_labels: Label for each phase (length must match *phase_boundaries*).
    """
    import matplotlib.pyplot as plt

    pos_np = positions.cpu().numpy()
    n_steps = pos_np.shape[0]
    t = [i * dt for i in range(n_steps)]

    fig, ax = plt.subplots(figsize=(10, 5))
    for j, name in enumerate(joint_names):
        ax.plot(t, pos_np[:, j], label=name)

    if phase_boundaries and phase_labels:
        for idx, label in zip(phase_boundaries, phase_labels):
            ax.axvline(x=idx * dt, color="grey", linestyle="--", linewidth=0.8)
            ax.text(
                idx * dt, ax.get_ylim()[1], f" {label}", fontsize=8, va="top",
            )

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Joint position (rad)")
    ax.set_title(title)
    ax.legend(fontsize=7, ncol=2, loc="upper right")
    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)
    print(f"Trajectory plot saved to: {save_path}")


def pose_planning_example():
    """Plan a collision-free trajectory to a goal pose.

    Args:
        output_dir: Where to save output files. Defaults to
            ``<runtime.cache_dir>/examples/motion_planning/``.

    Returns:
        True if planning succeeded.
    """
    config = MotionPlannerCfg.create(
        robot="franka.yml",
        scene_model="collision_test.yml",
    )
    planner = MotionPlanner(config)
    planner.warmup(enable_graph=True, num_warmup_iterations=5)

    q_start = JointState.from_position(
        planner.default_joint_state.position.unsqueeze(0),
        joint_names=planner.joint_names,
    )

    goal_pose = GoalToolPose(
        tool_frames=planner.tool_frames,
        position=torch.tensor([[[[[0.5, 0.0, 0.3]]]]], device="cuda", dtype=torch.float32),
        quaternion=torch.tensor([[[[[1.0, 0.0, 0.0, 0.0]]]]], device="cuda", dtype=torch.float32),
    )

    result = planner.plan_pose(goal_pose, q_start)

    interp_dt = planner.trajopt_solver.config.interpolation_dt
    if result is not None and result.success.any():
        print("✓ Planning succeeded!")
        interpolated = result.get_interpolated_plan()
        n_waypoints = interpolated.position.shape[-2]
        print(f"Trajectory has {n_waypoints} waypoints")
        print(f"Duration: {n_waypoints * interp_dt:.2f}s")

        export_curobo_to_csv(interpolated, filename="curobo_trajectory.csv")
        execute_trajectory_ros2(interpolated, interp_dt, 10)

        return True
    else:
        print("✗ Planning failed - try adjusting the goal or obstacles")
        return False


def interactive_motion_planning(robot_file="franka.yml", scene_file="collision_test.yml", port=8080):
    """Launch an interactive Viser viewer for motion planning.

    Provides a web-based 3D viewer where you can:
    - Drag the target frame to set the goal pose
    - Drag obstacles to reposition them
    - Click "Move" to plan and execute a collision-free trajectory
    - Click "Grasp" to plan a three-phase grasp motion (approach, grasp, lift)
    """
    import threading

    viser_viz = ViserVisualizer(
        content_path=ContentPath(robot_config_file=robot_file),
        connect_ip="0.0.0.0",
        connect_port=port,
        add_control_frames=True,
        visualize_robot_spheres=False,
    )

    config = MotionPlannerCfg.create(robot=robot_file, scene_model=scene_file)
    config.scene_collision_cfg.scene_model = SceneCfg(superquadric=[Superquadric(
                name=f"cube",
                pose=[0.4, 0, 0.3, 1, 0, 0, 0],
                radii=[0.1, 0.1, 0.1],
                shape=[0.1, 0.1],
            )])
    planner = MotionPlanner(config)

    scene_cfg = config.scene_collision_cfg.scene_model
    obstacle_frames = viser_viz.add_scene(scene_cfg, add_control_frames=True)

    old_obstacle_poses = {
        k: Pose.from_numpy(obstacle_frames[k].position, obstacle_frames[k].wxyz)
        for k in obstacle_frames.keys()
    }

    current_state = planner.default_joint_state.clone().unsqueeze(0)

    print("Warming up motion planner...")
    planner.warmup(enable_graph=True, num_warmup_iterations=5)

    is_moving = False

    def _create_trajectory_image(trajectory, joint_names, title=""):
        """Render a joint trajectory as a PNG image array for the Viser GUI."""
        import io

        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np

        traj = trajectory.squeeze(0)
        pos = np.atleast_2d(traj.position[0].cpu().numpy())  # (horizon, dof)
        dt_val = traj.dt.item() if traj.dt is not None else 0.02
        t = np.arange(pos.shape[0]) * dt_val

        vel = np.atleast_2d(traj.velocity[0].cpu().numpy()) if traj.velocity is not None else None
        acc = np.atleast_2d(traj.acceleration[0].cpu().numpy()) if traj.acceleration is not None else None
        jrk = np.atleast_2d(traj.jerk[0].cpu().numpy()) if traj.jerk is not None else None

        n_plots = 1 + (vel is not None) + (acc is not None) + (jrk is not None)
        fig, axes = plt.subplots(n_plots, 1, figsize=(5, 2 * n_plots), dpi=100, sharex=True)
        if n_plots == 1:
            axes = [axes]

        plot_data = [(pos, "Position (rad)")]
        if vel is not None:
            plot_data.append((vel, "Velocity (rad/s)"))
        if acc is not None:
            plot_data.append((acc, "Accel (rad/s²)"))
        if jrk is not None:
            plot_data.append((jrk, "Jerk (rad/s³)"))

        for ax, (data, ylabel) in zip(axes, plot_data):
            for j in range(data.shape[1]):
                label = joint_names[j] if j < len(joint_names) else f"J{j}"
                if len(label) > 8:
                    label = label[:6] + ".."
                ax.plot(t, data[:, j], linewidth=1.0, label=label)
            ax.set_ylabel(ylabel, fontsize=9)
            ax.grid(True, alpha=0.3)
            ax.tick_params(labelsize=8)

        axes[0].legend(loc="upper right", fontsize=7, ncol=2)
        axes[-1].set_xlabel("Time (s)", fontsize=9)
        if title:
            fig.suptitle(title, fontsize=11, fontweight="bold")
        plt.tight_layout()

        buf = io.BytesIO()
        fig.savefig(buf, format="png")
        buf.seek(0)
        from PIL import Image
        img_array = np.array(Image.open(buf))
        plt.close(fig)
        buf.close()
        return img_array

    server = viser_viz._server
    traj_plot = server.gui.add_image(
        _create_trajectory_image(
            JointState.from_position(
                planner.default_joint_state.position.unsqueeze(0).unsqueeze(0),
                joint_names=planner.joint_names,
            ),
            planner.joint_names,
            title="No trajectory yet",
        ),
        label="Joint Trajectory",
        format="png",
    )

    def update_obstacles():
        for k in obstacle_frames.keys():
            new_pose = Pose.from_numpy(obstacle_frames[k].position, obstacle_frames[k].wxyz)
            if new_pose != old_obstacle_poses[k]:
                planner.scene_collision_checker.update_obstacle_pose(k, new_pose)
                old_obstacle_poses[k] = new_pose.clone()

    def execute_trajectory(trajectory):
        nonlocal current_state, is_moving
        traj = trajectory.squeeze(0)
        for i in range(traj.position.shape[-2]):
            if not is_moving:
                return
            waypoint = JointState.from_position(
                traj.position[0, i, :].unsqueeze(0),
                joint_names=traj.joint_names,
            )
            viser_viz.set_joint_state(waypoint.squeeze(0))
            time.sleep(0.02)
        current_state = JointState.from_position(
            traj.position[0, -1, :].unsqueeze(0),
            joint_names=traj.joint_names,
        )

    def on_move(_):
        nonlocal is_moving
        if is_moving:
            return

        def plan_and_execute():
            nonlocal is_moving
            is_moving = True
            update_obstacles()
            target_poses = viser_viz.get_control_frame_pose()
            active_js = planner.kinematics.get_active_js(current_state.clone())
            result = planner.plan_pose(
                GoalToolPose.from_poses(target_poses, num_goalset=1),
                active_js,
                use_implicit_goal=True,
                max_attempts=3,
            )
            if result is not None and result.success.any():
                interp = result.get_interpolated_plan()
                traj_plot.image = _create_trajectory_image(
                    interp, planner.joint_names,
                    title=f"Pose Plan  |  {result.total_time:.3f}s",
                )

                execute_trajectory(interp)
                execute_trajectory_ros2(interp, interp.dt, 10)
            else:
                print("Motion planning failed")
            is_moving = False

        threading.Thread(target=plan_and_execute, daemon=True).start()

    def on_grasp(_):
        nonlocal is_moving
        if is_moving:
            return

        def plan_grasp_and_execute():
            nonlocal is_moving
            is_moving = True
            update_obstacles()
            target_poses = viser_viz.get_control_frame_pose()
            active_js = planner.kinematics.get_active_js(current_state.clone())
            grasp_poses = GoalToolPose.from_poses(target_poses, num_goalset=1)
            result = planner.plan_grasp(
                grasp_poses,
                active_js,
                plan_approach_to_grasp=True,
                plan_grasp_to_lift=True,
                grasp_lift_in_tool_frame=True,
            )
            if result is not None and result.success is not None and result.success.any():
                traj_plot.image = _create_trajectory_image(
                    result.approach_interpolated_trajectory, planner.joint_names,
                    title="Approach",
                )
                execute_trajectory(result.approach_interpolated_trajectory)
                traj_plot.image = _create_trajectory_image(
                    result.grasp_interpolated_trajectory, planner.joint_names,
                    title="Grasp",
                )
                execute_trajectory(result.grasp_interpolated_trajectory)
                traj_plot.image = _create_trajectory_image(
                    result.lift_interpolated_trajectory, planner.joint_names,
                    title="Lift",
                )
                execute_trajectory(result.lift_interpolated_trajectory)
            else:
                print("Grasp planning failed")
            is_moving = False

        threading.Thread(target=plan_grasp_and_execute, daemon=True).start()

    move_btn = server.gui.add_button("Move", color="green")
    move_btn.on_click(on_move)
    grasp_btn = server.gui.add_button("Grasp", color="blue")
    grasp_btn.on_click(on_grasp)

    print(f"\nInteractive Motion Planner running at http://localhost:{port}")
    print("  - Drag the target frame to set goal pose")
    print("  - Drag obstacles to reposition them")
    print("  - Click 'Move' for pose-to-pose planning")
    print("  - Click 'Grasp' for approach-grasp-lift planning")
    print("Press Ctrl+C to exit.\n")

    try:
        while True:
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("\nShutting down...")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Motion Planning with cuRobo",
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Launch interactive Viser viewer with Move and Grasp buttons",
    )
    parser.add_argument(
        "--mode",
        choices=["pose", "grasp", "all"],
        default="all",
        help="Which example to run (default: all)",
    )
    parser.add_argument(
        "--robot",
        type=str,
        default="franka.yml",
        help="Robot config file (default: franka.yml)",
    )
    parser.add_argument(
        "--scene",
        type=str,
        default="collision_test.yml",
        help="Scene config file (default: collision_test.yml)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8080,
        help="Viser server port (default: 8080)",
    )
    args = parser.parse_args()

    if args.interactive:
        interactive_motion_planning(
            robot_file=args.robot, scene_file=args.scene, port=args.port,
        )
        return

    if args.mode in ("pose", "all"):
        print("=== Pose-to-Pose Motion Planning ===")
        pose_planning_example()
        print()



if __name__ == "__main__":
    main()
