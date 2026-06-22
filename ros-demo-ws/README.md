# ROS Manipulation Demo

This folder is a workspace for a small tabletop demonstration demo. There are a few quirks to the setup to get ROS packages working with python dependencies that conflict with ROS/system python. **Ensure that you have pulled the submodules in this repo.**

## System Requirements

- Ubuntu 24
- GPU with CUDA support

## Setup
### ROS/Gazebo Installation
This demo is supposed to run under ROS2 Jazzy and Gazebo Harmonic. Follow the
instructions [here](https://docs.ros.org/en/jazzy/Installation.html) and
[here](https://gazebosim.org/docs/harmonic/getstarted/). The Franka Driver
repository is the basis of this workspace, so follow steps 4 and 5 form
[here](https://github.com/frankarobotics/franka_ros2/tree/jazzy), treating this
workspace as the one in the instructions.

Next, install the following ROS packages through apt:
- `ros-jazzy-ros-gz`
- `ros-jazzy-ros2-control`
- `ros-jazzy-ros2-controllers`
- `ros-jazzy-parallel-gripper-controller`
- `ros-jazzy-gz-ros2-control`


### venv setup

Incompatible Python dependencies are installed into a venv. First, install
[uv](https://docs.astral.sh/uv/getting-started/installation/). Then install the
python packages in this repository into the venv, assuming you start in this
directory:

``` sh
uv venv --python 3.11
source .venv/bin/activate

cd ../curobov2&/curobo
uv pip install .[cu13-torch]

cd - 
cd ../project_3dv
uv pip install -e .
```

The demo also depends on SuperDec. After acquiring a copy of the source code,
also install it into the venv.

### Building the workspace

Finally, simply run the following commands in this directory.

``` sh
. /opt/ros/jazzy/setup.bash
colcon build
```

### Running the demo

In seperate terminals, run each of the following sections. There is no single
launchfile, as some nodes need to be run with the venv python interpreter.

``` sh
# simulator
. install/setup.bash
ros2 launch motion_planning_3dv gazebo_franka.launch.py

# perception - Needs venv
. install/setup.bash
. .venv/bin/activate
python3 install/motion_planning_3dv/lib/motion_planning_3dv/sq_perception

# scene manager
. install/setup.bash
ros2 run motion_planning_3dv scene_manager

# planner - Needs venv
. install/setup.bash
. .venv/bin/activate
python3 install/motion_planning_3dv/lib/motion_planning_3dv/curobo_ros_executor 

# demo script
. install/setup.bash
ros2 run motion_planning_3dv demo
```

