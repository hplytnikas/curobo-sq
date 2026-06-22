# ROS Manipulation Demo

This folder is a workspace for a small tabletop demonstration demo. There are a few quirks to the setup to get ROS packages working with python dependencies that conflict with ROS/system python. **Ensure that you have pulled the submodules in this repo.** The results of the demo are shown in ![this video](demo.mp4).


## Setup
### System Requirements

- Ubuntu 24
- GPU with CUDA support

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

cd ../curobov2/curobo
uv pip install .[cu13-torch]

cd - 
cd ../project_3dv
uv pip install -e .
```

The demo also depends on SuperDec/Superflex. After acquiring a copy of the source code,
also install it into the venv as described in the finetuning section of this repo.

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

# Files

Most of the files are parts of `franka_ros2`. Our contributions are the following:

## `motion_planning_3dv`
This package is the core of the demo, containing all the relevant nodes as well as
some utilities.

### `curobo_ros_executor.py`

ROS interface to cuRobo. This node consumes a scene representation via a topic
(`/scence_superquadrics`) and provides moving the arm and manipulating objects
via actions (`/move_arm`, `/grasp_object`, `/release_object`). The
manipultion actions also update the robot model for the purposes of
collision-free planning.

The node interfaces with the arm through standard `ros2_control` actions and topics.

### `sq_utils.py`

Contains a helper to mesh a superquadric for visualization.

### `sq_renderer.py`

Republishes the data from a superquadric topic as visualization messages vor RViz.

### `scene_renderer.py`

This node manages a superquadric scene representation, consumed on-demand from
a topic (`/scene`). Updates can be triggered via a service, and the current
scene is regularly published as a flat list of superquadrics
(`/scene_superquadrics`). The manager enables editing of the scene through
services for querying, inserting and deleting objects - By deleting and then
reinserting an object with an updated position, objects can be relocated in the
scene.

### `sq_perception.py`

This node uses our perception pipeline to process incoming depth images
(`/camera/depth/image`) and publishes a superquadric scene for the rest of the
system, both as a structrued and as a flattened representation. It also
publishes intermediate results for visualization.

### `depth_image_m_to_mm.py`

This node simply converts the units in a depth image. This node is somewhat
redundant as most depth image related software exposes unit configurations, but
this node is used for tidy isolation.

### `demo.py`

This node takes the role of an upstream application using our software to
control the arm. It sequences the different APIs in our project to demonstrate
its functionality.

## `superquadric_interfaces`

This package contains all the custom message definitions (messages, services and 
actions) used in this project.
