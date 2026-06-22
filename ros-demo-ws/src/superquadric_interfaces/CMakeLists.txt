cmake_minimum_required(VERSION 3.8)
project(superquadric_interfaces)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# Find dependencies
find_package(ament_cmake REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(action_msgs REQUIRED)

# Generate the messages
rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/Superquadric.msg"
  "msg/SuperquadricArray.msg"
  "msg/SceneObject.msg"
  "msg/Scene.msg"
  "srv/ListObjects.srv"
  "srv/InsertObject.srv"
  "srv/DeleteObject.srv"
  "action/Move.action"
  "action/Grasp.action"
  "action/Release.action"
  DEPENDENCIES geometry_msgs std_msgs
)

ament_package()
