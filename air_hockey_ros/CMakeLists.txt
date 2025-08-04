cmake_minimum_required(VERSION 3.5)
project(air_hockey_ros)

# Dependencies for building and runtime
find_package(ament_cmake REQUIRED)
find_package(rclpy REQUIRED)
find_package(rosidl_default_generators REQUIRED)

# Generate interfaces from .msg and .srv files
rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/AgentCommand.msg"
  "msg/WorldState.msg"
  "srv/StartGame.srv"
)

# Install Python code
ament_python_install_package(${PROJECT_NAME})

ament_package()
