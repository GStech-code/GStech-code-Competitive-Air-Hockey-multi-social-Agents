cmake_minimum_required(VERSION 3.16)
project(air_hockey_ros)

find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)
find_package(rclpy REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(builtin_interfaces REQUIRED)

rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/AgentCommand.msg"
  "msg/WorldState.msg"
  "srv/StartGame.srv"
  DEPENDENCIES builtin_interfaces
)

if(BUILD_TESTING)
  find_package(ament_cmake_pytest REQUIRED)
endif()

ament_get_python_install_dir(PYTHON_INSTALL_DIR)

# Python packages
install(DIRECTORY air_hockey_ros/ DESTINATION ${PYTHON_INSTALL_DIR}/air_hockey_ros)
install(DIRECTORY policies/       DESTINATION ${PYTHON_INSTALL_DIR}/policies)
install(DIRECTORY simulations/    DESTINATION ${PYTHON_INSTALL_DIR}/simulations)

# Data + launch
install(DIRECTORY game_scenarios DESTINATION share/${PROJECT_NAME})
install(DIRECTORY launch         DESTINATION share/${PROJECT_NAME})

# Executable scripts (must be chmod +x)
install(PROGRAMS
  air_hockey_ros/agent_node.py
  air_hockey_ros/game_manager_node.py
  DESTINATION lib/${PROJECT_NAME}
)

ament_export_dependencies(rosidl_default_runtime rclpy)

ament_package()
