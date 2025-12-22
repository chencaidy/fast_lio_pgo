cmake_minimum_required(VERSION 3.8)
project(fast_lio_pgo)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

add_definitions(-DROOT_DIR="${CMAKE_CURRENT_SOURCE_DIR}")

# Dependencies
find_package(ament_cmake_auto REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(OpenMP REQUIRED)
find_package(GTSAM REQUIRED)
find_package(Sophus REQUIRED)
ament_auto_find_build_dependencies()

# Build messages
rosidl_generate_interfaces(${PROJECT_NAME} msg/Pose6D.msg)
rosidl_get_typesupport_target(fastlio_idl ${PROJECT_NAME} rosidl_typesupport_cpp)

# Build mapping
ament_auto_add_executable(fastlio_mapping
  src/laser_mapping.cpp
  src/esekfom.cpp
  src/ikd_tree.cpp
  src/laser_process.cpp
  src/imu_process.cpp
)
target_link_libraries(fastlio_mapping ${fastlio_idl} ${OpenMP_CXX_LIBRARIES} Sophus::Sophus)
target_compile_options(fastlio_mapping PRIVATE ${OpenMP_CXX_FLAGS})

# Build pgo
ament_auto_add_executable(laser_pgo
  src/laser_posegraph_optimization.cpp
)
target_link_libraries(laser_pgo gtsam ${OpenMP_CXX_LIBRARIES})
target_compile_options(laser_pgo PRIVATE ${OpenMP_CXX_FLAGS})

# Install
ament_auto_package(
  INSTALL_TO_SHARE config launch rviz
  USE_SCOPED_HEADER_INSTALL_DIR
)
