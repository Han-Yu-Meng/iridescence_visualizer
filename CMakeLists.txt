cmake_minimum_required(VERSION 3.14)
project(iridescence_visualizer LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(PCL REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(Iridescence REQUIRED)

fins_add_node(${PROJECT_NAME} SHARED 
  src/nodes.cpp
)

target_include_directories(${PROJECT_NAME} PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  ${PCL_INCLUDE_DIRS}
  ${iridescence_INCLUDE_DIRS}
  /usr/include/iridescence
)

target_link_libraries(${PROJECT_NAME} PRIVATE
  ${PCL_LIBRARIES}
  Iridescence::Iridescence
  boost_system
  boost_filesystem
)

fins_link_ros_dependencies(${PROJECT_NAME}
  std_msgs
  sensor_msgs
  nav_msgs
  geometry_msgs
)
