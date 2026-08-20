cmake_minimum_required(VERSION 3.16)
project(super_ik_tsinghua_1dqp LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(Eigen3 REQUIRED)

add_library(super_ik_tsinghua_1dqp
  src/solver.cpp
)
target_include_directories(super_ik_tsinghua_1dqp PUBLIC include)
target_link_libraries(super_ik_tsinghua_1dqp PUBLIC Eigen3::Eigen)

add_executable(super_ik_tsinghua_1dqp_demo app/main.cpp)
target_link_libraries(super_ik_tsinghua_1dqp_demo PRIVATE super_ik_tsinghua_1dqp)
