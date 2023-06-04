cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(fitting VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_BUILD_TYPE Debug)

# Eigen library
find_package(Eigen3 3.3 REQUIRED NO_MODULE)
find_package(fmt REQUIRED)
find_package(spdlog REQUIRED)
find_package(Ceres REQUIRED)

add_executable(sphere_fitting
    src/sphere_fitting.cpp
)
target_link_libraries(sphere_fitting
    Ceres::ceres
    Eigen3::Eigen
    fmt::fmt
    spdlog::spdlog
)
