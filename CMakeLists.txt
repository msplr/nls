cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(nls VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_BUILD_TYPE Debug)

# Eigen library
find_package(Eigen3 3.3 REQUIRED NO_MODULE)
find_package(fmt REQUIRED)
find_package(spdlog REQUIRED)

add_executable(${PROJECT_NAME}
	src/main.cpp
	src/cholesky.cpp
)
target_link_libraries(${PROJECT_NAME}
	Eigen3::Eigen
    fmt::fmt
    spdlog::spdlog
)
target_include_directories(${PROJECT_NAME} PUBLIC
  include
#   eigen-3.3.9
)
target_compile_features(${PROJECT_NAME} PUBLIC cxx_std_20)
set_target_properties(${PROJECT_NAME} PROPERTIES CMAKE_BUILD_TYPE "Debug")
