cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(nls VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Eigen library
find_package(Eigen3 3.3 REQUIRED NO_MODULE)
find_package(fmt REQUIRED)
find_package(spdlog REQUIRED)
find_package(Ceres REQUIRED)

# NLS test
add_executable(nls_test
  src/nls_test.cpp
  src/cholesky.cpp
)
target_link_libraries(nls_test
  Eigen3::Eigen
  fmt::fmt
  spdlog::spdlog
)
target_include_directories(nls_test PUBLIC
  include
)
target_compile_features(nls_test PUBLIC cxx_std_20)
# set_target_properties(nls_test PROPERTIES CMAKE_BUILD_TYPE "Debug")

# Robust NLS
add_executable(robust_nls
  src/levenberg_marquardt.cpp
  src/robust_nls.cpp
)
target_link_libraries(robust_nls
  Eigen3::Eigen
  fmt::fmt
  spdlog::spdlog
)
target_include_directories(robust_nls PUBLIC
  include
)
target_compile_features(robust_nls PUBLIC cxx_std_20)

# Sphere fitting
add_executable(ceres_test
    src/ceres_test.cpp
)
target_link_libraries(ceres_test
    Ceres::ceres
    Eigen3::Eigen
    fmt::fmt
    spdlog::spdlog
)
target_compile_features(ceres_test PUBLIC cxx_std_20)
