cmake_minimum_required(VERSION 3.5 FATAL_ERROR)
project(nls VERSION 0.1.0 LANGUAGES CXX)

#############
# Configure #
#############

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_BUILD_TYPE Release)

# Eigen library
find_package(Eigen3 3.3 REQUIRED NO_MODULE)
find_package(fmt REQUIRED)
find_package(spdlog REQUIRED)
find_package(Ceres REQUIRED)

#########
# Build #
#########

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

###########
# Install #
###########

install(TARGETS nls_test robust_nls ceres_test
  RUNTIME DESTINATION bin
)
