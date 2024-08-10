cmake_minimum_required(VERSION 3.5 FATAL_ERROR)
project(nls VERSION 0.1.0 LANGUAGES CXX)

#############
# Configure #
#############

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_BUILD_TYPE Release)

option(BUILD_TESTING "Build tests" ON)
option(BUILD_EXAMPLES "Build examples" ON)

# Eigen library
find_package(Eigen3 3.3 REQUIRED NO_MODULE)
find_package(fmt REQUIRED)
find_package(spdlog REQUIRED)
find_package(Ceres REQUIRED)

#########
# Build #
#########

# NLS test
add_executable(prototype
  src/nls_test.cpp
  src/cholesky.cpp
)
target_link_libraries(prototype
  Eigen3::Eigen
  fmt::fmt
  spdlog::spdlog
)
target_include_directories(prototype PUBLIC
  include
)

# NLS library
add_library(nls SHARED
  src/levenberg_marquardt.cpp
)
target_link_libraries(nls PUBLIC
  Eigen3::Eigen
  spdlog::spdlog
)
target_include_directories(nls PUBLIC
  include
)

if (BUILD_EXAMPLES)
  add_executable(sphere_fitting
    examples/sphere_fitting.cpp
  )
  target_link_libraries(sphere_fitting
    nls
    spdlog::spdlog
  )
endif()

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
target_include_directories(ceres_test PUBLIC
  include
)

########
# Test #
########

enable_testing()
if (BUILD_TESTING)
  find_package(GTest REQUIRED)
  add_executable(nls_test
    test/nls_test.cpp
  )
  target_link_libraries(nls_test
    nls
    GTest::gtest
    GTest::gtest_main
  )
  add_test(NAME nls_test COMMAND nls_test)
endif()

###########
# Install #
###########

install(TARGETS nls_test ceres_test
  RUNTIME DESTINATION bin
)
install(TARGETS nls
  LIBRARY DESTINATION lib
)

if (TARGET)
  install(TARGETS sphere_fitting
    RUNTIME DESTINATION bin
  )
endif()
