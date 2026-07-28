cmake_minimum_required(VERSION 3.28)
project(lunar_path_planner_v3 VERSION 0.1.0 LANGUAGES CXX)

option(LPP_V3_BUILD_TESTS "Build C++ tests" ON)
option(LPP_V3_BUILD_BENCHMARKS "Build C++ benchmarks" ON)
option(LPP_V3_ENABLE_OSQP "Build the optional OSQP backend" OFF)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(Eigen3 5.0.1 EXACT CONFIG REQUIRED)
find_package(nlohmann_json 3.12.0 EXACT CONFIG REQUIRED)

add_library(lpp_v3_project_options INTERFACE)
target_compile_features(lpp_v3_project_options INTERFACE cxx_std_20)
add_library(lpp_v3_project_warnings INTERFACE)

include(cmake/ProjectOptions.cmake)
include(cmake/CompilerWarnings.cmake)
lpp_v3_set_project_options(lpp_v3_project_options)
lpp_v3_set_project_warnings(lpp_v3_project_warnings)

if(LPP_V3_BUILD_TESTS)
  enable_testing()
  find_package(GTest 1.17.0 EXACT CONFIG REQUIRED)
  add_executable(lpp_v3_toolchain_tests tests/toolchain/smoke_test.cpp)
  target_link_libraries(
    lpp_v3_toolchain_tests
    PRIVATE Eigen3::Eigen nlohmann_json::nlohmann_json
            GTest::gtest_main lpp_v3_project_options lpp_v3_project_warnings)
  include(GoogleTest)
  gtest_discover_tests(
    lpp_v3_toolchain_tests
    TEST_PREFIX "lpp_v3_toolchain.")
endif()

if(LPP_V3_BUILD_BENCHMARKS)
  find_package(benchmark 1.9.5 EXACT CONFIG REQUIRED)
  add_executable(
    lpp_v3_shared_core_benchmark benchmarks/shared_core_benchmark.cpp)
  target_link_libraries(
    lpp_v3_shared_core_benchmark
    PRIVATE benchmark::benchmark_main lpp_v3_project_options)
endif()
