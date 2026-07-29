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
find_package(double-conversion 3.4.0 EXACT CONFIG REQUIRED)
find_path(LPP_V3_PICOSHA2_INCLUDE_DIR NAMES picosha2.h REQUIRED)

add_library(lpp_v3_project_options INTERFACE)
target_compile_features(lpp_v3_project_options INTERFACE cxx_std_20)
add_library(lpp_v3_project_warnings INTERFACE)

include(cmake/ProjectOptions.cmake)
include(cmake/CompilerWarnings.cmake)
lpp_v3_set_project_options(lpp_v3_project_options)
lpp_v3_set_project_warnings(lpp_v3_project_warnings)

add_library(lpp_v3_contracts src/contracts/base_types.cpp)
target_include_directories(
  lpp_v3_contracts
  PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include")
target_link_libraries(
  lpp_v3_contracts
  PRIVATE Eigen3::Eigen lpp_v3_project_options lpp_v3_project_warnings)

add_library(
  lpp_v3_wire_primitives
  src/codec/jcs_canonicalizer.cpp
  src/crypto/sha256.cpp)
target_include_directories(
  lpp_v3_wire_primitives
  PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include"
  PRIVATE "${LPP_V3_PICOSHA2_INCLUDE_DIR}")
target_link_libraries(
  lpp_v3_wire_primitives
  PUBLIC lpp_v3_contracts nlohmann_json::nlohmann_json
  PRIVATE double-conversion::double-conversion
          lpp_v3_project_options lpp_v3_project_warnings)

add_library(
  lpp_v3_map_snapshot
  src/map/immutable_snapshot.cpp)
target_include_directories(
  lpp_v3_map_snapshot
  PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include")
target_link_libraries(
  lpp_v3_map_snapshot
  PUBLIC lpp_v3_contracts lpp_v3_wire_primitives
         nlohmann_json::nlohmann_json
  PRIVATE lpp_v3_project_options lpp_v3_project_warnings)

add_library(
  lpp_v3_cost
  src/cost/learned_cost_snapshot.cpp)
target_include_directories(
  lpp_v3_cost
  PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include")
target_link_libraries(
  lpp_v3_cost
  PUBLIC lpp_v3_contracts lpp_v3_map_snapshot
  PRIVATE lpp_v3_project_options lpp_v3_project_warnings)

add_library(
  lpp_v3_map
  src/map/safe_projection.cpp)
target_include_directories(
  lpp_v3_map
  PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include")
target_link_libraries(
  lpp_v3_map
  PUBLIC lpp_v3_contracts lpp_v3_cost lpp_v3_map_snapshot
  PRIVATE lpp_v3_project_options lpp_v3_project_warnings)

add_library(
  lpp_v3_search
  src/search/candidate_ranker.cpp)
target_include_directories(
  lpp_v3_search
  PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include")
target_link_libraries(
  lpp_v3_search
  PUBLIC lpp_v3_contracts
  PRIVATE lpp_v3_project_options lpp_v3_project_warnings)

add_library(
  lpp_v3_goal
  src/goal/terminal_resolver.cpp)
target_include_directories(
  lpp_v3_goal
  PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include")
target_link_libraries(
  lpp_v3_goal
  PUBLIC lpp_v3_contracts lpp_v3_map lpp_v3_search
  PRIVATE lpp_v3_project_options lpp_v3_project_warnings)

add_library(
  lpp_v3_corridor
  src/corridor/convex_corridor.cpp)
target_include_directories(
  lpp_v3_corridor
  PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include")
target_link_libraries(
  lpp_v3_corridor
  PUBLIC lpp_v3_contracts lpp_v3_map
  PRIVATE lpp_v3_project_options lpp_v3_project_warnings)

add_library(
  lpp_v3_optimization
  src/optimization/bounded_qp_solver.cpp)
target_include_directories(
  lpp_v3_optimization
  PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include")
target_link_libraries(
  lpp_v3_optimization
  PUBLIC Eigen3::Eigen lpp_v3_contracts
  PRIVATE lpp_v3_project_options lpp_v3_project_warnings)

add_library(lpp_v3_cache INTERFACE)
target_include_directories(
  lpp_v3_cache
  INTERFACE "${CMAKE_CURRENT_SOURCE_DIR}/include")
target_link_libraries(
  lpp_v3_cache
  INTERFACE lpp_v3_contracts lpp_v3_project_options)

if(LPP_V3_ENABLE_OSQP)
  # The vcpkg manifest and override pin OSQP to 1.0.0. Its upstream
  # generated package-version file reports 0.0.0 when built from the
  # release archive, so version enforcement must remain at the registry
  # boundary instead of using find_package(... EXACT).
  find_package(osqp CONFIG REQUIRED)
  add_library(
    lpp_v3_osqp_backend
    src/optimization/osqp_bounded_qp_solver.cpp)
  target_compile_definitions(
    lpp_v3_osqp_backend
    PUBLIC LPP_V3_HAS_OSQP=1)
  target_include_directories(
    lpp_v3_osqp_backend
    PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include")
  target_link_libraries(
    lpp_v3_osqp_backend
    PUBLIC lpp_v3_optimization osqp::osqp
    PRIVATE lpp_v3_project_options lpp_v3_project_warnings)
endif()

add_library(
  lpp_v3_codec
  src/codec/json_codec.cpp
  src/codec/semantic_validator.cpp)
target_include_directories(
  lpp_v3_codec
  PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include")
target_link_libraries(
  lpp_v3_codec
  PUBLIC lpp_v3_contracts lpp_v3_cost lpp_v3_map
         lpp_v3_wire_primitives
         nlohmann_json::nlohmann_json
  PRIVATE lpp_v3_project_options lpp_v3_project_warnings)

add_library(lpp_v3_common INTERFACE)
target_link_libraries(
  lpp_v3_common
  INTERFACE
    lpp_v3_contracts
    lpp_v3_codec
    lpp_v3_map
    lpp_v3_cost
    lpp_v3_goal
    lpp_v3_search
    lpp_v3_corridor
    lpp_v3_optimization
    lpp_v3_cache)

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

  add_executable(
    lpp_v3_contracts_tests tests/contracts/planning_request_test.cpp)
  target_link_libraries(
    lpp_v3_contracts_tests
    PRIVATE lpp_v3_contracts GTest::gtest_main
            lpp_v3_project_options lpp_v3_project_warnings)
  gtest_discover_tests(
    lpp_v3_contracts_tests
    TEST_PREFIX "lpp_v3_contracts.")

  add_executable(
    lpp_v3_references_tests tests/contracts/reference_bundle_test.cpp)
  target_link_libraries(
    lpp_v3_references_tests
    PRIVATE lpp_v3_contracts GTest::gtest_main
            lpp_v3_project_options lpp_v3_project_warnings)
  gtest_discover_tests(
    lpp_v3_references_tests
    TEST_PREFIX "lpp_v3_references.")

  add_executable(
    lpp_v3_map_tests
    tests/map/immutable_snapshot_test.cpp
    tests/map/safe_projection_test.cpp)
  target_link_libraries(
    lpp_v3_map_tests
    PRIVATE lpp_v3_map GTest::gtest_main
            lpp_v3_project_options lpp_v3_project_warnings)
  gtest_discover_tests(
    lpp_v3_map_tests
    TEST_PREFIX "lpp_v3_map.")

  add_executable(
    lpp_v3_cost_tests tests/cost/learned_cost_snapshot_test.cpp)
  target_link_libraries(
    lpp_v3_cost_tests
    PRIVATE lpp_v3_cost GTest::gtest_main
            lpp_v3_project_options lpp_v3_project_warnings)
  gtest_discover_tests(
    lpp_v3_cost_tests
    TEST_PREFIX "lpp_v3_cost.")

  add_executable(
    lpp_v3_search_tests
    tests/search/ara_star_test.cpp
    tests/search/candidate_ranker_test.cpp)
  target_link_libraries(
    lpp_v3_search_tests
    PRIVATE lpp_v3_search GTest::gtest_main
            lpp_v3_project_options lpp_v3_project_warnings)
  gtest_discover_tests(
    lpp_v3_search_tests
    TEST_PREFIX "lpp_v3_search.")

  add_executable(
    lpp_v3_goal_tests
    tests/goal/terminal_resolver_test.cpp)
  target_link_libraries(
    lpp_v3_goal_tests
    PRIVATE lpp_v3_goal GTest::gtest_main
            lpp_v3_project_options lpp_v3_project_warnings)
  gtest_discover_tests(
    lpp_v3_goal_tests
    TEST_PREFIX "lpp_v3_goal.")

  add_executable(
    lpp_v3_corridor_tests
    tests/corridor/convex_corridor_test.cpp)
  target_link_libraries(
    lpp_v3_corridor_tests
    PRIVATE lpp_v3_corridor GTest::gtest_main
            lpp_v3_project_options lpp_v3_project_warnings)
  gtest_discover_tests(
    lpp_v3_corridor_tests
    TEST_PREFIX "lpp_v3_corridor.")

  add_executable(
    lpp_v3_optimization_tests
    tests/optimization/bounded_qp_solver_test.cpp)
  target_link_libraries(
    lpp_v3_optimization_tests
    PRIVATE lpp_v3_optimization GTest::gtest_main
            lpp_v3_project_options lpp_v3_project_warnings)
  gtest_discover_tests(
    lpp_v3_optimization_tests
    TEST_PREFIX "lpp_v3_optimization.")

  add_executable(
    lpp_v3_cache_tests
    tests/cache/deterministic_cache_test.cpp)
  target_link_libraries(
    lpp_v3_cache_tests
    PRIVATE lpp_v3_cache GTest::gtest_main
            lpp_v3_project_options lpp_v3_project_warnings)
  gtest_discover_tests(
    lpp_v3_cache_tests
    TEST_PREFIX "lpp_v3_cache.")

  if(LPP_V3_ENABLE_OSQP)
    add_executable(
      lpp_v3_osqp_tests
      tests/optimization/osqp_bounded_qp_solver_test.cpp)
    target_link_libraries(
      lpp_v3_osqp_tests
      PRIVATE lpp_v3_osqp_backend GTest::gtest_main
              lpp_v3_project_options lpp_v3_project_warnings)
    gtest_discover_tests(
      lpp_v3_osqp_tests
      TEST_PREFIX "lpp_v3_osqp.")
  endif()

  add_executable(
    lpp_v3_crypto_tests tests/crypto/sha256_test.cpp)
  target_link_libraries(
    lpp_v3_crypto_tests
    PRIVATE lpp_v3_wire_primitives GTest::gtest_main
            lpp_v3_project_options lpp_v3_project_warnings)
  gtest_discover_tests(
    lpp_v3_crypto_tests
    TEST_PREFIX "lpp_v3_crypto.")

  add_executable(
    lpp_v3_codec_tests
    tests/codec/jcs_canonicalizer_test.cpp
    tests/codec/json_codec_test.cpp
    tests/codec/semantic_validator_test.cpp)
  target_compile_definitions(
    lpp_v3_codec_tests
    PRIVATE
      LPP_V3_SCHEMA_ROOT="${CMAKE_CURRENT_SOURCE_DIR}/../schemas/v3")
  target_link_libraries(
    lpp_v3_codec_tests
    PRIVATE lpp_v3_codec GTest::gtest_main
            lpp_v3_project_options lpp_v3_project_warnings)
  add_dependencies(lpp_v3_codec_tests lpp_v3_crypto_tests)
  gtest_discover_tests(
    lpp_v3_codec_tests
    TEST_PREFIX "lpp_v3_codec.")
endif()

if(LPP_V3_BUILD_BENCHMARKS)
  find_package(benchmark 1.9.5 EXACT CONFIG REQUIRED)
  add_executable(
    lpp_v3_shared_core_benchmark benchmarks/shared_core_benchmark.cpp)
  target_link_libraries(
    lpp_v3_shared_core_benchmark
    PRIVATE benchmark::benchmark_main lpp_v3_project_options)
endif()
