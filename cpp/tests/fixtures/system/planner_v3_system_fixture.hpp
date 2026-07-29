#pragma once

#include <memory>
#include <string>

#include "lunar_path_planner/v3/api/planner_v3.hpp"
#include "lunar_path_planner/v3/codec/json_codec.hpp"
#include "lunar_path_planner/v3/codec/semantic_validator.hpp"

namespace lunar::planning::v3::system_test {

enum class MapScenario {
  kOpenKnown,
  kKnownObstacleAtGoal,
  kUnknownGoalWithSafeFrontier,
};

enum class RegistryFault {
  kNone,
  kMissingCapability,
  kCapabilityBindingsMismatch,
};

struct SystemScenario final {
  PlanningRequest request;
  std::shared_ptr<const ContractObjectRegistry> registry;
  std::unique_ptr<SafeProjectionCache> projection_cache;
};

[[nodiscard]] SystemScenario MakeSystemScenario(
    PlatformType platform_type,
    MapScenario map_scenario = MapScenario::kOpenKnown,
    RegistryFault registry_fault = RegistryFault::kNone);

[[nodiscard]] std::unique_ptr<PlannerV3> MakePlanner(
    SystemScenario& scenario);

[[nodiscard]] PreviousExecutionContext MakeGroundExecutionContext(
    const SystemScenario& scenario,
    bool stale_source_map = false);

[[nodiscard]] Result<Sha256Digest> ResponseHash(
    const PlanningResponse& response);

[[nodiscard]] std::string DescribeIssues(
    const ValidationReport& report);

}  // namespace lunar::planning::v3::system_test
