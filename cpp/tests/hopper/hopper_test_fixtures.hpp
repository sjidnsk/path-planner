#pragma once

#include <chrono>
#include <memory>
#include <string>

#include "lunar_path_planner/v3/hopper/hopper_config.hpp"
#include "lunar_path_planner/v3/map/immutable_snapshot.hpp"

namespace lunar::planning::v3::hopper_test {

[[nodiscard]] ContentRef Ref(std::string id, char digit);
[[nodiscard]] AxisAlignedBox3 Box(
    Vec3 center = {}, Vec3 half_extent = {});
[[nodiscard]] DeterministicVectorSet3 ZeroVectorSet();
[[nodiscard]] HopperErrorBounds ZeroHopperError();
[[nodiscard]] ConvexPolytope3 UnitBoxPolytope(
    double half_x, double half_y, double minimum_z,
    double maximum_z);
[[nodiscard]] SafetyCapabilityProfile ValidCapability();
[[nodiscard]] PlannerAlgorithmConfig ValidAlgorithm();
[[nodiscard]] ResolvedCapabilityBindings ValidBindings();
[[nodiscard]] std::shared_ptr<const ImmutableMapSnapshot> FlatMap(
    std::size_t width = 12U, std::size_t height = 12U,
    double resolution_m = 0.5);

}  // namespace lunar::planning::v3::hopper_test
