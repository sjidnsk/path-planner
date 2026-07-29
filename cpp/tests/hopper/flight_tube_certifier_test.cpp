#include <gtest/gtest.h>

#include "hopper_test_fixtures.hpp"
#include "lunar_path_planner/v3/hopper/flight_tube_certifier.hpp"

namespace lunar::planning::v3 {
namespace {

TEST(FlightTubeCertifierTest,
     NullMapFailsClosedWithoutPartialTube) {
  const auto capability = std::get<HopperCapabilityView>(
      bind_hopper_capability(
          hopper_test::ValidCapability(),
          hopper_test::ValidBindings()));
  const auto limits = std::get<HopperPlannerLimits>(
      bind_hopper_limits(hopper_test::ValidAlgorithm()));
  FlightTubeCertificationInput input{};
  input.map = nullptr;

  const auto result =
      FlightTubeCertifier{capability, limits}.certify(input);
  EXPECT_FALSE(result.certified_tube.has_value());
  EXPECT_EQ(result.diagnostics.rejection_reason, "missing_map");
}

}  // namespace
}  // namespace lunar::planning::v3
