#include <array>
#include <cstddef>
#include <string>

#include <gtest/gtest.h>

#include "fixtures/system/planner_v3_system_fixture.hpp"

namespace lunar::planning::v3 {
namespace {

TEST(PlannerV3DeterminismPropertyTest,
     FixedTypedInputsHaveStableResponseHashesForOneHundredCalls) {
  constexpr std::array<PlatformType, 3U> kPlatforms{
      PlatformType::kWheeled,
      PlatformType::kLegged,
      PlatformType::kHopper,
  };

  for (const PlatformType platform_type : kPlatforms) {
    SCOPED_TRACE(static_cast<int>(platform_type));
    auto scenario =
        system_test::MakeSystemScenario(platform_type);
    auto planner = system_test::MakePlanner(scenario);
    const PlanningResponse baseline =
        planner->Plan(scenario.request);
    const auto baseline_hash =
        system_test::ResponseHash(baseline);
    ASSERT_TRUE(IsOk(baseline_hash));
    const Sha256Digest expected =
        std::get<Sha256Digest>(baseline_hash);

    for (std::size_t iteration = 0U;
         iteration < 100U; ++iteration) {
      const PlanningResponse response =
          planner->Plan(scenario.request);
      const auto response_hash =
          system_test::ResponseHash(response);
      ASSERT_TRUE(IsOk(response_hash));
      EXPECT_EQ(
          std::get<Sha256Digest>(response_hash),
          expected)
          << "iteration=" << iteration;
    }
  }
}

}  // namespace
}  // namespace lunar::planning::v3
