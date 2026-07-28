#include <Eigen/Core>
#include <gtest/gtest.h>
#include <nlohmann/json.hpp>

TEST(ToolchainSmoke, UsesPinnedDependenciesAndCpp20) {
  static_assert(__cplusplus >= 202002L);
  Eigen::Vector2d value{1.0, 2.0};
  nlohmann::json payload{{"x", value.x()}, {"y", value.y()}};
  EXPECT_EQ(payload.at("x").get<double>(), 1.0);
  EXPECT_EQ(payload.at("y").get<double>(), 2.0);
}
