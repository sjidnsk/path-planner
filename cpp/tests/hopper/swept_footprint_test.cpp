#include <array>
#include <cmath>
#include <numbers>

#include <gtest/gtest.h>

#include "lunar_path_planner/v3/hopper/swept_footprint.hpp"

namespace lunar::planning::v3 {
namespace {

TEST(SweptFootprintTest,
     ContainsEveryRotatedRectangleAcrossYawInterval) {
  const ConvexPolygon2d rectangle{
      {{-0.6, -0.3}, {0.6, -0.3},
       {0.6, 0.3}, {-0.6, 0.3}}};
  const CircularYawInterval yaw{
      .start_rad = -0.35,
      .span_rad = 1.4,
  };
  const auto directions = make_uniform_unit_directions(32U);
  const auto result = outer_approximate_rotated_footprint(
      rectangle, yaw, directions, 0.08);
  ASSERT_TRUE(IsOk(result));
  const auto& envelope =
      std::get<SweptFootprintEnvelope>(result);

  for (int sample = 0; sample <= 200; ++sample) {
    const double angle =
        yaw.start_rad + yaw.span_rad *
            static_cast<double>(sample) / 200.0;
    const Eigen::Rotation2Dd rotation(angle);
    ConvexPolygon2d rotated;
    for (const Eigen::Vector2d& vertex :
         rectangle.vertices_ccw) {
      rotated.vertices_ccw.push_back(rotation * vertex);
    }
    EXPECT_TRUE(polygon_is_contained(
        rotated, envelope, 1.0e-9));
  }
}

TEST(SweptFootprintTest, RejectsNonUnitSupportDirection) {
  const std::array invalid{
      Eigen::Vector2d{1.0, 0.0},
      Eigen::Vector2d{0.0, 2.0},
      Eigen::Vector2d{-1.0, 0.0},
      Eigen::Vector2d{0.0, -1.0},
  };
  EXPECT_FALSE(IsOk(outer_approximate_rotated_footprint(
      ConvexPolygon2d{
          {{-0.5, -0.25}, {0.5, -0.25},
           {0.5, 0.25}, {-0.5, 0.25}}},
      CircularYawInterval{
          .start_rad = -std::numbers::pi,
          .span_rad = 2.0 * std::numbers::pi},
      invalid, 0.0)));
}

}  // namespace
}  // namespace lunar::planning::v3
