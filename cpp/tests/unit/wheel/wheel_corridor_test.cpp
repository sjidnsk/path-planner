#include <chrono>
#include <cstdint>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "lunar_path_planner/v3/wheel/wheel_corridor.hpp"

namespace lunar::planning::v3 {
namespace {

using namespace std::chrono_literals;

[[nodiscard]] ContentRef Ref(std::string id, char digit) {
  return {std::move(id), 1U, std::string(64U, digit)};
}

[[nodiscard]] SafeProjection Projection(float clearance_m) {
  SafeProjection projection;
  projection.geometry = {
      .width = 6U,
      .height = 6U,
      .resolution_m = 1.0,
      .origin_m = {0.0, 0.0},
      .frame_id = "map",
  };
  const std::size_t count = projection.geometry.CellCount();
  projection.known_mask.assign(count, 1U);
  projection.hard_feasible_mask.assign(count, 1U);
  projection.esdf_clearance_m.assign(count, clearance_m);
  return projection;
}

[[nodiscard]] WheelLatticeEdge Edge(
    std::int32_t source_x,
    std::int32_t target_x) {
  return {
      .source =
          {source_x, 2, 0, WheelMotionMode::kForward},
      .target =
          {target_x, 2, 0, WheelMotionMode::kForward},
      .primitive_id = "drive",
      .capability_primitive_id = "drive",
      .primitive_kind = WheelPrimitiveKind::kDriveLine,
      .source_pose =
          {{static_cast<double>(source_x) + 0.5, 2.5, 0.0},
           0.0},
      .target_pose =
          {{static_cast<double>(target_x) + 0.5, 2.5, 0.0},
           0.0},
      .transition_time = DurationNanoseconds{1s},
      .validation_ref = Ref("sweep", 'a'),
  };
}

[[nodiscard]] WheelCollisionEnvelope Envelope() {
  return {
      .footprint =
          {
              .vertices_xy_m =
                  {{-0.1, -0.1}, {0.1, -0.1},
                   {0.1, 0.1}, {-0.1, 0.1}},
              .minimum_z_m = -0.1,
              .maximum_z_m = 0.4,
          },
      .horizontal_tracking_error_m = 0.0,
      .minimum_clearance_m = 0.0,
  };
}

TEST(WheelCorridorTest, CertifiesConnectedOpenRoute) {
  auto projection = Projection(3.0F);
  const std::vector edges{Edge(1, 2), Edge(2, 3)};

  const auto result = BuildWheelCorridor(
      {
          .edges = edges,
          .projection = projection,
          .envelope = Envelope(),
          .config =
              {
                  .maximum_regions = 8U,
                  .maximum_inflation_iterations = 128U,
                  .maximum_halfplanes_per_region = 16U,
                  .maximum_centerline_samples = 32U,
                  .sampling_spacing_m = 0.5,
                  .additional_margin_m = 0.0,
                  .maximum_curvature_per_m = 2.0,
              },
      });

  EXPECT_EQ(result.status, CorridorStatus::kCertified);
  EXPECT_EQ(result.fallback, CorridorFallback::kNone);
  EXPECT_FALSE(result.cells.empty());
}

TEST(WheelCorridorTest,
     FailsClosedWhenConservativeSectionsDoNotOverlap) {
  auto projection = Projection(0.5F);
  const std::vector edges{Edge(1, 2)};

  const auto result = BuildWheelCorridor(
      {
          .edges = edges,
          .projection = projection,
          .envelope = Envelope(),
          .config =
              {
                  .maximum_regions = 8U,
                  .maximum_inflation_iterations = 128U,
                  .maximum_halfplanes_per_region = 16U,
                  .maximum_centerline_samples = 8U,
                  .sampling_spacing_m = 10.0,
                  .additional_margin_m = 0.0,
                  .maximum_curvature_per_m = 2.0,
              },
      });

  EXPECT_EQ(result.status, CorridorStatus::kFallbackRequired);
  EXPECT_EQ(result.fallback,
            CorridorFallback::kUseDiscreteValidatedPrimitives);
  EXPECT_EQ(result.reason_code,
            "adjacent_cells_do_not_overlap");
}

TEST(WheelCorridorTest, CurvatureOverCapabilityFailsClosed) {
  auto projection = Projection(3.0F);
  auto curved = Edge(1, 2);
  curved.target_pose.yaw_rad = 1.0;
  const std::vector edges{curved};

  const auto result = BuildWheelCorridor(
      {
          .edges = edges,
          .projection = projection,
          .envelope = Envelope(),
          .config =
              {
                  .maximum_regions = 8U,
                  .maximum_inflation_iterations = 128U,
                  .maximum_halfplanes_per_region = 16U,
                  .maximum_centerline_samples = 8U,
                  .sampling_spacing_m = 0.5,
                  .additional_margin_m = 0.0,
                  .maximum_curvature_per_m = 0.5,
              },
      });

  EXPECT_EQ(result.status, CorridorStatus::kFallbackRequired);
  EXPECT_EQ(result.reason_code, "curvature_limit_exceeded");
}

}  // namespace
}  // namespace lunar::planning::v3
