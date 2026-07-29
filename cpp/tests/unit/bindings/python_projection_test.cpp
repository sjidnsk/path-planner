#include <chrono>
#include <cmath>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include <gtest/gtest.h>

#include "lunar_path_planner/v3/bindings/python_projection.hpp"

namespace lpp = lunar::planning::v3;
namespace {

using namespace std::chrono_literals;

lpp::GridGeometry MakeGrid() {
  return {
      .width = 10U,
      .height = 10U,
      .resolution_m = 1.0,
      .origin_m = {0.0, 0.0},
      .frame_id = "map",
  };
}

lpp::ValidatedPrimitive MakePrimitive(
    std::string id, lpp::PrimitiveKind kind,
    lpp::PoseXyzYaw start, lpp::PoseXyzYaw finish) {
  return {
      .primitive_id = std::move(id),
      .capability_primitive_id = "capability-primitive",
      .primitive_kind = kind,
      .start_pose = start,
      .end_pose = finish,
      .nominal_duration = lpp::DurationNanoseconds{1s},
  };
}

lpp::WheeledReference MakeWheelReference() {
  const lpp::ValidatedPrimitive primitive = MakePrimitive(
      "wheel-drive", lpp::PrimitiveKind::kDriveForward,
      {{1.25, 1.25, 0.0}, 0.0},
      {{3.25, 1.25, 0.0}, 0.75});
  return {
      .reference_id = "wheel-reference",
      .reference_hash = std::string(64U, '1'),
      .reference_time_origin = {"mission", 0ns},
      .segments =
          {
              lpp::DriveSegment{
                  .segment_id = "drive",
                  .time_interval =
                      {
                          lpp::DurationNanoseconds{0ns},
                          lpp::DurationNanoseconds{1s},
                      },
                  .direction = lpp::DriveDirection::kForward,
                  .geometric_path =
                      lpp::ValidatedPrimitiveChain{{primitive}},
              },
          },
  };
}

lpp::WheeledReference MakeWheelSplineReference() {
  return {
      .reference_id = "wheel-spline-reference",
      .reference_hash = std::string(64U, '4'),
      .reference_time_origin = {"mission", 0ns},
      .segments =
          {
              lpp::DriveSegment{
                  .segment_id = "spline-drive",
                  .time_interval =
                      {
                          lpp::DurationNanoseconds{0ns},
                          lpp::DurationNanoseconds{1s},
                      },
                  .direction = lpp::DriveDirection::kForward,
                  .geometric_path =
                      lpp::ClampedCubicBSplinePath{
                          .knots =
                              {0.0, 0.0, 0.0, 0.0,
                               1.0, 1.0, 1.0, 1.0},
                          .control_points =
                              {
                                  {{1.25, 2.25, 0.0}, 0.0},
                                  {{2.25, 2.25, 0.0}, 0.3},
                                  {{3.25, 2.25, 0.0}, 0.7},
                                  {{4.25, 2.25, 0.0}, 1.0},
                              },
                      },
              },
          },
  };
}

lpp::WheeledReference MakeWheelSpinReference() {
  return {
      .reference_id = "wheel-spin-reference",
      .reference_hash = std::string(64U, '5'),
      .reference_time_origin = {"mission", 0ns},
      .segments =
          {
              lpp::SpinSegment{
                  .segment_id = "spin",
                  .time_interval =
                      {
                          lpp::DurationNanoseconds{0ns},
                          lpp::DurationNanoseconds{1s},
                      },
                  .fixed_position_m = {2.25, 2.25, 0.0},
                  .unwrapped_yaw_rad =
                      {
                          .value_semantics =
                              "unwrapped_yaw_rad",
                          .segments =
                              {
                                  {
                                      .start_offset =
                                          lpp::DurationNanoseconds{
                                              0ns},
                                      .end_offset =
                                          lpp::DurationNanoseconds{
                                              1s},
                                      .coefficients =
                                          {0.25, 1.0, 0.0,
                                           0.0},
                                  },
                              },
                      },
              },
          },
  };
}

lpp::LeggedBodyReference MakeLeggedReference() {
  const lpp::ValidatedPrimitive primitive = MakePrimitive(
      "legged-body", lpp::PrimitiveKind::kBodyTranslation,
      {{1.25, 1.25, 0.5}, 0.0},
      {{1.25, 3.25, 0.5}, -0.5});
  return {
      .reference_id = "legged-reference",
      .reference_hash = std::string(64U, '2'),
      .reference_point_id = "base_link",
      .reference_time_origin = {"mission", 0ns},
      .geometric_path =
          lpp::ValidatedPrimitiveChain{{primitive}},
  };
}

lpp::HopperReference MakeHopperReference() {
  return {
      .reference_id = "hopper-reference",
      .reference_hash = std::string(64U, '3'),
      .reference_time_origin = {"mission", 0ns},
      .ground_hold_anchor =
          {
              .anchor_id = "hold",
              .hold_state =
                  {
                      .position_m = {1.25, 1.25, 0.0},
                  },
          },
      .next_landing_region =
          {
              .region_id = "landing-region",
              .frame_id = "map",
              .allowed_yaw_interval =
                  {
                      .start_rad = 0.2,
                      .span_rad = 0.4,
                  },
          },
      .jump_boundary =
          {
              .boundary_id = "jump-boundary",
              .ballistic_flight_time =
                  lpp::DurationNanoseconds{1s},
          },
  };
}

lpp::PlanningResponse MakeActivatedResponse(
    lpp::PlatformType platform_type,
    lpp::PlatformReference reference) {
  lpp::ReferenceBundle bundle{
      .bundle_id = "bundle-7",
      .bundle_revision = 1U,
      .bundle_hash = std::string(64U, 'b'),
      .source_request_id = "request-7",
      .platform_type = platform_type,
      .platform_reference = std::move(reference),
  };
  bundle.route_skeleton.content.waypoints = {
      {{8.25, 8.25, 0.0}, 2.5},
      {{9.25, 8.25, 0.0}, 2.5},
  };
  return {
      .request_id = "request-7",
      .response_time = {"mission", 2ns},
      .planning_outcome =
          lpp::PlanningOutcome::kNewReferenceReady,
      .execution_directive =
          lpp::ExecutionDirective::kActivateNewBundle,
      .reason_code = "NEW_REFERENCE_READY",
      .new_reference_bundle = std::move(bundle),
  };
}

const lpp::PpoPathProjection& ProjectionOf(
    const lpp::Result<lpp::PpoPathProjection>& result) {
  return std::get<lpp::PpoPathProjection>(result);
}

}  // namespace

TEST(PythonProjectionTest,
     ContinuedBundleWithoutInlinePayloadIsDiagnosticOnly) {
  lpp::PlanningResponse response{
      .request_id = "request-7",
      .response_time = {"mission", 2ns},
      .planning_outcome =
          lpp::PlanningOutcome::kNoKnownSafeRoute,
      .execution_directive =
          lpp::ExecutionDirective::kContinueActiveBundle,
      .reason_code = "CONTINUE_ACTIVE_REFERENCE",
      .active_bundle_ref =
          lpp::ContentRef{
              .id = "active-bundle",
              .revision = 4U,
              .content_hash = std::string(64U, 'a'),
          },
  };

  const auto result =
      lpp::ProjectForPpoDiagnostics(response, MakeGrid());

  ASSERT_TRUE(lpp::IsOk(result));
  const auto& projection = ProjectionOf(result);
  EXPECT_FALSE(projection.executable_reference_present);
  EXPECT_TRUE(projection.path_cells.empty());
  EXPECT_DOUBLE_EQ(projection.path_length_m, 0.0);
  EXPECT_EQ(projection.source_bundle_id, "active-bundle");
}

TEST(PythonProjectionTest,
     RouteSkeletonCannotOverrideAuthoritativeWheelGeometry) {
  const auto response = MakeActivatedResponse(
      lpp::PlatformType::kWheeled, MakeWheelReference());

  const auto result =
      lpp::ProjectForPpoDiagnostics(response, MakeGrid());

  ASSERT_TRUE(lpp::IsOk(result));
  const auto& projection = ProjectionOf(result);
  ASSERT_TRUE(projection.executable_reference_present);
  ASSERT_FALSE(projection.path_cells.empty());
  for (const auto& cell : projection.path_cells) {
    EXPECT_EQ(cell.y, 1U);
    EXPECT_NE(cell.y, 8U);
  }
}

TEST(PythonProjectionTest,
     ProjectsWheelPrimitiveGeometryLengthAndFinalYaw) {
  const auto response = MakeActivatedResponse(
      lpp::PlatformType::kWheeled, MakeWheelReference());

  const auto result =
      lpp::ProjectForPpoDiagnostics(response, MakeGrid());

  ASSERT_TRUE(lpp::IsOk(result));
  const auto& projection = ProjectionOf(result);
  EXPECT_TRUE(projection.executable_reference_present);
  EXPECT_EQ(
      projection.path_cells,
      (std::vector<lpp::GridCell>{{1U, 1U}, {2U, 1U},
                                  {3U, 1U}}));
  EXPECT_NEAR(projection.path_length_m, 2.0, 1.0e-12);
  EXPECT_NEAR(projection.final_yaw_rad, 0.75, 1.0e-12);
  EXPECT_EQ(projection.source_bundle_id, "bundle-7");
}

TEST(PythonProjectionTest,
     ProjectsWheelSplineGeometryDeterministically) {
  const auto response = MakeActivatedResponse(
      lpp::PlatformType::kWheeled,
      MakeWheelSplineReference());

  const auto result =
      lpp::ProjectForPpoDiagnostics(response, MakeGrid());

  ASSERT_TRUE(lpp::IsOk(result));
  const auto& projection = ProjectionOf(result);
  EXPECT_EQ(
      projection.path_cells,
      (std::vector<lpp::GridCell>{{1U, 2U}, {2U, 2U},
                                  {3U, 2U}, {4U, 2U}}));
  EXPECT_NEAR(projection.path_length_m, 3.0, 1.0e-12);
  EXPECT_NEAR(projection.final_yaw_rad, 1.0, 1.0e-12);
}

TEST(PythonProjectionTest,
     SpinOnlyReferenceHasOneCellZeroLengthAndTerminalYaw) {
  const auto response = MakeActivatedResponse(
      lpp::PlatformType::kWheeled, MakeWheelSpinReference());

  const auto result =
      lpp::ProjectForPpoDiagnostics(response, MakeGrid());

  ASSERT_TRUE(lpp::IsOk(result));
  const auto& projection = ProjectionOf(result);
  EXPECT_TRUE(projection.executable_reference_present);
  EXPECT_EQ(
      projection.path_cells,
      (std::vector<lpp::GridCell>{{2U, 2U}}));
  EXPECT_DOUBLE_EQ(projection.path_length_m, 0.0);
  EXPECT_NEAR(projection.final_yaw_rad, 1.25, 1.0e-12);
}

TEST(PythonProjectionTest,
     ProjectsLeggedBodyGeometryWithoutClaimingFootsteps) {
  const auto response = MakeActivatedResponse(
      lpp::PlatformType::kLegged, MakeLeggedReference());

  const auto result =
      lpp::ProjectForPpoDiagnostics(response, MakeGrid());

  ASSERT_TRUE(lpp::IsOk(result));
  const auto& projection = ProjectionOf(result);
  EXPECT_TRUE(projection.executable_reference_present);
  EXPECT_EQ(
      projection.path_cells,
      (std::vector<lpp::GridCell>{{1U, 1U}, {1U, 2U},
                                  {1U, 3U}}));
  EXPECT_NEAR(projection.path_length_m, 2.0, 1.0e-12);
  EXPECT_NEAR(projection.final_yaw_rad, -0.5, 1.0e-12);
}

TEST(PythonProjectionTest,
     HopperNeverFabricatesContinuousGroundPath) {
  const auto response = MakeActivatedResponse(
      lpp::PlatformType::kHopper, MakeHopperReference());

  const auto result =
      lpp::ProjectForPpoDiagnostics(response, MakeGrid());

  ASSERT_TRUE(lpp::IsOk(result));
  const auto& projection = ProjectionOf(result);
  EXPECT_FALSE(projection.executable_reference_present);
  EXPECT_TRUE(projection.path_cells.empty());
  EXPECT_DOUBLE_EQ(projection.path_length_m, 0.0);
  EXPECT_DOUBLE_EQ(projection.final_yaw_rad, 0.0);
  EXPECT_EQ(projection.source_bundle_id, "bundle-7");
}

TEST(PythonProjectionTest,
     SameBundleAndGridProduceStableProjection) {
  const auto response = MakeActivatedResponse(
      lpp::PlatformType::kWheeled, MakeWheelReference());
  const auto grid = MakeGrid();

  const auto first =
      lpp::ProjectForPpoDiagnostics(response, grid);
  const auto second =
      lpp::ProjectForPpoDiagnostics(response, grid);

  ASSERT_TRUE(lpp::IsOk(first));
  ASSERT_TRUE(lpp::IsOk(second));
  EXPECT_EQ(ProjectionOf(first), ProjectionOf(second));
}

TEST(PythonProjectionTest, RejectsInvalidGridGeometry) {
  const auto response = MakeActivatedResponse(
      lpp::PlatformType::kWheeled, MakeWheelReference());
  auto grid = MakeGrid();
  grid.resolution_m = 0.0;

  const auto result =
      lpp::ProjectForPpoDiagnostics(response, grid);

  ASSERT_FALSE(lpp::IsOk(result));
  EXPECT_EQ(std::get<lpp::Error>(result).code,
            lpp::ErrorCode::kInvalidArgument);
  EXPECT_EQ(std::get<lpp::Error>(result).field_path,
            "grid_geometry.resolution_m");
}
